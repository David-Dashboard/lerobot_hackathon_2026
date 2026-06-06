"""Turn an object's image pixel into an (x, y, z) coordinate in the robot frame.

Assumes the object lies on a known plane (the table, ``z = plane_z`` in world/robot
coords). Back-projects the pixel into a ray and intersects it with that plane.
Combined with the live camera pose from `markers.py`, this is the whole vision
front-end: pixel -> robot-frame coordinate, the only thing the policy ever sees.
Pure numpy.
"""

from __future__ import annotations

import numpy as np


def pixel_to_plane(u: float, v: float, camera_matrix, T_cam_world, plane_z: float = 0.0) -> np.ndarray:
    """Pixel (u, v) -> world point on the plane z = plane_z.

    `T_cam_world` maps world points into the camera frame (as returned by
    `markers.estimate_camera_pose`).
    """
    K = np.asarray(camera_matrix, dtype=float)
    T = np.asarray(T_cam_world, dtype=float)
    R, t = T[:3, :3], T[:3, 3]

    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])  # direction in camera frame
    cam_center_world = -R.T @ t
    ray_world = R.T @ ray_cam
    if abs(ray_world[2]) < 1e-9:
        raise ValueError("camera ray is parallel to the plane")
    s = (plane_z - cam_center_world[2]) / ray_world[2]
    return cam_center_world + s * ray_world


def project_point(point_world, camera_matrix, T_cam_world) -> np.ndarray:
    """Inverse of `pixel_to_plane`: world point -> pixel (u, v). Handy for tests
    and for drawing the estimated target back onto the image."""
    K = np.asarray(camera_matrix, dtype=float)
    T = np.asarray(T_cam_world, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    p_cam = R @ np.asarray(point_world, dtype=float) + t
    if p_cam[2] <= 0:
        raise ValueError("point is behind the camera")
    uvw = K @ p_cam
    return uvw[:2] / uvw[2]
