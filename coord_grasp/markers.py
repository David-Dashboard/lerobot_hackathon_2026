"""ArUco marker detection + live camera-pose estimation.

Stick markers at KNOWN positions in the robot/table frame; this estimates the
camera extrinsics (``T_cam_world``) from them every frame via solvePnP. Because
it's recomputed live, the system is camera-pose-agnostic: move the camera and the
next frame re-estimates the pose. cv2 is imported lazily.
"""

from __future__ import annotations

import numpy as np

from . import frames


def make_detector(dict_name: str = "DICT_4X4_50"):
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    return cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())


def detect_markers(image, detector=None):
    """Return (corners, ids) for an image (grayscale or color)."""
    import cv2

    detector = detector or make_detector()
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def square_marker_corners(center, side: float) -> np.ndarray:
    """The 4 corner coords (world frame, z=0 plane) of a flat square marker,
    in OpenCV's order: top-left, top-right, bottom-right, bottom-left."""
    cx, cy = center[0], center[1]
    h = side / 2.0
    return np.array([
        [cx - h, cy + h, 0.0],
        [cx + h, cy + h, 0.0],
        [cx + h, cy - h, 0.0],
        [cx - h, cy - h, 0.0],
    ], dtype=np.float64)


def estimate_camera_pose(corners, ids, marker_world_corners, camera_matrix, dist_coeffs):
    """Estimate ``T_cam_world`` (maps world points into the camera frame) from
    detected markers whose world corners are known.

    marker_world_corners: dict ``{marker_id: (4,3) world corner coords}``.
    Raises ValueError if no known marker is in view.
    """
    import cv2

    if ids is None:
        raise ValueError("no markers detected")
    obj_pts, img_pts = [], []
    for c, i in zip(corners, ids.flatten()):
        if int(i) in marker_world_corners:
            obj_pts.append(np.asarray(marker_world_corners[int(i)], dtype=np.float64))
            img_pts.append(np.asarray(c, dtype=np.float64).reshape(4, 2))
    if not obj_pts:
        raise ValueError("no known markers in view")

    ok, rvec, tvec = cv2.solvePnP(
        np.vstack(obj_pts), np.vstack(img_pts),
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(dist_coeffs, dtype=np.float64),
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise ValueError("solvePnP failed to estimate camera pose")
    return frames.rvec_tvec_to_T(rvec, tvec)
