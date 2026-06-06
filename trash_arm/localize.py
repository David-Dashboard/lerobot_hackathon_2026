"""Pixel -> table-plane (X, Y) mapping via a calibrated homography.

The homography is exact only AT the table surface (z = z_table). Tall objects'
tops project to the wrong (X,Y) -- grasp toward the base, or account for height
(see the plan's "parallax" note). Build the homography with
scripts/calibrate_camera.py.

Applying a homography is pure numpy (so this is unit-testable with no OpenCV);
only building one or undistorting needs cv2, imported lazily there.
"""

from __future__ import annotations

import numpy as np


def compute_homography(pixel_points, table_points) -> np.ndarray:
    """Least-squares homography mapping image pixels -> table (X,Y).

    `pixel_points`: Nx2 image coords (u,v) of known markers (N >= 4).
    `table_points`: Nx2 corresponding table coords (X,Y) in metres.
    """
    import cv2

    src = np.asarray(pixel_points, dtype=np.float64)
    dst = np.asarray(table_points, dtype=np.float64)
    if src.shape[0] < 4 or src.shape != dst.shape:
        raise ValueError("need >=4 matching pixel/table point pairs")
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("findHomography failed (degenerate points?)")
    return H


def save_homography(H: np.ndarray, path) -> None:
    np.save(path, np.asarray(H, dtype=np.float64))


def load_homography(path) -> np.ndarray:
    return np.load(path)


def apply_homography(H: np.ndarray, u: float, v: float) -> tuple[float, float]:
    """Map one pixel (u,v) to table (X,Y). Pure numpy."""
    p = np.asarray(H, dtype=np.float64) @ np.array([u, v, 1.0])
    if abs(p[2]) < 1e-12:
        raise ValueError("homography mapped point to infinity")
    return float(p[0] / p[2]), float(p[1] / p[2])


class Localizer:
    """Turns scene-camera pixels into table coordinates, with optional undistortion
    and an ROI gate (so the arm/bin regions can be excluded)."""

    def __init__(self, H, camera_matrix=None, dist_coeffs=None, roi: dict | None = None):
        self.H = np.asarray(H, dtype=np.float64)
        self.camera_matrix = None if camera_matrix is None else np.asarray(camera_matrix)
        self.dist_coeffs = None if dist_coeffs is None else np.asarray(dist_coeffs)
        self.roi = roi

    def _undistort(self, u: float, v: float) -> tuple[float, float]:
        if self.camera_matrix is None or self.dist_coeffs is None:
            return u, v
        import cv2

        pts = np.array([[[u, v]]], dtype=np.float64)
        out = cv2.undistortPoints(pts, self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        return float(out[0, 0, 0]), float(out[0, 0, 1])

    def in_roi(self, u: float, v: float) -> bool:
        if not self.roi:
            return True
        return (
            self.roi["x_min"] <= u <= self.roi["x_max"]
            and self.roi["y_min"] <= v <= self.roi["y_max"]
        )

    def to_table(self, u: float, v: float) -> tuple[float, float]:
        """Pixel (u,v) -> table (X,Y) in metres (undistorting first if configured)."""
        uu, vv = self._undistort(u, v)
        return apply_homography(self.H, uu, vv)
