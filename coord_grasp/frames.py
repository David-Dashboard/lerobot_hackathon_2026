"""Minimal SE(3) rigid-transform math (4x4 homogeneous matrices). Pure numpy.

Convention: a transform ``T_a_b`` maps a point expressed in frame *b* into frame
*a*:  ``p_a = apply_transform(T_a_b, p_b)``. Composition reads right-to-left:
``T_a_c = compose(T_a_b, T_b_c)``.
"""

from __future__ import annotations

import numpy as np


def make_transform(R, t) -> np.ndarray:
    """Build a 4x4 transform from a 3x3 rotation and a 3-vector translation."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_transform(T) -> np.ndarray:
    """Inverse of a rigid transform (cheaper + more stable than np.linalg.inv)."""
    T = np.asarray(T, dtype=float)
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def compose(*transforms) -> np.ndarray:
    """Chain transforms left-to-right: compose(A, B, C) == A @ B @ C."""
    out = np.eye(4)
    for T in transforms:
        out = out @ np.asarray(T, dtype=float)
    return out


def apply_transform(T, points) -> np.ndarray:
    """Apply T to a point (3,) or a set of points (N,3). Returns same leading shape."""
    pts = np.atleast_2d(np.asarray(points, dtype=float))
    homog = np.hstack([pts, np.ones((pts.shape[0], 1))])
    out = (np.asarray(T, dtype=float) @ homog.T).T[:, :3]
    return out[0] if out.shape[0] == 1 else out


def rvec_tvec_to_T(rvec, tvec) -> np.ndarray:
    """OpenCV (rvec, tvec) pose -> 4x4 transform (lazy cv2 for the Rodrigues map)."""
    import cv2

    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float))
    return make_transform(R, np.asarray(tvec, dtype=float).reshape(3))
