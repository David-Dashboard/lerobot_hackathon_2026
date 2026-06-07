"""Homography pixel->table mapping (pure numpy + cv2 fit)."""

import numpy as np

from trash_arm.localize import (
    Localizer,
    apply_homography,
    compute_homography,
)


def test_apply_identity_homography_is_passthrough():
    H = np.eye(3)
    assert apply_homography(H, 12.0, 34.0) == (12.0, 34.0)


def test_compute_and_apply_round_trip():
    # A square on the table seen as a square in pixels -> recover the mapping.
    pix = [[100, 100], [300, 100], [300, 300], [100, 300]]
    tab = [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]]
    H = compute_homography(pix, tab)
    for (u, v), (tx, ty) in zip(pix, tab):
        x, y = apply_homography(H, u, v)
        assert abs(x - tx) < 1e-6
        assert abs(y - ty) < 1e-6
    # A point in the middle maps to the middle of the table.
    mx, my = apply_homography(H, 200, 200)
    assert abs(mx - 0.1) < 1e-6 and abs(my - 0.1) < 1e-6


def test_compute_homography_requires_four_points():
    import pytest

    with pytest.raises(ValueError):
        compute_homography([[0, 0], [1, 0], [1, 1]], [[0, 0], [1, 0], [1, 1]])


def test_localizer_roi_gate():
    loc = Localizer(np.eye(3), roi={"x_min": 10, "y_min": 10, "x_max": 100, "y_max": 100})
    assert loc.in_roi(50, 50) is True
    assert loc.in_roi(5, 50) is False
    assert loc.in_roi(50, 200) is False


def test_localizer_to_table_without_undistort():
    H = compute_homography(
        [[100, 100], [300, 100], [300, 300], [100, 300]],
        [[0.0, 0.0], [0.2, 0.0], [0.2, 0.2], [0.0, 0.2]],
    )
    loc = Localizer(H)
    x, y = loc.to_table(200, 200)
    assert abs(x - 0.1) < 1e-6 and abs(y - 0.1) < 1e-6
