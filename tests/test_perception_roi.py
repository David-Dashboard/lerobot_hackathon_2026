"""Perception geometry helpers (the OWL-ViT model itself isn't exercised here)."""

from trash_arm.perception import Detection, center_of, filter_roi


def test_center_of():
    assert center_of((10, 20, 30, 40)) == (20.0, 30.0)


def _det(cx, cy, label="can"):
    # bbox centered at (cx, cy), 10x10
    return Detection(label=label, confidence=0.5, pixel_xy=(cx, cy),
                     bbox=(cx - 5, cy - 5, cx + 5, cy + 5))


def test_filter_roi_keeps_inside_only():
    roi = {"x_min": 0, "y_min": 0, "x_max": 100, "y_max": 100}
    dets = [_det(50, 50, "in"), _det(150, 50, "out_x"), _det(50, 150, "out_y")]
    kept = filter_roi(dets, roi)
    assert [d.label for d in kept] == ["in"]


def test_filter_roi_none_keeps_all():
    dets = [_det(50, 50), _det(999, 999)]
    assert len(filter_roi(dets, None)) == 2
