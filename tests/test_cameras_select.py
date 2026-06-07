"""Camera selection logic -- pure, no hardware."""

from so101.cameras import CameraInfo, select_cameras


def _fixture():
    return [
        CameraInfo(kind="oak", name="OAK-D (OAK-D-PRO)", mxid="14442C10"),
        CameraInfo(kind="uvc", name="USB2.0 HD IR UVC WebCam", index=0),  # laptop
        CameraInfo(kind="uvc", name="Logitech BRIO", index=1),            # wrist-ish
        CameraInfo(kind="uvc", name="Generic USB Camera", index=2),
    ]


def test_laptop_excluded_by_default():
    sel = select_cameras(_fixture())
    assert all("IR UVC" not in c.name for c in sel)
    assert {c.kind for c in sel} == {"oak", "uvc"}


def test_default_roles_oak_is_scene_first_webcam_is_wrist():
    sel = select_cameras(_fixture())
    by_role = {c.role: c for c in sel}
    assert by_role["scene"].kind == "oak"
    assert by_role["wrist"].index == 1  # Logitech, first non-laptop webcam
    assert by_role["side1"].index == 2


def test_include_allowlist_by_index_and_name():
    sel = select_cameras(_fixture(), include=[1, "OAK"])
    keys = {c.key() for c in sel}
    assert keys == {"uvc:1", "oak:14442C10"}


def test_explicit_role_override_wins():
    sel = select_cameras(_fixture(), roles={2: "wrist", "OAK": "scene"})
    by_role = {c.role: c for c in sel}
    assert by_role["wrist"].index == 2
    assert by_role["scene"].kind == "oak"


def test_extra_ignore_drops_named_camera():
    sel = select_cameras(_fixture(), exclude=("IR UVC", "Generic"))
    assert all("Generic" not in c.name for c in sel)
    assert any(c.index == 1 for c in sel)  # Logitech survives
