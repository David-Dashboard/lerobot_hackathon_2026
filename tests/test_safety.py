"""Workspace/joint limits, speed cap, e-stop (pure)."""

import pytest

from trash_arm import safety

WS = {"x_min": 0.1, "x_max": 0.3, "y_min": -0.15, "y_max": 0.15, "z_floor": 0.01}
LIMITS = {"shoulder_pan": [-110, 110], "gripper": [0, 100]}


def test_in_workspace():
    assert safety.in_workspace(0.2, 0.0, WS) is True
    assert safety.in_workspace(0.05, 0.0, WS) is False
    assert safety.in_workspace(0.2, 0.5, WS) is False


def test_require_workspace_raises_outside():
    safety.require_workspace(0.2, 0.0, WS)  # ok
    with pytest.raises(safety.SafetyError):
        safety.require_workspace(0.0, 0.0, WS)


def test_clamp_z_respects_floor():
    assert safety.clamp_z(-0.05, WS) == 0.01
    assert safety.clamp_z(0.08, WS) == 0.08


def test_clamp_and_check_joints():
    clamped = safety.clamp_joints({"shoulder_pan": 200, "gripper": -10, "wrist_roll": 999}, LIMITS)
    assert clamped["shoulder_pan"] == 110
    assert clamped["gripper"] == 0
    assert clamped["wrist_roll"] == 999  # no limit -> passes through
    with pytest.raises(safety.SafetyError):
        safety.check_joints({"shoulder_pan": 200}, LIMITS)


def test_step_toward_caps_delta():
    out = safety.step_toward({"a": 0.0}, {"a": 100.0}, max_step_deg=6.0)
    assert out["a"] == 6.0
    out2 = safety.step_toward({"a": 0.0}, {"a": -100.0}, max_step_deg=6.0)
    assert out2["a"] == -6.0


def test_interpolate_reaches_target():
    wps = safety.interpolate({"a": 0.0}, {"a": 20.0}, max_step_deg=6.0)
    assert wps[-1]["a"] == pytest.approx(20.0)
    # every step is within the cap
    prev = 0.0
    for wp in wps:
        assert abs(wp["a"] - prev) <= 6.0 + 1e-9
        prev = wp["a"]


def test_estop():
    e = safety.EStop()
    e.check()  # not tripped -> no raise
    e.trip()
    assert e.tripped is True
    with pytest.raises(safety.SafetyError):
        e.check()
