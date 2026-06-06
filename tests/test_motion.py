"""Planar IK: forward-kinematics consistency and reach limits."""

import math

import pytest

from trash_arm.motion import ReachError, plan_planar_ik

IK = {
    "l1": 0.116,
    "l2": 0.135,
    "base_height": 0.06,
    "wrist_length": 0.05,
    "elbow_up": False,
    "sign": {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1},
    "offset": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0},
}


def _fk_tip(j, ik):
    """Forward kinematics: joint degrees -> gripper tip (x, y, z)."""
    pan = math.radians(j["shoulder_pan"])
    sh = math.radians(j["shoulder_lift"])
    a2 = sh + math.radians(j["elbow_flex"])
    a3 = a2 + math.radians(j["wrist_flex"])
    r = ik["l1"] * math.cos(sh) + ik["l2"] * math.cos(a2) + ik["wrist_length"] * math.cos(a3)
    z = (
        ik["base_height"]
        + ik["l1"] * math.sin(sh)
        + ik["l2"] * math.sin(a2)
        + ik["wrist_length"] * math.sin(a3)
    )
    return r * math.cos(pan), r * math.sin(pan), z, a3


@pytest.mark.parametrize("target", [(0.20, 0.05, 0.02), (0.18, -0.04, 0.05), (0.22, 0.0, 0.0)])
def test_ik_round_trips_through_fk(target):
    x, y, z = target
    joints = plan_planar_ik(x, y, z, IK)
    fx, fy, fz, a3 = _fk_tip(joints, IK)
    assert fx == pytest.approx(x, abs=1e-6)
    assert fy == pytest.approx(y, abs=1e-6)
    assert fz == pytest.approx(z, abs=1e-6)
    # gripper points straight down: last-link angle == -90 deg
    assert math.degrees(a3) == pytest.approx(-90.0, abs=1e-6)


def test_ik_sets_pan_from_xy():
    j = plan_planar_ik(0.12, 0.12, 0.02, IK)  # 45 deg azimuth, within reach
    assert j["shoulder_pan"] == pytest.approx(45.0, abs=1e-6)


def test_unreachable_target_raises():
    with pytest.raises(ReachError):
        plan_planar_ik(1.0, 0.0, 0.0, IK)


def test_sign_and_offset_mapping_applied():
    ik = {**IK, "sign": {**IK["sign"], "shoulder_pan": -1},
          "offset": {**IK["offset"], "shoulder_pan": 10}}
    j = plan_planar_ik(0.12, 0.12, 0.02, ik)
    # raw pan = +45 deg; sign -1, offset +10 -> -35
    assert j["shoulder_pan"] == pytest.approx(-35.0, abs=1e-6)
