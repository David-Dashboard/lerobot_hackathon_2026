"""Pure observation-helper tests (no torch, no hardware, no network)."""

from so101.obs import (
    SO101_JOINTS,
    format_joint_line,
    joints_to_action,
    observation_to_joints,
)


def test_observation_to_joints_filters_non_joints():
    obs = {
        "shoulder_pan.pos": 1.0,
        "gripper.pos": 2,           # int -> float
        "front": object(),          # camera frame, ignored
        "task": "pick up the cube",  # metadata, ignored
    }
    assert observation_to_joints(obs) == {"shoulder_pan": 1.0, "gripper": 2.0}


def test_action_roundtrip():
    joints = {"shoulder_pan": 1.5, "gripper": -2.0}
    assert observation_to_joints(joints_to_action(joints)) == joints


def test_all_joints_roundtrip():
    joints = {j: float(i) for i, j in enumerate(SO101_JOINTS)}
    assert observation_to_joints(joints_to_action(joints)) == joints


def test_format_joint_line_readable():
    line = format_joint_line({"shoulder_pan": 12.3, "gripper": -3.0})
    assert "shoulder_pan=" in line and "gripper=" in line


def test_empty_observation():
    assert observation_to_joints({}) == {}
