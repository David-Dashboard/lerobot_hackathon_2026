"""Unit tests for the pure observation helpers (no hardware, no imports of torch/lerobot)."""

from arm_utils import SO101_JOINTS, extract_joint_positions, format_joint_line


def test_extract_ignores_non_joint_keys():
    obs = {
        "shoulder_pan.pos": 1.0,
        "gripper.pos": 2,            # int -> coerced to float
        "front": object(),          # a camera frame, must be ignored
        "task": "pick up the cube",  # metadata, must be ignored
    }
    joints = extract_joint_positions(obs)
    assert joints == {"shoulder_pan": 1.0, "gripper": 2.0}
    assert all(isinstance(v, float) for v in joints.values())


def test_extract_strips_pos_suffix_for_all_joints():
    obs = {f"{j}.pos": float(i) for i, j in enumerate(SO101_JOINTS)}
    joints = extract_joint_positions(obs)
    assert set(joints) == set(SO101_JOINTS)


def test_format_joint_line_is_readable():
    line = format_joint_line({"shoulder_pan": 12.345, "gripper": -3.0})
    assert "shoulder_pan=" in line
    assert "gripper=" in line


def test_empty_observation_yields_empty():
    assert extract_joint_positions({}) == {}
