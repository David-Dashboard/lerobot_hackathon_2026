"""Tests for the hardware-free mock arm, and that it plays nicely with arm_utils."""

import pytest

from arm_utils import SO101_JOINTS, extract_joint_positions
from mock_robot import MockSO101Follower


def test_lifecycle():
    robot = MockSO101Follower()
    assert robot.is_connected is False
    robot.connect()
    assert robot.is_connected is True
    robot.disconnect()
    assert robot.is_connected is False


def test_observation_has_all_joints():
    robot = MockSO101Follower()
    robot.connect()
    obs = robot.get_observation()
    joints = extract_joint_positions(obs)
    assert set(joints) == set(SO101_JOINTS)
    assert all(isinstance(v, float) for v in joints.values())


def test_observation_is_deterministic():
    a = MockSO101Follower(); a.connect()
    b = MockSO101Follower(); b.connect()
    assert a.get_observation() == b.get_observation()


def test_observation_changes_over_time():
    robot = MockSO101Follower()
    robot.connect()
    first = robot.get_observation()
    second = robot.get_observation()
    assert first != second  # joints move (sine waves), so plots animate


def test_send_action_records_command():
    robot = MockSO101Follower()
    robot.connect()
    cmd = {f"{j}.pos": 0.0 for j in SO101_JOINTS}
    robot.send_action(cmd)
    assert robot.last_action == cmd


def test_io_before_connect_raises():
    robot = MockSO101Follower()
    with pytest.raises(RuntimeError):
        robot.get_observation()
    with pytest.raises(RuntimeError):
        robot.send_action({})
