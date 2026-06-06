"""Tests for the mock arm and its conformance to the RobotArm interface."""

import pytest

from so101 import SO101_JOINTS, MockArm, RobotArm


def test_mock_satisfies_interface():
    assert isinstance(MockArm(), RobotArm)


def test_lifecycle():
    arm = MockArm()
    assert arm.is_connected is False
    arm.connect()
    assert arm.is_connected is True
    arm.disconnect()
    assert arm.is_connected is False


def test_read_returns_all_joints():
    arm = MockArm()
    arm.connect()
    joints = arm.read_joints()
    assert set(joints) == set(SO101_JOINTS)
    assert all(isinstance(v, float) for v in joints.values())


def test_read_is_deterministic():
    a, b = MockArm(), MockArm()
    a.connect()
    b.connect()
    assert a.read_joints() == b.read_joints()


def test_read_changes_over_time():
    arm = MockArm()
    arm.connect()
    assert arm.read_joints() != arm.read_joints()


def test_write_records_command():
    arm = MockArm()
    arm.connect()
    cmd = {j: 0.0 for j in SO101_JOINTS}
    arm.write_joints(cmd)
    assert arm.last_command == cmd


def test_io_before_connect_raises():
    arm = MockArm()
    with pytest.raises(RuntimeError):
        arm.read_joints()
    with pytest.raises(RuntimeError):
        arm.write_joints({})
