"""The factory must pick the right arm and stay hardware/torch-free for the mock path."""

import pytest

from so101 import MockArm, RobotArm, make_arm


def test_mock_path_returns_interface():
    arm = make_arm(mock=True)
    assert isinstance(arm, RobotArm)
    assert isinstance(arm, MockArm)
    assert arm.is_connected is False  # factory does not connect


def test_real_path_requires_port():
    # Must fail fast WITHOUT importing LeRobot/torch (raises before the lazy import).
    with pytest.raises(ValueError):
        make_arm(mock=False)
