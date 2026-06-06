"""Teleoperation data-gathering: mock leader + the record loop, no hardware."""

import numpy as np
import pytest

from so101 import MockArm, MockTeleop, SO101_JOINTS, Teleoperator, make_teleop
from so101.record import (
    build_teleop_features,
    parse_camera_spec,
    record_teleop_dataset,
)


def test_mock_teleop_satisfies_interface():
    assert isinstance(MockTeleop(), Teleoperator)


def test_mock_teleop_lifecycle_and_read():
    t = MockTeleop()
    assert t.is_connected is False
    t.connect()
    assert t.is_connected is True
    action = t.read_action()
    assert set(action) == set(SO101_JOINTS)
    assert all(isinstance(v, float) for v in action.values())
    t.disconnect()
    assert t.is_connected is False


def test_mock_teleop_read_before_connect_raises():
    with pytest.raises(RuntimeError):
        MockTeleop().read_action()


def test_make_teleop_requires_port_for_real():
    with pytest.raises(ValueError):
        make_teleop(mock=False, port=None)


def test_leader_action_differs_from_follower_state():
    # A leader should "lead": its action must not be identical to the follower pose.
    arm, leader = MockArm(), MockTeleop()
    arm.connect()
    leader.connect()
    assert arm.read_joints() != leader.read_action()


def test_parse_camera_spec_variants():
    name, cfg = parse_camera_spec("scene=0")
    assert name == "scene"
    assert cfg == {"index": 0, "width": 640, "height": 480, "fps": 30}

    _, cfg = parse_camera_spec("wrist=2:1280x720@15")
    assert cfg == {"index": 2, "width": 1280, "height": 720, "fps": 15}

    # Non-integer index (e.g. a device path) stays a string.
    _, cfg = parse_camera_spec("usb=/dev/video0")
    assert cfg["index"] == "/dev/video0"


def test_parse_camera_spec_rejects_bad_input():
    with pytest.raises(ValueError):
        parse_camera_spec("noequalssign")


def test_build_teleop_features_has_one_image_per_camera():
    cameras = {"scene": {"width": 64, "height": 48}, "wrist": {"width": 32, "height": 24}}
    feats = build_teleop_features(cameras)
    assert feats["observation.images.scene"]["shape"] == (48, 64, 3)
    assert feats["observation.images.wrist"]["shape"] == (24, 32, 3)
    assert feats["action"]["shape"] == (len(SO101_JOINTS),)


def test_record_teleop_creates_valid_vision_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cameras = {"scene": {"index": 0, "width": 64, "height": 48, "fps": 30}}
    summary = record_teleop_dataset(
        MockArm(),
        MockTeleop(),
        repo_id="local/test_teleop",
        task="pick up the trash",
        cameras=cameras,
        num_episodes=2,
        episode_steps=4,
        reset_steps=2,  # exercised but never recorded
        root=tmp_path / "ds",
        synthetic_frames=True,
    )
    assert summary["num_episodes"] == 2
    assert summary["num_frames"] == 8  # reset frames are not recorded
    assert summary["stopped_early"] is False

    ds = LeRobotDataset(repo_id="local/test_teleop", root=tmp_path / "ds")
    assert ds.num_episodes == 2
    assert ds.num_frames == 8
    feats = set(ds.features)
    assert {"action", "observation.state", "observation.images.scene"} <= feats
    frame = ds[0]
    assert tuple(frame["observation.state"].shape) == (len(SO101_JOINTS),)


def test_record_teleop_progress_and_stop(tmp_path):
    seen = []
    # Stop right after the first frame; should end cleanly with one episode saved.
    state = {"n": 0}

    def should_stop():
        state["n"] += 1
        return state["n"] >= 1

    summary = record_teleop_dataset(
        MockArm(),
        MockTeleop(),
        repo_id="local/test_teleop_stop",
        task="t",
        cameras={},
        num_episodes=3,
        episode_steps=5,
        root=tmp_path / "ds2",
        progress=lambda done, total: seen.append((done, total)),
        should_stop=should_stop,
    )
    assert summary["stopped_early"] is True
    assert summary["num_episodes"] == 1
    assert seen[0] == (1, 15)


def test_record_teleop_manual_episode_control(tmp_path):
    # Simulate ENTER-controlled mode: await_start gates episodes, end_episode ends them.
    starts = []

    def await_start(ep):
        starts.append(ep)
        return ep < 2  # run episodes 0 and 1, then "quit" at episode 2

    def end_episode(step):
        return step >= 3  # operator presses ENTER after 3 frames

    summary = record_teleop_dataset(
        MockArm(),
        MockTeleop(),
        repo_id="local/test_manual",
        task="t",
        cameras={},
        num_episodes=5,
        episode_steps=None,  # unbounded: rely on end_episode
        root=tmp_path / "ds_manual",
        await_start=await_start,
        end_episode=end_episode,
    )
    assert starts == [0, 1, 2]
    assert summary["num_episodes"] == 2
    assert summary["num_frames"] == 6  # 2 episodes x 3 frames
