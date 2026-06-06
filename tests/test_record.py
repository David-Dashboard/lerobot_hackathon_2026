"""Recording a real LeRobotDataset from the mock arm (no hardware, no network)."""

from so101.mock import MockArm
from so101.record import record_dataset


def test_record_creates_valid_vision_dataset(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    arm = MockArm()
    arm.connect()
    summary = record_dataset(
        arm,
        repo_id="local/test_rec",
        task="pick up the cube",
        num_episodes=2,
        episode_steps=4,
        root=tmp_path / "ds",
        push_to_hub=False,
    )
    assert summary["num_episodes"] == 2
    assert summary["num_frames"] == 8

    ds = LeRobotDataset(repo_id="local/test_rec", root=tmp_path / "ds")
    assert ds.num_episodes == 2
    assert ds.num_frames == 8
    feats = set(ds.features)
    assert "action" in feats
    assert "observation.state" in feats
    assert "observation.images.front" in feats  # VLA-trainable
    frame = ds[0]
    assert tuple(frame["observation.state"].shape) == (6,)


def test_progress_callback_is_called(tmp_path):
    arm = MockArm()
    arm.connect()
    seen = []
    record_dataset(
        arm, repo_id="local/test_rec2", task="t",
        num_episodes=1, episode_steps=3, root=tmp_path / "ds2",
        progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1] == (3, 3)
