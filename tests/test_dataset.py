"""Validate the LeRobot data pipeline using a real public SO-101 dataset.

This confirms the format our recorded dataset must match, and that training input
will load. It needs the dataset cached locally (we pulled it during the Rerun demo)
or network access; otherwise it SKIPS rather than fails.
"""

import pytest

REPO_ID = "lerobot/svla_so101_pickplace"


@pytest.fixture(scope="module")
def dataset():
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset(REPO_ID)
    except Exception as e:  # noqa: BLE001 - offline / not cached / hub error
        pytest.skip(f"SO-101 dataset unavailable (offline?): {e}")


def test_has_episodes_and_frames(dataset):
    assert dataset.num_episodes > 0
    assert dataset.num_frames > 0


def test_has_action_and_state_features(dataset):
    feats = set(dataset.features)
    assert "action" in feats, f"no 'action' feature; got {sorted(feats)}"
    # state lives under an 'observation.state' style key
    assert any("state" in f for f in feats), f"no state feature; got {sorted(feats)}"


def test_has_camera_feature(dataset):
    feats = list(dataset.features)
    assert any("image" in f or "observation.images" in f for f in feats), (
        f"expected a camera feature for a vision policy; got {feats}"
    )


def test_first_frame_loads(dataset):
    frame = dataset[0]
    assert "action" in frame
