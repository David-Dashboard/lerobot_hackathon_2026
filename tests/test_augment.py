"""Synthetic data augmentation: pure helpers (no deps) + dataset round-trip.

The per-frame helpers are pure numpy and always run. The dataset-level test
builds a real LeRobotDataset from the mock arm (like test_record.py) and is
skipped if lerobot isn't installed. The build_observation test needs torch.
"""

import numpy as np
import pytest

from so101.augment import (
    MIRROR_JOINTS,
    _make_variant,
    augment_image,
    augment_trajectory,
    mirror_image,
    mirror_joint_signs,
)
from so101.obs import SO101_JOINTS


# --- pure helpers (numpy only) ---------------------------------------------
def test_mirror_joint_signs_negates_only_lateral_joints():
    signs = mirror_joint_signs()
    assert signs.shape == (len(SO101_JOINTS),)
    for i, name in enumerate(SO101_JOINTS):
        expected = -1.0 if name in MIRROR_JOINTS else 1.0
        assert signs[i] == expected, name


def test_mirror_image_flips_horizontally():
    img = np.arange(2 * 4 * 3, dtype=np.uint8).reshape(2, 4, 3)
    flipped = mirror_image(img)
    assert flipped.shape == img.shape
    assert np.array_equal(flipped, img[:, ::-1])
    assert np.array_equal(mirror_image(flipped), img)  # involution


def test_augment_image_keeps_shape_and_dtype_and_can_change_pixels():
    base = np.tile(np.linspace(0, 255, 40, dtype=np.uint8)[None, :, None], (30, 1, 3))
    changed = False
    for seed in range(10):
        rng = np.random.default_rng(seed)
        out = augment_image(base, rng)
        assert out.shape == base.shape
        assert out.dtype == np.uint8
        if not np.array_equal(out, base):
            changed = True
    assert changed, "augment_image never altered the image across 10 seeds"


def test_augment_trajectory_preserves_shape_and_keeps_action_smoother():
    rng = np.random.default_rng(0)
    t, j = 50, len(SO101_JOINTS)
    state = np.zeros((t, j), dtype=np.float32)
    action = np.zeros((t, j), dtype=np.float32)
    new_state, new_action = augment_trajectory(state, action, rng,
                                               state_scale=1.5, action_scale=0.3)
    assert new_state.shape == state.shape
    assert new_action.shape == action.shape
    # action target is perturbed less than the state input
    assert np.abs(new_action).max() < np.abs(new_state).max()
    # correlated (smooth): step-to-step change stays well under the amplitude
    step_diff = np.abs(np.diff(new_state, axis=0)).max()
    assert step_diff < np.abs(new_state).max()


def test_make_variant_mirror_only_is_exact():
    """Mirror with no noise: lateral joints negated, others identical, images flipped."""
    t, j = 3, len(SO101_JOINTS)
    state = np.random.default_rng(1).standard_normal((t, j)).astype(np.float32)
    action = np.random.default_rng(2).standard_normal((t, j)).astype(np.float32)
    img_key = "observation.images.front"
    images = {img_key: [np.full((4, 6, 3), v, dtype=np.uint8) for v in (10, 20, 30)]}
    ep = {"state": state.copy(), "action": action.copy(), "images": images, "task": "pick"}

    variant = _make_variant(ep, [img_key], np.random.default_rng(0),
                            mirror=True, image_aug=False, traj_aug=False)

    signs = mirror_joint_signs(j)
    assert np.allclose(variant["state"], state * signs)
    assert np.allclose(variant["action"], action * signs)
    for i, name in enumerate(SO101_JOINTS):
        if name in MIRROR_JOINTS:
            assert np.allclose(variant["state"][:, i], -state[:, i])
        else:
            assert np.allclose(variant["state"][:, i], state[:, i])
    for t_i in range(t):
        assert np.array_equal(variant["images"][img_key][t_i],
                              mirror_image(images[img_key][t_i]))
    assert variant["task"] == "pick"


# --- dataset round-trip (needs lerobot) ------------------------------------
def test_augment_dataset_multiplies_and_preserves_schema(tmp_path):
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101.augment import augment_dataset
    from so101.mock import MockArm
    from so101.record import record_dataset

    arm = MockArm()
    arm.connect()
    record_dataset(arm, repo_id="local/aug_src", task="pick up the cube",
                   num_episodes=2, episode_steps=4, root=tmp_path / "src")

    summary = augment_dataset(
        src_repo_id="local/aug_src", src_root=tmp_path / "src",
        out_repo_id="local/aug_out", out_root=tmp_path / "out",
        multiplier=3, keep_original=True, mirror=True, seed=0,
    )

    assert summary["src_episodes"] == 2
    assert summary["out_episodes"] == 2 * (3 + 1)  # original + 3 variants each
    assert summary["out_frames"] == summary["src_frames"] * (3 + 1)

    out = LeRobotDataset(repo_id="local/aug_out", root=tmp_path / "out")
    assert set(out.features) == set(
        LeRobotDataset(repo_id="local/aug_src", root=tmp_path / "src").features
    )
    assert out.num_episodes == 8


# --- observation builder (needs torch) -------------------------------------
def test_build_observation_shapes_and_range():
    pytest.importorskip("torch")
    from so101.deploy import build_observation

    joints = {j: float(i) for i, j in enumerate(SO101_JOINTS)}
    images = {"front": np.full((8, 10, 3), 255, dtype=np.uint8)}
    obs = build_observation(joints, images, device="cpu", task="pick")

    assert tuple(obs["observation.state"].shape) == (1, len(SO101_JOINTS))
    img = obs["observation.images.front"]
    assert tuple(img.shape) == (1, 3, 8, 10)  # batched CHW
    assert float(img.max()) <= 1.0 and float(img.min()) >= 0.0
    assert obs["task"] == ["pick"]
