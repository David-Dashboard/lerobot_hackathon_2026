"""Synthetic data augmentation for SO-101 ``LeRobotDataset``s.

The motivation: collecting real demonstrations by teleoperation is slow, and a
handful of episodes is too few to train a robust pick-up policy. This module
multiplies an existing dataset into a larger one by emitting several augmented
variants of every episode -- so you get more training data with no extra
teleoperation and no hardware.

What augmentation can and cannot do
-----------------------------------
Augmentation improves *robustness/generalization* from the demos you already
have. It does **not** invent new grasp strategies or object positions that are
absent from the source demos. For genuinely new coverage you need either more
real demos or a physics simulator (see the README "physics sim" note).

The augmentations, in order of value (per the design review):

* **Trajectory jitter (primary):** temporally-correlated, low-frequency noise on
  ``observation.state`` (the policy *input*). The ``action`` target is kept
  smooth -- only a tiny correlated perturbation -- because a policy like ACT
  learns to predict smooth action chunks and independent per-frame action noise
  would just teach it to predict noise.
* **Mirror (opt-in, experimental):** horizontally flip every camera image and
  negate the laterally-symmetric joints (``shoulder_pan``, ``wrist_roll``). This
  is the only augmentation that adds genuinely new spatial coverage, but its
  validity depends on the arm's calibration sign/offset and a centred camera --
  so it is **off by default** and should be eyeballed before training on it.
* **Image jitter (optional, light):** small geometric crop/translate/cutout --
  things LeRobot's train-time transforms don't do by default. Photometric jitter
  is intentionally left to the trainer's ``ImageTransforms`` to avoid baking
  frozen, redundant augmentation onto disk.

Only the dataset read/write functions touch LeRobot; the per-frame helpers are
pure numpy so they stay unit-testable with no heavy dependencies.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .obs import SO101_JOINTS
from .record import prepare_dataset_dir

# Joints that flip sign under a left<->right mirror of the workspace.
MIRROR_JOINTS = ("shoulder_pan", "wrist_roll")


# --- pure per-frame helpers (numpy only; no torch, no lerobot) --------------
def mirror_joint_signs(n_joints: int = len(SO101_JOINTS)) -> np.ndarray:
    """A ``(n_joints,)`` vector of +1/-1: -1 for laterally-symmetric joints.

    Multiplying a state/action row by this mirrors it left<->right.
    """
    signs = np.ones(n_joints, dtype=np.float32)
    for name in MIRROR_JOINTS:
        if name in SO101_JOINTS:
            signs[SO101_JOINTS.index(name)] = -1.0
    return signs


def _correlated_noise(n_steps: int, n_dim: int, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency (temporally-correlated) noise, ``(n_steps, n_dim)``.

    A random walk, mean-removed and scaled, so consecutive frames drift together
    instead of jittering independently -- this keeps trajectories smooth.
    """
    if n_steps == 0:
        return np.zeros((0, n_dim), dtype=np.float32)
    steps = rng.standard_normal((n_steps, n_dim)).astype(np.float32)
    walk = np.cumsum(steps, axis=0)
    walk -= walk.mean(axis=0, keepdims=True)
    denom = np.abs(walk).max(axis=0, keepdims=True)
    denom[denom == 0] = 1.0
    return (walk / denom) * scale


def augment_trajectory(
    state: np.ndarray,
    action: np.ndarray,
    rng: np.random.Generator,
    *,
    state_scale: float = 1.5,
    action_scale: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Perturb a ``(T, J)`` state/action pair with correlated, smooth noise.

    The input ``state`` is perturbed more freely than the supervised ``action``
    target. Scales are in the dataset's joint units (degrees for SO-101).
    """
    state = np.asarray(state, dtype=np.float32)
    action = np.asarray(action, dtype=np.float32)
    t, j = state.shape
    new_state = state + _correlated_noise(t, j, state_scale, rng)
    new_action = action + _correlated_noise(t, j, action_scale, rng)
    return new_state, new_action


def augment_image(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Light *geometric* augmentation of one ``HxWxC`` uint8 image.

    A small random translate (with edge padding) plus an optional cutout patch.
    Photometric jitter is deliberately omitted -- the trainer does that. Always
    returns a uint8 array of the same shape.
    """
    img = np.asarray(img)
    out = img.astype(np.uint8, copy=True)
    h, w = out.shape[:2]

    # small translation: shift up to ~6% of each dimension, pad with edge pixels
    max_dy, max_dx = max(1, h // 16), max(1, w // 16)
    dy = int(rng.integers(-max_dy, max_dy + 1))
    dx = int(rng.integers(-max_dx, max_dx + 1))
    if dy or dx:
        out = np.roll(out, shift=(dy, dx), axis=(0, 1))
        if dy > 0:
            out[:dy] = out[dy:dy + 1]
        elif dy < 0:
            out[dy:] = out[dy - 1:dy]
        if dx > 0:
            out[:, :dx] = out[:, dx:dx + 1]
        elif dx < 0:
            out[:, dx:] = out[:, dx - 1:dx]

    # occasional cutout: zero a small random patch (occlusion robustness)
    if rng.random() < 0.5:
        ph, pw = max(1, h // 8), max(1, w // 8)
        cy = int(rng.integers(0, h - ph + 1))
        cx = int(rng.integers(0, w - pw + 1))
        out[cy:cy + ph, cx:cx + pw] = 0
    return out


def mirror_image(img: np.ndarray) -> np.ndarray:
    """Horizontally flip an ``HxWxC`` image (left<->right)."""
    return np.ascontiguousarray(np.asarray(img)[:, ::-1])


# --- episode I/O (lerobot imported lazily) ----------------------------------
def _load_episode(src, ep_idx: int) -> dict[str, Any]:
    """Load one source episode as raw arrays/images (uint8), keyed for rebuild.

    Reads from ``src.hf_dataset`` -- NOT ``src[i]`` -- so images come back as
    stored uint8 ``HxWxC`` numpy (``src[i]`` would give CHW float[0,1] tensors).
    Returns ``{"state","action","images":{name:list[uint8]},"task"}``.
    """
    bounds = src.episode_data_index
    start = int(bounds["from"][ep_idx])
    end = int(bounds["to"][ep_idx])

    image_keys = [k for k in src.features if k.startswith("observation.images.")]
    states, actions, task = [], [], None
    images: dict[str, list] = {k: [] for k in image_keys}

    for i in range(start, end):
        row = src.hf_dataset[i]
        states.append(np.asarray(row["observation.state"], dtype=np.float32))
        actions.append(np.asarray(row["action"], dtype=np.float32))
        for k in image_keys:
            images[k].append(np.asarray(row[k]).astype(np.uint8))
        if task is None:
            task = _row_task(src, row)

    return {
        "state": np.stack(states) if states else np.zeros((0, len(SO101_JOINTS)), np.float32),
        "action": np.stack(actions) if actions else np.zeros((0, len(SO101_JOINTS)), np.float32),
        "images": {k: v for k, v in images.items()},
        "task": task or "",
    }


def _row_task(src, row: dict) -> str | None:
    """Recover a frame's task string from the hf row (column or task_index map)."""
    if "task" in row:
        return str(row["task"])
    idx = row.get("task_index")
    if idx is not None:
        try:
            tasks = src.meta.tasks
            # tasks may be a list or a mapping index->task
            return str(tasks[int(idx)])
        except Exception:
            return None
    return None


def _write_episode(dataset, ep: dict, image_keys: list[str]) -> None:
    """Append one prepared episode (frame-by-frame) and save it."""
    state, action, images, task = ep["state"], ep["action"], ep["images"], ep["task"]
    for t in range(len(state)):
        frame: dict[str, Any] = {
            "observation.state": state[t].astype(np.float32),
            "action": action[t].astype(np.float32),
            "task": task,
        }
        for k in image_keys:
            frame[k] = images[k][t]
        dataset.add_frame(frame)
    dataset.save_episode()


def _make_variant(ep: dict, image_keys: list[str], rng: np.random.Generator,
                  *, mirror: bool, image_aug: bool, traj_aug: bool) -> dict:
    """Build one augmented copy of a loaded episode."""
    state = ep["state"].copy()
    action = ep["action"].copy()
    images = {k: [np.asarray(img).astype(np.uint8) for img in ep["images"][k]] for k in image_keys}

    if mirror:
        signs = mirror_joint_signs(state.shape[1] if state.size else len(SO101_JOINTS))
        state = state * signs
        action = action * signs
        images = {k: [mirror_image(img) for img in images[k]] for k in image_keys}

    if traj_aug:
        state, action = augment_trajectory(state, action, rng)

    if image_aug:
        images = {k: [augment_image(img, rng) for img in images[k]] for k in image_keys}

    return {"state": state, "action": action, "images": images, "task": ep["task"]}


def augment_dataset(
    *,
    src_repo_id: str,
    out_repo_id: str,
    src_root=None,
    out_root=None,
    multiplier: int = 4,
    mirror: bool = False,
    keep_original: bool = True,
    image_aug: bool = True,
    traj_aug: bool = True,
    seed: int = 0,
    push_to_hub: bool = False,
    overwrite: bool = False,
    progress=None,
) -> dict:
    """Read a source ``LeRobotDataset`` and write a larger, augmented one.

    For each source episode we emit the original (if ``keep_original``) plus
    ``multiplier`` augmented variants. The output schema, fps and ``robot_type``
    are copied from the source, so this works on any dataset (the real
    scene+wrist 8-episode set included) with no hardcoding.

    Returns a summary dict. ``progress(done_episodes, total_episodes)`` is called
    as it runs (for a UI/CLI bar).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(repo_id=src_repo_id, root=src_root)
    image_keys = [k for k in src.features if k.startswith("observation.images.")]

    prepare_dataset_dir(out_repo_id, out_root, overwrite)
    out = LeRobotDataset.create(
        repo_id=out_repo_id,
        fps=src.fps,
        features=src.features,
        root=out_root,
        robot_type=getattr(src.meta, "robot_type", None) or "so101_follower",
        use_videos=False,
    )

    rng = np.random.default_rng(seed)
    per_episode = (1 if keep_original else 0) + multiplier
    total_out_eps = src.num_episodes * per_episode
    done_eps = 0

    for ep_idx in range(src.num_episodes):
        ep = _load_episode(src, ep_idx)
        if keep_original:
            _write_episode(out, ep, image_keys)
            done_eps += 1
            if progress:
                progress(done_eps, total_out_eps)
        for _ in range(multiplier):
            variant = _make_variant(
                ep, image_keys, rng,
                mirror=mirror, image_aug=image_aug, traj_aug=traj_aug,
            )
            _write_episode(out, variant, image_keys)
            done_eps += 1
            if progress:
                progress(done_eps, total_out_eps)

    if push_to_hub:
        out.push_to_hub()

    return {
        "src_repo_id": src_repo_id,
        "out_repo_id": out_repo_id,
        "src_episodes": src.num_episodes,
        "out_episodes": out.num_episodes,
        "src_frames": src.num_frames,
        "out_frames": out.num_frames,
        "multiplier": multiplier,
        "mirror": mirror,
        "kept_original": keep_original,
        "root": str(out.root),
        "pushed": push_to_hub,
    }
