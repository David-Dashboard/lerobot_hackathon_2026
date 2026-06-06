"""
Evaluate a trained policy OFFLINE -- no robot required.

With no hardware you can't measure task success directly, but you can measure how
well the policy reproduces a held-out demonstration: replay the episode frame by
frame, ask the policy for an action at each step, and compare to the recorded
("ground-truth") action. This gives a per-joint / overall error that **ranks
checkpoints** -- e.g. "did augmenting the 8 episodes lower the error?" -- and
produces predicted-vs-true trajectory plots.

Caveats
-------
This is *open-loop next-action error*: the policy is fed the real recorded states,
not its own rollout, so there's no compounding error and no physics. A low number
means "predicts the demo's actions well", NOT "grasps successfully". Use it to
compare checkpoints, then confirm real performance on the arm when available.

The eval episode MUST be one you held OUT of augmentation/training, or the number
just measures memorization. With 8 episodes, reserve 1-2 as the holdout.

Example
-------
    python eval_offline.py \
      --checkpoint outputs/train/demos_aug_act/checkpoints/last/pretrained_model \
      --dataset-root recorded/demos --dataset-repo-id local/demos \
      --episode-index 7 --out-dir outputs/eval/demos_ep7
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless (Colab/Qualia) -- must be set before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from so101.augment import _load_episode  # shared raw-frame loader  # noqa: E402
from so101.deploy import LeRobotPolicy  # noqa: E402
from so101.obs import SO101_JOINTS  # noqa: E402


def eval_offline(
    *,
    checkpoint: str,
    dataset_repo_id: str,
    episode_index: int,
    out_dir: str,
    dataset_root=None,
    device: str = "cpu",
) -> dict:
    """Roll a policy open-loop over one held-out episode; report action error."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    src = LeRobotDataset(repo_id=dataset_repo_id, root=dataset_root)
    if not (0 <= episode_index < src.num_episodes):
        raise IndexError(
            f"episode_index {episode_index} out of range (dataset has {src.num_episodes})"
        )
    ep = _load_episode(src, episode_index)
    image_keys = [k for k in src.features if k.startswith("observation.images.")]

    policy = LeRobotPolicy(checkpoint, device=device, task=ep["task"] or None)
    policy.reset()

    n = len(ep["state"])
    preds = np.zeros((n, len(SO101_JOINTS)), dtype=np.float32)
    for t in range(n):
        joints = {j: float(ep["state"][t][i]) for i, j in enumerate(SO101_JOINTS)}
        images = {k: ep["images"][k][t] for k in image_keys}
        action = policy.select_action(joints, images)
        preds[t] = [action[j] for j in SO101_JOINTS]

    truth = ep["action"].astype(np.float32)
    err = preds - truth
    per_joint_mse = (err ** 2).mean(axis=0)
    per_joint_mae = np.abs(err).mean(axis=0)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "checkpoint": checkpoint,
        "dataset_repo_id": dataset_repo_id,
        "episode_index": episode_index,
        "num_frames": n,
        "overall_mse": float((err ** 2).mean()),
        "overall_mae": float(np.abs(err).mean()),
        "per_joint_mse": {j: float(v) for j, v in zip(SO101_JOINTS, per_joint_mse)},
        "per_joint_mae": {j: float(v) for j, v in zip(SO101_JOINTS, per_joint_mae)},
        "note": "open-loop next-action error (not task success); compare across checkpoints",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(preds, truth, out / "trajectories.png", episode_index)

    print(json.dumps(summary, indent=2))
    print(f"\n  wrote {out / 'summary.json'} and {out / 'trajectories.png'}")
    return summary


def _plot(preds: np.ndarray, truth: np.ndarray, path: Path, episode_index: int) -> None:
    """One subplot per joint: predicted vs ground-truth action over time."""
    j = len(SO101_JOINTS)
    fig, axes = plt.subplots(j, 1, figsize=(9, 1.8 * j), sharex=True)
    steps = np.arange(len(truth))
    for i, ax in enumerate(np.atleast_1d(axes)):
        ax.plot(steps, truth[:, i], label="ground truth", linewidth=1.5)
        ax.plot(steps, preds[:, i], label="predicted", linewidth=1.2, linestyle="--")
        ax.set_ylabel(SO101_JOINTS[i], fontsize=8)
        ax.grid(True, alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("frame")
    fig.suptitle(f"Open-loop action prediction -- episode {episode_index}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate a trained policy offline against a held-out episode.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True,
                   help="trained policy dir (…/pretrained_model) or HF repo id")
    p.add_argument("--dataset-repo-id", required=True, help="evaluation dataset repo id")
    p.add_argument("--dataset-root", default=None,
                   help="dataset dir (omit to use the Hub / LeRobot cache)")
    p.add_argument("--episode-index", type=int, required=True,
                   help="held-out episode to evaluate (exclude it from training!)")
    p.add_argument("--out-dir", required=True, help="where to write summary.json + plot")
    p.add_argument("--device", default="cpu", help="torch device for inference")
    return p


def main() -> None:
    args = build_parser().parse_args()
    eval_offline(
        checkpoint=args.checkpoint,
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        episode_index=args.episode_index,
        out_dir=args.out_dir,
        device=args.device,
    )


if __name__ == "__main__":
    main()
