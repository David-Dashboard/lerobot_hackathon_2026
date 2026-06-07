"""06 - train ACT on the recorded demos.

    python pipeline/06_train_act.py --name trash_v1                 # train locally (CPU/GPU)
    python pipeline/06_train_act.py --name trash_v1 --steps 60000
    python pipeline/06_train_act.py --name trash_v1 --qualia        # cloud (Qualia)
    python pipeline/06_train_act.py --name trash_v1 --push          # push dataset to HF first

Default trains ACT locally via LeRobot's CLI from recorded/<name>/. --qualia pushes
to the Hub and launches a managed finetune instead (spends credits). Either way the
output is a policy you point 07_deploy.py at.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Robust HF uploads: skip Xet (stalls on flaky networks) -> mature LFS multipart.
# Must be set before huggingface_hub is imported. Override with HF_HUB_DISABLE_XET=0.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _hf_user():
    from huggingface_hub import HfApi

    return HfApi().whoami()["name"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--push", action="store_true", help="push the dataset to HF before training")
    ap.add_argument("--qualia", action="store_true", help="train on Qualia cloud instead of locally")
    ap.add_argument("--hours", type=float, default=2.0)
    args = ap.parse_args()

    root = REPO / "recorded" / args.name
    if not root.exists():
        sys.exit(f"no dataset at {root} -- record it first (05_record_demos.py).")

    if args.qualia:
        from so101 import qualia_client as q
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id = f"{_hf_user()}/{args.name}"
        ds = LeRobotDataset(repo_id=repo_id, root=str(root))
        img_keys = [k for k in ds.features if k.startswith("observation.images.")]
        ds.push_to_hub(upload_large_folder=True)  # resumable + per-file retries
        cam_map = {}
        pref = {"scene": "image_top", "wrist": "image_wrist"}
        slots = ["image_top", "image_wrist", "image_side"]
        for k in img_keys:
            name = k.rsplit(".", 1)[-1]
            slot = pref.get(name) or next((s for s in slots if s not in cam_map), None)
            if slot:
                cam_map[slot] = k
        print(f"Launching Qualia ACT finetune on {repo_id} ({args.hours}h), cameras {cam_map}")
        if input("Type 'yes' to spend credits: ").strip().lower() != "yes":
            sys.exit("aborted.")
        print(q.launch_finetune(dataset_id=repo_id, vla_type="act", model_id=None,
                                hours=args.hours, camera_mappings=cam_map))
        return

    repo_id = f"{_hf_user()}/{args.name}" if args.push else f"local/{args.name}"
    if args.push:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(repo_id=repo_id, root=str(root)).push_to_hub(upload_large_folder=True)

    out = REPO / "outputs" / "train" / f"{args.name}_act"
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={root}",
        "--policy.type=act", f"--policy.device={args.device}",
        f"--output_dir={out}", f"--steps={args.steps}",
        f"--batch_size={args.batch_size}", "--num_workers=0",
        "--save_freq=2000", "--wandb.enable=false",
    ]
    print("== " + " ".join(cmd) + " ==")
    subprocess.call(cmd)
    print(f"\nTrained policy under {out}")
    print(f"Next: python pipeline/07_deploy.py --policy {out}/checkpoints/last/pretrained_model --dry-run")


if __name__ == "__main__":
    main()
