"""Re-encode an existing IMAGE (inline-PNG) LeRobotDataset to VIDEO (MP4).

Inline-PNG datasets are ~50-100x larger than video, which is why a multi-GB
push_to_hub stalls. This rebuilds the dataset with use_videos=True (MP4-encoded
cameras) so it uploads in seconds.

    python pipeline/reencode_video.py --name batch2              # -> recorded/batch2_video
    python pipeline/reencode_video.py --name batch2 --push       # + push to HF (robust)
    python pipeline/reencode_video.py --name batch2 --dst b2v --overwrite

Then train on the small dataset, e.g.:
    python pipeline/06_train_act.py --name batch2_video --qualia
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # robust uploads; before any HF import
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _dir_mb(p: Path) -> float:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="source dataset under recorded/")
    ap.add_argument("--dst", default=None, help="output name (default: <name>_video)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--push", action="store_true", help="push the video dataset to HF afterwards")
    ap.add_argument("--hf-user", default=None)
    args = ap.parse_args()

    import numpy as np
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101.record import build_teleop_features, prepare_dataset_dir

    src_root = REPO / "recorded" / args.name
    if not src_root.exists():
        sys.exit(f"no dataset at {src_root}")
    dst_name = args.dst or f"{Path(args.name).name}_video"
    dst_root = REPO / "recorded" / dst_name

    print(f"Loading source {src_root} ...")
    src = LeRobotDataset(repo_id=f"local/{Path(args.name).name}", root=str(src_root))
    img_keys = [k for k in src.features if k.startswith("observation.images.")]
    if not img_keys:
        sys.exit("source has no camera images -- nothing to re-encode.")

    cams = {}
    for k in img_keys:
        shp = list(src.features[k]["shape"])
        h, w = (shp[1], shp[2]) if shp[0] == 3 else (shp[0], shp[1])  # CHW or HWC
        cams[k.rsplit(".", 1)[-1]] = {"width": int(w), "height": int(h)}

    prepare_dataset_dir(f"local/{dst_name}", str(dst_root), args.overwrite)
    dst = LeRobotDataset.create(
        repo_id=f"local/{dst_name}", fps=src.fps,
        features=build_teleop_features(cams, use_videos=True),
        root=str(dst_root), robot_type="so101_follower", use_videos=True,
    )

    n = src.num_frames
    print(f"Re-encoding {src.num_episodes} episode(s) / {n} frames -> video ...")
    prev_ep = None
    for i in range(n):
        f = src[i]
        ep = int(f["episode_index"])
        if prev_ep is not None and ep != prev_ep:
            dst.save_episode()
        frame = {
            "observation.state": f["observation.state"].numpy().astype(np.float32),
            "action": f["action"].numpy().astype(np.float32),
            "task": f.get("task", "task"),
        }
        for k in img_keys:
            chw = f[k]  # float CHW in [0,1]
            frame[k] = (chw.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        dst.add_frame(frame)
        prev_ep = ep
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{n}")
    dst.save_episode()

    print(f"\nDone: {src_root.name} {_dir_mb(src_root):.0f} MB  ->  {dst_name} {_dir_mb(dst_root):.0f} MB")

    if args.push:
        from huggingface_hub import HfApi

        user = args.hf_user or HfApi().whoami()["name"]
        repo = f"{user}/{dst_name}"
        print(f"Pushing {repo} (Xet off, resumable) ...")
        LeRobotDataset(repo_id=repo, root=str(dst_root)).push_to_hub(upload_large_folder=True)
        print(f"Pushed: https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
