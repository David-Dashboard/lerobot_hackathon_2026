"""
Augment a recorded LeRobotDataset into a larger one -- no robot, no teleoperation.

Collecting demonstrations by teleoperation is slow, and a handful of episodes is
too few to train a robust pick-up policy. This multiplies an existing dataset by
emitting several augmented variants of every episode (trajectory jitter, optional
left<->right mirror, light geometric image augmentation). Train ACT/SmolVLA on the
result; evaluate with ``eval_offline.py``.

See ``so101/augment.py`` for what augmentation can and cannot do -- in short, it
improves robustness from the demos you have; it does not invent new grasp poses.

Examples
--------
Local dataset on disk, 8 episodes -> ~40 (8 originals + 4 variants each):

    python augment_dataset.py \
      --src-root recorded/demos --src-repo-id local/demos \
      --out-root recorded/demos_aug --out-repo-id local/demos_aug \
      --multiplier 4

Pull the source from the Hub and push the augmented set back (required for the
Qualia training path, which trains from a Hub dataset_id):

    python augment_dataset.py \
      --src-repo-id you/demos --out-repo-id you/demos_aug \
      --multiplier 4 --push-to-hub

Add the experimental left<->right mirror (doubles spatial coverage -- VERIFY one
mirrored episode in Rerun before training on it):

    python augment_dataset.py --src-root recorded/demos --src-repo-id local/demos \
      --out-root recorded/demos_aug --out-repo-id local/demos_aug --mirror
"""

from __future__ import annotations

import argparse

from so101.augment import augment_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Augment a LeRobotDataset into a larger training set (no hardware).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src-repo-id", required=True, help="source dataset repo id")
    p.add_argument("--src-root", default=None,
                   help="source dataset dir (omit to use the Hub / LeRobot cache)")
    p.add_argument("--out-repo-id", required=True, help="output dataset repo id")
    p.add_argument("--out-root", default=None,
                   help="output dataset dir (omit to use the LeRobot cache)")
    p.add_argument("--multiplier", type=int, default=4,
                   help="augmented variants emitted per source episode")
    p.add_argument("--mirror", action="store_true",
                   help="EXPERIMENTAL: also left<->right mirror (flip images + negate "
                        "shoulder_pan/wrist_roll). Verify one episode before trusting it.")
    p.add_argument("--no-keep-original", dest="keep_original", action="store_false",
                   help="drop the unaugmented originals from the output")
    p.add_argument("--no-image-aug", dest="image_aug", action="store_false",
                   help="skip geometric image augmentation (trajectory aug only)")
    p.add_argument("--no-traj-aug", dest="traj_aug", action="store_false",
                   help="skip trajectory jitter (image aug only)")
    p.add_argument("--push-to-hub", action="store_true",
                   help="push the augmented dataset to the Hub (needed for Qualia)")
    p.add_argument("--overwrite", action="store_true",
                   help="replace the output dir if it already exists")
    p.add_argument("--seed", type=int, default=0, help="RNG seed for reproducibility")
    return p


def main() -> None:
    args = build_parser().parse_args()

    if args.mirror:
        print(
            "\n  [!] --mirror is EXPERIMENTAL. It negates shoulder_pan/wrist_roll and flips\n"
            "      every image. Validity depends on the arm's calibration sign/offset and a\n"
            "      centred camera. Open ONE mirrored episode in Rerun and sanity-check it\n"
            "      before training on this dataset.\n"
        )

    def progress(done: int, total: int) -> None:
        print(f"\r  episodes {done}/{total}", end="", flush=True)

    summary = augment_dataset(
        src_repo_id=args.src_repo_id,
        src_root=args.src_root,
        out_repo_id=args.out_repo_id,
        out_root=args.out_root,
        multiplier=args.multiplier,
        mirror=args.mirror,
        keep_original=args.keep_original,
        image_aug=args.image_aug,
        traj_aug=args.traj_aug,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        overwrite=args.overwrite,
        progress=progress,
    )
    print("\n\n  Done.")
    print(f"    episodes: {summary['src_episodes']} -> {summary['out_episodes']}")
    print(f"    frames:   {summary['src_frames']} -> {summary['out_frames']}")
    print(f"    mirror:   {'on' if summary['mirror'] else 'off'}    "
          f"kept originals: {summary['kept_original']}")
    print(f"    root:     {summary['root']}")
    if summary["pushed"]:
        print(f"    pushed to Hub as: {summary['out_repo_id']}")
    else:
        print("    (not pushed; add --push-to-hub to train on Qualia)")


if __name__ == "__main__":
    main()
