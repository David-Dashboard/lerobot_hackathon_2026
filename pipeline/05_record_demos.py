"""05 - record demos: teleoperate (leader -> follower) into an ACT-ready dataset.

    python pipeline/05_record_demos.py --name trash_v1 --episodes 40
    python pipeline/05_record_demos.py --name trash_v1 --ignore "Integrated"

Auto-detects which arm is leader/follower and which cameras to record (OAK scene +
wrist by default; tune with --use/--ignore/--scene/--wrist as in step 01). Manual
ENTER-controlled episodes: ENTER starts, ENTER ends, 'q' finishes. Ctrl+C saves the
current episode then stops. Records locally to recorded/<name>/; push + train in 06.

Cameras recorded: ACT learns from the wrist (gripper-relative -> robust) plus the
scene view. The detector/world-coords channel is NOT recorded here (that is the
optional "feed detections into ACT" upgrade -- add it as an observation feature if
you want pose-conditioning; see PIPELINE.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from so101 import arm_id, cameras, make_arm, make_teleop  # noqa: E402
from so101.discovery import LAPTOP_NAME_HINTS  # noqa: E402
from so101.reliability import connect_with_retry, graceful_stop  # noqa: E402
from so101.record import prepare_dataset_dir, record_teleop_dataset  # noqa: E402


def _enter_poller():
    try:
        import msvcrt

        def drain():
            while msvcrt.kbhit():
                msvcrt.getwch()

        def pressed():
            hit = False
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    hit = True
            return hit
    except ImportError:
        import select

        def drain():
            while select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline():
                pass

        def pressed():
            hit = False
            while select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline():
                hit = True
            return hit

    return drain, pressed


def _coerce(keys):
    return [int(k) if str(k).lstrip("-").isdigit() else k for k in (keys or [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="trash_v1")
    ap.add_argument("--task", default="pick up the trash and drop it in the bin")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--use", nargs="+", default=None)
    ap.add_argument("--ignore", nargs="+", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--wrist", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((Path(args.config) if args.config else REPO / "config.yaml").read_text())
    arms = arm_id.identify_arm_roles(cfg)

    roles = {}
    if args.scene is not None:
        roles[int(args.scene) if str(args.scene).isdigit() else args.scene] = "scene"
    if args.wrist is not None:
        roles[int(args.wrist) if str(args.wrist).isdigit() else args.wrist] = "wrist"
    selected = cameras.select_cameras(
        cameras.enumerate_cameras(),
        include=_coerce(args.use),
        exclude=list(LAPTOP_NAME_HINTS) + _coerce(args.ignore),
        roles=roles or None,
    )
    print("Recording cameras:\n" + cameras.describe(selected))
    extra = cameras.open_selected(selected, fps=args.fps)

    repo_id = f"local/{args.name}"
    root = REPO / "recorded" / args.name
    prepare_dataset_dir(repo_id, str(root), args.overwrite)

    follower = make_arm(port=arms["follower"]["port"], arm_id=cfg["robot"]["id"], calibrate=False)
    teleop = make_teleop(port=arms["leader"]["port"], teleop_id=cfg["teleop"]["id"], calibrate=False)

    drain, pressed = _enter_poller()

    def await_start(ep: int) -> bool:
        try:
            r = input(f"\n=== Episode {ep + 1}/{args.episodes} === ENTER to START (or 'q' + ENTER to finish): ")
        except (EOFError, KeyboardInterrupt):
            return False
        if r.strip().lower() == "q":
            return False
        drain()
        print("  recording... move the LEADER. ENTER to END this episode.")
        return True

    summary = None
    try:
        connect_with_retry(follower, "follower")
        connect_with_retry(teleop, "leader")
        for cam in extra.values():
            connect_with_retry(cam, "camera")
        with graceful_stop() as should_stop:
            summary = record_teleop_dataset(
                follower, teleop, repo_id=repo_id, task=args.task,
                cameras={}, extra_cameras=extra,
                num_episodes=args.episodes, episode_steps=None, fps=args.fps,
                root=str(root), overwrite=args.overwrite, push_to_hub=False,
                progress=lambda d, t: print(f"\r  frame {d}", end="", flush=True),
                should_stop=should_stop, await_start=await_start,
                end_episode=lambda step: pressed(),
            )
    finally:
        for dev in (teleop, follower):
            try:
                dev.disconnect()
            except Exception:
                pass
        for cam in extra.values():
            try:
                cam.close()
            except Exception:
                pass
        print("\nDisconnected.")

    if summary and summary.get("num_episodes"):
        print(f"\nRecorded {summary['num_episodes']} episode(s) -> {root}")
        print(f"Next: python pipeline/06_train_act.py --name {args.name}")
    else:
        print("No episodes recorded.")


if __name__ == "__main__":
    main()
