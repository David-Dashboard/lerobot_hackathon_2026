"""07 - deploy: clear the table. Two modes.

    python pipeline/07_deploy.py --dry-run        # SCRIPTED pick, motion DISABLED (rehearse targets)
    python pipeline/07_deploy.py                   # SCRIPTED pick, motion ENABLED (the robust default)
    python pipeline/07_deploy.py --policy outputs/train/trash_v1_act/checkpoints/last/pretrained_model

SCRIPTED mode (default) runs the coord-grasp Orchestrator: zero-shot detect -> ROI
-> table coords -> reach filter -> scripted IK pick into the bin. This is "the
robust idea" deployed and needs no trained model -- the safest thing to demo.

POLICY mode (--policy PATH) runs your trained ACT on the follower via LeRobot, which
builds the camera observation and steps the policy. The exact LeRobot run-a-policy
flags are version-specific (this repo pins LeRobot 0.4.4) -- the command is printed
for you to confirm; pass --go to execute it.

SAFETY: starts slow (speeds from config.safety -- keep max_step_deg low). First
Ctrl+C trips the e-stop and returns home. Keep the workspace clear and the e-stop
reachable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def run_scripted(cfg, *, execute: bool) -> None:
    from build_pipeline import build_detector, build_follower, build_localizer
    from trash_arm.camera import open_scene_camera
    from trash_arm.motion import Motion
    from trash_arm.orchestrator import Orchestrator
    from trash_arm.safety import EStop

    if cfg["calibration"] and build_localizer(cfg) is None:
        sys.exit("No homography -- run pipeline/03_calibrate_scene.py first.")

    estop = EStop()
    estop.install_sigint()
    camera = open_scene_camera(cfg)
    detector = build_detector(cfg)
    localizer = build_localizer(cfg)
    arm = build_follower(cfg)
    motion = Motion(arm=arm, cfg=cfg, estop=estop, execute=execute)
    orch = Orchestrator(camera, detector, localizer, motion, cfg, estop=estop)
    try:
        motion.go_home()
        orch.run()
    except Exception as e:
        print(f"\nstopped: {e}")
    finally:
        try:
            motion.go_home()
        except Exception:
            pass
        camera.close()
        arm.disconnect()
        print("arm disconnected.")


def run_policy(cfg, policy_path: str, *, go: bool) -> None:
    import arm_id  # noqa: F401 -- so101.arm_id via sys.path
    from so101 import arm_id as aid

    arms = aid.identify_arm_roles(cfg)
    r = cfg["robot"]
    cams = cfg["cameras"]
    # LeRobot runs a trained policy on the robot through `lerobot-record` with a
    # --policy.path. Camera/flag syntax is version-specific -- CONFIRM for LeRobot 0.4.4.
    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={arms['follower']['port']}",
        f"--robot.id={r['id']}",
        f"--robot.cameras={{ wrist: {{type: opencv, index_or_path: {cams['wrist']['index']}, "
        f"width: {cams['wrist']['width']}, height: {cams['wrist']['height']}, fps: {cams['wrist']['fps']}}} }}",
        f"--policy.path={policy_path}",
        "--dataset.repo_id=eval/trash_run",
        "--dataset.num_episodes=1",
        "--dataset.single_task=clear the table",
    ]
    print("== LeRobot policy run (confirm flags for your LeRobot version) ==")
    print(" \\\n  ".join(cmd))
    if not go:
        print("\n(dry preview -- re-run with --go to execute)")
        return
    subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default=None, help="path/HF id of a trained ACT policy (POLICY mode)")
    ap.add_argument("--dry-run", action="store_true", help="scripted mode with motion disabled")
    ap.add_argument("--go", action="store_true", help="actually execute the LeRobot policy command")
    ap.add_argument("--yes", action="store_true", help="skip the 'will move the arm' confirmation")
    args = ap.parse_args()

    from trash_arm.config import load_config

    cfg = load_config(args.config)

    if args.policy:
        run_policy(cfg, args.policy, go=args.go)
        return

    if not args.dry_run and not args.yes:
        if input("This will MOVE the arm. Workspace clear, e-stop reachable? [y/N] ").strip().lower() != "y":
            sys.exit("aborted.")
    run_scripted(cfg, execute=not args.dry_run)


if __name__ == "__main__":
    main()
