"""02 - calibrate arms: run LeRobot's joint-range calibration for each arm.

    python pipeline/02_calibrate_arms.py            # auto-detect roles, calibrate both
    python pipeline/02_calibrate_arms.py --only leader

Auto-resolves which port is leader/follower (so you never calibrate the wrong one),
then shells out to LeRobot's interactive `lerobot-calibrate` for each. Calibration
is saved by LeRobot keyed by the arm id from config.yaml and reused everywhere.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from so101 import arm_id  # noqa: E402


def _cfg(path):
    import yaml

    p = Path(path) if path else REPO / "config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", choices=["leader", "follower"], default=None)
    args = ap.parse_args()

    cfg = _cfg(args.config)
    arms = arm_id.identify_arm_roles(cfg)
    fid = cfg["robot"]["id"]
    lid = cfg["teleop"]["id"]

    jobs = []
    if args.only in (None, "follower"):
        jobs.append(["lerobot-calibrate", "--robot.type=so101_follower",
                     f"--robot.port={arms['follower']['port']}", f"--robot.id={fid}"])
    if args.only in (None, "leader"):
        jobs.append(["lerobot-calibrate", "--teleop.type=so101_leader",
                     f"--teleop.port={arms['leader']['port']}", f"--teleop.id={lid}"])

    for cmd in jobs:
        print(f"\n== {' '.join(cmd)} ==")
        print("Follow the prompts: move each joint through its full range, then set the middle pose.")
        subprocess.call(cmd)
    print("\nCalibration done. Next: python pipeline/03_calibrate_scene.py")


if __name__ == "__main__":
    main()
