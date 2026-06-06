"""The real autonomous trash-clearing loop (milestone M5+).

  python scripts/run.py            # clear the table into the bin, then stop

SAFETY: starts slow (speeds come from config.safety -- keep max_step_deg low until
you trust it). Ctrl+C trips the e-stop and the arm returns home. Before running
this, get green results from:
  1. scripts/calibrate_camera.py --check   (table coords within ~1 cm)
  2. scripts/test_perception.py            (boxes the items, ignores arm/bin)
  3. scripts/dry_run.py                     (targets look right, motion disabled)
  4. motion tuned on hardcoded coords       (M4)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pipeline import build_detector, build_follower, build_localizer  # noqa: E402
from trash_arm.camera import open_scene_camera  # noqa: E402
from trash_arm.config import load_config  # noqa: E402
from trash_arm.motion import Motion  # noqa: E402
from trash_arm.orchestrator import Orchestrator  # noqa: E402
from trash_arm.safety import EStop  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--yes", action="store_true", help="skip the 'enable motion?' confirmation")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if cfg["calibration"] and build_localizer(cfg) is None:
        raise SystemExit("No homography -- run scripts/calibrate_camera.py first.")

    if not args.yes:
        ans = input("This will MOVE the arm. Workspace clear, e-stop reachable? [y/N] ")
        if ans.strip().lower() != "y":
            raise SystemExit("aborted.")

    estop = EStop()
    estop.install_sigint()  # first Ctrl+C -> clean stop, not a kill

    camera = open_scene_camera(cfg)
    detector = build_detector(cfg)
    localizer = build_localizer(cfg)
    arm = build_follower(cfg)
    motion = Motion(arm=arm, cfg=cfg, estop=estop, execute=True)

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


if __name__ == "__main__":
    main()
