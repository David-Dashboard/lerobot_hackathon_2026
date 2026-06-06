"""Full perceive -> localize -> decide loop with MOTION DISABLED (milestone M5 prep).

Runs the real camera + detector + localizer and prints which item it WOULD pick,
in table coordinates -- but never moves the arm. This is the safe integration test:
run it until the targets look right before ever enabling motion in run.py.

  python scripts/dry_run.py
  python scripts/dry_run.py --once     # one perception pass, then exit

Requires the homography (scripts/calibrate_camera.py) and the detector model
(transformers). Without a homography it still runs, printing pixel targets only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_pipeline import build_detector, build_localizer  # noqa: E402
from trash_arm.camera import open_scene_camera  # noqa: E402
from trash_arm.config import load_config  # noqa: E402
from trash_arm.motion import Motion  # noqa: E402
from trash_arm.orchestrator import Orchestrator  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--once", action="store_true", help="single perception pass, no loop")
    args = ap.parse_args()

    cfg = load_config(args.config)
    camera = open_scene_camera(cfg)
    detector = build_detector(cfg)
    localizer = build_localizer(cfg)  # None if no homography yet

    # execute=False -> Motion plans + logs but never commands the arm (arm not needed).
    motion = Motion(arm=None, cfg=cfg, execute=False)
    orch = Orchestrator(camera, detector, localizer, motion, cfg)

    print("DRY RUN -- motion disabled. Ctrl+C to stop.\n")
    try:
        if args.once:
            dets = orch.perceive()
            for d in dets:
                print(f"  would consider {d.label} @ "
                      f"{'pixel ' + str(d.pixel_xy) if d.table_xy is None else d.table_xy}")
        else:
            orch.run()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        camera.close()


if __name__ == "__main__":
    main()
