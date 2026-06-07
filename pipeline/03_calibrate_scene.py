"""03 - calibrate the scene camera to the table (pixel -> robot-frame metres).

    python pipeline/03_calibrate_scene.py          # interactive: click >=4 markers, type their XY
    python pipeline/03_calibrate_scene.py --check   # validate: click a point, read its table XY
    python pipeline/03_calibrate_scene.py --points calibration/points.json

This is a thin pass-through to the repo's existing, working calibrator
(scripts/calibrate_camera.py) so the numbered flow stays in one place. It builds
the homography saved at config.calibration.homography_path; aim for <~1 cm
reprojection error.

MULTI-CAMERA / 3D: for two scene cameras you can triangulate instead of relying on
the single-plane homography -- use the coord_grasp.markers + coord_grasp.localize3d
modules (AprilTag hand-eye + triangulation). The homography path below is the
fastest robust option for a flat table and is what 04/07 use by default.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main() -> None:
    target = REPO / "scripts" / "calibrate_camera.py"
    if not target.exists():
        sys.exit("scripts/calibrate_camera.py not found -- merge the coord-grasp branch "
                 "(it holds the calibrator + trash_arm/ localization).")
    # Hand argv straight through (--check / --points / --config all supported there).
    sys.argv = [str(target)] + sys.argv[1:]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
