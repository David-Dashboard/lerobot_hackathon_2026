"""Build the pixel -> table homography for the scene camera (milestone M2).

Place >=4 markers at KNOWN table coordinates (metres, robot frame). Two ways:

  Interactive (click in the live image):
    python scripts/calibrate_camera.py
  then click each marker; for each click, type its table X Y in the terminal.

  From a file (repeatable, non-interactive) -- a JSON list of pairs:
    python scripts/calibrate_camera.py --points calibration/points.json
    # [{"pixel": [u, v], "table": [x, y]}, ...]

Saves calibration/homography_scene.npy. Validate with:
  python scripts/calibrate_camera.py --check          # click a point -> prints table XY

The homography is exact only at the table surface -- calibrate at table height
and grasp toward an object's base (see the parallax note in the plan).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trash_arm.camera import open_scene_camera  # noqa: E402
from trash_arm.config import load_config, resolve_path  # noqa: E402
from trash_arm.localize import (  # noqa: E402
    apply_homography,
    compute_homography,
    load_homography,
    save_homography,
)


def _grab_frame(cfg):
    cam = open_scene_camera(cfg)
    try:
        return cam.read()
    finally:
        cam.close()


def from_file(path: str):
    pairs = json.loads(Path(path).read_text())
    pix = [p["pixel"] for p in pairs]
    tab = [p["table"] for p in pairs]
    return pix, tab


def interactive_collect(cfg):
    import cv2

    frame_rgb = _grab_frame(cfg)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    pix, tab = [], []

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"\nclicked pixel ({x}, {y})")
            raw = input("  enter table X Y (metres), or blank to skip: ").strip()
            if raw:
                xs, ys = raw.split()
                pix.append([x, y])
                tab.append([float(xs), float(ys)])
                cv2.circle(frame_bgr, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow("calibrate (click markers, ESC when done)", frame_bgr)

    cv2.imshow("calibrate (click markers, ESC when done)", frame_bgr)
    cv2.setMouseCallback("calibrate (click markers, ESC when done)", on_click)
    print("Click each marker, type its table X Y. Need >=4. ESC to finish.")
    while True:
        if cv2.waitKey(20) == 27:  # ESC
            break
    cv2.destroyAllWindows()
    return pix, tab


def check(cfg):
    import cv2

    H = load_homography(resolve_path(cfg, cfg["calibration"]["homography_path"]))
    frame_rgb = _grab_frame(cfg)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def on_click(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            tx, ty = apply_homography(H, x, y)
            print(f"pixel ({x},{y}) -> table ({tx:.3f}, {ty:.3f}) m")

    cv2.imshow("check (click to read table coords, ESC to quit)", frame_bgr)
    cv2.setMouseCallback("check (click to read table coords, ESC to quit)", on_click)
    while cv2.waitKey(20) != 27:
        pass
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--points", default=None, help="JSON correspondences file (non-interactive)")
    ap.add_argument("--check", action="store_true", help="validate an existing homography")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.check:
        check(cfg)
        return

    pix, tab = from_file(args.points) if args.points else interactive_collect(cfg)
    if len(pix) < 4:
        raise SystemExit(f"need >=4 point pairs, got {len(pix)}")

    H = compute_homography(pix, tab)
    out = resolve_path(cfg, cfg["calibration"]["homography_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    save_homography(H, out)

    # Report reprojection error so you can judge the fit (<~1 cm is good).
    errs = [np.hypot(*(np.subtract(apply_homography(H, u, v), t))) for (u, v), t in zip(pix, tab)]
    print(f"\nsaved {out}")
    print(f"reprojection error: mean {np.mean(errs) * 100:.2f} cm, max {np.max(errs) * 100:.2f} cm")


if __name__ == "__main__":
    main()
