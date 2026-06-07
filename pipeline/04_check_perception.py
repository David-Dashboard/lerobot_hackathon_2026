"""04 - check perception: prove the robust front-end before any motion.

    python pipeline/04_check_perception.py            # one frame from the scene cam
    python pipeline/04_check_perception.py --image table.jpg

Runs the zero-shot detector (config.perception.model_id / classes), drops anything
outside the pixel ROI, maps each surviving detection to a table (x, y) via the
homography, and flags which ones fall inside the arm's reachable workspace. Writes
an annotated PNG. This is the exact chain 07 uses to choose targets -- if the boxes
and table coords look right here, deployment will too.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_pipeline import build_detector, build_localizer  # noqa: E402
from trash_arm.config import load_config  # noqa: E402
from trash_arm.perception import filter_roi  # noqa: E402


def in_reach(x: float, y: float, ws: dict) -> bool:
    return ws["x_min"] <= x <= ws["x_max"] and ws["y_min"] <= y <= ws["y_max"]


def grab_frame(cfg):
    import cv2

    if args.image:
        bgr = cv2.imread(args.image)
        if bgr is None:
            sys.exit(f"could not read {args.image}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    from trash_arm.camera import open_scene_camera

    cam = open_scene_camera(cfg)
    try:
        return cam.read()
    finally:
        cam.close()


def main() -> None:
    global args
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None)
    ap.add_argument("--out", default="perception_check.png")
    args = ap.parse_args()

    cfg = load_config(args.config)
    image = grab_frame(cfg)

    detector = build_detector(cfg)
    localizer = build_localizer(cfg)  # None if no homography yet
    ws = cfg["workspace"]
    roi = cfg["perception"].get("roi")

    dets = filter_roi(detector.detect(image), roi)
    print(f"\n{len(dets)} detection(s) in ROI:")
    annotated = []
    for d in dets:
        line = f"  {d.label:22s} conf={d.confidence:.2f} px={tuple(round(v) for v in d.pixel_xy)}"
        if localizer is not None:
            x, y = localizer.to_table(*d.pixel_xy)
            reach = "IN-REACH" if in_reach(x, y, ws) else "out-of-reach"
            line += f"  table=({x:+.3f},{y:+.3f})m  [{reach}]"
            annotated.append((d, (x, y), in_reach(x, y, ws)))
        else:
            line += "  (no homography -- run 03_calibrate_scene.py for table coords)"
        print(line)

    # annotate
    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if roi:
        cv2.rectangle(bgr, (roi["x_min"], roi["y_min"]), (roi["x_max"], roi["y_max"]), (255, 120, 0), 1)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        reachable = next((r for dd, _, r in annotated if dd is d), True)
        color = (0, 200, 0) if reachable else (0, 0, 220)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr, f"{d.label} {d.confidence:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(args.out, bgr)
    print(f"\nannotated -> {args.out}")
    print("green = in reach, red = detected but out of the workspace.")


if __name__ == "__main__":
    main()
