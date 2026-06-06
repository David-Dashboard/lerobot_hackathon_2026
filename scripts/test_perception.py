"""Run the detector on a saved image or the live scene camera (milestone M3).

  python scripts/test_perception.py --list                 # enumerate camera indices
  python scripts/test_perception.py --image table.jpg      # detect on a saved image
  python scripts/test_perception.py                         # one frame from the scene cam

Draws boxes + labels and writes an annotated PNG (and shows it if a display is
available). Classes / confidence / device come from config.yaml.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trash_arm.config import load_config  # noqa: E402
from trash_arm.perception import OwlVitDetector, filter_roi  # noqa: E402


def list_cameras(max_index: int = 6):
    import cv2

    print("probing camera indices 0..", max_index)
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        print(f"  index {i}: {'AVAILABLE' if ok else '-'}")


def load_image(path: str) -> np.ndarray:
    import cv2

    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def grab_frame(cfg) -> np.ndarray:
    from trash_arm.camera import open_scene_camera

    cam = open_scene_camera(cfg)
    try:
        return cam.read()
    finally:
        cam.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--image", default=None, help="detect on a saved image instead of the camera")
    ap.add_argument("--list", action="store_true", help="list available camera indices and exit")
    ap.add_argument("--out", default="perception_annotated.png")
    args = ap.parse_args()

    if args.list:
        list_cameras()
        return

    cfg = load_config(args.config)
    p = cfg["perception"]
    image = load_image(args.image) if args.image else grab_frame(cfg)

    detector = OwlVitDetector(p["classes"], p["confidence"], p["device"], p["model_id"])
    dets = detector.detect(image)
    dets = filter_roi(dets, p.get("roi"))

    print(f"\n{len(dets)} detection(s):")
    for d in dets:
        print(f"  {d.label:20s} conf={d.confidence:.2f}  center={d.pixel_xy}  bbox={tuple(round(b) for b in d.bbox)}")

    import cv2

    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    roi = p.get("roi")
    if roi:
        cv2.rectangle(bgr, (roi["x_min"], roi["y_min"]), (roi["x_max"], roi["y_max"]), (255, 0, 0), 1)
    for d in dets:
        x1, y1, x2, y2 = (int(v) for v in d.bbox)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(bgr, f"{d.label} {d.confidence:.2f}", (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.imwrite(args.out, bgr)
    print(f"\nwrote {args.out}")
    try:
        cv2.imshow("perception (any key to close)", bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass  # headless: the PNG is enough


if __name__ == "__main__":
    main()
