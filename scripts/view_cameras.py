"""Live-view one or more cameras in Rerun -- no arm, just verify the feeds.

Streams each given camera index to Rerun so you can confirm it works and see which
index is which (scene vs wrist vs laptop).

    # needs rerun.exe on PATH first (PowerShell):
    #   $env:Path = "$PWD\\.venv\\Scripts;$env:Path"
    python scripts/view_cameras.py --camera scene=0 --camera wrist=2
    python scripts/view_cameras.py --camera a=0 --camera b=1 --camera c=2   # probe several

Each --camera is name=index[:WxH][@fps]. Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from so101.record import parse_camera_spec  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", action="append", default=[], metavar="NAME=INDEX[:WxH][@FPS]",
                    help="camera to view (repeatable)")
    ap.add_argument("--fps", type=float, default=15.0, help="display refresh rate")
    args = ap.parse_args()

    if not args.camera:
        ap.error("give at least one --camera, e.g. --camera scene=0")

    import cv2
    import rerun as rr

    specs = dict(parse_camera_spec(s) for s in args.camera)

    caps = {}
    for name, cfg in specs.items():
        cap = cv2.VideoCapture(cfg["index"])
        if not cap.isOpened():
            print(f"[{name}] index {cfg['index']}: FAILED to open -- skipping")
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg["height"])
        caps[name] = cap
        print(f"[{name}] index {cfg['index']}: opened")

    if not caps:
        raise SystemExit("no cameras opened")

    rr.init("camera_view", spawn=True)
    print(f"Streaming {list(caps)} to Rerun. Ctrl+C to stop.")

    period = 1.0 / args.fps
    try:
        while True:
            for name, cap in caps.items():
                ok, frame_bgr = cap.read()
                if ok:
                    rr.log(f"cameras/{name}", rr.Image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nstopping.")
    finally:
        for cap in caps.values():
            cap.release()


if __name__ == "__main__":
    main()
