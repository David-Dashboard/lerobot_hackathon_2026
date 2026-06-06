"""Live-view the OAK-D-PRO RGB camera in Rerun.

    # needs rerun.exe on PATH first (PowerShell):
    #   $env:Path = "$PWD\\.venv\\Scripts;$env:Path"
    python scripts/oak_view.py
    python scripts/oak_view.py --width 1280 --height 800   # full sensor res

Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coord_grasp.oak import OakCamera  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=400)
    args = ap.parse_args()

    import rerun as rr

    rr.init("oak_rgb", spawn=True)
    print(f"Streaming OAK-D-PRO RGB ({args.width}x{args.height}) to Rerun. Ctrl+C to stop.")
    with OakCamera(size=(args.width, args.height)) as cam:
        try:
            while True:
                rr.log("oak/rgb", rr.Image(cam.read()))
        except KeyboardInterrupt:
            print("\nstopping.")


if __name__ == "__main__":
    main()
