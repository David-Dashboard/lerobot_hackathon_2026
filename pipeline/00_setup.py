"""00 - setup: install dependencies into the active venv via uv.

    python pipeline/00_setup.py

Thin convenience wrapper -- the same as `uv pip install -r requirements.txt`, but
pinned to the interpreter you ran it with so install and run can't diverge.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    req = REPO / "requirements.txt"
    print(f"Installing {req} into {sys.executable} ...")
    try:
        subprocess.check_call(["uv", "pip", "install", "--python", sys.executable, "-r", str(req)])
    except FileNotFoundError:
        print("uv not found; falling back to pip.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req)])
    print("done. Next: python pipeline/01_detect_hardware.py")


if __name__ == "__main__":
    main()
