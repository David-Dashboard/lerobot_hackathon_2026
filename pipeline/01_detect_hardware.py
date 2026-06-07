"""01 - detect hardware: which arm is leader/follower, and which cameras to use.

    python pipeline/01_detect_hardware.py                  # detect + print
    python pipeline/01_detect_hardware.py --no-interactive  # serial fast-path only
    python pipeline/01_detect_hardware.py --use 0 "OAK"     # only these cameras
    python pipeline/01_detect_hardware.py --ignore "Integrated"  # drop these
    python pipeline/01_detect_hardware.py --scene OAK --wrist 2   # force roles
    python pipeline/01_detect_hardware.py --json            # machine-readable

Detects roles with the wiggle test (or config serials), enumerates EVERY camera,
applies your include/exclude/role choices, and prints a config.yaml snippet you can
paste in. It never moves the follower and never overwrites your commented config.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from so101 import arm_id, cameras  # noqa: E402


def _load_cfg(path: str | None) -> dict:
    import yaml

    p = Path(path) if path else REPO / "config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}


def _coerce(keys):
    """Turn CLI matcher tokens into ints where they look like indices."""
    return [int(k) if str(k).lstrip("-").isdigit() else k for k in (keys or [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-interactive", action="store_true",
                    help="don't run the wiggle test; require serials in config")
    ap.add_argument("--use", nargs="+", default=None, metavar="KEY",
                    help="allowlist cameras by index or name-substring")
    ap.add_argument("--ignore", nargs="+", default=None, metavar="KEY",
                    help="drop cameras by index or name-substring (adds to laptop defaults)")
    ap.add_argument("--scene", default=None, help="force the scene camera (index or name)")
    ap.add_argument("--wrist", default=None, help="force the wrist camera (index or name)")
    ap.add_argument("--json", action="store_true", help="print JSON and exit")
    args = ap.parse_args()

    cfg = _load_cfg(args.config)

    # ---- arms (roles) -----------------------------------------------------
    arms = arm_id.identify_arm_roles(cfg, interactive=not args.no_interactive)

    # ---- cameras (detect + choose) ---------------------------------------
    from so101.discovery import LAPTOP_NAME_HINTS

    found = cameras.enumerate_cameras()
    roles = {}
    if args.scene is not None:
        roles[int(args.scene) if str(args.scene).isdigit() else args.scene] = "scene"
    if args.wrist is not None:
        roles[int(args.wrist) if str(args.wrist).isdigit() else args.wrist] = "wrist"
    selected = cameras.select_cameras(
        found,
        include=_coerce(args.use),
        exclude=list(LAPTOP_NAME_HINTS) + _coerce(args.ignore),
        roles=roles or None,
    )

    if args.json:
        print(json.dumps({
            "arms": arms,
            "cameras": [vars(c) for c in selected],
        }, indent=2))
        return

    # ---- report -----------------------------------------------------------
    print("\n== Arms ==")
    print(f"  follower : {arms['follower']['port']}  (serial {arms['follower']['serial']})")
    print(f"  leader   : {arms['leader']['port']}  (serial {arms['leader']['serial']})")
    print("\n== Cameras found ==")
    print(cameras.describe(found))
    print("\n== Cameras selected (after include/exclude/roles) ==")
    print(cameras.describe(selected))

    # ---- paste-ready snippet (does NOT clobber your commented config) -----
    scene = next((c for c in selected if c.role == "scene"), None)
    wrist = next((c for c in selected if c.role == "wrist"), None)
    print("\n== Paste into config.yaml (review first) ==")
    print(f"robot:  {{serial: \"{arms['follower']['serial']}\"}}")
    print(f"teleop: {{serial: \"{arms['leader']['serial']}\"}}")
    print("cameras:")
    if scene:
        loc = f"index: {scene.index}" if scene.kind == "uvc" else f"# OAK mxid {scene.mxid} (DepthAI, not an index)"
        print(f"  scene: {{ {loc} }}")
    if wrist:
        print(f"  wrist: {{ index: {wrist.index} }}")
    Path(REPO / "calibration").mkdir(exist_ok=True)
    (REPO / "calibration" / "detected_hardware.json").write_text(
        json.dumps({"arms": arms, "cameras": [vars(c) for c in selected]}, indent=2)
    )
    print("\n(also written to calibration/detected_hardware.json)")


if __name__ == "__main__":
    main()
