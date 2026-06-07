"""Shared builders that turn config.yaml into pipeline objects (used by dry_run/run)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trash_arm.config import resolve_path  # noqa: E402
from trash_arm.localize import Localizer, load_homography  # noqa: E402
from trash_arm.perception import OwlVitDetector  # noqa: E402


def build_detector(cfg: dict) -> OwlVitDetector:
    p = cfg["perception"]
    return OwlVitDetector(p["classes"], p["confidence"], p["device"], p["model_id"])


def build_localizer(cfg: dict):
    """Localizer from the saved homography (+ optional undistortion). None if absent."""
    cal = cfg["calibration"]
    hpath = resolve_path(cfg, cal["homography_path"])
    if not hpath.exists():
        print(f"(no homography at {hpath} -- run scripts/calibrate_camera.py; "
              "localization disabled)")
        return None
    H = load_homography(hpath)

    cam_mtx = dist = None
    mpath = resolve_path(cfg, cal.get("camera_matrix_path", ""))
    dpath = resolve_path(cfg, cal.get("dist_coeffs_path", ""))
    if mpath.exists() and dpath.exists():
        load = load_homography  # np.load under the hood
        cam_mtx, dist = load(mpath), load(dpath)

    return Localizer(H, camera_matrix=cam_mtx, dist_coeffs=dist, roi=cfg["perception"].get("roi"))


def build_follower(cfg: dict):
    """Connect the real follower arm (calibration loaded by id)."""
    from so101 import make_arm

    r = cfg["robot"]
    arm = make_arm(mock=False, port=r["port"], arm_id=r["id"], calibrate=False)
    arm.connect()
    return arm
