# SO-101 trash-pickup — full plan & pipeline

_Single-file plan: the rationale, the design, and the complete source of every script. Copy each script out of the fenced blocks in the appendix into the path shown in its heading._

# Why: robustness and reproducibility

Both goals come down to one idea applied on two different axes: **decouple what
varies from what matters.**

## Robustness — invariance to changes in the world

End-to-end ACT on raw pixels *entangles the task with the conditions it was filmed
under*. It doesn't learn "pick up trash" — it learns "when these pixels light up in
this camera, move this way." Camera pose, table geometry, and lighting get baked into
the policy implicitly, so any drift in them breaks it. Every design choice keeps the
volatile stuff away from the fragile learned function:

- **Detection → world coordinates** (via the calibration transform) turns "trash in a
  specific view" into "trash at (x, y) on the table." The camera angle is absorbed into
  a cheap, re-estimable transform instead of into the policy — bump the camera, re-run
  calibration, no retrain.
- **The wrist camera** is gripper-relative, so it is viewpoint-invariant by
  construction; the physical mounting does the canonicalization for free.
- **The reach filter and class blacklist** stop the arm chasing unreachable detections
  or grabbing a hand or the bin.
- **Scripted IK as the default deploy path** is robustness-as-predictability:
  deterministic, debuggable, no learned distribution shift. The policy is the upgrade,
  not the foundation.
- **E-stop, slow speeds, dry-run, joint limits** are robustness in the *fail-safe*
  sense — when something goes wrong (and at a hackathon it will), the failure is bounded.

The deeper principle: you can't remove variation from the world, so the only real
question is *where the variance lands*. Route it into explicit, cheap, re-estimable
components (a homography, a calibration) and keep it out of brittle ones (a
pixel-conditioned net).

## Reproducibility — invariance to changes in the setup

Robotics setups are notoriously un-reproducible because the hardware handles drift: COM
ports renumber on replug or reboot, camera indices shuffle, and nobody remembers which
arm is which. Hard-code `COM5` and the setup dies the moment you reboot or a teammate
plugs in their laptop. So the pipeline pins everything volatile to a stable handle:

- **Arms matched by stable board serial**, with the wiggle test as the zero-knowledge
  fallback — role identity survives replug, reboot, and a different machine.
- **Cameras found by enumeration + role** rather than fragile indices — the laptop cam
  can't sneak in, the OAK is found wherever it lands.
- **One `config.yaml` as the single source of truth** — every magic number (workspace,
  IK, ROI, gripper thresholds) in one committed file, so the whole team runs identical
  parameters.
- **The numbered `01→07` scripts** encode the order of operations and the dependency
  chain: you physically can't deploy before calibrating, and you validate perception
  (`04`) before the arm is allowed to move (`07`). That turns "a pile of scripts someone
  ran once in some order" into a documented, re-runnable procedure.
- **The data → model → deploy chain is traceable**: a named `LeRobotDataset` is the
  artifact, the same dataset retrains the same policy, a specific checkpoint is what you
  deploy. (The catch: deploy must feed the policy the same observation pipeline it was
  trained on.)
- **Pure functions with hardware-free tests** (`movement_score`, `decide_roles`,
  `select_cameras`) make the decision logic verifiable in CI with no arm plugged in —
  reproducibility of *correctness*. The repo's "only `real.py` touches LeRobot"
  structure exists precisely so the logic can be tested in isolation.

## Why they're the same discipline

Robustness is invariance to changes in the *world*; reproducibility is invariance to
changes in the *context*. Both are won by the identical move — find what varies, pin it
to a stable or explicit handle, and don't let the fragile parts depend on the volatile
ones. Decoupling is the throughline of the whole design.

And the hackathon is what makes both non-negotiable rather than nice-to-have: the camera
*will* get bumped, ports *will* renumber, a teammate *will* run it on another laptop,
and there's no time to re-debug on demo day. These two properties are exactly what stop
you losing hours to setup churn when it counts.


---


# Robust pickup pipeline (`01_*` → `07_*`)

A numbered, linear path from bare hardware to a deployed table-clearing robot, built
on top of what your repo already has. Each script is a thin orchestrator over verified
entry points — the heavy lifting (zero-shot detection, homography localization,
`Orchestrator`, `Motion`, `EStop`, recording, Qualia) already exists.

## What's new vs. what's reused

**New (added because they didn't exist):**
- `so101/cameras.py` — discover **every** camera (OAK + all UVC), then **choose which
  to use or ignore** by index or name-substring, and assign roles (scene/wrist/sideN).
  Generalises the old "find the OAK + one wrist by name" into a real selector. The
  selection logic is pure and unit-tested (`tests/test_cameras_select.py`).
- `so101/arm_id.py` — **automatically detect which arm is leader vs follower.** Fast
  path matches config serials; otherwise a safe *wiggle test*: both ports are opened as
  leaders (a leader never enables torque, so both arms stay hand-movable), you move one
  arm, and the one that moved is the leader. Decision logic is pure and unit-tested
  (`tests/test_arm_id.py`).

**Reused (already in your repo):** `OwlVitDetector` (zero-shot), `Localizer`
(`to_table` / `in_roi`), `Orchestrator`, `Motion`, `EStop`, `record_teleop_dataset`,
`qualia_client`, `deploy.run_policy`, `coord_grasp/{markers,handeye,localize3d}` (marker
hand-eye + triangulation), and `scripts/{calibrate_camera,build_pipeline}.py`.

## ⚠️ Branch merge

The recording half lives on **`slim-auto-record`**; the autonomous half
(`trash_arm/`, `coord_grasp/`, `scripts/`, `so101/deploy.py`) lives on **`coord-grasp`**.
This pipeline needs **both**. Merge them into one working tree (e.g. branch off
`coord-grasp`, then bring in `slim-auto-record`'s `discovery.py`, `record.py`,
`reliability.py`, `auto_record.py`), and drop the two new `so101/*.py` files in. Steps
03/04/07 import from `trash_arm`/`scripts`; 05 imports from `so101.record`.

## The flow

| # | Script | Does | Wraps |
|---|--------|------|-------|
| 00 | `00_setup.py` | install deps into the venv | `uv pip install` |
| 01 | `01_detect_hardware.py` | **auto-detect arm roles + all cameras; choose/ignore; print a config snippet** | `arm_id`, `cameras` (new) |
| 02 | `02_calibrate_arms.py` | role-aware joint calibration for each arm | `lerobot-calibrate` |
| 03 | `03_calibrate_scene.py` | scene-camera → table homography (or markers/triangulation) | `scripts/calibrate_camera.py` |
| 04 | `04_check_perception.py` | **validate the robust front-end**: detect → ROI → table coords → reach filter | `OwlVitDetector`, `Localizer` |
| 05 | `05_record_demos.py` | teleop → ACT-ready `LeRobotDataset` (role arms + selected cams) | `record_teleop_dataset` |
| 06 | `06_train_act.py` | train ACT locally, or on Qualia cloud | `lerobot-train` / `qualia_client` |
| 07 | `07_deploy.py` | clear the table: scripted pick (default) or run trained ACT | `Orchestrator` / `lerobot-record` |

Gate each step on the previous one being green (esp. 04 before 07).

## The robust idea, wired

The whole point of the earlier discussion was to make the policy **agnostic to scene
camera pose**. Three paths, in increasing order of effort, all supported here:

- **C — scripted (default, most robust for the hackathon).** `07 --dry-run` then `07`.
  Detector → `Localizer.to_table` → reach filter (`config.workspace`) → scripted IK
  pick. No learned model; camera pose is fully decoupled because the policy only ever
  sees **table coordinates**, never raw pixels. This is `coord-grasp`'s `run.py`, fronted
  by the new auto-detection.
- **A — image ACT (what 05/06 record & train today).** ACT on **wrist + scene** images.
  Robustness comes from the wrist camera (gripper-relative → viewpoint-stable) plus
  keeping the scene camera fixed (or warping it to a canonical view). The detector is
  used at deploy as a **workspace gate / target picker**, not as a policy input.
- **B — detection-conditioned ACT (the strongest decoupling; an upgrade).** Feed the
  **world (x, y)** from `Localizer.to_table` into ACT as an extra observation feature, so
  the policy is conditioned on geometry, not pixels. This means adding a non-image
  feature to the `LeRobotDataset` in `record.py` and to the policy config — a real but
  contained change. Feed the **table coordinate**, never the raw pixel box (a pixel box
  still secretly encodes the camera angle).

## Camera selection cheatsheet (steps 01 and 05)

```
--use 0 "OAK"          # allowlist: ONLY these cameras (index or name-substring)
--ignore "Integrated"  # drop these (added to the built-in laptop-cam defaults)
--scene OAK --wrist 2  # force roles explicitly
```
The laptop built-in camera is always excluded by default (name hints in `discovery.py`).

## Before autonomous runs — fill the `config.yaml` TODOs

`config.yaml` already enumerates everything, but the `# TODO: MEASURE` values must be
real before motion is safe: `workspace` (reachable x/y + table/floor z), `ik` link
lengths + sign/offset, `bin.xy`, `home.joints`, `gripper.miss_below`, and the pixel
`perception.roi`. Tune with `07 --dry-run` (targets print, arm doesn't move) and keep
`safety.max_step_deg` low until you trust it.

## Tests (no hardware)

```
python -m pytest tests/test_cameras_select.py tests/test_arm_id.py -q
```


---

# Appendix — full source

Create these files at the paths shown (the `pipeline/NN_*.py` are run directly, e.g. `python pipeline/01_detect_hardware.py`; they are not imported, so the leading digits are fine).


### `so101/cameras.py`

```python
"""General camera auto-discovery + selection.

Generalises discovery.py's "find the OAK + one wrist webcam" into:
  * enumerate_cameras()  -- list EVERY camera the machine can see (OAK devices via
    DepthAI + all UVC indices via pygrabber names), as plain CameraInfo records.
  * select_cameras(...)  -- a PURE function (no IO) that decides which to USE and
    which to IGNORE, and assigns each a role (scene / wrist / sideN). You can drive
    it with include / exclude lists (by name-substring OR index) and an explicit
    role map. The laptop built-in cam is excluded by default.
  * open_selected(...)   -- open the chosen cameras and return the {role: camera}
    dict the recorder wants (extra_cameras) -- OAK and webcams behind one interface.

select_cameras is deliberately IO-free so it is unit-tested with no hardware
(see tests/test_cameras_select.py). All heavy imports stay lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .discovery import LAPTOP_NAME_HINTS, Cv2Camera, list_video_devices


@dataclass
class CameraInfo:
    """One discovered camera. `index` is set for UVC webcams, `mxid` for OAKs."""

    kind: str                      # "oak" | "uvc"
    name: str                      # human-readable device name
    index: int | None = None       # OpenCV index (UVC only)
    mxid: str | None = None         # DepthAI device id (OAK only)
    role: str | None = None         # assigned by select_cameras: scene/wrist/sideN

    def key(self) -> str:
        return f"oak:{self.mxid}" if self.kind == "oak" else f"uvc:{self.index}"


# --------------------------------------------------------------------------- #
# discovery (IO)
# --------------------------------------------------------------------------- #
def _enumerate_oaks() -> list[CameraInfo]:
    try:
        import depthai as dai
    except Exception:
        return []
    out: list[CameraInfo] = []
    try:
        for d in dai.Device.getAllAvailableDevices():
            mxid = getattr(d, "getMxId", lambda: None)() or getattr(d, "mxid", None)
            name = getattr(d, "name", None) or "OAK"
            out.append(CameraInfo(kind="oak", name=f"OAK-D ({name})", mxid=str(mxid)))
    except Exception:
        pass
    return out


def _enumerate_uvc() -> list[CameraInfo]:
    names = list_video_devices()  # pygrabber: names in OpenCV index order (or None)
    if names:
        return [CameraInfo(kind="uvc", name=n, index=i) for i, n in enumerate(names)]
    # Fallback (no pygrabber, e.g. Linux): probe openable indices and name generically.
    try:
        import cv2
    except Exception:
        return []
    out: list[CameraInfo] = []
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            out.append(CameraInfo(kind="uvc", name=f"video{i}", index=i))
        cap.release()
    return out


def enumerate_cameras() -> list[CameraInfo]:
    """Every camera visible right now: OAK devices first, then UVC webcams."""
    return _enumerate_oaks() + _enumerate_uvc()


# --------------------------------------------------------------------------- #
# selection (PURE -- unit-tested without hardware)
# --------------------------------------------------------------------------- #
def _matches(info: CameraInfo, key) -> bool:
    """A matcher key is either an int (== UVC index) or a str (case-insensitive
    substring of the device name)."""
    if isinstance(key, int):
        return info.index == key
    return str(key).lower() in info.name.lower()


def _matches_any(info: CameraInfo, keys) -> bool:
    return any(_matches(info, k) for k in (keys or []))


def select_cameras(
    infos: list[CameraInfo],
    *,
    include=None,
    exclude=LAPTOP_NAME_HINTS,
    roles: dict | None = None,
) -> list[CameraInfo]:
    """Choose which cameras to use and assign each a role. Pure / no IO.

    include : if given, an allowlist of matcher keys -- ONLY cameras matching one of
              them are kept (applied before exclude).
    exclude : matcher keys to drop (defaults to laptop built-in cam name hints).
    roles   : {matcher_key: role} -- force a role on a camera (e.g. {0: "scene",
              "Logitech": "wrist"}). Unforced cameras get default roles: the first
              OAK -> "scene", the first remaining webcam -> "wrist", then side1, side2...

    Returns the kept cameras (with `.role` set) in a stable order (forced roles first
    in scene/wrist/side order, then the rest).
    """
    kept = list(infos)
    if include:
        kept = [c for c in kept if _matches_any(c, include)]
    kept = [c for c in kept if not _matches_any(c, exclude)]

    # 1) forced roles
    forced: dict[str, CameraInfo] = {}
    roles = roles or {}
    for c in kept:
        for key, role in roles.items():
            if _matches(c, key):
                c.role = role
                forced[c.key()] = c
                break

    # 2) default roles for the rest
    rest = [c for c in kept if c.key() not in forced]
    side_n = 1
    used_roles = {c.role for c in forced.values()}
    for c in rest:
        if c.kind == "oak" and "scene" not in used_roles:
            c.role = "scene"
        elif "wrist" not in used_roles:
            c.role = "wrist"
        else:
            c.role = f"side{side_n}"
            side_n += 1
        used_roles.add(c.role)

    order = {"scene": 0, "wrist": 1}
    return sorted(kept, key=lambda c: (order.get(c.role, 2 + (c.index or 0)),))


# --------------------------------------------------------------------------- #
# opening (IO)
# --------------------------------------------------------------------------- #
def open_camera(info: CameraInfo, size=(640, 480), fps: int = 30):
    """Open one CameraInfo and return an object exposing .read()->RGB / .close().

    OAK -> coord_grasp.oak.OakCamera ; UVC -> discovery.Cv2Camera. (Single-OAK
    assumption: OakCamera grabs the default DepthAI device. Multi-OAK rigs need a
    device-id arg added to OakCamera -- left as a TODO.)
    """
    w, h = size
    if info.kind == "oak":
        from coord_grasp.oak import OakCamera

        return OakCamera(size=(w, h))
    if info.index is None:
        raise ValueError(f"UVC camera {info.name!r} has no index")
    return Cv2Camera(info.index, width=w, height=h, fps=fps)


def open_selected(selected: list[CameraInfo], *, sizes: dict | None = None, fps: int = 30) -> dict:
    """Open all selected cameras -> {role: camera_obj} for record's extra_cameras.

    `sizes` optionally maps role -> (w, h); otherwise 640x480 is used. Cameras are
    NOT connected here (the recorder/connect-with-retry does that).
    """
    sizes = sizes or {}
    out: dict[str, object] = {}
    for c in selected:
        role = c.role or c.key()
        out[role] = open_camera(c, size=sizes.get(role, (640, 480)), fps=fps)
    return out


def describe(infos: list[CameraInfo]) -> str:
    """One line per camera for console output."""
    if not infos:
        return "  (no cameras found)"
    lines = []
    for c in infos:
        where = f"index {c.index}" if c.kind == "uvc" else f"mxid {c.mxid}"
        role = f"  -> {c.role}" if c.role else ""
        lines.append(f"  [{c.kind:3}] {where:14}  {c.name}{role}")
    return "\n".join(lines)
```

### `so101/arm_id.py`

```python
"""Automatic leader / follower role detection for the two SO-101 arms.

Today the repo maps role -> port by a serial hard-coded in config.yaml. This adds
ZERO-config role detection so a fresh machine (or a teammate's) "just works":

  identify_arm_roles(cfg) returns {"follower": {...}, "leader": {...}} with each
  arm's port + serial, using, in order:
    1. FAST PATH  -- if config has serials and both are plugged in, match by serial
       (discovery.find_arm_ports). No interaction, no motion.
    2. WIGGLE TEST -- otherwise, open BOTH candidate ports as *leaders* (an SO-101
       leader never enables torque, so both arms stay freely hand-movable and this
       is safe), ask the user to gently move ONE arm, sample joint positions for a
       couple of seconds, and the arm that MOVED is the leader; the other is the
       follower. We then read each port's USB serial so it can be saved to config.

The movement maths is split into pure functions (movement_score / decide_roles)
so the decision logic is unit-tested with no hardware (tests/test_arm_id.py).
"""

from __future__ import annotations

import time

SO101_VID_PID = (0x1A86, 0x55D3)  # Waveshare/CH343 board used by the SO-101


# --------------------------------------------------------------------------- #
# pure decision logic (unit-tested, no hardware)
# --------------------------------------------------------------------------- #
def movement_score(samples: list[dict]) -> float:
    """How much an arm moved across a series of joint-dict samples.

    Sum of per-joint peak-to-peak (max - min) range, in degrees. Robust to slow
    drift and to a missing joint in some frames.
    """
    if not samples:
        return 0.0
    joints: set[str] = set()
    for s in samples:
        joints.update(s)
    total = 0.0
    for j in joints:
        vals = [s[j] for s in samples if j in s]
        if len(vals) >= 2:
            total += max(vals) - min(vals)
    return total


def decide_roles(
    score_a: float,
    port_a: str,
    score_b: float,
    port_b: str,
    *,
    min_move_deg: float = 5.0,
    ratio: float = 3.0,
) -> dict | None:
    """Given each port's movement score, decide which is leader/follower.

    Leader = the clearly-more-moved port. Returns None (ambiguous -> caller should
    re-prompt or fall back) unless the winner moved >= `min_move_deg` AND at least
    `ratio`x more than the other.
    """
    hi_port, hi, lo_port, lo = (
        (port_a, score_a, port_b, score_b)
        if score_a >= score_b
        else (port_b, score_b, port_a, score_a)
    )
    if hi < min_move_deg:
        return None  # nothing moved enough -- did the user move an arm?
    if lo > 0 and hi < ratio * lo:
        return None  # both moved similarly -- ambiguous
    return {"leader": hi_port, "follower": lo_port}


# --------------------------------------------------------------------------- #
# hardware helpers (IO)
# --------------------------------------------------------------------------- #
def candidate_ports(vid_pid=SO101_VID_PID) -> list:
    """COM/tty devices that look like SO-101 boards (by VID/PID), with serials."""
    from serial.tools import list_ports

    vid, pid = vid_pid
    return [p for p in list_ports.comports() if p.vid == vid and p.pid == pid]


def _serial_for_port(device: str) -> str | None:
    from serial.tools import list_ports

    for p in list_ports.comports():
        if p.device == device:
            return (p.serial_number or "").strip() or None
    return None


def _sample_motion(port: str, *, seconds: float, hz: float = 20.0) -> list[dict]:
    """Open `port` as a (torque-free) leader and sample joint dicts for `seconds`."""
    from .factory import make_teleop

    teleop = make_teleop(mock=False, port=port, teleop_id="roleprobe", calibrate=False)
    samples: list[dict] = []
    teleop.connect()
    try:
        t_end = time.time() + seconds
        period = 1.0 / hz
        while time.time() < t_end:
            try:
                samples.append(teleop.read_action())
            except Exception:
                pass
            time.sleep(period)
    finally:
        try:
            teleop.disconnect()
        except Exception:
            pass
    return samples


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def identify_arm_roles(cfg: dict | None = None, *, interactive: bool = True,
                       seconds: float = 2.5) -> dict:
    """Resolve {"follower": {port, serial}, "leader": {port, serial}}.

    Fast path uses config serials; otherwise runs the wiggle test. Raises
    RuntimeError with guidance if it can't decide.
    """
    cfg = cfg or {}

    # 1) FAST PATH: serials known + both present.
    serials = {}
    if cfg.get("robot", {}).get("serial"):
        serials["follower"] = cfg["robot"]["serial"]
    if cfg.get("teleop", {}).get("serial"):
        serials["leader"] = cfg["teleop"]["serial"]
    if len(serials) == 2:
        try:
            from .discovery import find_arm_ports

            ports = find_arm_ports(serials)
            return {
                "follower": {"port": ports["follower"], "serial": serials["follower"]},
                "leader": {"port": ports["leader"], "serial": serials["leader"]},
            }
        except Exception as e:
            print(f"(serial fast-path failed: {e}; falling back to wiggle test)")

    # 2) WIGGLE TEST.
    ports = candidate_ports()
    if len(ports) != 2:
        raise RuntimeError(
            f"Expected exactly 2 SO-101 boards (VID/PID {SO101_VID_PID}); found "
            f"{len(ports)}: {[(p.device, p.serial_number) for p in ports]}. "
            "Check both USB + power, or set serials in config.yaml."
        )
    pa, pb = ports[0].device, ports[1].device

    if interactive:
        input(
            "\nRole detection: when you press ENTER, GENTLY move ONE arm "
            "(the leader) through a few joints for ~2-3 s.\nPress ENTER to start..."
        )
    print("  sampling both arms... move one arm now.")
    samples_a = _sample_motion(pa, seconds=seconds)
    samples_b = _sample_motion(pb, seconds=seconds)
    # ^ sampled sequentially; tell the user to keep moving across both windows, or
    #   run twice. (Sequential keeps it single-threaded and avoids two SDK opens at
    #   once; for simultaneous sampling, thread these two calls.)

    score_a, score_b = movement_score(samples_a), movement_score(samples_b)
    print(f"  movement: {pa}={score_a:.1f} deg, {pb}={score_b:.1f} deg")
    decided = decide_roles(score_a, pa, score_b, pb)
    if decided is None:
        raise RuntimeError(
            "Could not tell the arms apart (no clear movement). Re-run and move ONE "
            "arm clearly, keep it moving across BOTH sampling windows, or set serials "
            "in config.yaml."
        )

    return {
        "follower": {"port": decided["follower"], "serial": _serial_for_port(decided["follower"])},
        "leader": {"port": decided["leader"], "serial": _serial_for_port(decided["leader"])},
    }
```

### `pipeline/00_setup.py`

```python
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
```

### `pipeline/01_detect_hardware.py`

```python
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
```

### `pipeline/02_calibrate_arms.py`

```python
"""02 - calibrate arms: run LeRobot's joint-range calibration for each arm.

    python pipeline/02_calibrate_arms.py            # auto-detect roles, calibrate both
    python pipeline/02_calibrate_arms.py --only leader

Auto-resolves which port is leader/follower (so you never calibrate the wrong one),
then shells out to LeRobot's interactive `lerobot-calibrate` for each. Calibration
is saved by LeRobot keyed by the arm id from config.yaml and reused everywhere.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from so101 import arm_id  # noqa: E402


def _cfg(path):
    import yaml

    p = Path(path) if path else REPO / "config.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", choices=["leader", "follower"], default=None)
    args = ap.parse_args()

    cfg = _cfg(args.config)
    arms = arm_id.identify_arm_roles(cfg)
    fid = cfg["robot"]["id"]
    lid = cfg["teleop"]["id"]

    jobs = []
    if args.only in (None, "follower"):
        jobs.append(["lerobot-calibrate", "--robot.type=so101_follower",
                     f"--robot.port={arms['follower']['port']}", f"--robot.id={fid}"])
    if args.only in (None, "leader"):
        jobs.append(["lerobot-calibrate", "--teleop.type=so101_leader",
                     f"--teleop.port={arms['leader']['port']}", f"--teleop.id={lid}"])

    for cmd in jobs:
        print(f"\n== {' '.join(cmd)} ==")
        print("Follow the prompts: move each joint through its full range, then set the middle pose.")
        subprocess.call(cmd)
    print("\nCalibration done. Next: python pipeline/03_calibrate_scene.py")


if __name__ == "__main__":
    main()
```

### `pipeline/03_calibrate_scene.py`

```python
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
```

### `pipeline/04_check_perception.py`

```python
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
```

### `pipeline/05_record_demos.py`

```python
"""05 - record demos: teleoperate (leader -> follower) into an ACT-ready dataset.

    python pipeline/05_record_demos.py --name trash_v1 --episodes 40
    python pipeline/05_record_demos.py --name trash_v1 --ignore "Integrated"

Auto-detects which arm is leader/follower and which cameras to record (OAK scene +
wrist by default; tune with --use/--ignore/--scene/--wrist as in step 01). Manual
ENTER-controlled episodes: ENTER starts, ENTER ends, 'q' finishes. Ctrl+C saves the
current episode then stops. Records locally to recorded/<name>/; push + train in 06.

Cameras recorded: ACT learns from the wrist (gripper-relative -> robust) plus the
scene view. The detector/world-coords channel is NOT recorded here (that is the
optional "feed detections into ACT" upgrade -- add it as an observation feature if
you want pose-conditioning; see PIPELINE.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from so101 import arm_id, cameras, make_arm, make_teleop  # noqa: E402
from so101.discovery import LAPTOP_NAME_HINTS  # noqa: E402
from so101.reliability import connect_with_retry, graceful_stop  # noqa: E402
from so101.record import prepare_dataset_dir, record_teleop_dataset  # noqa: E402


def _enter_poller():
    try:
        import msvcrt

        def drain():
            while msvcrt.kbhit():
                msvcrt.getwch()

        def pressed():
            hit = False
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    hit = True
            return hit
    except ImportError:
        import select

        def drain():
            while select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline():
                pass

        def pressed():
            hit = False
            while select.select([sys.stdin], [], [], 0)[0] and sys.stdin.readline():
                hit = True
            return hit

    return drain, pressed


def _coerce(keys):
    return [int(k) if str(k).lstrip("-").isdigit() else k for k in (keys or [])]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--name", default="trash_v1")
    ap.add_argument("--task", default="pick up the trash and drop it in the bin")
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--use", nargs="+", default=None)
    ap.add_argument("--ignore", nargs="+", default=None)
    ap.add_argument("--scene", default=None)
    ap.add_argument("--wrist", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    import yaml

    cfg = yaml.safe_load((Path(args.config) if args.config else REPO / "config.yaml").read_text())
    arms = arm_id.identify_arm_roles(cfg)

    roles = {}
    if args.scene is not None:
        roles[int(args.scene) if str(args.scene).isdigit() else args.scene] = "scene"
    if args.wrist is not None:
        roles[int(args.wrist) if str(args.wrist).isdigit() else args.wrist] = "wrist"
    selected = cameras.select_cameras(
        cameras.enumerate_cameras(),
        include=_coerce(args.use),
        exclude=list(LAPTOP_NAME_HINTS) + _coerce(args.ignore),
        roles=roles or None,
    )
    print("Recording cameras:\n" + cameras.describe(selected))
    extra = cameras.open_selected(selected, fps=args.fps)

    repo_id = f"local/{args.name}"
    root = REPO / "recorded" / args.name
    prepare_dataset_dir(repo_id, str(root), args.overwrite)

    follower = make_arm(port=arms["follower"]["port"], arm_id=cfg["robot"]["id"], calibrate=False)
    teleop = make_teleop(port=arms["leader"]["port"], teleop_id=cfg["teleop"]["id"], calibrate=False)

    drain, pressed = _enter_poller()

    def await_start(ep: int) -> bool:
        try:
            r = input(f"\n=== Episode {ep + 1}/{args.episodes} === ENTER to START (or 'q' + ENTER to finish): ")
        except (EOFError, KeyboardInterrupt):
            return False
        if r.strip().lower() == "q":
            return False
        drain()
        print("  recording... move the LEADER. ENTER to END this episode.")
        return True

    summary = None
    try:
        connect_with_retry(follower, "follower")
        connect_with_retry(teleop, "leader")
        for cam in extra.values():
            connect_with_retry(cam, "camera")
        with graceful_stop() as should_stop:
            summary = record_teleop_dataset(
                follower, teleop, repo_id=repo_id, task=args.task,
                cameras={}, extra_cameras=extra,
                num_episodes=args.episodes, episode_steps=None, fps=args.fps,
                root=str(root), overwrite=args.overwrite, push_to_hub=False,
                progress=lambda d, t: print(f"\r  frame {d}", end="", flush=True),
                should_stop=should_stop, await_start=await_start,
                end_episode=lambda step: pressed(),
            )
    finally:
        for dev in (teleop, follower):
            try:
                dev.disconnect()
            except Exception:
                pass
        for cam in extra.values():
            try:
                cam.close()
            except Exception:
                pass
        print("\nDisconnected.")

    if summary and summary.get("num_episodes"):
        print(f"\nRecorded {summary['num_episodes']} episode(s) -> {root}")
        print(f"Next: python pipeline/06_train_act.py --name {args.name}")
    else:
        print("No episodes recorded.")


if __name__ == "__main__":
    main()
```

### `pipeline/06_train_act.py`

```python
"""06 - train ACT on the recorded demos.

    python pipeline/06_train_act.py --name trash_v1                 # train locally (CPU/GPU)
    python pipeline/06_train_act.py --name trash_v1 --steps 60000
    python pipeline/06_train_act.py --name trash_v1 --qualia        # cloud (Qualia)
    python pipeline/06_train_act.py --name trash_v1 --push          # push dataset to HF first

Default trains ACT locally via LeRobot's CLI from recorded/<name>/. --qualia pushes
to the Hub and launches a managed finetune instead (spends credits). Either way the
output is a policy you point 07_deploy.py at.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _hf_user():
    from huggingface_hub import HfApi

    return HfApi().whoami()["name"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu | cuda")
    ap.add_argument("--push", action="store_true", help="push the dataset to HF before training")
    ap.add_argument("--qualia", action="store_true", help="train on Qualia cloud instead of locally")
    ap.add_argument("--hours", type=float, default=2.0)
    args = ap.parse_args()

    root = REPO / "recorded" / args.name
    if not root.exists():
        sys.exit(f"no dataset at {root} -- record it first (05_record_demos.py).")

    if args.qualia:
        from so101 import qualia_client as q
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id = f"{_hf_user()}/{args.name}"
        ds = LeRobotDataset(repo_id=repo_id, root=str(root))
        img_keys = [k for k in ds.features if k.startswith("observation.images.")]
        ds.push_to_hub()
        cam_map = {}
        pref = {"scene": "image_top", "wrist": "image_wrist"}
        slots = ["image_top", "image_wrist", "image_side"]
        for k in img_keys:
            name = k.rsplit(".", 1)[-1]
            slot = pref.get(name) or next((s for s in slots if s not in cam_map), None)
            if slot:
                cam_map[slot] = k
        print(f"Launching Qualia ACT finetune on {repo_id} ({args.hours}h), cameras {cam_map}")
        if input("Type 'yes' to spend credits: ").strip().lower() != "yes":
            sys.exit("aborted.")
        print(q.launch_finetune(dataset_id=repo_id, vla_type="act", model_id=None,
                                hours=args.hours, camera_mappings=cam_map))
        return

    repo_id = f"{_hf_user()}/{args.name}" if args.push else f"local/{args.name}"
    if args.push:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        LeRobotDataset(repo_id=repo_id, root=str(root)).push_to_hub()

    out = REPO / "outputs" / "train" / f"{args.name}_act"
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={repo_id}", f"--dataset.root={root}",
        "--policy.type=act", f"--policy.device={args.device}",
        f"--output_dir={out}", f"--steps={args.steps}",
        f"--batch_size={args.batch_size}", "--num_workers=0",
        "--save_freq=2000", "--wandb.enable=false",
    ]
    print("== " + " ".join(cmd) + " ==")
    subprocess.call(cmd)
    print(f"\nTrained policy under {out}")
    print(f"Next: python pipeline/07_deploy.py --policy {out}/checkpoints/last/pretrained_model --dry-run")


if __name__ == "__main__":
    main()
```

### `pipeline/07_deploy.py`

```python
"""07 - deploy: clear the table. Two modes.

    python pipeline/07_deploy.py --dry-run        # SCRIPTED pick, motion DISABLED (rehearse targets)
    python pipeline/07_deploy.py                   # SCRIPTED pick, motion ENABLED (the robust default)
    python pipeline/07_deploy.py --policy outputs/train/trash_v1_act/checkpoints/last/pretrained_model

SCRIPTED mode (default) runs the coord-grasp Orchestrator: zero-shot detect -> ROI
-> table coords -> reach filter -> scripted IK pick into the bin. This is "the
robust idea" deployed and needs no trained model -- the safest thing to demo.

POLICY mode (--policy PATH) runs your trained ACT on the follower via LeRobot, which
builds the camera observation and steps the policy. The exact LeRobot run-a-policy
flags are version-specific (this repo pins LeRobot 0.4.4) -- the command is printed
for you to confirm; pass --go to execute it.

SAFETY: starts slow (speeds from config.safety -- keep max_step_deg low). First
Ctrl+C trips the e-stop and returns home. Keep the workspace clear and the e-stop
reachable.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))


def run_scripted(cfg, *, execute: bool) -> None:
    from build_pipeline import build_detector, build_follower, build_localizer
    from trash_arm.camera import open_scene_camera
    from trash_arm.motion import Motion
    from trash_arm.orchestrator import Orchestrator
    from trash_arm.safety import EStop

    if cfg["calibration"] and build_localizer(cfg) is None:
        sys.exit("No homography -- run pipeline/03_calibrate_scene.py first.")

    estop = EStop()
    estop.install_sigint()
    camera = open_scene_camera(cfg)
    detector = build_detector(cfg)
    localizer = build_localizer(cfg)
    arm = build_follower(cfg)
    motion = Motion(arm=arm, cfg=cfg, estop=estop, execute=execute)
    orch = Orchestrator(camera, detector, localizer, motion, cfg, estop=estop)
    try:
        motion.go_home()
        orch.run()
    except Exception as e:
        print(f"\nstopped: {e}")
    finally:
        try:
            motion.go_home()
        except Exception:
            pass
        camera.close()
        arm.disconnect()
        print("arm disconnected.")


def run_policy(cfg, policy_path: str, *, go: bool) -> None:
    import arm_id  # noqa: F401 -- so101.arm_id via sys.path
    from so101 import arm_id as aid

    arms = aid.identify_arm_roles(cfg)
    r = cfg["robot"]
    cams = cfg["cameras"]
    # LeRobot runs a trained policy on the robot through `lerobot-record` with a
    # --policy.path. Camera/flag syntax is version-specific -- CONFIRM for LeRobot 0.4.4.
    cmd = [
        "lerobot-record",
        "--robot.type=so101_follower",
        f"--robot.port={arms['follower']['port']}",
        f"--robot.id={r['id']}",
        f"--robot.cameras={{ wrist: {{type: opencv, index_or_path: {cams['wrist']['index']}, "
        f"width: {cams['wrist']['width']}, height: {cams['wrist']['height']}, fps: {cams['wrist']['fps']}}} }}",
        f"--policy.path={policy_path}",
        "--dataset.repo_id=eval/trash_run",
        "--dataset.num_episodes=1",
        "--dataset.single_task=clear the table",
    ]
    print("== LeRobot policy run (confirm flags for your LeRobot version) ==")
    print(" \\\n  ".join(cmd))
    if not go:
        print("\n(dry preview -- re-run with --go to execute)")
        return
    subprocess.call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--policy", default=None, help="path/HF id of a trained ACT policy (POLICY mode)")
    ap.add_argument("--dry-run", action="store_true", help="scripted mode with motion disabled")
    ap.add_argument("--go", action="store_true", help="actually execute the LeRobot policy command")
    ap.add_argument("--yes", action="store_true", help="skip the 'will move the arm' confirmation")
    args = ap.parse_args()

    from trash_arm.config import load_config

    cfg = load_config(args.config)

    if args.policy:
        run_policy(cfg, args.policy, go=args.go)
        return

    if not args.dry_run and not args.yes:
        if input("This will MOVE the arm. Workspace clear, e-stop reachable? [y/N] ").strip().lower() != "y":
            sys.exit("aborted.")
    run_scripted(cfg, execute=not args.dry_run)


if __name__ == "__main__":
    main()
```

### `tests/test_cameras_select.py`

```python
"""Camera selection logic -- pure, no hardware."""

from so101.cameras import CameraInfo, select_cameras


def _fixture():
    return [
        CameraInfo(kind="oak", name="OAK-D (OAK-D-PRO)", mxid="14442C10"),
        CameraInfo(kind="uvc", name="USB2.0 HD IR UVC WebCam", index=0),  # laptop
        CameraInfo(kind="uvc", name="Logitech BRIO", index=1),            # wrist-ish
        CameraInfo(kind="uvc", name="Generic USB Camera", index=2),
    ]


def test_laptop_excluded_by_default():
    sel = select_cameras(_fixture())
    assert all("IR UVC" not in c.name for c in sel)
    assert {c.kind for c in sel} == {"oak", "uvc"}


def test_default_roles_oak_is_scene_first_webcam_is_wrist():
    sel = select_cameras(_fixture())
    by_role = {c.role: c for c in sel}
    assert by_role["scene"].kind == "oak"
    assert by_role["wrist"].index == 1  # Logitech, first non-laptop webcam
    assert by_role["side1"].index == 2


def test_include_allowlist_by_index_and_name():
    sel = select_cameras(_fixture(), include=[1, "OAK"])
    keys = {c.key() for c in sel}
    assert keys == {"uvc:1", "oak:14442C10"}


def test_explicit_role_override_wins():
    sel = select_cameras(_fixture(), roles={2: "wrist", "OAK": "scene"})
    by_role = {c.role: c for c in sel}
    assert by_role["wrist"].index == 2
    assert by_role["scene"].kind == "oak"


def test_extra_ignore_drops_named_camera():
    sel = select_cameras(_fixture(), exclude=("IR UVC", "Generic"))
    assert all("Generic" not in c.name for c in sel)
    assert any(c.index == 1 for c in sel)  # Logitech survives
```

### `tests/test_arm_id.py`

```python
"""Arm role-detection decision logic -- pure, no hardware."""

from so101.arm_id import decide_roles, movement_score


def test_movement_score_is_summed_peak_to_peak():
    samples = [
        {"shoulder_pan": 0.0, "elbow_flex": 10.0},
        {"shoulder_pan": 5.0, "elbow_flex": 10.0},
        {"shoulder_pan": -3.0, "elbow_flex": 12.0},
    ]
    # pan range = 5 - (-3) = 8 ; elbow range = 12 - 10 = 2
    assert movement_score(samples) == 10.0


def test_movement_score_handles_empty_and_singletons():
    assert movement_score([]) == 0.0
    assert movement_score([{"shoulder_pan": 1.0}]) == 0.0  # need >=2 samples per joint


def test_decide_roles_picks_the_moved_arm_as_leader():
    out = decide_roles(40.0, "COM4", 1.0, "COM5")
    assert out == {"leader": "COM4", "follower": "COM5"}
    out2 = decide_roles(0.5, "COM4", 38.0, "COM5")
    assert out2 == {"leader": "COM5", "follower": "COM4"}


def test_decide_roles_ambiguous_when_both_move():
    # both moved a lot and within ratio -> can't tell them apart
    assert decide_roles(30.0, "COM4", 25.0, "COM5") is None


def test_decide_roles_none_when_nothing_moved():
    assert decide_roles(2.0, "COM4", 1.0, "COM5") is None  # below min_move_deg
```
