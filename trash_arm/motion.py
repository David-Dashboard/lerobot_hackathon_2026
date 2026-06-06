"""Motion: planar inverse kinematics + the `pick_and_place(x, y)` primitive.

Geometry (simplified, good enough for a flat table):
  * shoulder_pan sets the azimuth:           pan = atan2(y, x)
  * shoulder_lift + elbow_flex form a 2-link planar arm reaching radius r, height z
  * wrist_flex keeps the gripper pointing straight down

The raw IK angles are mapped to servo degrees via per-joint sign/offset in
config (`ik.sign`, `ik.offset`) because the servo zero/direction depends on YOUR
calibration -- tune those with scripts/dry_run.py and slow jogging before trusting
real motion. Link lengths (`ik.l1/l2/base_height/wrist_length`) likewise need
measuring. The IK math itself (reach/branch) is unit-tested.

This module is intentionally kept independent of *how* the arm arrived at the
target, so Phase 3 (the mobile base) can drive to an item and call the unchanged
`pick_and_place`.
"""

from __future__ import annotations

import math
import time

from . import safety


class ReachError(Exception):
    """Target is outside the arm's geometric reach (or near a singularity)."""


def plan_planar_ik(x: float, y: float, z: float, ik: dict) -> dict[str, float]:
    """Solve for shoulder_pan/lift, elbow_flex, wrist_flex (servo degrees) so the
    gripper tip reaches table point (x, y) at height z, pointing straight down.

    Raises ReachError if (x, y, z) is unreachable.
    """
    l1, l2 = ik["l1"], ik["l2"]
    base_h, wrist_len = ik["base_height"], ik["wrist_length"]

    pan = math.atan2(y, x)
    r = math.hypot(x, y)

    # Wrist pivot sits `wrist_len` above the tip (gripper points down), measured
    # from the shoulder pivot height.
    dr = r
    dz = (z + wrist_len) - base_h

    reach = math.hypot(dr, dz)
    if reach > (l1 + l2) or reach < abs(l1 - l2):
        raise ReachError(
            f"({x:.3f},{y:.3f},{z:.3f}) -> planar reach {reach:.3f} m outside "
            f"[{abs(l1 - l2):.3f}, {l1 + l2:.3f}]"
        )

    cos_elbow = (dr * dr + dz * dz - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    cos_elbow = max(-1.0, min(1.0, cos_elbow))
    sin_elbow = math.sqrt(1.0 - cos_elbow * cos_elbow)
    if not ik.get("elbow_up", False):
        sin_elbow = -sin_elbow
    elbow = math.atan2(sin_elbow, cos_elbow)

    shoulder = math.atan2(dz, dr) - math.atan2(l2 * math.sin(elbow), l1 + l2 * math.cos(elbow))
    # Keep the gripper vertical: the three planar angles must sum to -90 deg (down).
    wrist = -math.pi / 2 - (shoulder + elbow)

    raw = {  # radians, before per-joint sign/offset mapping
        "shoulder_pan": pan,
        "shoulder_lift": shoulder,
        "elbow_flex": elbow,
        "wrist_flex": wrist,
    }
    sign, offset = ik["sign"], ik["offset"]
    return {j: sign[j] * math.degrees(a) + offset[j] for j, a in raw.items()}


class Motion:
    """Commands a follower `RobotArm` through safe, speed-capped moves and the
    full pick-and-place sequence. `execute=False` plans + logs but never moves
    (used by dry_run)."""

    def __init__(self, arm, cfg: dict, estop: safety.EStop | None = None, execute: bool = True, log=print):
        self.arm = arm
        self.cfg = cfg
        self.estop = estop
        self.execute = execute
        self.log = log
        self.limits = cfg["safety"]["joint_limits"]
        self.max_step = cfg["safety"]["max_step_deg"]
        self.step_pause = cfg["safety"]["step_pause_s"]
        self.settle = cfg["safety"]["settle_s"]

    # --- low-level moves -------------------------------------------------

    def move_to_joints(self, target: dict[str, float]) -> None:
        """Interpolate from the current pose to `target`, capped to max_step_deg/tick.
        Every waypoint is clamped to joint limits and checked against the e-stop."""
        target = safety.clamp_joints(target, self.limits)
        safety.check_joints(target, self.limits)
        if not self.execute:
            self.log(f"   [plan] move_to_joints {_fmt(target)}")
            return
        current = self.arm.read_joints()
        for wp in safety.interpolate(current, {**current, **target}, self.max_step):
            if self.estop:
                self.estop.check()
            self.arm.write_joints(wp)
            time.sleep(self.step_pause)
        time.sleep(self.settle)

    def move_to_xyz(self, x: float, y: float, z: float, gripper: float | None = None) -> None:
        """Move the gripper tip to table (x, y) at height z (pointing down)."""
        ws = self.cfg["workspace"]
        safety.require_workspace(x, y, ws)
        z = safety.clamp_z(z, ws)
        joints = plan_planar_ik(x, y, z, self.cfg["ik"])
        joints["wrist_roll"] = self.arm.read_joints().get("wrist_roll", 0.0) if self.execute else 0.0
        if gripper is not None:
            joints["gripper"] = gripper
        self.move_to_joints(joints)

    def set_gripper(self, value: float) -> None:
        cur = self.arm.read_joints() if self.execute else {}
        self.move_to_joints({**cur, "gripper": value})

    def open_gripper(self) -> None:
        self.set_gripper(self.cfg["gripper"]["open"])

    def close_gripper(self) -> None:
        self.set_gripper(self.cfg["gripper"]["closed"])

    def go_home(self) -> None:
        self.log(" -> home")
        self.move_to_joints(dict(self.cfg["home"]["joints"]))

    # --- grasp check -----------------------------------------------------

    def grasp_succeeded(self) -> bool:
        """Heuristic: if the gripper closed almost fully it caught nothing.
        (Tune `gripper.miss_below`; later phases can also check servo load / wrist cam.)"""
        if not self.execute:
            return True
        pos = self.arm.read_joints().get("gripper", 0.0)
        ok = pos > self.cfg["gripper"]["miss_below"]
        self.log(f"   gripper={pos:.1f} -> {'holding' if ok else 'MISS'}")
        return ok

    # --- the primitive ---------------------------------------------------

    def pick_and_place(self, x: float, y: float) -> bool:
        """approach above -> descend -> grasp -> lift -> move to bin -> release -> home.
        Returns True if the grasp looked successful. Raises on out-of-workspace target."""
        ws, b = self.cfg["workspace"], self.cfg["bin"]
        safety.require_workspace(x, y, ws)
        self.log(f"pick_and_place at ({x:.3f}, {y:.3f})")

        self.open_gripper()
        self.move_to_xyz(x, y, ws["z_approach"])              # hover above target
        self.move_to_xyz(x, y, ws["z_grasp"])                 # descend
        self.close_gripper()                                  # grasp
        self.move_to_xyz(x, y, ws["z_lift"])                  # lift

        if not self.grasp_succeeded():
            self.open_gripper()
            self.go_home()
            return False

        bx, by = b["xy"]
        self.move_to_xyz(bx, by, b["z_release"])              # over the bin
        self.open_gripper()                                   # release
        self.go_home()
        return True


def _fmt(joints: dict[str, float]) -> str:
    return " ".join(f"{k}={v:.1f}" for k, v in joints.items())
