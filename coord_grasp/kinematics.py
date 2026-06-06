"""SO-101 forward kinematics: joint angles -> gripper-tip position in the robot
base frame. This is the "robot arm estimation" piece -- read the encoders, get the
end-effector location, with no vision.

It is the exact inverse of ``trash_arm.motion.plan_planar_ik`` and shares the same
geometry config (``ik`` block), so FK(IK(x,y,z)) == (x,y,z). Tuning the link
lengths / sign / offset there fixes both.
"""

from __future__ import annotations

import math

import numpy as np

_PLANAR_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")


def _raw_angles(joints: dict, ik: dict) -> dict:
    """Undo the servo deg = sign*degrees(angle) + offset mapping -> raw radians."""
    sign, offset = ik["sign"], ik["offset"]
    return {
        j: math.radians((joints[j] - offset[j]) / sign[j]) for j in _PLANAR_JOINTS
    }


def forward_kinematics(joints: dict, ik: dict) -> np.ndarray:
    """Gripper-tip (x, y, z) in the robot base frame for the given joint degrees."""
    raw = _raw_angles(joints, ik)
    pan = raw["shoulder_pan"]
    sh = raw["shoulder_lift"]
    a2 = sh + raw["elbow_flex"]
    a3 = a2 + raw["wrist_flex"]
    l1, l2, wl, bh = ik["l1"], ik["l2"], ik["wrist_length"], ik["base_height"]
    r = l1 * math.cos(sh) + l2 * math.cos(a2) + wl * math.cos(a3)
    z = bh + l1 * math.sin(sh) + l2 * math.sin(a2) + wl * math.sin(a3)
    return np.array([r * math.cos(pan), r * math.sin(pan), z])
