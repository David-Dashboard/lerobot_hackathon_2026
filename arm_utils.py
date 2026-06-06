"""
Pure, hardware-free helpers for working with SO-101 observations.

Kept free of any robot/serial imports so they can be unit-tested without an arm.
"""

from __future__ import annotations

# The six SO-101 joints, in bus order.
SO101_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def extract_joint_positions(observation: dict) -> dict[str, float]:
    """Pull just the joint positions out of a LeRobot observation dict.

    A LeRobot observation looks like ``{"shoulder_pan.pos": 12.3, ..., "front": <img>}``.
    We keep only the ``*.pos`` entries and strip the suffix, ignoring camera frames
    and anything else.
    """
    joints: dict[str, float] = {}
    for key, value in observation.items():
        if key.endswith(".pos"):
            joints[key[: -len(".pos")]] = float(value)
    return joints


def format_joint_line(joints: dict[str, float]) -> str:
    """Render joints as a single aligned status line for console output."""
    return "  ".join(f"{name}={value:7.2f}" for name, value in joints.items())
