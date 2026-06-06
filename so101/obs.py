"""Pure helpers for translating LeRobot observations <-> plain joint dicts.

No torch, no serial, no network -- trivially unit-testable. Used by the real-arm
adapter to keep LeRobot's ``"<joint>.pos"`` naming out of the rest of the code.
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


def observation_to_joints(observation: dict) -> dict[str, float]:
    """Extract joint positions from a LeRobot observation, dropping cameras/metadata.

    ``{"shoulder_pan.pos": 12.3, "front": <img>}`` -> ``{"shoulder_pan": 12.3}``
    """
    return {
        key[: -len(".pos")]: float(value)
        for key, value in observation.items()
        if key.endswith(".pos")
    }


def joints_to_action(positions: dict[str, float]) -> dict[str, float]:
    """Inverse: ``{"shoulder_pan": 12.3}`` -> ``{"shoulder_pan.pos": 12.3}``."""
    return {f"{name}.pos": float(value) for name, value in positions.items()}


def format_joint_line(positions: dict[str, float]) -> str:
    """A single aligned status line for console output."""
    return "  ".join(f"{name}={value:7.2f}" for name, value in positions.items())
