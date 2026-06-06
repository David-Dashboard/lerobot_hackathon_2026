"""The one abstraction everything else depends on.

A `RobotArm` speaks in plain joint dictionaries -- ``{"shoulder_pan": 12.3, ...}`` --
with NO knowledge of LeRobot, serial ports, or HTTP. The mock arm, the real arm,
the server, and the client all depend on *this* and nothing heavier. That keeps the
codebase decoupled: only `real.py` ever imports LeRobot.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

# Joint name -> position (degrees). e.g. {"shoulder_pan": 12.3, "gripper": -4.0}
JointPositions = dict[str, float]


@runtime_checkable
class RobotArm(Protocol):
    """Minimal interface for anything we can read from and command."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_joints(self) -> JointPositions:
        """Current joint positions."""

    def write_joints(self, positions: JointPositions) -> None:
        """Command target joint positions."""
