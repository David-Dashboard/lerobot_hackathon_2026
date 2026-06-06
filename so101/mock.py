"""A hardware-free SO-101 that satisfies `RobotArm`.

Produces deterministic, smoothly-varying joint positions so tests are reproducible
and Rerun plots actually move. Use it to develop the read loop, the server, the
client, and any control logic with NO arm plugged in -- then swap in `SO101Arm`.
"""

from __future__ import annotations

import math

from .obs import SO101_JOINTS


class MockArm:
    def __init__(self, arm_id: str = "mock_follower"):
        self.id = arm_id
        self._connected = False
        self._step = 0
        self.last_command: dict[str, float] | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_joints(self) -> dict[str, float]:
        self._require_connected()
        # Distinct phase per joint so the plots look different; fully deterministic.
        joints = {
            joint: 30.0 * math.sin(0.05 * self._step + i)
            for i, joint in enumerate(SO101_JOINTS)
        }
        self._step += 1
        return joints

    def write_joints(self, positions: dict[str, float]) -> None:
        self._require_connected()
        self.last_command = dict(positions)

    def _require_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("arm not connected -- call connect() first")
