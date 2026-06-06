"""
A hardware-free stand-in for an SO-101 follower arm.

`MockSO101Follower` duck-types the parts of the LeRobot robot interface that our
scripts actually use (connect / get_observation / send_action / disconnect), so you
can develop and test the read loop, Rerun visualization, and any control logic with
NO arm plugged in. Swap it for the real `SO101Follower` when hardware is ready.

It produces deterministic, smoothly-varying joint positions (sine waves) so tests
are reproducible and the Rerun plots actually move.
"""

from __future__ import annotations

import math

from arm_utils import SO101_JOINTS


class MockSO101Follower:
    def __init__(self, port: str = "MOCK", robot_id: str = "mock_follower"):
        self.port = port
        self.id = robot_id
        self._connected = False
        self._step = 0
        self.last_action: dict[str, float] | None = None

    # --- lifecycle -------------------------------------------------------
    def connect(self, calibrate: bool = False) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    # --- I/O -------------------------------------------------------------
    def get_observation(self) -> dict:
        """Return a LeRobot-style observation: one ``<joint>.pos`` per joint."""
        if not self._connected:
            raise RuntimeError("MockSO101Follower.get_observation() before connect()")
        obs: dict[str, float] = {}
        for i, joint in enumerate(SO101_JOINTS):
            # Each joint a different phase so the plots look distinct; deterministic.
            obs[f"{joint}.pos"] = 30.0 * math.sin(0.05 * self._step + i)
        self._step += 1
        return obs

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        """Pretend to actuate; just record what was commanded."""
        if not self._connected:
            raise RuntimeError("MockSO101Follower.send_action() before connect()")
        self.last_action = dict(action)
        return self.last_action
