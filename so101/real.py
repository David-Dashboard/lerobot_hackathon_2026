"""Adapter: a real SO-101 follower exposed through the `RobotArm` interface.

This is the ONLY module that imports LeRobot (and therefore torch). The import is
lazy -- done inside methods -- so the rest of the package (mock, server, client,
tests) stays importable on a machine without the heavy deps or without an arm.
"""

from __future__ import annotations

from .obs import joints_to_action, observation_to_joints


class SO101Arm:
    def __init__(self, port: str, arm_id: str = "follower", calibrate: bool = True):
        self.port = port
        self.id = arm_id
        self._calibrate = calibrate
        self._robot = None  # constructed on connect()

    @property
    def is_connected(self) -> bool:
        return self._robot is not None and self._robot.is_connected

    def connect(self) -> None:
        # Lazy import keeps LeRobot/torch out of the import path everywhere else.
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

        cfg = SO101FollowerConfig(port=self.port, id=self.id, cameras={})
        self._robot = SO101Follower(cfg)
        self._robot.connect(calibrate=self._calibrate)

    def disconnect(self) -> None:
        if self._robot is not None:
            self._robot.disconnect()

    def read_joints(self) -> dict[str, float]:
        self._require_connected()
        return observation_to_joints(self._robot.get_observation())

    def write_joints(self, positions: dict[str, float]) -> None:
        self._require_connected()
        self._robot.send_action(joints_to_action(positions))

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("arm not connected -- call connect() first")
