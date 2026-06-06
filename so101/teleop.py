"""Teleoperator (leader arm) adapters -- the counterpart to the follower adapters.

During data collection a human moves the SO-101 *leader*; we read its joint
targets, mirror them onto the follower (so the follower physically tracks), and
log both: the leader targets become ``action`` and the follower's measured pose
becomes ``observation.state``. That is exactly what `lerobot-teleoperate` does,
plus we save every frame into a LeRobotDataset.

Like `real.py`, only `SO101Teleop` imports LeRobot, and it does so lazily, so the
rest of the package (and the mock-driven record loop) stays importable with no
hardware and no torch. `MockTeleop` lets the whole pipeline be exercised offline.
"""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

from .interface import JointPositions
from .obs import SO101_JOINTS, observation_to_joints


@runtime_checkable
class Teleoperator(Protocol):
    """Anything that can supply joint-target actions (a leader arm, a script, ...)."""

    @property
    def is_connected(self) -> bool: ...

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def read_action(self) -> JointPositions:
        """Current commanded joint targets (degrees), e.g. ``{"shoulder_pan": 12.3}``."""


class SO101Teleop:
    """A real SO-101 leader exposed through the `Teleoperator` interface.

    This is the only teleop module that imports LeRobot; the import is lazy so the
    package stays light where the leader isn't needed.
    """

    def __init__(self, port: str, teleop_id: str = "leader", calibrate: bool = True):
        self.port = port
        self.id = teleop_id
        self._calibrate = calibrate
        self._leader = None  # constructed on connect()

    @property
    def is_connected(self) -> bool:
        return self._leader is not None and self._leader.is_connected

    def connect(self) -> None:
        from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

        cfg = SO101LeaderConfig(port=self.port, id=self.id)
        self._leader = SO101Leader(cfg)
        self._leader.connect(calibrate=self._calibrate)

    def disconnect(self) -> None:
        if self._leader is not None:
            self._leader.disconnect()

    def read_action(self) -> JointPositions:
        self._require_connected()
        # get_action() returns ``{"<motor>.pos": value}`` -- same shape as an
        # observation, so the existing helper strips the ".pos" suffix for us.
        return observation_to_joints(self._leader.get_action())

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("teleop not connected -- call connect() first")


class MockTeleop:
    """A hardware-free leader that satisfies `Teleoperator`.

    Emits deterministic, smoothly-varying targets (distinct from `MockArm`'s own
    motion) so the record loop, dataset writing, and Rerun plots can be developed
    and tested with nothing plugged in.
    """

    def __init__(self, teleop_id: str = "mock_leader"):
        self.id = teleop_id
        self._connected = False
        self._step = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def read_action(self) -> JointPositions:
        if not self._connected:
            raise RuntimeError("teleop not connected -- call connect() first")
        # Phase-shifted from MockArm so action != state, like a real leader leads.
        action = {
            joint: 25.0 * math.sin(0.05 * self._step + i + 0.5)
            for i, joint in enumerate(SO101_JOINTS)
        }
        self._step += 1
        return action
