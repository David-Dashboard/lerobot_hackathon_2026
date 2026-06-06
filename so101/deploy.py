"""Run a policy on an arm -- the deployment loop.

A `Policy` maps current joints -> target joints. We ship stub policies so the
deploy path is demoable with NO trained model and NO hardware (e.g. on the mock
arm, `SinePolicy` makes it move autonomously). A real LeRobot policy plugs in via
`LeRobotPolicy` without changing the loop.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol

from .interface import RobotArm
from .obs import SO101_JOINTS


class Policy(Protocol):
    def select_action(self, joints: dict[str, float]) -> dict[str, float]: ...


class HoldPolicy:
    """Command the current position (no motion)."""

    def select_action(self, joints: dict[str, float]) -> dict[str, float]:
        return dict(joints)


class SinePolicy:
    """Autonomous sine sweep -- a visible 'it's running' demo on the mock arm."""

    def __init__(self, amplitude: float = 40.0, period_s: float = 4.0):
        self.amplitude = amplitude
        self.period_s = period_s
        self._t0: float | None = None

    def select_action(self, joints: dict[str, float]) -> dict[str, float]:
        if self._t0 is None:
            self._t0 = time.time()
        phase = 2 * math.pi * (time.time() - self._t0) / self.period_s
        return {j: self.amplitude * math.sin(phase + i) for i, j in enumerate(SO101_JOINTS)}


class LeRobotPolicy:
    """Adapter around a trained LeRobot policy loaded from a path/HF repo.

    Lazy-imports LeRobot. Needs the policy's expected observation (incl. camera),
    so it's used with a real arm + cameras -- not exercised in the hardware-free tests.
    """

    def __init__(self, path: str, device: str = "cpu"):
        self.path = path
        self.device = device
        self._policy = None

    def _ensure(self):
        if self._policy is None:
            from lerobot.policies.factory import make_policy  # noqa: F401

            raise NotImplementedError(
                "Wire make_policy(path) + observation tensors here for your trained model."
            )

    def select_action(self, joints: dict[str, float]) -> dict[str, float]:
        self._ensure()
        raise NotImplementedError


def load_policy(name_or_path: str) -> Policy:
    """'sine'/'hold' -> stub policy; anything else -> a trained LeRobot policy path."""
    key = (name_or_path or "").strip().lower()
    if key in ("sine", "demo", ""):
        return SinePolicy()
    if key in ("hold", "zero"):
        return HoldPolicy()
    return LeRobotPolicy(name_or_path)


def run_policy(
    arm: RobotArm,
    policy: Policy,
    *,
    steps: int = 200,
    fps: int = 30,
    should_stop: Callable[[], bool] | None = None,
    on_step: Callable[[int, int], None] | None = None,
) -> dict:
    """Closed loop: read -> policy -> command, for `steps` (or until should_stop)."""
    if not arm.is_connected:
        arm.connect()
    period = 1.0 / fps
    done = 0
    for t in range(steps):
        if should_stop and should_stop():
            break
        action = policy.select_action(arm.read_joints())
        arm.write_joints(action)
        done = t + 1
        if on_step:
            on_step(done, steps)
        time.sleep(period)
    return {"steps": done}
