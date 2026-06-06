"""One place to choose mock vs. real, so callers never branch on it themselves."""

from __future__ import annotations

from .interface import RobotArm


def make_arm(
    mock: bool = False,
    port: str | None = None,
    arm_id: str = "follower",
    calibrate: bool = True,
) -> RobotArm:
    """Return a (not-yet-connected) `RobotArm`.

    mock=True  -> MockArm (no hardware)
    mock=False -> SO101Arm on `port` (requires LeRobot + a wired arm)
    """
    if mock:
        from .mock import MockArm

        return MockArm(arm_id=arm_id)

    if not port:
        raise ValueError("port is required for a real arm (or pass mock=True)")

    from .real import SO101Arm

    return SO101Arm(port=port, arm_id=arm_id, calibrate=calibrate)
