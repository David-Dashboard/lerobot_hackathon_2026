"""One place to choose mock vs. real, so callers never branch on it themselves."""

from __future__ import annotations

from .interface import RobotArm


def make_arm(
    mock: bool = False,
    port: str | None = None,
    arm_id: str = "follower",
    calibrate: bool = True,
    cameras: dict | None = None,
) -> RobotArm:
    """Return a (not-yet-connected) `RobotArm`.

    mock=True  -> MockArm (no hardware)
    mock=False -> SO101Arm on `port` (requires LeRobot + a wired arm)

    `cameras` (real arm only) maps a name to ``{"index", "width", "height", "fps"}``
    so the follower also streams frames -- see `record.parse_camera_spec`.
    """
    if mock:
        from .mock import MockArm

        return MockArm(arm_id=arm_id)

    if not port:
        raise ValueError("port is required for a real arm (or pass mock=True)")

    from .real import SO101Arm

    return SO101Arm(port=port, arm_id=arm_id, calibrate=calibrate, cameras=cameras)


def make_teleop(
    mock: bool = False,
    port: str | None = None,
    teleop_id: str = "leader",
    calibrate: bool = True,
):
    """Return a (not-yet-connected) `Teleoperator` (the leader arm).

    mock=True  -> MockTeleop (no hardware)
    mock=False -> SO101Teleop on `port` (requires LeRobot + a wired leader)
    """
    if mock:
        from .teleop import MockTeleop

        return MockTeleop(teleop_id=teleop_id)

    if not port:
        raise ValueError("port is required for a real leader (or pass mock=True)")

    from .teleop import SO101Teleop

    return SO101Teleop(port=port, teleop_id=teleop_id, calibrate=calibrate)
