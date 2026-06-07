"""Workspace / joint limits, speed caps, and an e-stop. Pure + dependency-free.

Every commanded pose should pass through here: clamp/check joint limits, verify
the target is inside the reachable workspace, and step toward it no faster than
`max_step_deg` per tick. `EStop` lets Ctrl+C (or a hook) abort cleanly.
"""

from __future__ import annotations

import signal


class SafetyError(Exception):
    """Raised when a command would violate a safety limit."""


def in_workspace(x: float, y: float, ws: dict) -> bool:
    return ws["x_min"] <= x <= ws["x_max"] and ws["y_min"] <= y <= ws["y_max"]


def require_workspace(x: float, y: float, ws: dict) -> None:
    if not in_workspace(x, y, ws):
        raise SafetyError(
            f"target ({x:.3f}, {y:.3f}) outside workspace "
            f"x[{ws['x_min']},{ws['x_max']}] y[{ws['y_min']},{ws['y_max']}]"
        )


def clamp_z(z: float, ws: dict) -> float:
    """Never allow a commanded height below the hard floor."""
    return max(z, ws["z_floor"])


def clamp_joints(joints: dict[str, float], limits: dict[str, list]) -> dict[str, float]:
    """Clamp each joint into its [lo, hi] limit (joints without a limit pass through)."""
    out = {}
    for name, value in joints.items():
        if name in limits:
            lo, hi = limits[name]
            out[name] = max(lo, min(hi, value))
        else:
            out[name] = value
    return out


def check_joints(joints: dict[str, float], limits: dict[str, list]) -> None:
    """Raise SafetyError if any joint is outside its limit."""
    for name, value in joints.items():
        if name in limits:
            lo, hi = limits[name]
            if not (lo <= value <= hi):
                raise SafetyError(f"joint {name}={value:.1f} outside [{lo}, {hi}]")


def step_toward(
    current: dict[str, float], target: dict[str, float], max_step_deg: float
) -> dict[str, float]:
    """One capped move: each joint advances toward its target by <= max_step_deg."""
    out = dict(current)
    for name, tgt in target.items():
        cur = current.get(name, tgt)
        delta = tgt - cur
        if abs(delta) > max_step_deg:
            delta = max_step_deg if delta > 0 else -max_step_deg
        out[name] = cur + delta
    return out


def interpolate(
    current: dict[str, float], target: dict[str, float], max_step_deg: float
) -> list[dict[str, float]]:
    """Sequence of capped waypoints from `current` to `target` (inclusive of target)."""
    if max_step_deg <= 0:
        raise ValueError("max_step_deg must be > 0")
    waypoints: list[dict[str, float]] = []
    pos = dict(current)
    # Bound the iteration so a bad input can never spin forever.
    max_iters = 1 + int(
        max((abs(target[j] - pos.get(j, target[j])) for j in target), default=0.0)
        / max_step_deg
    )
    for _ in range(max_iters):
        pos = step_toward(pos, target, max_step_deg)
        waypoints.append(pos)
        if all(abs(target[j] - pos[j]) < 1e-6 for j in target):
            break
    return waypoints


class EStop:
    """A trip flag you can wire to Ctrl+C. Call `.check()` between motion steps."""

    def __init__(self) -> None:
        self._tripped = False

    @property
    def tripped(self) -> bool:
        return self._tripped

    def trip(self, *_args) -> None:
        self._tripped = True

    def check(self) -> None:
        if self._tripped:
            raise SafetyError("e-stop tripped")

    def install_sigint(self) -> None:
        """Make the first Ctrl+C trip the e-stop instead of killing the process."""
        signal.signal(signal.SIGINT, self.trip)
