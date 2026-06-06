"""Choose which detected item to pick next. Pure functions -- no hardware, no I/O.

Works on any object exposing ``.table_xy`` (an (x, y) tuple) and optionally
``.confidence`` -- e.g. perception.Detection, or a lightweight stand-in in tests.
"""

from __future__ import annotations

import math

STRATEGIES = ("nearest", "most_isolated", "confidence")


def _xy(det) -> tuple[float, float] | None:
    xy = det["table_xy"] if isinstance(det, dict) else getattr(det, "table_xy", None)
    return None if xy is None else (float(xy[0]), float(xy[1]))


def _conf(det) -> float:
    if isinstance(det, dict):
        return float(det.get("confidence", 0.0))
    return float(getattr(det, "confidence", 0.0))


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _localized(detections) -> list:
    return [d for d in detections if _xy(d) is not None]


def choose_next(detections, strategy: str = "nearest", reference=(0.0, 0.0)):
    """Return the next detection to pick (or None if there are none localized).

    nearest      -- closest to `reference` (e.g. the arm base / home). Default.
    most_isolated -- largest clearance to its nearest neighbour (least likely to
                     disturb other items when grasped).
    confidence   -- highest detector confidence.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; pick one of {STRATEGIES}")

    items = _localized(detections)
    if not items:
        return None
    ref = (float(reference[0]), float(reference[1]))

    if strategy == "nearest":
        return min(items, key=lambda d: _dist(_xy(d), ref))
    if strategy == "confidence":
        return max(items, key=_conf)

    # most_isolated: maximise distance to the closest *other* item.
    def clearance(d) -> float:
        others = [o for o in items if o is not d]
        if not others:
            return math.inf
        return min(_dist(_xy(d), _xy(o)) for o in others)

    return max(items, key=clearance)
