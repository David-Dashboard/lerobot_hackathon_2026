"""The perceive -> localize -> decide -> act -> loop controller.

Holds no hardware itself: it's handed a `camera`, `detector`, `localizer`, and
`motion`, each behind a small interface, so the whole loop is unit-testable with
fakes and each stage is independently swappable (e.g. scripted grasp -> ACT).

Stop conditions: no trash detected (table clear), the `max_items` cap, or the
e-stop. Failed grasps are retried up to `max_retries`, then skipped.
"""

from __future__ import annotations

from . import decide, safety
from .perception import filter_roi


class Orchestrator:
    def __init__(self, camera, detector, localizer, motion, cfg, estop=None, log=print):
        self.camera = camera
        self.detector = detector
        self.localizer = localizer
        self.motion = motion
        self.cfg = cfg
        self.estop = estop
        self.log = log

    def perceive(self) -> list:
        """Grab a frame, detect, drop ROI/unreachable, and attach table coords.
        Returns detections that are reachable trash, with `.table_xy` set."""
        frame = self.camera.read()
        dets = self.detector.detect(frame)
        dets = filter_roi(dets, self.cfg["perception"].get("roi"))

        ws = self.cfg["workspace"]
        reachable = []
        for d in dets:
            u, v = d.pixel_xy
            if self.localizer is not None and not self.localizer.in_roi(u, v):
                continue
            d.table_xy = self.localizer.to_table(u, v) if self.localizer else None
            if d.table_xy is None:
                self.log(f"   skip {d.label} (no table coords -- localizer/homography missing)")
                continue
            if safety.in_workspace(d.table_xy[0], d.table_xy[1], ws):
                reachable.append(d)
            else:
                self.log(f"   skip {d.label} @ {_fmt_xy(d.table_xy)} (out of workspace)")
        return reachable

    def run(self) -> dict:
        """Clear the table. Returns a summary dict (picked/skipped/reason)."""
        loop = self.cfg["loop"]
        strategy = loop.get("strategy", "nearest")
        reference = tuple(self.cfg["bin"]["xy"])  # pick nearest to bin side by default
        picked = skipped = 0
        reason = "max_items"

        for i in range(loop["max_items"]):
            if self.estop:
                self.estop.check()
            self.log(f"\n[iter {i + 1}] perceiving ...")
            dets = self.perceive()
            self.log(f"   {len(dets)} item(s): " + ", ".join(_describe(d) for d in dets))

            if not dets:
                reason = "table_clear"
                self.log("   table clear -- done.")
                break

            target = decide.choose_next(dets, strategy=strategy, reference=reference)
            self.log(f"   target: {target.label} @ {_fmt_xy(target.table_xy)}")

            if self._attempt_pick(target):
                picked += 1
            else:
                skipped += 1
                self.log(f"   skipping {target.label} after retries")
        else:
            self.log(f"   reached max_items={loop['max_items']}")

        summary = {"picked": picked, "skipped": skipped, "reason": reason}
        self.log(f"\nRun complete: {summary}")
        return summary

    def _attempt_pick(self, target) -> bool:
        """pick_and_place with retries. A dry-run motion (execute=False) just plans."""
        retries = self.cfg["loop"]["max_retries"]
        x, y = target.table_xy
        for attempt in range(retries + 1):
            if self.estop:
                self.estop.check()
            try:
                if self.motion.pick_and_place(x, y):
                    return True
            except safety.SafetyError:
                raise
            except Exception as e:  # a slipped grasp / transient fault -> retry
                self.log(f"   attempt {attempt + 1} error: {e}")
            self.log(f"   attempt {attempt + 1} failed, retrying ...")
        return False


def _fmt_xy(xy) -> str:
    return "n/a" if xy is None else f"({xy[0]:.3f}, {xy[1]:.3f})"


def _describe(d) -> str:
    return f"{d.label}{_fmt_xy(d.table_xy)}@{d.confidence:.2f}"
