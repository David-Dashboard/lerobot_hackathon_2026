"""The perceive->decide->act loop, driven by fakes (no camera, model, or arm)."""

import numpy as np

from trash_arm.orchestrator import Orchestrator
from trash_arm.perception import Detection

CFG = {
    "workspace": {"x_min": 0.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0, "z_floor": 0.0},
    "perception": {"roi": None},
    "bin": {"xy": [0.0, 0.0]},
    "loop": {"strategy": "nearest", "max_items": 10, "max_retries": 2},
}


class FakeCamera:
    def read(self):
        return np.zeros((10, 10, 3), dtype=np.uint8)


class ShrinkingDetector:
    """Returns N items, then one fewer each perceive -- simulates clearing the table."""

    def __init__(self, n):
        self.n = n

    def detect(self, _img):
        dets = [
            Detection(label=f"item{i}", confidence=0.9, pixel_xy=(i, 0), bbox=(i, 0, i + 1, 1))
            for i in range(self.n)
        ]
        if self.n > 0:
            self.n -= 1
        return dets


class FakeLocalizer:
    def in_roi(self, u, v):
        return True

    def to_table(self, u, v):
        return (0.1 + 0.05 * u, 0.0)  # all inside CFG workspace


class FakeMotion:
    def __init__(self, succeed=True):
        self.calls = []
        self.succeed = succeed

    def pick_and_place(self, x, y):
        self.calls.append((x, y))
        return self.succeed


def test_loop_clears_table_then_stops():
    motion = FakeMotion(succeed=True)
    orch = Orchestrator(FakeCamera(), ShrinkingDetector(3), FakeLocalizer(), motion, CFG, log=lambda *_: None)
    summary = orch.run()
    assert summary["picked"] == 3
    assert summary["reason"] == "table_clear"
    assert len(motion.calls) == 3


def test_failed_grasp_is_retried_then_skipped():
    motion = FakeMotion(succeed=False)
    # one persistent item: detector always returns 1
    class OneItem:
        def detect(self, _img):
            return [Detection("stuck", 0.9, (1, 0), (1, 0, 2, 1))]

    cfg = {**CFG, "loop": {**CFG["loop"], "max_items": 1, "max_retries": 2}}
    orch = Orchestrator(FakeCamera(), OneItem(), FakeLocalizer(), motion, cfg, log=lambda *_: None)
    summary = orch.run()
    assert summary["skipped"] == 1
    assert len(motion.calls) == 3  # 1 try + 2 retries
