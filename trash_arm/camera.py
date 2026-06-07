"""Thin OpenCV camera wrapper returning RGB frames.

Kept tiny and behind `.read() -> HxWx3 RGB uint8` so the orchestrator depends on
an interface, not on cv2 -- tests pass a fake camera instead.
"""

from __future__ import annotations

import numpy as np


class Camera:
    def __init__(self, index, width: int = 640, height: int = 480, fps: int = 30):
        import cv2

        self._cv2 = cv2
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera index {index!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

    def read(self) -> np.ndarray:
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed (frame dropped?)")
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def open_scene_camera(cfg: dict) -> Camera:
    """Build the scene camera from config."""
    c = cfg["cameras"]["scene"]
    return Camera(c["index"], c["width"], c["height"], c.get("fps", 30))
