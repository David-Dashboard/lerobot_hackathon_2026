"""Minimal DepthAI wrapper for the OAK-D-PRO **RGB** camera.

Exposes the same ``.read() -> HxWx3 RGB uint8`` interface as
``trash_arm.camera.Camera``, so the OAK can stand in for an OpenCV webcam in our
own scripts. (LeRobot's ``opencv`` camera path can't see the OAK -- it isn't a UVC
device -- so we go through DepthAI here.)

RGB only for now; depth is available on the same device and can be added later.
"""

from __future__ import annotations

import numpy as np


class OakCamera:
    def __init__(self, size: tuple[int, int] = (640, 400), socket: str = "CAM_A"):
        self.size = size
        self.socket_name = socket
        self._dai = None
        self._pipeline = None
        self._queue = None

    @property
    def is_connected(self) -> bool:
        return self._pipeline is not None

    def connect(self) -> None:
        import depthai as dai

        self._dai = dai
        socket = getattr(dai.CameraBoardSocket, self.socket_name)
        self._pipeline = dai.Pipeline()
        cam = self._pipeline.create(dai.node.Camera).build(socket)
        out = cam.requestOutput(self.size)  # default type; getCvFrame() yields BGR
        self._queue = out.createOutputQueue()
        self._pipeline.start()

    def read(self) -> np.ndarray:
        """Blocking read of one RGB frame (HxWx3 uint8)."""
        if self._queue is None:
            raise RuntimeError("OakCamera not connected -- call connect() first")
        import cv2

        pkt = self._queue.get()
        return cv2.cvtColor(pkt.getCvFrame(), cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
            self._queue = None

    def __enter__(self) -> "OakCamera":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
