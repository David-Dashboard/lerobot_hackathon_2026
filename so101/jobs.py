"""A dead-simple single-slot background job runner for the server.

The server controls one arm, so one long-running activity (record OR deploy) runs
at a time. This tracks its status and lets the UI poll and stop it.
"""

from __future__ import annotations

import threading
from typing import Callable


class BackgroundJob:
    def __init__(self):
        self._thread: threading.Thread | None = None
        self.kind: str | None = None
        self.running: bool = False
        self.error: str | None = None
        self.info: dict = {}
        self._stop = False

    def start(self, kind: str, target: Callable[["BackgroundJob"], None]) -> None:
        if self.running:
            raise RuntimeError(f"a '{self.kind}' job is already running")
        self.kind = kind
        self.running = True
        self.error = None
        self.info = {}
        self._stop = False

        def run():
            try:
                target(self)
            except Exception as e:  # noqa: BLE001 - report to UI
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self.running = False

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def request_stop(self) -> None:
        self._stop = True

    @property
    def should_stop(self) -> bool:
        return self._stop

    def status(self) -> dict:
        return {
            "kind": self.kind,
            "running": self.running,
            "error": self.error,
            **self.info,
        }

    def join(self, timeout: float | None = None) -> None:
        if self._thread:
            self._thread.join(timeout)
