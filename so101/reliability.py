"""Reliability helpers for hardware sessions: connect-with-retry + graceful Ctrl+C.

LeRobot 0.4.4 surfaces transient bus problems as ConnectionError (port open /
TxRx "There is no status packet!"), RuntimeError (handshake "Missing motor IDs",
or a per-packet error), or OSError (port). The common real failure we hit is a
single motor dropping out at connect/torque-enable from a loose daisy-chain cable
-- a retry (after clearing the half-open serial state) usually recovers it.
"""

from __future__ import annotations

import logging
import signal
import time
from contextlib import contextmanager

log = logging.getLogger("auto_record")

# Transient hardware/bus errors worth a retry.
DROPOUT_ERRORS = (ConnectionError, RuntimeError, OSError)


def connect_with_retry(device, name: str = "device", attempts: int = 3, delay: float = 2.0):
    """Connect anything with .connect()/.is_connected, retrying transient bus errors.

    Disconnects between attempts to clear half-open serial state, then re-raises a
    clear RuntimeError after the last failure so the caller aborts rather than
    running half-connected. Works for SO101Arm, SO101Teleop, OakCamera, Cv2Camera.
    """
    last = None
    for i in range(1, attempts + 1):
        try:
            if not getattr(device, "is_connected", False):
                device.connect()
            log.info("[%s] connected", name)
            return device
        except DROPOUT_ERRORS as e:
            last = e
            first_line = str(e).splitlines()[0] if str(e) else e.__class__.__name__
            log.warning(
                "[%s] connect attempt %d/%d failed: %s\n"
                "   -> check USB + barrel-jack power, the right device is plugged in, "
                "and (for arms) that all 6 motors are wired.",
                name, i, attempts, first_line,
            )
            try:
                device.disconnect()
            except Exception:
                try:
                    device.close()
                except Exception:
                    pass
            if i < attempts:
                time.sleep(delay)
    raise RuntimeError(
        f"[{name}] could not connect after {attempts} attempts; aborting (last error: {last})."
    ) from last


@contextmanager
def graceful_stop():
    """Yield a should_stop() predicate that flips True on the first Ctrl+C.

    The recorder checks should_stop() after each frame, so the current frame
    finishes and the episode is saved before stopping. A SECOND Ctrl+C restores
    default SIGINT for a hard abort. Restores the previous handler on exit.
    """
    state = {"flag": False}
    prev = signal.getsignal(signal.SIGINT)

    def handler(_sig, _frame):
        if state["flag"]:
            signal.signal(signal.SIGINT, prev)  # second hit -> hard abort next time
            raise KeyboardInterrupt
        state["flag"] = True
        log.info("Ctrl+C received -- finishing the current frame, saving, then stopping "
                 "(press Ctrl+C again to abort hard).")

    signal.signal(signal.SIGINT, handler)
    try:
        yield (lambda: state["flag"])
    finally:
        signal.signal(signal.SIGINT, prev)
