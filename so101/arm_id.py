"""Automatic leader / follower role detection for the two SO-101 arms.

Today the repo maps role -> port by a serial hard-coded in config.yaml. This adds
ZERO-config role detection so a fresh machine (or a teammate's) "just works":

  identify_arm_roles(cfg) returns {"follower": {...}, "leader": {...}} with each
  arm's port + serial, using, in order:
    1. FAST PATH  -- if config has serials and both are plugged in, match by serial
       (discovery.find_arm_ports). No interaction, no motion.
    2. WIGGLE TEST -- otherwise, open BOTH candidate ports as *leaders* (an SO-101
       leader never enables torque, so both arms stay freely hand-movable and this
       is safe), ask the user to gently move ONE arm, sample joint positions for a
       couple of seconds, and the arm that MOVED is the leader; the other is the
       follower. We then read each port's USB serial so it can be saved to config.

The movement maths is split into pure functions (movement_score / decide_roles)
so the decision logic is unit-tested with no hardware (tests/test_arm_id.py).
"""

from __future__ import annotations

import time

SO101_VID_PID = (0x1A86, 0x55D3)  # Waveshare/CH343 board used by the SO-101


# --------------------------------------------------------------------------- #
# pure decision logic (unit-tested, no hardware)
# --------------------------------------------------------------------------- #
def movement_score(samples: list[dict]) -> float:
    """How much an arm moved across a series of joint-dict samples.

    Sum of per-joint peak-to-peak (max - min) range, in degrees. Robust to slow
    drift and to a missing joint in some frames.
    """
    if not samples:
        return 0.0
    joints: set[str] = set()
    for s in samples:
        joints.update(s)
    total = 0.0
    for j in joints:
        vals = [s[j] for s in samples if j in s]
        if len(vals) >= 2:
            total += max(vals) - min(vals)
    return total


def decide_roles(
    score_a: float,
    port_a: str,
    score_b: float,
    port_b: str,
    *,
    min_move_deg: float = 5.0,
    ratio: float = 3.0,
) -> dict | None:
    """Given each port's movement score, decide which is leader/follower.

    Leader = the clearly-more-moved port. Returns None (ambiguous -> caller should
    re-prompt or fall back) unless the winner moved >= `min_move_deg` AND at least
    `ratio`x more than the other.
    """
    hi_port, hi, lo_port, lo = (
        (port_a, score_a, port_b, score_b)
        if score_a >= score_b
        else (port_b, score_b, port_a, score_a)
    )
    if hi < min_move_deg:
        return None  # nothing moved enough -- did the user move an arm?
    if lo > 0 and hi < ratio * lo:
        return None  # both moved similarly -- ambiguous
    return {"leader": hi_port, "follower": lo_port}


# --------------------------------------------------------------------------- #
# hardware helpers (IO)
# --------------------------------------------------------------------------- #
def candidate_ports(vid_pid=SO101_VID_PID) -> list:
    """COM/tty devices that look like SO-101 boards (by VID/PID), with serials."""
    from serial.tools import list_ports

    vid, pid = vid_pid
    return [p for p in list_ports.comports() if p.vid == vid and p.pid == pid]


def _serial_for_port(device: str) -> str | None:
    from serial.tools import list_ports

    for p in list_ports.comports():
        if p.device == device:
            return (p.serial_number or "").strip() or None
    return None


def _sample_motion(port: str, *, seconds: float, hz: float = 20.0) -> list[dict]:
    """Open `port` as a (torque-free) leader and sample joint dicts for `seconds`."""
    from .factory import make_teleop

    teleop = make_teleop(mock=False, port=port, teleop_id="roleprobe", calibrate=False)
    samples: list[dict] = []
    teleop.connect()
    try:
        t_end = time.time() + seconds
        period = 1.0 / hz
        while time.time() < t_end:
            try:
                samples.append(teleop.read_action())
            except Exception:
                pass
            time.sleep(period)
    finally:
        try:
            teleop.disconnect()
        except Exception:
            pass
    return samples


# --------------------------------------------------------------------------- #
# top-level entry point
# --------------------------------------------------------------------------- #
def identify_arm_roles(cfg: dict | None = None, *, interactive: bool = True,
                       seconds: float = 2.5) -> dict:
    """Resolve {"follower": {port, serial}, "leader": {port, serial}}.

    Fast path uses config serials; otherwise runs the wiggle test. Raises
    RuntimeError with guidance if it can't decide.
    """
    cfg = cfg or {}

    # 1) FAST PATH: serials known + both present.
    serials = {}
    if cfg.get("robot", {}).get("serial"):
        serials["follower"] = cfg["robot"]["serial"]
    if cfg.get("teleop", {}).get("serial"):
        serials["leader"] = cfg["teleop"]["serial"]
    if len(serials) == 2:
        try:
            from .discovery import find_arm_ports

            ports = find_arm_ports(serials)
            return {
                "follower": {"port": ports["follower"], "serial": serials["follower"]},
                "leader": {"port": ports["leader"], "serial": serials["leader"]},
            }
        except Exception as e:
            print(f"(serial fast-path failed: {e}; falling back to wiggle test)")

    # 2) WIGGLE TEST.
    ports = candidate_ports()
    if len(ports) != 2:
        raise RuntimeError(
            f"Expected exactly 2 SO-101 boards (VID/PID {SO101_VID_PID}); found "
            f"{len(ports)}: {[(p.device, p.serial_number) for p in ports]}. "
            "Check both USB + power, or set serials in config.yaml."
        )
    pa, pb = ports[0].device, ports[1].device

    if interactive:
        input(
            "\nRole detection: when you press ENTER, GENTLY move ONE arm "
            "(the leader) through a few joints for ~2-3 s.\nPress ENTER to start..."
        )
    print("  sampling both arms... move one arm now.")
    samples_a = _sample_motion(pa, seconds=seconds)
    samples_b = _sample_motion(pb, seconds=seconds)
    # ^ sampled sequentially; tell the user to keep moving across both windows, or
    #   run twice. (Sequential keeps it single-threaded and avoids two SDK opens at
    #   once; for simultaneous sampling, thread these two calls.)

    score_a, score_b = movement_score(samples_a), movement_score(samples_b)
    print(f"  movement: {pa}={score_a:.1f} deg, {pb}={score_b:.1f} deg")
    decided = decide_roles(score_a, pa, score_b, pb)
    if decided is None:
        raise RuntimeError(
            "Could not tell the arms apart (no clear movement). Re-run and move ONE "
            "arm clearly, keep it moving across BOTH sampling windows, or set serials "
            "in config.yaml."
        )

    return {
        "follower": {"port": decided["follower"], "serial": _serial_for_port(decided["follower"])},
        "leader": {"port": decided["leader"], "serial": _serial_for_port(decided["leader"])},
    }
