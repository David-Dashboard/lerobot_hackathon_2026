"""Hardware-discovery logic — port matching by serial, with no hardware."""

from dataclasses import dataclass

import pytest

from so101.discovery import find_arm_ports


@dataclass
class FakePort:
    device: str
    serial_number: str | None
    vid: int | None
    pid: int | None
    description: str = "USB-Enhanced-SERIAL CH343"


def _patch_ports(monkeypatch, ports):
    import serial.tools.list_ports as lp

    monkeypatch.setattr(lp, "comports", lambda: ports)


def test_match_by_serial_case_insensitive(monkeypatch):
    _patch_ports(monkeypatch, [
        FakePort("COM4", "5B41531811", 0x1A86, 0x55D3),
        FakePort("COM5", "5B41531706", 0x1A86, 0x55D3),
    ])
    got = find_arm_ports({"follower": "5b41531706", "leader": "5B41531811"})
    assert got == {"follower": "COM5", "leader": "COM4"}


def test_missing_serial_raises_with_available_ports(monkeypatch):
    _patch_ports(monkeypatch, [FakePort("COM5", "5B41531706", 0x1A86, 0x55D3)])
    with pytest.raises(RuntimeError) as e:
        find_arm_ports({"follower": "5B41531706", "leader": "DEADBEEF"})
    assert "leader" in str(e.value) and "COM5" in str(e.value)


def test_duplicate_serial_raises(monkeypatch):
    _patch_ports(monkeypatch, [FakePort("COM4", "AAA", 0x1A86, 0x55D3)])
    with pytest.raises(ValueError):
        find_arm_ports({"follower": "AAA", "leader": "AAA"})


def test_vid_pid_fallback_when_serial_none(monkeypatch):
    # One role unmatched by serial, exactly one no-serial CH343 port -> assigned.
    _patch_ports(monkeypatch, [
        FakePort("COM5", "5B41531706", 0x1A86, 0x55D3),   # follower (by serial)
        FakePort("COM9", None, 0x1A86, 0x55D3),           # leader (no serial -> fallback)
    ])
    got = find_arm_ports({"follower": "5B41531706", "leader": "WHATEVER"})
    assert got == {"follower": "COM5", "leader": "COM9"}


def test_find_wrist_camera_excludes_laptop_by_name(monkeypatch):
    import so101.discovery as d

    # External present -> pick it (skip the laptop entry).
    monkeypatch.setattr(d, "list_video_devices", lambda: ["USB2.0 HD IR UVC WebCam", "USB2.0_CAM1"])
    assert d.find_wrist_camera() == 1
    # Only the laptop -> None (don't record the built-in cam).
    monkeypatch.setattr(d, "list_video_devices", lambda: ["USB2.0 HD IR UVC WebCam"])
    assert d.find_wrist_camera() is None
    # Explicit override always wins.
    assert d.find_wrist_camera(override_index=3) == 3


def test_fallback_not_used_when_ambiguous(monkeypatch):
    # Two no-serial ports + one missing role -> ambiguous -> must raise, not guess.
    _patch_ports(monkeypatch, [
        FakePort("COM5", "5B41531706", 0x1A86, 0x55D3),
        FakePort("COM8", None, 0x1A86, 0x55D3),
        FakePort("COM9", None, 0x1A86, 0x55D3),
    ])
    with pytest.raises(RuntimeError):
        find_arm_ports({"follower": "5B41531706", "leader": "MISSING"})
