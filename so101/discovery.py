"""Auto-discovery of hardware so a recording script needs zero hardcoded device IDs.

  * find_arm_ports(serials)  -- map role -> COM port by STABLE board serial (COM
    numbers drift across replug/reboot; serials don't).
  * oak_present()            -- is a Luxonis OAK enumerable right now?
  * find_wrist_camera()      -- pick the external webcam's OpenCV index, excluding
    the laptop built-in cam (by DirectShow name via pygrabber).
  * Cv2Camera                -- minimal OpenCV webcam with the same .read()->RGB
    interface as coord_grasp.oak.OakCamera, so both feed the recorder as cameras.

All heavy/optional imports (cv2, pygrabber, depthai, pyserial) are lazy, so
importing this module stays cheap and unit tests can patch the enumerators.
"""

from __future__ import annotations

import numpy as np

# Substrings identifying common LAPTOP built-in cameras (excluded from wrist pick).
LAPTOP_NAME_HINTS = (
    "USB2.0 HD IR UVC WebCam",   # this machine's laptop cam (VID_13D3 PID_56CB)
    "Integrated Camera",
    "Integrated Webcam",
    "IR Camera",
    "HD User Facing",
    "Windows Hello",
)


def find_arm_ports(
    serials: dict[str, str],
    vid_pid_fallback: tuple[int, int] | None = (0x1A86, 0x55D3),
) -> dict[str, str]:
    """Map each role -> COM device by stable USB board serial (case-insensitive).

    `serials` is role -> expected serial, e.g. {"follower": "5B41531706", ...}.
    Raises RuntimeError (listing what's missing + all available ports) if any role
    can't be matched. A board whose serial_number is None is matched by VID/PID only
    when unambiguous (exactly one role left, one candidate).
    """
    from serial.tools import list_ports

    if not serials:
        raise ValueError("`serials` must not be empty")

    seen: dict[str, str] = {}
    for role, sn in serials.items():
        key = str(sn).strip().upper()
        if key in seen:
            raise ValueError(f"Duplicate serial {sn!r} for roles {seen[key]!r} and {role!r}")
        seen[key] = role

    ports = list(list_ports.comports())
    by_serial = {p.serial_number.strip().upper(): p.device for p in ports if p.serial_number}

    result: dict[str, str] = {}
    missing: list[tuple[str, str]] = []
    for role, sn in serials.items():
        dev = by_serial.get(str(sn).strip().upper())
        (result.__setitem__(role, dev) if dev else missing.append((role, sn)))

    if missing and vid_pid_fallback is not None:
        vid, pid = vid_pid_fallback
        no_serial = [p.device for p in ports if p.serial_number is None and p.vid == vid and p.pid == pid]
        if len(missing) == 1 and len(no_serial) == 1:
            role, _ = missing.pop()
            result[role] = no_serial[0]

    if missing:
        available = [
            (p.device, p.serial_number, hex(p.vid or 0), hex(p.pid or 0), p.description)
            for p in ports
        ]
        raise RuntimeError(
            "Could not match SO-101 arm ports by serial: "
            + ", ".join(f"{role}={sn!r}" for role, sn in missing)
            + f". Available ports: {available!r}"
        )
    return result


def oak_present() -> bool:
    """True if at least one DepthAI/OAK device is enumerable now (non-destructive)."""
    try:
        import depthai as dai

        return len(dai.Device.getAllAvailableDevices()) > 0
    except Exception:
        return False


def list_video_devices() -> list[str] | None:
    """DirectShow camera names in OpenCV index order (via pygrabber), or None."""
    try:
        from pygrabber.dshow_graph import FilterGraph

        return list(FilterGraph().get_input_devices())
    except Exception:
        return None


def find_wrist_camera(
    exclude_name_substr: tuple[str, ...] = LAPTOP_NAME_HINTS,
    override_index: int | None = None,
) -> int | None:
    """OpenCV index of the external WRIST webcam, excluding the laptop built-in cam.

    Preferred: pygrabber gives device names in cv2's DirectShow index order, so we
    drop any whose name matches a laptop hint and take the first survivor. Fallback
    (no pygrabber): assume index 0 is the laptop and take the next openable index.
    Returns None if no external camera is found. `override_index` always wins.
    """
    if override_index is not None:
        return override_index

    excl = tuple(s.lower() for s in exclude_name_substr)
    names = list_video_devices()
    if names:
        for idx, name in enumerate(names):
            if not any(e in name.lower() for e in excl):
                return idx
        return None  # only the laptop cam was present

    # Fallback (no pygrabber): probe openable indices, assume index 0 is the laptop
    # and take the first OTHER openable index. Returns None if the laptop is the only
    # camera (honouring the 'None when no external cam' contract).
    import cv2

    openable = []
    for i in range(8):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            openable.append(i)
        cap.release()
    external = [i for i in openable if i != 0]
    return external[0] if external else None


class Cv2Camera:
    """Minimal OpenCV webcam (DirectShow on Windows) exposing .read()->RGB.

    Same interface as coord_grasp.oak.OakCamera, so the recorder treats the OAK and
    a USB webcam identically (both as `extra_cameras`).
    """

    def __init__(self, index: int, width: int = 640, height: int = 480, fps: int = 30, backend=None):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._backend = backend
        self._cap = None
        self._cv2 = None

    @property
    def is_connected(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def connect(self) -> None:
        import cv2

        self._cv2 = cv2
        backend = self._backend if self._backend is not None else cv2.CAP_DSHOW
        cap = cv2.VideoCapture(self.index, backend)
        if not cap.isOpened():
            raise RuntimeError(f"could not open webcam index {self.index!r} (backend {backend})")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        # Warm up — first frames are often dark/empty.
        for _ in range(5):
            cap.read()
        self._cap = cap

    def read(self) -> np.ndarray:
        if self._cap is None:
            raise RuntimeError("Cv2Camera not connected -- call connect() first")
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"webcam index {self.index} read failed")
        return self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Cv2Camera":
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
