"""General camera auto-discovery + selection.

Generalises discovery.py's "find the OAK + one wrist webcam" into:
  * enumerate_cameras()  -- list EVERY camera the machine can see (OAK devices via
    DepthAI + all UVC indices via pygrabber names), as plain CameraInfo records.
  * select_cameras(...)  -- a PURE function (no IO) that decides which to USE and
    which to IGNORE, and assigns each a role (scene / wrist / sideN). You can drive
    it with include / exclude lists (by name-substring OR index) and an explicit
    role map. The laptop built-in cam is excluded by default.
  * open_selected(...)   -- open the chosen cameras and return the {role: camera}
    dict the recorder wants (extra_cameras) -- OAK and webcams behind one interface.

select_cameras is deliberately IO-free so it is unit-tested with no hardware
(see tests/test_cameras_select.py). All heavy imports stay lazy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .discovery import LAPTOP_NAME_HINTS, Cv2Camera, list_video_devices


@dataclass
class CameraInfo:
    """One discovered camera. `index` is set for UVC webcams, `mxid` for OAKs."""

    kind: str                      # "oak" | "uvc"
    name: str                      # human-readable device name
    index: int | None = None       # OpenCV index (UVC only)
    mxid: str | None = None         # DepthAI device id (OAK only)
    role: str | None = None         # assigned by select_cameras: scene/wrist/sideN

    def key(self) -> str:
        return f"oak:{self.mxid}" if self.kind == "oak" else f"uvc:{self.index}"


# --------------------------------------------------------------------------- #
# discovery (IO)
# --------------------------------------------------------------------------- #
def _enumerate_oaks() -> list[CameraInfo]:
    try:
        import depthai as dai
    except Exception:
        return []
    out: list[CameraInfo] = []
    try:
        for d in dai.Device.getAllAvailableDevices():
            mxid = getattr(d, "getMxId", lambda: None)() or getattr(d, "mxid", None)
            name = getattr(d, "name", None) or "OAK"
            out.append(CameraInfo(kind="oak", name=f"OAK-D ({name})", mxid=str(mxid)))
    except Exception:
        pass
    return out


def _enumerate_uvc() -> list[CameraInfo]:
    names = list_video_devices()  # pygrabber: names in OpenCV index order (or None)
    if names:
        return [CameraInfo(kind="uvc", name=n, index=i) for i, n in enumerate(names)]
    # Fallback (no pygrabber, e.g. Linux): probe openable indices and name generically.
    try:
        import cv2
    except Exception:
        return []
    out: list[CameraInfo] = []
    for i in range(8):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            out.append(CameraInfo(kind="uvc", name=f"video{i}", index=i))
        cap.release()
    return out


def enumerate_cameras() -> list[CameraInfo]:
    """Every camera visible right now: OAK devices first, then UVC webcams."""
    return _enumerate_oaks() + _enumerate_uvc()


# --------------------------------------------------------------------------- #
# selection (PURE -- unit-tested without hardware)
# --------------------------------------------------------------------------- #
def _matches(info: CameraInfo, key) -> bool:
    """A matcher key is either an int (== UVC index) or a str (case-insensitive
    substring of the device name)."""
    if isinstance(key, int):
        return info.index == key
    return str(key).lower() in info.name.lower()


def _matches_any(info: CameraInfo, keys) -> bool:
    return any(_matches(info, k) for k in (keys or []))


def select_cameras(
    infos: list[CameraInfo],
    *,
    include=None,
    exclude=LAPTOP_NAME_HINTS,
    roles: dict | None = None,
) -> list[CameraInfo]:
    """Choose which cameras to use and assign each a role. Pure / no IO.

    include : if given, an allowlist of matcher keys -- ONLY cameras matching one of
              them are kept (applied before exclude).
    exclude : matcher keys to drop (defaults to laptop built-in cam name hints).
    roles   : {matcher_key: role} -- force a role on a camera (e.g. {0: "scene",
              "Logitech": "wrist"}). Unforced cameras get default roles: the first
              OAK -> "scene", the first remaining webcam -> "wrist", then side1, side2...

    Returns the kept cameras (with `.role` set) in a stable order (forced roles first
    in scene/wrist/side order, then the rest).
    """
    kept = list(infos)
    if include:
        kept = [c for c in kept if _matches_any(c, include)]
    kept = [c for c in kept if not _matches_any(c, exclude)]

    # 1) forced roles
    forced: dict[str, CameraInfo] = {}
    roles = roles or {}
    for c in kept:
        for key, role in roles.items():
            if _matches(c, key):
                c.role = role
                forced[c.key()] = c
                break

    # 2) default roles for the rest
    rest = [c for c in kept if c.key() not in forced]
    side_n = 1
    used_roles = {c.role for c in forced.values()}
    for c in rest:
        if c.kind == "oak" and "scene" not in used_roles:
            c.role = "scene"
        elif "wrist" not in used_roles:
            c.role = "wrist"
        else:
            c.role = f"side{side_n}"
            side_n += 1
        used_roles.add(c.role)

    order = {"scene": 0, "wrist": 1}
    return sorted(kept, key=lambda c: (order.get(c.role, 2 + (c.index or 0)),))


# --------------------------------------------------------------------------- #
# opening (IO)
# --------------------------------------------------------------------------- #
def open_camera(info: CameraInfo, size=(640, 480), fps: int = 30):
    """Open one CameraInfo and return an object exposing .read()->RGB / .close().

    OAK -> coord_grasp.oak.OakCamera ; UVC -> discovery.Cv2Camera. (Single-OAK
    assumption: OakCamera grabs the default DepthAI device. Multi-OAK rigs need a
    device-id arg added to OakCamera -- left as a TODO.)
    """
    w, h = size
    if info.kind == "oak":
        from coord_grasp.oak import OakCamera

        return OakCamera(size=(w, h))
    if info.index is None:
        raise ValueError(f"UVC camera {info.name!r} has no index")
    return Cv2Camera(info.index, width=w, height=h, fps=fps)


def open_selected(selected: list[CameraInfo], *, sizes: dict | None = None, fps: int = 30) -> dict:
    """Open all selected cameras -> {role: camera_obj} for record's extra_cameras.

    `sizes` optionally maps role -> (w, h); otherwise 640x480 is used. Cameras are
    NOT connected here (the recorder/connect-with-retry does that).
    """
    sizes = sizes or {}
    out: dict[str, object] = {}
    for c in selected:
        role = c.role or c.key()
        out[role] = open_camera(c, size=sizes.get(role, (640, 480)), fps=fps)
    return out


def describe(infos: list[CameraInfo]) -> str:
    """One line per camera for console output."""
    if not infos:
        return "  (no cameras found)"
    lines = []
    for c in infos:
        where = f"index {c.index}" if c.kind == "uvc" else f"mxid {c.mxid}"
        role = f"  -> {c.role}" if c.role else ""
        lines.append(f"  [{c.kind:3}] {where:14}  {c.name}{role}")
    return "\n".join(lines)
