"""Record a teleoperation LeRobotDataset (leader -> follower) with camera frames.

`record_teleop_dataset` mirrors the leader onto the follower and logs state +
action + every camera into a LeRobotDataset (the format ACT trains on). Mock runs
can synthesize a deterministic frame so the pipeline is testable with no hardware.
"""

from __future__ import annotations

import time

import numpy as np

from .interface import RobotArm
from .obs import SO101_JOINTS, observation_to_joints
from .teleop import Teleoperator

CAM_H, CAM_W = 120, 160  # default synthetic-frame size (mock fallback)


def _synthetic_frame(
    joints: dict[str, float], height: int = CAM_H, width: int = CAM_W
) -> np.ndarray:
    """A deterministic image that visibly depends on the joints (so it's not noise)."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 2] = np.linspace(20, 200, width, dtype=np.uint8)[None, :]  # blue gradient
    # a green marker whose position tracks two joints
    pan = joints.get("shoulder_pan", 0.0)
    grip = joints.get("gripper", 0.0)
    cx = int(np.clip((pan + 100) / 200 * (width - 1), 0, width - 1))
    cy = int(np.clip((grip + 100) / 200 * (height - 1), 0, height - 1))
    img[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5, 1] = 255
    return img


def build_teleop_features(cameras: dict, use_videos: bool = True) -> dict:
    """Dataset schema for teleop recording: state, action, and one image per camera.

    `cameras` maps name -> ``{"width", "height", ...}``. Camera keys follow the
    LeRobot convention: ``observation.images.<name>``. With `use_videos` the camera
    feature dtype is ``video`` (frames encoded to MP4 -> ~50-100x smaller datasets,
    far faster/robuster HF uploads); otherwise ``image`` (one PNG per frame).
    """
    n = len(SO101_JOINTS)
    feats = {
        "observation.state": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
        "action": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
    }
    cam_dtype = "video" if use_videos else "image"
    for name, spec in cameras.items():
        feats[f"observation.images.{name}"] = {
            "dtype": cam_dtype,
            "shape": (spec["height"], spec["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def dataset_root_path(repo_id: str, root=None):
    """Where a dataset will live: `root` if given, else LeRobot's cache / repo_id."""
    from pathlib import Path

    if root is not None:
        return Path(root)
    try:
        from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except Exception:  # pragma: no cover - import path varies by version
        try:
            from lerobot.constants import HF_LEROBOT_HOME
        except Exception:
            HF_LEROBOT_HOME = Path.home() / ".cache" / "huggingface" / "lerobot"
    return Path(HF_LEROBOT_HOME) / repo_id


def prepare_dataset_dir(repo_id: str, root=None, overwrite: bool = False):
    """Validate/clear the target dataset dir BEFORE any hardware is touched.

    LeRobot's `create()` refuses to write into an existing directory. This checks
    that up front so a name clash fails fast (not after connecting the arms):
      * empty leftover dir  -> removed silently
      * non-empty + overwrite -> removed
      * non-empty, no overwrite -> FileExistsError with an actionable message
    Returns the resolved target path.
    """
    import shutil

    target = dataset_root_path(repo_id, root)
    if target.exists():
        if not any(target.iterdir()):
            shutil.rmtree(target)
        elif overwrite:
            shutil.rmtree(target)
        else:
            raise FileExistsError(
                f"A dataset already exists at:\n  {target}\n"
                "Re-run with --overwrite to replace it, or pick a new "
                "--repo-id / --root."
            )
    return target


def record_teleop_dataset(
    follower: RobotArm,
    teleop: Teleoperator,
    repo_id: str,
    task: str,
    *,
    cameras: dict | None = None,
    extra_cameras: dict | None = None,
    num_episodes: int = 2,
    episode_steps: int | None = 200,
    reset_steps: int = 0,
    fps: int = 30,
    root=None,
    overwrite: bool = False,
    push_to_hub: bool = False,
    use_videos: bool = True,
    synthetic_frames: bool = False,
    on_step=None,
    on_images=None,
    progress=None,
    should_stop=None,
    await_start=None,
    end_episode=None,
) -> dict:
    """Teleoperate (leader drives follower) and record every frame to a LeRobotDataset.

    Each step: read the leader's target (``action``), command it to the follower so
    it physically tracks, then read the follower's measured pose (``observation.state``)
    and camera frames -- one synchronized read per step. The loop is paced to `fps`.

    Args:
        follower: the arm being recorded; if it has cameras configured they are
            captured from its observation. (A `RobotArm`; ``SO101Arm`` for real
            frames, ``MockArm`` for offline testing.)
        teleop: the leader supplying actions (``SO101Teleop`` or ``MockTeleop``).
        cameras: name -> spec dict; OpenCV cameras captured via the follower's
            observation. Defines which image keys land in the dataset.
        extra_cameras: name -> camera object exposing ``.read() -> HxWx3 RGB`` (e.g.
            an OAK via DepthAI, which isn't UVC so can't go through `cameras`).
            Captured each step *alongside* `cameras`, so OpenCV + non-UVC cameras
            record simultaneously. The caller connects/closes them.
        episode_steps: frames recorded per episode (≈ episode_seconds * fps), or
            None to record until `end_episode` fires (manual / Enter-controlled mode).
        reset_steps: frames to keep teleoperating *without* recording between
            episodes, so the human can reset the scene (time-based mode only).
        synthetic_frames: if a camera frame is missing from the observation (e.g.
            the mock follower has no real cameras), synthesize a deterministic one
            instead of failing -- lets the whole pipeline run with no hardware.
        on_step(state_joints, action_joints): per-frame hook (e.g. Rerun logging).
        on_images(images): per-frame hook with ``{name: HxWx3 RGB}`` for every camera
            (OpenCV + extra), e.g. to stream the live feeds to Rerun.
        should_stop(): optional predicate; if it returns True the loop ends cleanly
            after finishing the current frame (e.g. a Ctrl+C / e-stop flag).
        await_start(ep_index): optional callable invoked BEFORE each episode (may
            block, e.g. wait for ENTER). Return False to finish the session early.
            When given, the per-episode reset window is skipped (the prompt is the
            reset point).
        end_episode(step): optional per-tick predicate; return True to end the
            current episode (e.g. operator pressed ENTER). Pairs with episode_steps=None.

    Returns a summary dict.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    cameras = cameras or {}
    extra_cameras = extra_cameras or {}

    # Fail fast on a name clash BEFORE connecting any hardware.
    prepare_dataset_dir(repo_id, root, overwrite)

    if not follower.is_connected:
        follower.connect()
    if not teleop.is_connected:
        teleop.connect()

    # Extra cameras (e.g. an OAK via DepthAI) are any object with .read() -> RGB and
    # run alongside the follower's OpenCV cameras. Read one frame each to learn their
    # shape for the dataset schema.
    feature_cams = dict(cameras)
    for name, cam in extra_cameras.items():
        h, w = np.asarray(cam.read()).shape[:2]
        feature_cams[name] = {"width": w, "height": h}

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_teleop_features(feature_cams, use_videos=use_videos),
        root=root,
        robot_type="so101_follower",
        use_videos=use_videos,
    )

    period = 1.0 / fps
    total = num_episodes * episode_steps if episode_steps is not None else None
    done = 0
    episodes_recorded = 0

    def _teleop_step() -> tuple[dict, dict]:
        """One leader->follower mirror; returns (state_joints, action_joints)."""
        action = teleop.read_action()
        follower.write_joints(action)
        if cameras and hasattr(follower, "read_observation"):
            obs = follower.read_observation()
            state_joints = observation_to_joints(obs)
        else:
            state_joints = follower.read_joints()
            obs = {}
        return state_joints, action, obs

    stopped = False
    for ep in range(num_episodes):
        if stopped:
            break
        if await_start is not None and not await_start(ep):
            break  # operator chose to finish the session

        step = 0
        while True:
            t0 = time.perf_counter()
            state_joints, action, obs = _teleop_step()

            state = np.array([state_joints[j] for j in SO101_JOINTS], dtype=np.float32)
            act = np.array([action[j] for j in SO101_JOINTS], dtype=np.float32)
            frame = {"observation.state": state, "action": act, "task": task}
            images = {}
            for name, spec in cameras.items():
                if name in obs:
                    img = np.asarray(obs[name])
                elif synthetic_frames:
                    img = _synthetic_frame(state_joints, spec["height"], spec["width"])
                else:
                    raise KeyError(
                        f"camera {name!r} not in follower observation and "
                        "synthetic_frames=False"
                    )
                frame[f"observation.images.{name}"] = img
                images[name] = img
            for name, cam in extra_cameras.items():
                img = np.asarray(cam.read())
                frame[f"observation.images.{name}"] = img
                images[name] = img
            dataset.add_frame(frame)

            done += 1
            step += 1
            if on_step:
                on_step(state_joints, action)
            if on_images and images:
                on_images(images)
            if progress:
                progress(done, total)

            _sleep_remainder(t0, period)
            if should_stop and should_stop():
                stopped = True
                break
            if end_episode is not None and end_episode(step):
                break
            if episode_steps is not None and step >= episode_steps:
                break

        if step == 0:
            # Nothing recorded this episode (e.g. ended immediately) -- don't save.
            continue
        dataset.save_episode()
        episodes_recorded += 1

        # Time-based reset window (manual mode uses the start prompt as the reset point).
        if await_start is None and reset_steps and not stopped and episodes_recorded < num_episodes:
            for _ in range(reset_steps):
                t0 = time.perf_counter()
                _teleop_step()
                _sleep_remainder(t0, period)
                if should_stop and should_stop():
                    stopped = True
                    break

    if push_to_hub:
        # upload_large_folder = resumable, per-file retries (survives flaky networks);
        # far more robust than the default single-commit push.
        dataset.push_to_hub(upload_large_folder=True)

    return {
        "repo_id": repo_id,
        "num_episodes": episodes_recorded,
        "num_frames": done,
        "root": str(dataset.root),
        "pushed": push_to_hub,
        "stopped_early": stopped,
    }


def _sleep_remainder(t0: float, period: float) -> None:
    """Sleep just long enough to hold the loop at the target period."""
    dt = time.perf_counter() - t0
    if dt < period:
        time.sleep(period - dt)
