"""Record a LeRobotDataset from any `RobotArm` -- mock or real.

For the mock, we synthesize a deterministic camera frame from the joint state so
the resulting dataset is a valid *vision* dataset (trainable by a VLA), with no
hardware. Stored as images (use_videos=False) so it needs no ffmpeg.
"""

from __future__ import annotations

import time

import numpy as np

from .interface import RobotArm
from .obs import SO101_JOINTS, observation_to_joints
from .teleop import Teleoperator

CAM_H, CAM_W = 120, 160
CAMERA_KEY = "observation.images.front"


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


def parse_camera_spec(spec: str) -> tuple[str, dict]:
    """Parse a CLI camera spec ``name=index[:WxH][@fps]`` into ``(name, config)``.

    Examples::

        "front=0"             -> ("front", {"index": 0, "width": 640, "height": 480, "fps": 30})
        "wrist=2:1280x720@15" -> ("wrist", {"index": 2, "width": 1280, "height": 720, "fps": 15})

    ``index`` stays a string if it isn't an integer (so device paths work too).
    """
    if "=" not in spec:
        raise ValueError(f"camera spec must be name=index[:WxH][@fps], got {spec!r}")
    name, rest = spec.split("=", 1)
    name = name.strip()
    if not name:
        raise ValueError(f"camera spec has empty name: {spec!r}")

    fps = 30
    if "@" in rest:
        rest, fps_str = rest.split("@", 1)
        fps = int(fps_str)

    width, height = 640, 480
    if ":" in rest:
        index_str, res = rest.split(":", 1)
        if "x" not in res:
            raise ValueError(f"resolution must be WxH, got {res!r} in {spec!r}")
        w_str, h_str = res.lower().split("x", 1)
        width, height = int(w_str), int(h_str)
    else:
        index_str = rest

    index_str = index_str.strip()
    index: int | str = int(index_str) if index_str.lstrip("-").isdigit() else index_str
    return name, {"index": index, "width": width, "height": height, "fps": fps}


def build_features(with_camera: bool = True) -> dict:
    n = len(SO101_JOINTS)
    feats = {
        "observation.state": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
        "action": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
    }
    if with_camera:
        feats[CAMERA_KEY] = {
            "dtype": "image",
            "shape": (CAM_H, CAM_W, 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def record_dataset(
    arm: RobotArm,
    repo_id: str,
    task: str,
    *,
    num_episodes: int = 2,
    episode_steps: int = 20,
    fps: int = 30,
    root=None,
    with_camera: bool = True,
    push_to_hub: bool = False,
    progress=None,
) -> dict:
    """Record `num_episodes` of mock/real teleop into a LeRobotDataset.

    `progress(done_frames, total_frames)` is called as it runs (for the UI).
    Returns a summary dict.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if not arm.is_connected:
        arm.connect()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_features(with_camera),
        root=root,
        robot_type="so101_follower",
        use_videos=False,
    )

    total = num_episodes * episode_steps
    done = 0
    for _ in range(num_episodes):
        for _ in range(episode_steps):
            joints = arm.read_joints()
            state = np.array([joints[j] for j in SO101_JOINTS], dtype=np.float32)
            frame = {"observation.state": state, "action": state.copy(), "task": task}
            if with_camera:
                frame[CAMERA_KEY] = _synthetic_frame(joints)
            dataset.add_frame(frame)
            done += 1
            if progress:
                progress(done, total)
        dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

    return {
        "repo_id": repo_id,
        "num_episodes": num_episodes,
        "num_frames": total,
        "root": str(dataset.root),
        "pushed": push_to_hub,
    }


def build_teleop_features(cameras: dict) -> dict:
    """Dataset schema for teleop recording: state, action, and one image per camera.

    `cameras` maps name -> ``{"width", "height", ...}`` (the dict from
    `parse_camera_spec`). Camera keys follow LeRobot convention:
    ``observation.images.<name>``.
    """
    n = len(SO101_JOINTS)
    feats = {
        "observation.state": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
        "action": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
    }
    for name, spec in cameras.items():
        feats[f"observation.images.{name}"] = {
            "dtype": "image",
            "shape": (spec["height"], spec["width"], 3),
            "names": ["height", "width", "channels"],
        }
    return feats


def record_teleop_dataset(
    follower: RobotArm,
    teleop: Teleoperator,
    repo_id: str,
    task: str,
    *,
    cameras: dict | None = None,
    num_episodes: int = 2,
    episode_steps: int | None = 200,
    reset_steps: int = 0,
    fps: int = 30,
    root=None,
    push_to_hub: bool = False,
    synthetic_frames: bool = False,
    on_step=None,
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
        cameras: name -> spec dict; defines which image keys land in the dataset.
        episode_steps: frames recorded per episode (≈ episode_seconds * fps), or
            None to record until `end_episode` fires (manual / Enter-controlled mode).
        reset_steps: frames to keep teleoperating *without* recording between
            episodes, so the human can reset the scene (time-based mode only).
        synthetic_frames: if a camera frame is missing from the observation (e.g.
            the mock follower has no real cameras), synthesize a deterministic one
            instead of failing -- lets the whole pipeline run with no hardware.
        on_step(state_joints, action_joints): per-frame hook (e.g. Rerun logging).
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

    if not follower.is_connected:
        follower.connect()
    if not teleop.is_connected:
        teleop.connect()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_teleop_features(cameras),
        root=root,
        robot_type="so101_follower",
        use_videos=False,
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
            for name, spec in cameras.items():
                if name in obs:
                    frame[f"observation.images.{name}"] = np.asarray(obs[name])
                elif synthetic_frames:
                    frame[f"observation.images.{name}"] = _synthetic_frame(
                        state_joints, spec["height"], spec["width"]
                    )
                else:
                    raise KeyError(
                        f"camera {name!r} not in follower observation and "
                        "synthetic_frames=False"
                    )
            dataset.add_frame(frame)

            done += 1
            step += 1
            if on_step:
                on_step(state_joints, action)
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
        dataset.push_to_hub()

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
