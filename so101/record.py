"""Record a LeRobotDataset from any `RobotArm` -- mock or real.

For the mock, we synthesize a deterministic camera frame from the joint state so
the resulting dataset is a valid *vision* dataset (trainable by a VLA), with no
hardware. Stored as images (use_videos=False) so it needs no ffmpeg.
"""

from __future__ import annotations

import numpy as np

from .interface import RobotArm
from .obs import SO101_JOINTS

CAM_H, CAM_W = 120, 160
CAMERA_KEY = "observation.images.front"


def _synthetic_frame(joints: dict[str, float]) -> np.ndarray:
    """A deterministic image that visibly depends on the joints (so it's not noise)."""
    img = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
    img[:, :, 2] = np.linspace(20, 200, CAM_W, dtype=np.uint8)[None, :]  # blue gradient
    # a green marker whose position tracks two joints
    pan = joints.get("shoulder_pan", 0.0)
    grip = joints.get("gripper", 0.0)
    cx = int(np.clip((pan + 100) / 200 * (CAM_W - 1), 0, CAM_W - 1))
    cy = int(np.clip((grip + 100) / 200 * (CAM_H - 1), 0, CAM_H - 1))
    img[max(0, cy - 5):cy + 5, max(0, cx - 5):cx + 5, 1] = 255
    return img


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
