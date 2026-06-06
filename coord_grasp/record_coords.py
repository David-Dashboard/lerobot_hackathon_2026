"""Record a COORDINATE-ONLY dataset: state + goal + action, no camera images.

The goal (the target object's (x,y,z) in the robot frame) is logged as
``observation.environment_state`` -- LeRobot's convention for low-dim, non-proprio
observations. A policy trained on this is pixel-free and camera-pose-agnostic.

Mirror of so101.record.record_teleop_dataset, but with a goal vector instead of
cameras. Works with MockArm/MockTeleop for hardware-free testing.
"""

from __future__ import annotations

import time

import numpy as np

from so101.obs import SO101_JOINTS


def build_coord_features(goal_dim: int, goal_names=None) -> dict:
    n = len(SO101_JOINTS)
    goal_names = list(goal_names) if goal_names else [f"g{i}" for i in range(goal_dim)]
    return {
        "observation.state": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
        "observation.environment_state": {"dtype": "float32", "shape": (goal_dim,), "names": goal_names},
        "action": {"dtype": "float32", "shape": (n,), "names": list(SO101_JOINTS)},
    }


def record_coord_dataset(
    follower,
    teleop,
    repo_id: str,
    task: str,
    goal,
    *,
    goal_names=None,
    num_episodes: int = 2,
    episode_steps: int | None = 200,
    fps: int = 30,
    root=None,
    push_to_hub: bool = False,
    progress=None,
    should_stop=None,
    await_start=None,
    end_episode=None,
) -> dict:
    """Teleoperate while logging (state, goal, action) each frame.

    `goal`: the target coordinate -- a fixed sequence of floats, or a callable
    ``goal_for(ep_index) -> sequence`` to vary it per episode (e.g. from the
    geometry front-end). `await_start`/`end_episode`/`episode_steps` behave as in
    record_teleop_dataset (ENTER-controlled or timed episodes).
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    goal_fn = goal if callable(goal) else (lambda _ep: goal)
    goal_dim = len(np.asarray(goal_fn(0), dtype=np.float32))

    if not follower.is_connected:
        follower.connect()
    if not teleop.is_connected:
        teleop.connect()

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=build_coord_features(goal_dim, goal_names),
        root=root,
        robot_type="so101_follower",
        use_videos=False,
    )

    period = 1.0 / fps
    done = 0
    episodes_recorded = 0
    stopped = False

    for ep in range(num_episodes):
        if stopped:
            break
        if await_start is not None and not await_start(ep):
            break
        g = np.asarray(goal_fn(ep), dtype=np.float32)

        step = 0
        while True:
            t0 = time.perf_counter()
            action = teleop.read_action()
            follower.write_joints(action)
            state_joints = follower.read_joints()

            state = np.array([state_joints[j] for j in SO101_JOINTS], dtype=np.float32)
            act = np.array([action[j] for j in SO101_JOINTS], dtype=np.float32)
            dataset.add_frame({
                "observation.state": state,
                "observation.environment_state": g.copy(),
                "action": act,
                "task": task,
            })
            done += 1
            step += 1
            if progress:
                progress(done)

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
            if should_stop and should_stop():
                stopped = True
                break
            if end_episode is not None and end_episode(step):
                break
            if episode_steps is not None and step >= episode_steps:
                break

        if step == 0:
            continue
        dataset.save_episode()
        episodes_recorded += 1

    if push_to_hub:
        dataset.push_to_hub()

    return {
        "repo_id": repo_id,
        "num_episodes": episodes_recorded,
        "num_frames": done,
        "root": str(dataset.root),
        "goal_dim": goal_dim,
    }
