"""coord_grasp experiment: geometry front-end + coordinate-only recorder."""

import math

import numpy as np
import pytest

from coord_grasp import frames, kinematics, localize3d

# Same SO-101 geometry config as trash_arm/test_motion (FK must invert that IK).
IK = {
    "l1": 0.116, "l2": 0.135, "base_height": 0.06, "wrist_length": 0.05,
    "elbow_up": False,
    "sign": {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1},
    "offset": {"shoulder_pan": 0, "shoulder_lift": 0, "elbow_flex": 0, "wrist_flex": 0},
}


# --- frames ---------------------------------------------------------------

def test_invert_round_trip():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)  # 90deg about z
    T = frames.make_transform(R, [0.1, 0.2, 0.3])
    np.testing.assert_allclose(frames.compose(T, frames.invert_transform(T)), np.eye(4), atol=1e-12)


def test_apply_transform_translation_and_rotation():
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    T = frames.make_transform(R, [1, 0, 0])
    np.testing.assert_allclose(frames.apply_transform(T, [1, 0, 0]), [1, 1, 0], atol=1e-12)


def test_rvec_zero_is_identity_rotation():
    T = frames.rvec_tvec_to_T([0, 0, 0], [1, 2, 3])
    np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(T[:3, 3], [1, 2, 3])


# --- kinematics: FK must invert trash_arm's IK ----------------------------

@pytest.mark.parametrize("target", [(0.20, 0.05, 0.02), (0.18, -0.04, 0.05), (0.12, 0.12, 0.02)])
def test_fk_inverts_ik(target):
    from trash_arm.motion import plan_planar_ik

    joints = plan_planar_ik(*target, IK)
    joints["wrist_roll"] = 0.0
    joints["gripper"] = 0.0
    xyz = kinematics.forward_kinematics(joints, IK)
    np.testing.assert_allclose(xyz, target, atol=1e-6)


# --- localize3d: pixel <-> robot-frame coordinate round trip --------------

def _top_down_camera(height=0.6):
    # Camera at (0,0,height) looking straight down (-z world). Proper rotation.
    R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=float)  # world->cam
    C = np.array([0.0, 0.0, height])
    t = -R @ C
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=float)
    return K, frames.make_transform(R, t)


def test_pixel_to_plane_recovers_projected_point():
    K, T_cam_world = _top_down_camera()
    for P in [(0.05, 0.03, 0.0), (-0.04, 0.06, 0.0), (0.0, 0.0, 0.0)]:
        uv = localize3d.project_point(P, K, T_cam_world)
        back = localize3d.pixel_to_plane(uv[0], uv[1], K, T_cam_world, plane_z=0.0)
        np.testing.assert_allclose(back, P, atol=1e-9)


# --- coordinate-only recorder (mock hardware) -----------------------------

def test_record_coord_dataset_has_goal_and_no_images(tmp_path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    from so101 import MockArm, MockTeleop
    from coord_grasp.record_coords import record_coord_dataset

    summary = record_coord_dataset(
        MockArm(), MockTeleop(),
        repo_id="local/test_coord", task="grasp at goal",
        goal=(0.20, 0.05), goal_names=["x", "y"],
        num_episodes=2, episode_steps=4, root=tmp_path / "ds",
    )
    assert summary["num_episodes"] == 2
    assert summary["num_frames"] == 8
    assert summary["goal_dim"] == 2

    ds = LeRobotDataset(repo_id="local/test_coord", root=tmp_path / "ds")
    feats = set(ds.features)
    assert "observation.environment_state" in feats
    assert not any(k.startswith("observation.images") for k in feats)  # pixel-free
    frame = ds[0]
    np.testing.assert_allclose(np.asarray(frame["observation.environment_state"]), [0.20, 0.05], atol=1e-6)
