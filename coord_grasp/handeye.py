"""One-time hand-eye calibration: estimate the camera<->robot-base transform.

For a STATIC (eye-to-hand) scene camera, stick a marker on the gripper, jog the
arm to N varied poses, and at each pose record:
  * the gripper pose in the base frame (from forward kinematics / the robot), and
  * the marker (target) pose in the camera frame (from solvePnP on the marker).
``cv2.calibrateHandEye`` then solves for the fixed transform.

Gathering the poses needs hardware, so this is a thin, documented wrapper; the
math (cv2) is trusted. See README.md for the collection routine.
"""

from __future__ import annotations

import numpy as np

from . import frames


def calibrate_eye_to_hand(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam, method: str = "TSAI") -> np.ndarray:
    """Return ``T_base_cam`` (the static camera's pose in the robot base frame).

    Each argument is a list of N rotations / translations, one per recorded pose.
    """
    import cv2

    method_flag = getattr(cv2, f"CALIB_HAND_EYE_{method.upper()}")
    # Eye-to-hand: invert the gripper->base motions, per the OpenCV recipe.
    R_base2gripper, t_base2gripper = [], []
    for R, t in zip(R_gripper2base, t_gripper2base):
        R = np.asarray(R, dtype=float)
        t = np.asarray(t, dtype=float).reshape(3, 1)
        R_base2gripper.append(R.T)
        t_base2gripper.append(-R.T @ t)

    R_cam2base, t_cam2base = cv2.calibrateHandEye(
        R_base2gripper, t_base2gripper,
        [np.asarray(R, dtype=float) for R in R_target2cam],
        [np.asarray(t, dtype=float).reshape(3, 1) for t in t_target2cam],
        method=method_flag,
    )
    return frames.make_transform(R_cam2base, t_cam2base)
