"""coord_grasp -- self-calibrating, coordinate-only grasping (experiment branch).

Pixel-free grasp policy: a geometry front-end (ArUco camera-pose + hand-eye +
forward kinematics) turns an object into an (x,y,z) in the robot base frame, and
that coordinate -- not images -- conditions the policy. See README.md.

Pure modules (`frames`, `kinematics`, `localize3d`) import with just numpy;
`markers`/`handeye` import cv2 lazily; `record_coords` pulls LeRobot lazily.
"""

from __future__ import annotations

__all__ = ["frames", "kinematics", "localize3d"]
