# calibration/

**Commit the files here** so the whole team shares one calibrated setup
(the plan calls a `robot.id` / calibration mismatch out as a classic demo-day bug).

Contents:
- `homography_scene.npy` — pixel → table (X,Y) map for the scene camera.
  Built by `scripts/calibrate_camera.py`. Rebuild if the camera ever moves.
- `camera_matrix.npy`, `dist_coeffs.npy` *(optional)* — intrinsics for lens
  undistortion, applied before the homography. Omit if you skip undistortion.
- `points.json` *(optional)* — saved pixel↔table correspondences for
  non-interactive recalibration: `[{"pixel": [u, v], "table": [x, y]}, ...]`.

The **arm** calibration (from `lerobot-calibrate`) is stored separately in
LeRobot's own calibration dir, keyed by `robot.id` / `teleop.id`, not here.
