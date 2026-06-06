# coord_grasp — self-calibrating, coordinate-only grasping (experiment)

**Branch:** `coord-grasp`. Isolated from the `trash_arm` Phase-1 pipeline on purpose;
it reuses only the `so101` robot wrapper.

## Idea

Train a grasp policy that **never sees pixels**. Vision lives entirely in a
*geometry front-end* that estimates camera + arm pose automatically and emits the
target object's location in the **robot base frame**. The policy's only inputs are
that coordinate plus proprioception, so:

- it's **camera-pose-agnostic** (move the camera → extrinsics are re-estimated live,
  the coordinate is unchanged, the policy never notices);
- it's tiny and CPU-trainable (no image backbone).

```
ArUco markers ─► estimate camera pose (solvePnP)         ┐
gripper marker ─► hand-eye calib (calibrateHandEye)      ├─ all automatic
joint encoders ─► forward kinematics (arm/EE pose)       ┘
                         │
                         ▼
   object pixel ─► pixel→table-plane ray ─► (x,y,z) in ROBOT frame   = GOAL
                         │
                         ▼
   POLICY:  in = observation.state (joints) + observation.environment_state (goal)
            out = action (joint targets)            ← no images anywhere
```

## Modules

| file | role | status |
|---|---|---|
| `frames.py` | SE(3) transform math (compose/invert/apply) | ✅ tested |
| `kinematics.py` | SO-101 forward kinematics: joints → EE pose ("arm estimation") | ✅ tested |
| `markers.py` | ArUco detection + live camera-pose estimation | needs real markers |
| `localize3d.py` | pixel → robot-frame coordinate via plane intersection | ✅ tested |
| `handeye.py` | one-time camera↔base hand-eye calibration | needs hardware data |
| `record_coords.py` | record a **coordinate-only** dataset (state+goal+action, no images) | ✅ tested (mock) |

## Workflow

1. **Calibrate intrinsics** once (`cv2.calibrateCamera` on a checkerboard) → `K`, dist.
2. **Place ArUco markers** at known table coordinates (robot frame). `markers.py`
   estimates camera extrinsics live from them.
3. **Hand-eye** (`handeye.py`): marker on gripper, jog to N poses → camera↔base transform.
4. **Record** demos with `record_coords.py`: each episode's goal `(x,y,z)` comes from
   the geometry front-end (or is typed in), logged as `observation.environment_state`.
5. **Train** a state-only, goal-conditioned policy:
   ```
   lerobot-train --dataset.repo_id=local/coord_grasp --dataset.root=recorded/coord_grasp \
     --policy.type=act --policy.device=cpu --policy.push_to_hub=false \
     --output_dir=outputs/train/coord_grasp --batch_size=16 --num_workers=0 \
     --steps=20000 --save_freq=2000 --log_freq=50 --wandb.enable=false
   ```
   (No image keys in the dataset → ACT trains as a small low-dim policy.)

## Honest limits

- Open-loop w.r.t. vision during the motion: the policy *trusts the goal*. Good
  calibration matters more here than in a closed-loop image policy.
- Accuracy stacks: marker size + camera resolution + FK + arm calibration. Expect
  cm-level until tuned. The SO-101 calibration being "a bit off" propagates here.
- Live extrinsics need markers **in view**; markerless live estimation is out of scope.
