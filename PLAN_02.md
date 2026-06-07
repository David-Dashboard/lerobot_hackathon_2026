# PLAN_02 — Train a pick-up policy from generated data (no robot)

## Context

Collecting demonstrations by teleoperation is slow: every demo means driving the
leader arm by hand, and a handful of episodes is too few to train a robust
pick-up policy. This plan covers training **without more teleoperation**, under
real constraints we hit during the hackathon: **no robot access, little time,
cloud-only compute (Qualia/Colab), and only ~8 recorded episodes** (6-DoF
`observation.state` + `action`, plus a `scene` image and a `wrist` image).

This repo has **no physics simulator** — only a sine-wave `MockArm`
(`so101/mock.py`) and placeholder frames (`so101/record.py:_synthetic_frame`)
for exercising the software pipeline. Those cannot teach grasping (no object,
gravity, or contact). Building a true MuJoCo/Isaac sim is a large lift and,
without the arm in hand, you cannot close the sim-to-real gap or validate it.

So Phase 2A delivers the high-payoff path that works with what we have —
**synthetic data augmentation + offline evaluation** — and Phase 2B documents the
**physics-sim foundation** for unlimited demos + sim-to-real once the arm is back.

> **What augmentation can and cannot do.** It improves *robustness /
> generalization* from the demos you already have. It does **not** invent new
> grasp strategies or object positions absent from the source demos. For
> genuinely new coverage you need more real demos or a simulator (Phase 2B).

## Phase 2A — Synthetic augmentation + offline eval (DONE)

Shipped on branch `claude/ai-training-synthetic-data-MBLQK`.

### What was built
- **`so101/augment.py`** — `augment_dataset()` reads a source `LeRobotDataset`
  and writes a larger one, emitting `multiplier` augmented variants per episode.
  Schema / fps / `robot_type` are copied from the source (no hardcoding), so it
  works on the real scene+wrist set. Pure-numpy, unit-testable per-frame helpers;
  lerobot is imported lazily. Augmentations, in order of value:
  - **Trajectory jitter (primary):** temporally-correlated, low-frequency noise
    on `observation.state` (the input); the `action` target is kept smooth so a
    chunked policy like ACT isn't taught to predict noise.
  - **Mirror (opt-in, experimental):** flip every image + negate the laterally
    symmetric joints (`shoulder_pan`, `wrist_roll`). The only aug that adds new
    spatial coverage, but validity depends on calibration sign/offset and a
    centred camera — **off by default**, verify one episode before trusting it.
  - **Image jitter (optional, light):** small geometric crop/translate/cutout —
    things the trainer's `ImageTransforms` don't do; photometric jitter is left
    to the trainer to avoid baking frozen, redundant aug onto disk.
- **`augment_dataset.py`** — CLI (style of `record_teleop.py`): `--multiplier`,
  `--mirror`, `--no-keep-original`, `--no-image-aug`, `--push-to-hub` (required
  for the Qualia path), `--overwrite`, `--seed`.
- **`so101/deploy.py`** — wired the previously-stubbed `LeRobotPolicy` to load a
  trained checkpoint via `from_pretrained`, and added `build_observation()`
  (state + CHW `[0,1]` images + `task`). Torch stays lazy; stub policies and
  `run_policy` are unchanged.
- **`eval_offline.py`** — the no-robot success metric: open-loop next-action
  error vs a **held-out** episode → `summary.json` (per-joint / overall MSE/MAE)
  + predicted-vs-true `trajectories.png` (matplotlib `Agg`, headless).
- **`tests/test_augment.py`** — pure-helper tests always run; dataset round-trip
  and `build_observation` tests skip when lerobot/torch are absent (CI pattern).

### How to use it (Colab/Qualia, where lerobot is installed)
```bash
# 0. Verify the pinned lerobot API once before relying on it:
python -c "import lerobot, inspect; from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
from lerobot.policies.factory import make_policy; print(lerobot.__version__)"

# 1. Augment: ~8 episodes -> ~40 (8 originals + 4 variants each). Hold 1-2 OUT.
python augment_dataset.py --src-root recorded/demos --src-repo-id local/demos \
  --out-root recorded/demos_aug --out-repo-id local/demos_aug --multiplier 4
#   add --mirror for extra coverage (experimental); --push-to-hub to train on Qualia

# 2. Train (local ACT, or Qualia on the pushed dataset id)
lerobot-train --dataset.repo_id=local/demos_aug --dataset.root=recorded/demos_aug --policy.type=act ...

# 3. Evaluate offline against a HELD-OUT episode (no robot)
python eval_offline.py --checkpoint outputs/train/demos_aug_act/checkpoints/last/pretrained_model \
  --dataset-root recorded/demos --dataset-repo-id local/demos --episode-index 7 \
  --out-dir outputs/eval/demos_ep7
```

### Caveats (must respect)
- **Holdout discipline:** the eval episode must be excluded from
  augmentation/training, or the number just measures memorization.
- **Open-loop metric:** `eval_offline` ranks checkpoints ("did augmentation lower
  the error?"); it does **not** prove grasping — no compounding error, no physics.
- **Mirror sign convention** is not guaranteed valid for the SO-101; eyeball one
  mirrored episode in Rerun before training on a `--mirror` dataset.
- **Defaults:** `multiplier=4`, keep originals on. Don't go very high — augmented
  copies are correlated, so returns diminish fast.

## Phase 2B — Physics-sim foundation (FUTURE, needs the arm)

Goal: generate **unlimited** demos and train a grasp policy transferable to the
real SO-101 (sim-to-real). Deferred because closing the sim-to-real gap needs the
physical arm on hand to match dynamics, calibration, and camera placement.

### Approach (when picked up)
1. **Pick a simulator + SO-101 asset.** Candidates:
   - [LeRobot sim envs](https://github.com/huggingface/lerobot) (native dataset/policy fit)
   - [ManiSkill](https://github.com/haosulab/ManiSkill) (SO-100 support, fast GPU sim)
   - [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) SO-ARM model
2. **Wrap the sim arm behind the existing `RobotArm` protocol**
   (`so101/interface.py`) — same `read_joints`/`write_joints`/`read_observation`
   surface as `MockArm`, so `record.py`, `deploy.py`, and `eval_offline.py` reuse
   unchanged. Add it to `so101/factory.py` (`make_arm(sim=True)`).
3. **Define a pick-up task** (graspable object + table + scene/wrist cameras
   matching the real layout) and **auto-generate demonstrations** — scripted
   waypoints / IK reach-grasp-lift, or RL — so no teleoperation is needed.
4. **Record sim demos into a `LeRobotDataset`** (reuse `record.py`), optionally
   mix with the real episodes, and train ACT/SmolVLA as in Phase 2A.
5. **Sim-to-real:** domain-randomize camera/lighting/object pose in sim; calibrate
   against the real arm; evaluate on hardware. This is the step that needs the arm.

### Risks
- Sim-to-real gap (dynamics, textures, camera) is the central challenge — budget
  most of the effort here, not on building the sim itself.
- Matching the sim camera intrinsics/placement to the real scene+wrist rig is
  required for the visual policy to transfer.

## Verification
- **Phase 2A unit tests:** `python -m pytest tests/test_augment.py -q` (pure
  helpers run anywhere; dataset/torch tests run where lerobot/torch are present).
- **Phase 2A end-to-end (no robot):** record 2 mock episodes → `augment_dataset.py
  --multiplier 4` → confirm 2→10 episodes and the output opens with
  `LeRobotDataset` → train a few steps → `eval_offline.py` writes summary + plot.
- **Lift check:** compare `eval_offline` MSE for a checkpoint trained on the raw
  episodes vs the augmented set on the same held-out episode.
