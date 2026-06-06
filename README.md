# LeRobot Hackathon 2026 — SO-101

[![CI](https://github.com/David-Dashboard/lerobot_hackathon_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/David-Dashboard/lerobot_hackathon_2026/actions/workflows/ci.yml)

Hello-world + workflow for talking to an SO-101 arm with [LeRobot](https://github.com/huggingface/lerobot) and visualizing with [Rerun](https://rerun.io).

## Setup

```powershell
# Create the environment (uv)
uv venv --python 3.11
uv pip install -r requirements.txt          # + pytest for tests
```

The `.venv/` is git-ignored — recreate it with the commands above.

## Architecture (decoupled)

Everything depends on one small interface — `RobotArm` (read/command plain joint
dicts). Only `so101/real.py` ever imports LeRobot/torch, so the mock, server,
client, and tests all run with **no hardware and no heavy deps**.

```
so101/
  interface.py   RobotArm protocol + JointPositions          (no deps)
  obs.py         LeRobot observation <-> joint-dict helpers   (no deps)
  mock.py        MockArm — simulated, implements RobotArm     (no deps)
  real.py        SO101Arm — adapter over LeRobot (lazy import; ONLY torch-touching file)
  factory.py     make_arm(mock=...) -> RobotArm
  server.py      create_app(arm) -> FastAPI (arm + qualia routes + dashboard)
  client.py      ArmClient — drive an arm from another machine
  qualia_client.py  thin Qualia SDK wrapper (VLA finetuning)
  static/index.html dashboard UI (arm control + finetune launcher)
hello_read.py    read + visualize in Rerun  (supports --mock)
serve.py         run the server + dashboard (--mock or --port COMx)
```

## What's here

| Path | Purpose |
|------|---------|
| `so101/` | The decoupled arm package (interface, mock, real adapter, server, client). |
| `hello_read.py` | Read joint positions → live Rerun plots (no motion). Supports `--mock`. |
| `serve.py` | Remote-control HTTP server (talk to the arm over the network). |
| `tests/` | Suite that runs **without a robot** (`pytest`) — incl. an HTTP round-trip test. |
| `RUNBOOK.md` | Full command reference: find ports, calibrate, teleop, record, train, deploy. |

## No-hardware development

Run the full read/Rerun hello-world against a simulated arm:

```powershell
.\.venv\Scripts\python.exe hello_read.py --mock
```

Run the tests (install + data pipeline + mock arm + HTTP round-trip — no robot needed):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Remote control ("talk to the arm over the network")

Start the server — mock now, real arm later (same command, swap `--mock` for `--port`):

```powershell
.\.venv\Scripts\python.exe serve.py --mock                 # no hardware
.\.venv\Scripts\python.exe serve.py --port COM3 --id my_follower   # real arm
```

Then from this or **another machine** (same network):

```bash
curl http://ROBOT_HOST:8000/health
curl http://ROBOT_HOST:8000/joints
curl -X POST http://ROBOT_HOST:8000/joints \
     -H "content-type: application/json" \
     -d '{"positions": {"gripper": 10.0}}'
```

…or in Python:

```python
from so101.client import ArmClient
arm = ArmClient("http://ROBOT_HOST:8000")
print(arm.read_joints())
arm.write_joints({"gripper": 10.0})
```

> ⚠️ `POST /joints` moves a real arm. It's safe against `--mock`; on hardware,
> make sure the arm is calibrated and the workspace is clear.

## Dashboard UI

`serve.py` also serves a browser dashboard at **`http://localhost:8000/`**:

- **Arm panel** — live joint readout + sliders to command the arm (works with `--mock`).
- **Finetune · Qualia panel** — shows credit balance, and launches a real
  [Qualia](https://qualiastudios.dev) VLA finetune on a Hugging Face dataset
  (pick VLA type / hours / base model), with a live job list.

```powershell
.\.venv\Scripts\python.exe serve.py --mock      # open http://localhost:8000/
```

Qualia needs a token in `.env`:

```
QUALIA_TOKEN=...      # from the Qualia dashboard (Settings)
HF_TOKEN=...          # already set
```

The dashboard loads even without a token — the Qualia panel just shows
"no token" instead of erroring. Finetuning happens on Qualia's GPUs, so it works
regardless of your local hardware.

## Full workflow — no hardware needed

A teammate can walk the **entire pipeline from the dashboard** using the mock arm:

```powershell
git clone https://github.com/David-Dashboard/lerobot_hackathon_2026.git
cd lerobot_hackathon_2026
uv venv --python 3.11; .\.venv\Scripts\activate
uv pip install -r requirements.txt
python serve.py --mock        # open http://localhost:8000/
```

In the dashboard, top to bottom:
1. **Arm** — sliders read/command the (mock) arm.
2. **Gather data → dataset** — set task + episodes, click **Record**. A real
   LeRobotDataset (state + action + synthetic camera) is created under `recorded/`,
   and the repo id auto-fills the finetune panel. Tick "push to Hugging Face" to upload.
3. **Finetune · Qualia** — click **Launch finetune** to start a real VLA job on
   Qualia (uses the dataset from step 2).
4. **Deploy** — pick `sine` and click **Run**; the mock arm moves autonomously and
   you see panel 1 react. Swap in a trained model path to run a real policy.

Swap `--mock` for `--port COM3` and the same dashboard drives the real SO-101.

## End-to-end on real hardware (SO-101 + cameras + OAK-D)

The full hardware flow: bring up the arms, set up cameras, teleoperate, record
demonstrations (OpenCV **and** OAK-D cameras at once), and train. PowerShell
shown; for WSL2 run `bash setup_wsl2.sh` first and swap `COM5`→`/dev/ttyACM0` etc.

> In PowerShell, run multi-line commands as **one line** — a dropped backtick
> (`` ` ``) silently truncates the command into defaults.

### 0. Environment
```powershell
uv venv --python 3.11
uv pip install -r requirements.txt        # lerobot, rerun, transformers (<5), opencv, pyyaml, ...
uv pip install depthai                    # OAK-D-PRO (DepthAI)
```
WSL2: `bash setup_wsl2.sh` does all of this + dialout/udev/usbipd setup.

### 1. Find the arm ports (stable, by board serial)
COM numbers drift on replug; the CH343 board **serial** is stable.
```powershell
Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'COM[0-9]+' } | Select-Object Name, DeviceID
```
Example mapping: **follower = COM5** (`…706`), **leader = COM4** (`…811`). Verify reads (no motion):
```powershell
.\.venv\Scripts\python.exe -c "from so101 import make_arm; a=make_arm(port='COM5',arm_id='my_follower',calibrate=False); a.connect(); print(a.read_joints()); a.disconnect()"
```

### 2. Calibrate (once per arm)
```powershell
.\.venv\Scripts\lerobot-calibrate.exe --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower
.\.venv\Scripts\lerobot-calibrate.exe --teleop.type=so101_leader  --teleop.port=COM4 --teleop.id=my_leader
```

### 3. Cameras
- **Enable camera access:** Settings → Privacy & security → Camera → "Camera access" **ON** (a global OFF blocks OpenCV entirely).
- **Find OpenCV camera indices:** `.\.venv\Scripts\python.exe scripts/test_perception.py --list`
- **View feeds in Rerun** (set PATH once per terminal so Rerun's viewer is found):
```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
.\.venv\Scripts\python.exe scripts/oak_view.py                              # live OAK-D RGB
.\.venv\Scripts\python.exe scripts/view_cameras.py --camera scene=0 --camera wrist=1   # OpenCV cams
```

### 4. Teleoperate (leader drives follower) with camera feed
```powershell
$env:Path = "$PWD\.venv\Scripts;$env:Path"
lerobot-teleoperate `
  --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower `
  --robot.cameras="{ scene: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" `
  --teleop.type=so101_leader  --teleop.port=COM4 --teleop.id=my_leader `
  --display_data=true --fps=30
```
> The OAK can't go through `--robot.cameras` (OpenCV-only). Record it via `record_teleop.py --oak`.

### 5. Record demonstrations (OpenCV + OAK cameras simultaneously)
`record_teleop.py` mirrors leader→follower and saves state + action + every camera into a LeRobotDataset.
```powershell
.\.venv\Scripts\python.exe record_teleop.py --robot-port COM5 --robot-id my_follower --teleop-port COM4 --teleop-id my_leader --oak --oak-name scene --camera wrist=1:640x480 --no-calibrate --display --overwrite --manual --episodes 20 --fps 30 --root recorded/demos --repo-id local/demos --task "pick up the trash and drop it in the bin"
```

| Flag | Meaning |
|---|---|
| `--oak [--oak-name N]` | record the OAK-D RGB (DepthAI) as image key `N` |
| `--camera N=IDX:WxH` | add an OpenCV camera (repeatable) |
| `--manual` | press ENTER to start/stop each episode, `q` to finish |
| `--episodes N` | number of demos (a max in manual mode) |
| `--episode-seconds` / `--reset-seconds` | timed mode only |
| `--display` | stream joints **and** all camera feeds to Rerun |
| `--overwrite` | replace an existing dataset (else it fails fast, *before* touching hardware) |
| `--no-calibrate` | load saved calibration (don't re-run the prompt) |
| `--mock` | simulated arms (no hardware) |

**Timed-mode smoke test** (fixed-length episodes, no ENTER prompts — OAK as `scene`, webcam as `wrist`):
```powershell
.\.venv\Scripts\python.exe record_teleop.py --robot-port COM5 --robot-id my_follower --teleop-port COM4 --teleop-id my_leader --oak --oak-name scene --camera wrist=1:640x480 --no-calibrate --display --overwrite --episodes 10 --episode-seconds 15 --reset-seconds 0 --fps 30 --root recorded/02_smoke_correct --repo-id local/02_smoke_correct --task "smoke test"
```

### 6. Verify a recorded dataset
```powershell
.\.venv\Scripts\python.exe -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset as D; d=D(repo_id='local/demos', root='recorded/demos'); print('episodes', d.num_episodes, 'frames', d.num_frames, list(d.features))"
```

### 7. Train
Local ACT (free, CPU-capable):
```powershell
.\.venv\Scripts\lerobot-train.exe --dataset.repo_id=local/demos --dataset.root=recorded/demos --policy.type=act --policy.device=cpu --output_dir=outputs/train/demos_act --job_name=demos_act --batch_size=4 --num_workers=0 --steps=100000 --save_freq=200 --wandb.enable=false
```
Qualia VLA finetune (cloud; needs `QUALIA_TOKEN` + dataset on the HF Hub) — use the dashboard (`serve.py`) or `so101.qualia_client.launch_finetune`.

## Train from generated data (few demos, no robot)

Teleoperation is slow, and a handful of episodes is too few to train a robust
pick-up policy. `augment_dataset.py` multiplies a recorded `LeRobotDataset` into a
larger one — emitting several augmented variants of each episode — so you get more
training data with **no extra teleoperation and no hardware**. `eval_offline.py`
then scores a trained checkpoint **without a robot**.

> **What this does and doesn't do.** Augmentation improves *robustness/generalization*
> from the demos you already have (trajectory jitter, optional left↔right mirror,
> light geometric image aug). It does **not** invent new grasp strategies or object
> positions absent from the source demos — for genuinely new coverage you need more
> real demos or a physics simulator (see *Foundation: physics sim*).

```powershell
# 1. Augment: 8 episodes -> ~40 (8 originals + 4 variants each). Hold an episode
#    or two OUT of training so you can evaluate on them.
python augment_dataset.py --src-root recorded/demos --src-repo-id local/demos `
  --out-root recorded/demos_aug --out-repo-id local/demos_aug --multiplier 4
#    (add --mirror for extra spatial coverage — EXPERIMENTAL; eyeball one mirrored
#     episode in Rerun first. add --push-to-hub to train on Qualia.)

# 2. Train on the augmented set (local ACT, or Qualia on the pushed dataset)
lerobot-train --dataset.repo_id=local/demos_aug --dataset.root=recorded/demos_aug --policy.type=act ...

# 3. Evaluate offline against a HELD-OUT episode (no robot)
python eval_offline.py --checkpoint outputs/train/demos_aug_act/checkpoints/last/pretrained_model `
  --dataset-root recorded/demos --dataset-repo-id local/demos --episode-index 7 `
  --out-dir outputs/eval/demos_ep7
```

`eval_offline` writes `summary.json` (per-joint / overall action error) and a
predicted-vs-ground-truth `trajectories.png`. It measures *open-loop next-action
error* — it **ranks checkpoints** (did augmentation lower the error?) but does not
prove grasping; confirm real performance on the arm when one is available.

### Foundation: physics sim (future)

The above squeezes the most out of few real demos. To generate *unlimited* demos
and train a grasp policy that can be transferred to the real SO-101 (sim-to-real),
the next step is a physics simulation of the arm + a graspable object — e.g.
[LeRobot sim envs](https://github.com/huggingface/lerobot),
[ManiSkill](https://github.com/haosulab/ManiSkill) (SO-100 support), or a
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) SO-ARM
model. A sim arm would conform to the same `RobotArm` protocol (`so101/interface.py`)
as `MockArm`, so recording/deploy/eval reuse unchanged. This is deferred because it
needs the physical arm on hand to calibrate the sim-to-real gap.

### 8. Scripted trash-collecting pipeline (perceive → localize → pick)
```powershell
.\.venv\Scripts\python.exe scripts/calibrate_camera.py     # build pixel→table homography (M2)
.\.venv\Scripts\python.exe scripts/test_perception.py      # detector on the live scene cam (M3)
.\.venv\Scripts\python.exe scripts/dry_run.py              # full loop, MOTION DISABLED (M5 prep)
.\.venv\Scripts\python.exe scripts/run.py                  # the real autonomous loop
```
Ports, camera indices, workspace limits, and bin coords all live in **`config.yaml`**.

> **Experiment branch `coord-grasp`** — a self-calibrating, *coordinate-only* grasp
> approach (ArUco/depth → metric XYZ → a pixel-free, goal-conditioned policy). See
> `coord_grasp/` on that branch.

## Quick start

- **Windows** → see **[RUNBOOK.md](RUNBOOK.md)**
- **WSL2 / Linux** (hardware + calibration machine) → see **[SETUP_WSL2.md](SETUP_WSL2.md)**
  ⚠️ WSL2 needs USB passthrough (`usbipd-win`) before the arm is visible — covered in step 3.

Both cover the end-to-end flow:
teleoperate → record dataset → push to Hugging Face → fine-tune a policy → run it on the arm.
