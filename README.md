# LeRobot Hackathon 2026 — SO-101 (`auto_record`)

[![CI](https://github.com/David-Dashboard/lerobot_hackathon_2026/actions/workflows/ci.yml/badge.svg)](https://github.com/David-Dashboard/lerobot_hackathon_2026/actions/workflows/ci.yml)

Teleoperate an SO-101 (leader → follower), record demonstrations into an **ACT-ready
LeRobotDataset** with an **OAK-D-PRO** scene camera + a wrist webcam, and push to the
Hugging Face Hub / launch a **Qualia ACT finetune** — all from one script:
**`auto_record.py`**.

## Setup

```powershell
uv venv --python 3.11
uv pip install -r requirements.txt
```

`.venv/` is git-ignored — recreate it with the commands above. WSL2: run
`bash setup_wsl2.sh` (installs deps + dialout/udev/usbipd) — see [SETUP_WSL2.md](SETUP_WSL2.md).

## One command — `auto_record.py`

Auto-detects the arms (by **stable board serial**), the **OAK-D-PRO** (scene cam) and
the **wrist webcam** (laptop cam auto-excluded), records manual ENTER-controlled
episodes, and can push to HF + launch a Qualia ACT finetune. It also auto-fixes the
Rerun PATH, retries flaky connects, and fails fast on dataset name clashes.

```powershell
.\.venv\Scripts\python.exe auto_record.py --check               # detect hardware only (no motion)
.\.venv\Scripts\python.exe auto_record.py                       # record episodes          [needs arms]
.\.venv\Scripts\python.exe auto_record.py --record --push       # record, then push to HF  [needs arms]
.\.venv\Scripts\python.exe auto_record.py --push  --name batch1 # push an existing dataset to HF   [no arms]
.\.venv\Scripts\python.exe auto_record.py --train --name batch1 # push existing + Qualia ACT job   [no arms]
.\.venv\Scripts\python.exe auto_record.py --install             # uv-bootstrap missing deps, then run
```

| Action | Needs arms? | What it does |
|---|---|---|
| *(none)* or `--record` | **yes** | record demos — ENTER to start/stop each episode, `q` to finish |
| `--record --push` | **yes** | record, then push the dataset to the HF Hub |
| `--push` | **no** | push an existing `--name` dataset to `<hf_user>/<name>` |
| `--train` | **no** | push existing + launch a Qualia ACT finetune (confirms before spending credits) |
| `--check` | reads only | print the detected arms / OAK / wrist and exit |

Common options: `--name`, `--episodes`, `--scene-size WxH`, `--wrist-size WxH`,
`--wrist-index N`, `--no-wrist`, `--no-display`, `--overwrite`, `--hours`,
`--yes-spend`, `--hf-user`, `--root`, `--config`.

## Hardware bring-up (one-time)

1. **Ports** — resolved automatically by stable board serial (set in `config.yaml`).
   Confirm the mapping anytime with `auto_record.py --check`. COM numbers drift on
   replug; the serial is the source of truth.
2. **Calibrate** each arm (interactive — move each joint through its range):
   ```powershell
   .\.venv\Scripts\lerobot-calibrate.exe --robot.type=so101_follower --robot.port=COM5 --robot.id=my_follower
   .\.venv\Scripts\lerobot-calibrate.exe --teleop.type=so101_leader  --teleop.port=COM4 --teleop.id=my_leader
   ```
   Calibration saves automatically (keyed by `--robot.id` / `--teleop.id`).
3. **Cameras** — enable Windows camera access (Settings → Privacy & security → Camera).
   The **OAK-D-PRO** is the `scene` cam (DepthAI, not UVC); the **wrist** is an external
   USB webcam (the laptop cam is excluded by name automatically).

## Config

`config.yaml` holds the arm `robot.id` / `teleop.id` and their **stable board serials**
(committed so the whole team shares one setup). The arm calibration files live in
LeRobot's own cache, keyed by id.

## Architecture

```
auto_record.py            one-command: detect -> record -> push -> train
so101/
  discovery.py            find arms (by serial), OAK, wrist webcam; Cv2Camera wrapper
  reliability.py          connect-with-retry + graceful Ctrl+C
  record.py               record_teleop_dataset (leader->follower, multi-camera)
  teleop.py               SO101 leader (real) + MockTeleop
  factory.py              make_arm / make_teleop (mock or real)
  real.py                 SO101 follower adapter (the ONLY file that imports LeRobot)
  mock.py, obs.py, interface.py   hardware-free arm + observation/joint helpers
  qualia_client.py        Qualia SDK wrapper (cloud ACT/VLA finetune)
coord_grasp/oak.py        OAK-D-PRO RGB camera over DepthAI
```

Only `real.py` imports LeRobot/torch, so the mock, discovery, and tests run with no
hardware and no heavy deps.

## Tests (no hardware)

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Local ACT training (alternative to Qualia)

Cloud is one command (`auto_record.py --train`). To train locally instead:

```powershell
.\.venv\Scripts\lerobot-train.exe --dataset.repo_id=<user>/<name> --dataset.root=recorded/<name> --policy.type=act --policy.device=cpu --output_dir=outputs/train/<name>_act --steps=100000 --batch_size=4 --num_workers=0 --save_freq=200 --wandb.enable=false
```

## More

- **Windows hardware runbook** → [RUNBOOK.md](RUNBOOK.md)
- **WSL2 / Linux setup** → [SETUP_WSL2.md](SETUP_WSL2.md) (needs `usbipd-win` USB passthrough)
- Experiment branch **`coord-grasp`** — a self-calibrating, coordinate-only grasp approach.
