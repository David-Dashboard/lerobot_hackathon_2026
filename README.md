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

## Quick start

- **Windows** → see **[RUNBOOK.md](RUNBOOK.md)**
- **WSL2 / Linux** (hardware + calibration machine) → see **[SETUP_WSL2.md](SETUP_WSL2.md)**
  ⚠️ WSL2 needs USB passthrough (`usbipd-win`) before the arm is visible — covered in step 3.

Both cover the end-to-end flow:
teleoperate → record dataset → push to Hugging Face → fine-tune a policy → run it on the arm.
