# LeRobot Hackathon 2026 — SO-101

Hello-world + workflow for talking to an SO-101 arm with [LeRobot](https://github.com/huggingface/lerobot) and visualizing with [Rerun](https://rerun.io).

## Setup

```powershell
# Create the environment (uv)
uv venv --python 3.11
uv pip install "lerobot[feetech]" rerun-sdk
```

The `.venv/` is git-ignored — recreate it with the commands above.

## What's here

| File | Purpose |
|------|---------|
| `hello_read.py` | Read SO-101 joint positions and stream them live to Rerun (no motion). Supports `--mock`. |
| `arm_utils.py` | Pure, hardware-free helpers for parsing observations (unit-tested). |
| `mock_robot.py` | A simulated SO-101 arm — develop/test with **no hardware**. |
| `tests/` | Test suite that runs **without a robot** (`pytest`). |
| `RUNBOOK.md` | Full command reference: find ports, calibrate, teleop, record, train, deploy. |

## No-hardware development

Run the full read/Rerun hello-world against a simulated arm:

```powershell
.\.venv\Scripts\python.exe hello_read.py --mock
```

Run the tests (install + data pipeline + mock arm — no robot needed):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Quick start

- **Windows** → see **[RUNBOOK.md](RUNBOOK.md)**
- **WSL2 / Linux** (hardware + calibration machine) → see **[SETUP_WSL2.md](SETUP_WSL2.md)**
  ⚠️ WSL2 needs USB passthrough (`usbipd-win`) before the arm is visible — covered in step 3.

Both cover the end-to-end flow:
teleoperate → record dataset → push to Hugging Face → fine-tune a policy → run it on the arm.
