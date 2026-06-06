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
| `hello_read.py` | Read SO-101 joint positions and stream them live to Rerun (no motion). |
| `RUNBOOK.md` | Full command reference: find ports, calibrate, teleop, record, train, deploy. |

## Quick start

See **[RUNBOOK.md](RUNBOOK.md)** for the end-to-end flow:
teleoperate → record dataset → push to Hugging Face → fine-tune a policy → run it on the arm.
