# WSL2 Setup (for the hardware / calibration machine)

This is the Linux/WSL2 counterpart to `RUNBOOK.md` (which uses Windows paths).
Follow this on the machine that has the SO-101 arms plugged in.

> ⚠️ **The #1 gotcha:** WSL2 does **not** see USB serial devices by default.
> You **must** pass the arm's USB through from Windows into WSL with `usbipd-win`
> (Step 3) or `lerobot-find-port` / calibration will find **no ports**.

---

## 1. Clone the repo

```bash
git clone https://github.com/David-Dashboard/lerobot_hackathon_2026.git
cd lerobot_hackathon_2026
```

## 2. Install uv + the environment

The `.venv/` is git-ignored, so each machine builds its own:

```bash
# install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env   # or restart the shell

# create env + install LeRobot (Feetech motor SDK) + Rerun
uv venv --python 3.11
source .venv/bin/activate
uv pip install "lerobot[feetech]" rerun-sdk
```

## 3. Pass the arm's USB into WSL2  ← the critical step

**On the Windows host** (PowerShell as Administrator):

```powershell
winget install usbipd          # one-time install of usbipd-win
usbipd list                    # find the arm: look for "USB Serial" / CH340 / etc.
usbipd bind   --busid <BUSID>  # e.g. 2-4   (one-time per device)
usbipd attach --wsl --busid <BUSID>
```

Re-run `usbipd attach --wsl --busid <BUSID>` after every replug/reboot.
Repeat for the **second arm** (a different BUSID) so both show up.

**Back in WSL2**, confirm the device(s) appeared:

```bash
ls /dev/ttyACM*        # or /dev/ttyUSB*  -> you should see one per arm
sudo chmod 666 /dev/ttyACM0   # grant access (per device), or:
sudo usermod -aG dialout $USER && newgrp dialout   # permanent fix
```

## 4. Find which port is which arm

```bash
lerobot-find-port
```
Unplug one arm when prompted to learn its port (e.g. `/dev/ttyACM0`).
Note them, e.g. `follower=/dev/ttyACM0  leader=/dev/ttyACM1`.

## 5. Calibrate each arm

```bash
# Follower
lerobot-calibrate --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower

# Leader
lerobot-calibrate --teleop.type=so101_leader --teleop.port=/dev/ttyACM1 --teleop.id=my_leader
```

> If the motors were never given IDs (brand-new arm), run `lerobot-setup-motors`
> first — see the HF SO-101 docs. Skip if the arms already worked before.

Calibration files are saved under `~/.cache/huggingface/lerobot/calibration/`.
To share a calibration across machines, copy that folder (it's not in this repo).

## 6. Sanity check: read positions (no motion)

```bash
python -c "from so101 import make_arm; a=make_arm(port='/dev/ttyACM0',arm_id='my_follower',calibrate=False); a.connect(); print(a.read_joints()); a.disconnect()"
```
Or, with the OAK + wrist cam attached, `python auto_record.py --check` verifies the
whole setup (arms by serial + cameras) at once.

## 7. Teleop with Rerun

```bash
lerobot-teleoperate \
  --robot.type=so101_follower --robot.port=/dev/ttyACM0 --robot.id=my_follower \
  --teleop.type=so101_leader  --teleop.port=/dev/ttyACM1 --teleop.id=my_leader \
  --display_data=true --fps=60
```

---

## Windows ↔ Linux cheat sheet

| Windows (RUNBOOK.md) | WSL2 / Linux (this file) |
|---|---|
| `COM3`, `COM5` | `/dev/ttyACM0`, `/dev/ttyACM1` |
| `.\.venv\Scripts\activate` | `source .venv/bin/activate` |
| `.\.venv\Scripts\lerobot-*.exe` | `lerobot-*` (after activate) |
| ports just appear | must `usbipd attach` first |

## Keeping in sync with the team

```bash
git pull                       # get latest scripts/docs
git add -A && git commit -m "..." && git push   # share your changes
```
