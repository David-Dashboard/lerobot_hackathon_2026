#!/usr/bin/env bash
# setup_wsl2.sh — one-shot environment setup for the SO-101 + OAK stack under WSL2.
#
# Automates SETUP_WSL2.md: installs uv + the venv + deps, grants serial access,
# installs the OAK udev rule, and checks which USB devices are attached (guiding
# you through the usbipd passthrough that must happen on the Windows host).
#
# Idempotent — safe to re-run after attaching more devices.
#
#   bash setup_wsl2.sh
#
# Device reality under WSL2 (important):
#   * SO-101 arms (USB serial / cdc-acm) ...... pass through via usbipd  ✓
#   * OAK-D-PRO (Luxonis, libusb/depthai) ...... pass through via usbipd  ✓
#   * Plain UVC webcams ........................ NOT supported — the default
#       WSL2 kernel has no `uvcvideo`, so /dev/video* won't appear. Use the OAK
#       for vision in WSL2, or run the OpenCV-camera path on Windows.

set -uo pipefail

note() { printf '\n\033[1;36m==>\033[0m %s\n' "$*"; }
ok()   { printf '   \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '   \033[33m!\033[0m %s\n' "$*"; }

cd "$(dirname "$0")"

# --- 0. WSL sanity -----------------------------------------------------------
if ! grep -qiE "microsoft|wsl" /proc/version 2>/dev/null; then
  warn "This doesn't look like WSL2 (per /proc/version). Continuing anyway."
fi

# --- 1. uv -------------------------------------------------------------------
note "Checking uv..."
if ! command -v uv >/dev/null 2>&1; then
  note "Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # shellcheck disable=SC1090
  source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi
ok "uv $(uv --version 2>/dev/null || echo '(not found — restart shell)')"

# --- 2. venv + deps ----------------------------------------------------------
note "Creating .venv (Python 3.11) + installing deps (this can take a few minutes)..."
uv venv --python 3.11
# shellcheck disable=SC1091
source .venv/bin/activate
if [ -f requirements.txt ]; then
  uv pip install -r requirements.txt
else
  uv pip install "lerobot[feetech]" rerun-sdk "transformers>=4.57.1,<5" pyyaml opencv-python
fi
uv pip install depthai           # OAK-D-PRO (DepthAI) — not in requirements.txt
uv pip install pytest            # dev/test
ok "Python environment ready"

# --- 3. serial (arm) permissions --------------------------------------------
note "Serial permissions (dialout group)..."
if id -nG 2>/dev/null | grep -qw dialout; then
  ok "already in 'dialout' group"
elif sudo usermod -aG dialout "$USER" 2>/dev/null; then
  warn "added you to 'dialout' — run 'newgrp dialout' or re-login to apply"
else
  warn "could not add to dialout (need sudo). Manual: sudo usermod -aG dialout \$USER"
fi

# --- 4. OAK-D udev rule (Movidius VID 03e7) so depthai runs without root -----
note "OAK-D udev rule..."
RULE=/etc/udev/rules.d/80-movidius.rules
if [ -f "$RULE" ]; then
  ok "udev rule already present ($RULE)"
elif echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee "$RULE" >/dev/null 2>&1; then
  sudo udevadm control --reload-rules 2>/dev/null && sudo udevadm trigger 2>/dev/null
  ok "installed $RULE"
else
  warn "could not write $RULE (need sudo). depthai may require running as root."
fi

# --- 5. attached-device check ------------------------------------------------
note "Attached USB devices (must be usbipd-attached from the Windows host):"
ports=$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true)
[ -n "$ports" ] && ok "arms (serial): $ports" || warn "no /dev/ttyACM*|ttyUSB* — arms not attached yet"

if lsusb 2>/dev/null | grep -qi '03e7'; then
  ok "OAK-D (Movidius) visible to lsusb"
else
  warn "OAK-D not visible — attach it via usbipd (note: it re-enumerates on boot, see below)"
fi

ls /dev/video* >/dev/null 2>&1 \
  && ok "video devices: $(ls /dev/video* | tr '\n' ' ')" \
  || warn "no /dev/video* — UVC webcams are NOT supported by the WSL2 kernel (expected)"

# --- 6. usbipd guidance ------------------------------------------------------
if command -v usbipd.exe >/dev/null 2>&1; then
  note "Bindable USB devices on the Windows host (usbipd.exe list):"
  usbipd.exe list 2>/dev/null || true
fi
cat <<'EOS'

To pass a device into WSL2, run on the WINDOWS host (PowerShell as Administrator):
    usbipd list                         # find BUSIDs (arms = CH343/USB-Serial, OAK = Movidius MyriadX)
    usbipd bind   --busid <BUSID>       # once per device
    usbipd attach --wsl --busid <BUSID> # re-run after every replug / reboot

OAK-D note: it re-enumerates when its firmware boots (unbooted -> booted USB id),
which can drop the usbipd attachment. If depthai can't find it, re-run
`usbipd attach` for the new BUSID, or keep it attached with `usbipd attach --wsl --auto`.

Then re-run this script to confirm the devices appear.
EOS

note "Done. Next: source .venv/bin/activate   (then follow SETUP_WSL2.md from step 4)"
