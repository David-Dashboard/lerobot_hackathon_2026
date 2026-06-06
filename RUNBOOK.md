# SO-101 + LeRobot + Rerun — Hackathon Runbook

Project venv: `.\.venv` (Python 3.11). LeRobot 0.4.4, Rerun 0.26.2.

All commands are run from this folder: `C:\Users\Succe\lerobot-hackathon`
The venv tools live in `.\.venv\Scripts\`. Either activate the venv first
(`.\.venv\Scripts\Activate.ps1`) or prefix commands with `.\.venv\Scripts\`.

---

## 0. Hardware checklist (do this first)

Each arm's controller board needs BOTH:
1. USB cable -> PC  (creates the COM port)
2. Power supply -> barrel jack  (lets the motors respond)

You should see **two** COM ports (one per arm). Check with PowerShell:
```powershell
[System.IO.Ports.SerialPort]::GetPortNames()
```
If you only see one, the other arm's USB or power is not connected.
(Boards enumerate as "USB Serial Device (COMx)" using the built-in Windows
`usbser` driver — no driver install needed.)

---

## 1. Identify which COM port is which arm

```powershell
.\.venv\Scripts\lerobot-find-port.exe
```
It lists the ports, asks you to UNPLUG one arm and press Enter, then tells you
which port that arm was. Run it twice (or note both) to label leader vs follower.

Write them down, e.g.:  follower = COM3 , leader = COM5

---

## 2. Hello world #1 — READ ONLY (no motion)

Proves comms without moving anything. Move the arm BY HAND and watch the
Rerun plots react.
```powershell
.\.venv\Scripts\python.exe hello_read.py --port COM3 --id my_follower
```
Ctrl+C to stop. (Swap the port for the leader to test that one too.)

---

## 3. Calibrate each arm (one-time, needed before teleop)

Follower:
```powershell
.\.venv\Scripts\lerobot-calibrate.exe --robot.type=so101_follower --robot.port=COM3 --robot.id=my_follower
```
Leader:
```powershell
.\.venv\Scripts\lerobot-calibrate.exe --teleop.type=so101_leader --teleop.port=COM5 --teleop.id=my_leader
```
Follow the on-screen prompts (move joints through their range, set middle pose).
Calibration is saved and reused automatically next time.

> Only if the motors were never ID'd (brand-new arm): run `lerobot-setup-motors`
> first — see the HF SO-101 docs. If the arms already worked before, skip it.

---

## 4. Hello world #2 — TELEOP with Rerun (leader drives follower)

Keep hands clear of the follower. `--display_data=true` opens the Rerun viewer
showing live leader + follower joint values.
```powershell
.\.venv\Scripts\lerobot-teleoperate.exe `
  --robot.type=so101_follower --robot.port=COM3 --robot.id=my_follower `
  --teleop.type=so101_leader  --teleop.port=COM5 --teleop.id=my_leader `
  --display_data=true --fps=60
```
Move the leader arm; the follower mirrors it. Ctrl+C to stop.

---

## Remote control (the "talk to it remotely" goal — next step)

LeRobot ships `lekiwi`-style host/client networking, but the simplest remote
hello-world is to run teleop/inference on this PC and expose a tiny HTTP or
WebSocket endpoint that calls `robot.send_action({...})`. Ask Claude to scaffold
that once the local teleop works.

## Troubleshooting
- No COM port: bad/charge-only USB cable, or board not powered. Try another port.
- Permission/busy error: another program (or a previous run) holds the port —
  close it. Only one process can open a COM port at a time.
- Waveshare board: set both jumpers to the **B (USB)** channel.
