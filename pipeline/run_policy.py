"""Run a trained ACT policy on the SO-101 — OAK scene + wrist webcam + joint state.

The model trained on Qualia expects observation.images.scene (OAK-D), .wrist (USB
webcam) and observation.state (6 joints); LeRobot's `lerobot-record` can't feed the
OAK, so this drives inference directly via LeRobot's `predict_action` and sends the
result to the follower.

    # SAFE rehearsal (reads arm + cameras, runs the policy, PRINTS actions, NO motion):
    python pipeline/run_policy.py --policy qualia-robotics/act-batch1-9b99bbf0 --dry-run

    # REAL motion (starts slow; Ctrl+C = e-stop -> torque off):
    python pipeline/run_policy.py --policy qualia-robotics/act-batch1-9b99bbf0 --go

Needs the follower arm + OAK + wrist webcam free (close any process holding them).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DEFAULT_POLICY = "qualia-robotics/act-batch1-9b99bbf0"


def clamp_action(target, current, joint_names, limits, max_step_deg):
    """Rate- and range-limit a commanded action for safety.

    Each joint moves at most `max_step_deg` from its current measured position this
    tick, then is clamped to its [lo, hi] limit. Returns a {joint: deg} dict.
    """
    out = {}
    for i, j in enumerate(joint_names):
        tgt = float(target[i])
        cur = float(current.get(j, tgt))
        delta = max(-max_step_deg, min(max_step_deg, tgt - cur))
        val = cur + delta
        if j in limits:
            lo, hi = limits[j]
            val = max(lo, min(hi, val))
        out[j] = val
    return out


def _detect_follower_port(cfg) -> str:
    """Find the follower COM port regardless of its number: match the configured
    board serial, else fall back to the sole SO-101 (CH343) port present."""
    from serial.tools import list_ports

    from so101.discovery import find_arm_ports

    serial = cfg.get("robot", {}).get("serial")
    if serial:
        try:
            return find_arm_ports({"follower": serial})["follower"]
        except Exception as e:
            print(f"(serial {serial} not matched: {str(e).splitlines()[0]})")
    cands = [p.device for p in list_ports.comports() if (p.vid == 0x1A86)]  # WCH CH343 = SO-101
    if len(cands) == 1:
        print(f"(using the sole SO-101 port found: {cands[0]})")
        return cands[0]
    raise SystemExit(
        f"Could not auto-detect the follower (SO-101 ports: {cands or 'none'}). "
        "Plug it in, or set robot.serial in config.yaml."
    )


def _load_policy(policy_path: str):
    import torch
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors

    policy = ACTPolicy.from_pretrained(policy_path)
    policy.to("cpu").eval()
    policy.reset()
    pre, post = make_pre_post_processors(
        policy_cfg=policy.config, pretrained_path=policy_path,
        preprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    return policy, pre, post, torch.device("cpu")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="HF id or local path of the trained ACT policy")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="read + infer + PRINT actions, never move the arm (default if --go absent)")
    ap.add_argument("--go", action="store_true", help="actually MOVE the arm (real deployment)")
    ap.add_argument("--fps", type=int, default=30, help="control rate (ACT was trained at 30)")
    ap.add_argument("--max-step-deg", type=float, default=8.0, help="max per-joint move per tick (speed cap / safety)")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after N steps (0 = until Ctrl+C)")
    ap.add_argument("--task", default="clear the table")
    args = ap.parse_args()

    move = args.go and not args.dry_run
    if not args.go and not args.dry_run:
        print("No mode given -> defaulting to --dry-run (no motion). Use --go to move the arm.")
    import os
    os.chdir(REPO)
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

    from dotenv import load_dotenv
    load_dotenv(str(REPO / ".env"))

    import yaml
    cfg = yaml.safe_load((Path(args.config) if args.config else REPO / "config.yaml").read_text(encoding="utf-8"))
    limits = cfg.get("safety", {}).get("joint_limits", {})

    from coord_grasp.oak import OakCamera
    from lerobot.utils.control_utils import predict_action
    from so101 import SO101_JOINTS, make_arm
    from so101.discovery import Cv2Camera, find_wrist_camera, oak_present
    from so101.reliability import connect_with_retry, graceful_stop

    print(f"Loading policy {args.policy} (CPU) ...")
    policy, pre, post, device = _load_policy(args.policy)
    print("policy + processors ready.")

    # --- detect hardware (port-agnostic: serial match, else the sole SO-101 port) ---
    follower_port = _detect_follower_port(cfg)
    if not oak_present():
        sys.exit("OAK-D-PRO (scene cam) not detected. Plug it into USB3 and retry.")
    wrist_idx = find_wrist_camera()
    if wrist_idx is None:
        sys.exit("Wrist webcam not found (the model needs observation.images.wrist).")
    print(f"follower={follower_port}  scene=OAK  wrist=index {wrist_idx}")

    follower = make_arm(port=follower_port, arm_id=cfg["robot"]["id"], calibrate=False)
    scene = OakCamera(size=(640, 480))
    wrist = Cv2Camera(wrist_idx, width=640, height=480, fps=args.fps)

    mode = "MOVE (real)" if move else "DRY-RUN (no motion)"
    print(f"\n== Deploying ACT — {mode} @ {args.fps} fps, max {args.max_step_deg}°/tick ==")
    if move:
        print("The arm WILL move. Keep clear; Ctrl+C = e-stop (torque off).")

    period = 1.0 / args.fps
    n = 0
    try:
        connect_with_retry(follower, "follower")
        connect_with_retry(scene, "scene(OAK)")
        connect_with_retry(wrist, "wrist")
        with graceful_stop() as should_stop:
            while not should_stop():
                t0 = time.perf_counter()
                state = follower.read_joints()
                obs = {
                    "observation.state": np.array([state[j] for j in SO101_JOINTS], dtype=np.float32),
                    "observation.images.scene": np.asarray(scene.read()),
                    "observation.images.wrist": np.asarray(wrist.read()),
                }
                action = np.asarray(
                    predict_action(obs, policy, device, pre, post, use_amp=False,
                                   task=args.task, robot_type="so101_follower")
                ).flatten()
                cmd = clamp_action(action, state, SO101_JOINTS, limits, args.max_step_deg)
                if move:
                    follower.write_joints(cmd)
                    tag = "->"
                else:
                    tag = "[dry]"
                n += 1
                if n % args.fps == 0 or not move:
                    print(f"  {tag} " + " ".join(f"{j}={cmd[j]:6.1f}" for j in SO101_JOINTS))
                if args.max_steps and n >= args.max_steps:
                    break
                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\nstopped.")
    except Exception as e:
        print(f"\nstopped: {type(e).__name__}: {e}")
    finally:
        try:
            follower.disconnect()  # disable_torque_on_disconnect=True -> arm goes limp
        except Exception:
            pass
        for cam in (scene, wrist):
            try:
                cam.close()
            except Exception:
                pass
        print(f"Disconnected. ({n} steps)")


if __name__ == "__main__":
    main()
