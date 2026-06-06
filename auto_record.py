"""auto_record.py — zero-config, reliable demonstration recording for ACT.

Auto-detects the SO-101 arms (by stable board serial), the OAK-D-PRO (scene
camera, via DepthAI), and an external wrist webcam (excluding the laptop cam),
then records manual ENTER-controlled episodes into an **ACT-ready LeRobotDataset**.
It can push that dataset to the Hugging Face Hub and launch an **ACT finetune on
Qualia** (cloud).

Run with the project venv:
    .\\.venv\\Scripts\\python.exe auto_record.py --check         # detect hardware only (no arm motion)
    .\\.venv\\Scripts\\python.exe auto_record.py                 # record locally (manual episodes)
    .\\.venv\\Scripts\\python.exe auto_record.py --push          # + push dataset to the HF Hub
    .\\.venv\\Scripts\\python.exe auto_record.py --train         # + launch a Qualia ACT finetune (spends credits)
    .\\.venv\\Scripts\\python.exe auto_record.py --install       # bootstrap missing deps via uv, then continue

Reliability: connect is retried (the usual loose-cable motor dropout), Ctrl+C
finishes the current frame and saves the episode (second Ctrl+C aborts), and a
dataset name clash fails fast before any hardware is touched. The venv's Scripts
dir is added to PATH automatically so Rerun's viewer is found (no manual $env:Path).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

CORE_DEPS = {  # import name -> pip hint
    "serial": "pyserial", "cv2": "opencv-python", "numpy": "numpy", "yaml": "pyyaml",
    "depthai": "depthai", "pygrabber": "pygrabber", "lerobot": "lerobot[feetech]",
}


# --------------------------------------------------------------------------- #
# dependency bootstrap (uv)
# --------------------------------------------------------------------------- #
def ensure_deps(install: bool, need_rerun: bool, need_hub: bool) -> None:
    deps = dict(CORE_DEPS)
    if need_rerun:
        deps["rerun"] = "rerun-sdk"
    if need_hub:
        deps["huggingface_hub"] = "huggingface_hub"
    missing = [n for n in deps if importlib.util.find_spec(n) is None]
    if not missing:
        return
    print("Missing dependencies:", ", ".join(deps[m] for m in missing))
    if not install:
        print("\nInstall them with:\n    uv pip install -r requirements.txt")
        print("...or re-run this script with --install to do it automatically.")
        sys.exit(1)
    if os.environ.get("AUTO_RECORD_BOOTSTRAPPED"):
        print("Still missing after install — check that uv targeted this venv.")
        sys.exit(1)
    print("Installing dependencies via uv (uv pip install -r requirements.txt)...")
    try:
        # --python pins uv to THIS interpreter (the one os.execv re-runs), so install
        # and re-exec can't diverge into different environments.
        subprocess.check_call(
            ["uv", "pip", "install", "--python", sys.executable, "-r", "requirements.txt"]
        )
    except FileNotFoundError:
        print("uv not found on PATH. Install uv (https://astral.sh/uv) then retry, or run:\n"
              f"    {sys.executable} -m pip install -r requirements.txt")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"uv install failed (exit {e.returncode}).")
        sys.exit(1)
    os.environ["AUTO_RECORD_BOOTSTRAPPED"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)  # re-run with deps available


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def parse_size(s: str) -> tuple[int, int]:
    w, h = s.lower().split("x")
    return int(w), int(h)


def _make_end_poller():
    """Thread-free ENTER detection for ending a manual episode.

    Returns (drain, pressed): `drain()` clears any buffered keystrokes (call at
    episode start so stray ENTERs don't instantly end it); `pressed()` is a cheap
    non-blocking check returning True once ENTER was hit. No background thread, so
    nothing can outlive the episode and race a later prompt (e.g. the Qualia confirm).
    """
    try:
        import msvcrt  # Windows

        def drain():
            while msvcrt.kbhit():
                msvcrt.getwch()

        def pressed():
            hit = False
            while msvcrt.kbhit():
                if msvcrt.getwch() in ("\r", "\n"):
                    hit = True
            return hit

        return drain, pressed
    except ImportError:  # POSIX (WSL2/Linux)
        import select

        def drain():
            while select.select([sys.stdin], [], [], 0)[0]:
                if not sys.stdin.readline():
                    break  # EOF

        def pressed():
            hit = False
            while select.select([sys.stdin], [], [], 0)[0]:
                if not sys.stdin.readline():
                    break  # EOF
                hit = True
            return hit

        return drain, pressed


def _rerun_logger():
    try:
        import rerun as rr
    except ImportError:
        print("(rerun not installed — skipping live view)")
        return None, None
    from so101 import SO101_JOINTS

    rr.init("auto_record", spawn=True)

    def on_step(state, action):
        for j in SO101_JOINTS:
            rr.log(f"follower/{j}", rr.Scalars(state[j]))
            rr.log(f"leader/{j}", rr.Scalars(action[j]))

    def on_images(images):
        for name, img in images.items():
            rr.log(f"cameras/{name}", rr.Image(img))

    return on_step, on_images


def detect(cfg, args):
    """Return (ports, wrist_index). Pure detection — does NOT connect/move the arms."""
    from so101.discovery import find_arm_ports, find_wrist_camera, list_video_devices, oak_present

    ports = find_arm_ports({"follower": cfg["robot"]["serial"], "leader": cfg["teleop"]["serial"]})
    print(f"  arms    : follower={ports['follower']} ({cfg['robot']['serial']}), "
          f"leader={ports['leader']} ({cfg['teleop']['serial']})")

    if not oak_present():
        raise SystemExit("ERROR: OAK-D-PRO (scene camera) not detected. Plug it into USB3 and retry.")
    print("  scene   : OAK-D-PRO detected (DepthAI)")

    wrist_idx = None
    if not args.no_wrist:
        names = list_video_devices()
        wrist_idx = find_wrist_camera(override_index=args.wrist_index)
        if wrist_idx is None:
            print("  wrist   : no external webcam found -> scene-only (pass --no-wrist to silence)")
        else:
            label = names[wrist_idx] if names and wrist_idx < len(names) else "?"
            print(f"  wrist   : OpenCV index {wrist_idx}  ({label})")
    return ports, wrist_idx


# --------------------------------------------------------------------------- #
# Qualia ACT finetune
# --------------------------------------------------------------------------- #
def launch_qualia_act(dataset_id: str, recorded_cams: list[str], args) -> None:
    from so101 import qualia_client as q

    try:
        balance = q.credits().get("balance")
    except Exception as e:
        print(f"Qualia not available ({e}); skipping finetune. Dataset is on the Hub at {dataset_id}.")
        return

    # camera_mappings: model camera slot -> dataset image key.
    cam_map = {"image_top": "observation.images.scene"}
    if "wrist" in recorded_cams:
        cam_map["image_wrist"] = "observation.images.wrist"

    # Validate slot names against the live catalog so a renamed slot fails fast
    # (before spending) instead of erroring server-side.
    try:
        models = {m["id"]: m for m in q.list_models()}
        slots = models.get("act", {}).get("camera_slots")
        if slots:
            bad = [k for k in cam_map if k not in slots]
            if bad:
                print(f"WARNING: camera slots {bad} not in ACT's slots {slots}; dropping them.")
                cam_map = {k: v for k, v in cam_map.items() if k in slots}
    except Exception:
        pass  # catalog check is best-effort; don't block on a transient API hiccup

    print("\n== Qualia ACT finetune ==")
    print(f"  dataset : {dataset_id}")
    print(f"  vla_type: act   (model_id=None — ACT uses a fixed base model)")
    print(f"  hours   : {args.hours}")
    print(f"  cameras : {cam_map}")
    print(f"  credits : {balance} available")
    if not args.yes_spend:
        if input("Type 'yes' to spend credits and launch: ").strip().lower() != "yes":
            print("Skipped finetune (dataset is already on the Hub; launch later if you want).")
            return
    job = q.launch_finetune(
        dataset_id=dataset_id, vla_type="act", model_id=None,
        hours=args.hours, camera_mappings=cam_map,
    )
    print(f"Launched: {job}")
    print(f"Poll: {sys.executable} -c \"from so101 import qualia_client as q; "
          f"print(q.job_status('{job['job_id']}'))\"")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="detect hardware and exit (no recording, no arm motion)")
    ap.add_argument("--install", action="store_true", help="install missing deps via uv, then continue")
    ap.add_argument("--config", default=None, help="path to config.yaml (default: repo config.yaml)")
    ap.add_argument("--name", default="trash_pick", help="dataset name")
    ap.add_argument("--task", default="pick up the trash and drop it in the bin")
    ap.add_argument("--episodes", type=int, default=40, help="max demos (manual mode: quit early with 'q')")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--scene-size", default="640x480", help="OAK scene resolution WxH")
    ap.add_argument("--wrist-size", default="640x480", help="wrist webcam resolution WxH")
    ap.add_argument("--wrist-index", type=int, default=None, help="force the wrist webcam OpenCV index")
    ap.add_argument("--no-wrist", action="store_true", help="record the OAK scene camera only")
    ap.add_argument("--no-display", action="store_true", help="don't stream to Rerun")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing dataset of the same name")
    # cloud
    ap.add_argument("--push", action="store_true", help="push the dataset to the HF Hub")
    ap.add_argument("--train", action="store_true", help="push + launch a Qualia ACT finetune (spends credits)")
    ap.add_argument("--hf-user", default=None, help="HF namespace for the dataset (auto-detected if omitted)")
    ap.add_argument("--hours", type=float, default=2.0, help="Qualia finetune duration (hours)")
    ap.add_argument("--yes-spend", action="store_true", help="skip the credit-spend confirmation prompt")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)  # config.yaml / recorded/ paths + Qualia .env all resolve from here
    # Put the venv's Scripts/bin dir on PATH so Rerun's viewer is found automatically.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

    push = args.push or args.train
    ensure_deps(install=args.install, need_rerun=not args.no_display and not args.check, need_hub=push)

    import yaml

    cfg_path = Path(args.config) if args.config else REPO_ROOT / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    print("== Auto-detecting hardware ==")
    ports, wrist_idx = detect(cfg, args)
    if args.check:
        print("\n--check: detection only, exiting (no recording).")
        return

    # Resolve dataset id / output dir.
    hf_user = args.hf_user
    if push and not hf_user:
        from huggingface_hub import HfApi

        hf_user = HfApi().whoami()["name"]
    repo_id = f"{hf_user}/{args.name}" if push else f"local/{args.name}"
    root = REPO_ROOT / "recorded" / args.name

    from coord_grasp.oak import OakCamera
    from so101 import make_arm, make_teleop
    from so101.discovery import Cv2Camera
    from so101.record import prepare_dataset_dir, record_teleop_dataset
    from so101.reliability import connect_with_retry, graceful_stop

    # Fail fast on a dataset name clash BEFORE connecting hardware.
    try:
        prepare_dataset_dir(repo_id, str(root), args.overwrite)
    except FileExistsError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if args.episodes < 10:
        print(f"  note: {args.episodes} episodes is light for ACT — aim for 30–50+ for a usable policy.")

    sw, sh = parse_size(args.scene_size)
    ww, wh = parse_size(args.wrist_size)
    follower = make_arm(port=ports["follower"], arm_id=cfg["robot"]["id"], calibrate=False)
    teleop = make_teleop(port=ports["leader"], teleop_id=cfg["teleop"]["id"], calibrate=False)
    scene = OakCamera(size=(sw, sh))
    extra = {"scene": scene}
    if wrist_idx is not None:
        extra["wrist"] = Cv2Camera(wrist_idx, width=ww, height=wh, fps=args.fps)

    drain_keys, end_pressed = _make_end_poller()

    def await_start(ep: int) -> bool:
        try:
            resp = input(f"\n=== Episode {ep + 1}/{args.episodes} === ENTER to START "
                         "(or type 'q' then ENTER to finish): ")
        except (EOFError, KeyboardInterrupt):
            return False  # Ctrl+C / EOF at the prompt -> finish the session cleanly
        if resp.strip().lower() == "q":
            return False
        drain_keys()  # clear stray keystrokes so the episode doesn't end instantly
        print("  recording... move the leader. Press ENTER to END this episode.")
        return True

    summary = None
    # Connects are INSIDE this try, so the finally tears down ANY device that
    # connected even if a later connect (or recording) fails -- no torque left on
    # and no COM port left locked.
    try:
        print("\n== Connecting (with retry) ==")
        connect_with_retry(follower, "follower")
        connect_with_retry(teleop, "leader")
        connect_with_retry(scene, "scene(OAK)")
        if "wrist" in extra:
            connect_with_retry(extra["wrist"], "wrist")

        on_step, on_images = (None, None) if args.no_display else _rerun_logger()

        print(f"\n== Recording into {repo_id}  (root: {root}) ==")
        print("Move the LEADER to demonstrate. Keep clear of the follower.\n")
        with graceful_stop() as should_stop:
            summary = record_teleop_dataset(
                follower, teleop, repo_id=repo_id, task=args.task,
                cameras={}, extra_cameras=extra,
                num_episodes=args.episodes, episode_steps=None,
                fps=args.fps, root=str(root), overwrite=args.overwrite,
                push_to_hub=push,
                on_step=on_step, on_images=on_images,
                progress=lambda d, t: print(f"\r  frame {d}", end="", flush=True),
                should_stop=should_stop, await_start=await_start,
                end_episode=lambda step: end_pressed(),
            )
    except KeyboardInterrupt:
        print("\nAborted (hard).")
    except Exception as e:
        print(f"\nConnect/recording error: {e}")
    finally:
        for dev, _ in ((teleop, "leader"), (follower, "follower")):
            try:
                dev.disconnect()
            except Exception:
                pass
        for cam in extra.values():
            try:
                cam.close()
            except Exception:
                pass
        print("\nDisconnected.")

    if not summary or summary.get("num_episodes", 0) == 0:
        print("No episodes recorded — nothing to push or train.")
        return
    print(f"\nRecorded {summary['num_episodes']} episode(s), {summary['num_frames']} frames -> {summary['root']}")
    if push:
        print(f"Pushed to the HF Hub: {repo_id}")
    if args.train:
        launch_qualia_act(repo_id, list(extra), args)


if __name__ == "__main__":
    main()
