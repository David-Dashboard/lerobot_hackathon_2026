"""
Gather demonstration data by TELEOPERATION (leader drives follower).

This is the data-collection counterpart to `lerobot-teleoperate` (see RUNBOOK
section 4): a human moves the SO-101 *leader*, the follower mirrors it, and every
frame -- follower pose (`observation.state`), leader target (`action`), and the
scene + wrist camera images -- is recorded into a LeRobotDataset. Those episodes
are the training data for the learned grasp skill in Phase 2 of the trash-arm plan.

Examples
--------
No hardware (validate the whole pipeline -- writes a real dataset with synthetic
frames):

    .\\.venv\\Scripts\\python.exe record_teleop.py --mock --episodes 2 --episode-seconds 3

Real arms + two cameras (scene cam on index 0, wrist cam on index 2):

    .\\.venv\\Scripts\\python.exe record_teleop.py `
      --robot-port COM3 --robot-id my_follower `
      --teleop-port COM5 --teleop-id my_leader `
      --camera scene=0:640x480 --camera wrist=2:640x480 `
      --repo-id local/trash_teleop --task "pick up the trash and drop it in the bin" `
      --episodes 10 --episode-seconds 20 --reset-seconds 8 --display

Move the leader to perform each demo. Ctrl+C stops cleanly (acts as an e-stop:
the current frame finishes, the episode is saved, and both arms disconnect so
torque releases and the COM ports free up).
"""

from __future__ import annotations

import argparse
import threading

from so101 import SO101_JOINTS, format_joint_line, make_arm, make_teleop
from so101.record import parse_camera_spec, prepare_dataset_dir, record_teleop_dataset


class _EnterWatcher:
    """Watches stdin for a single ENTER press on a background thread.

    Used for manual episode control: `arm()` starts waiting; `triggered` flips
    True once the operator presses ENTER, ending the current episode.
    """

    def __init__(self):
        self._evt = threading.Event()

    def arm(self) -> None:
        self._evt.clear()
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self) -> None:
        try:
            input()
        except (EOFError, RuntimeError):
            pass
        self._evt.set()

    @property
    def triggered(self) -> bool:
        return self._evt.is_set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # Follower (the recorded arm).
    parser.add_argument("--robot-port", help="follower COM port, e.g. COM3 (omit with --mock)")
    parser.add_argument("--robot-id", default="my_follower", help="follower arm id (must match its calibration)")
    # Leader (the teleoperator).
    parser.add_argument("--teleop-port", help="leader COM port, e.g. COM5 (omit with --mock)")
    parser.add_argument("--teleop-id", default="my_leader", help="leader arm id (must match its calibration)")

    # Cameras: repeatable name=index[:WxH][@fps], e.g. scene=0:640x480 wrist=2:640x480@15
    parser.add_argument(
        "--camera", action="append", default=[], metavar="NAME=INDEX[:WxH][@FPS]",
        help="add a camera (repeatable). Ignored under --mock unless --synthetic-frames.",
    )

    # Dataset.
    parser.add_argument("--repo-id", default="local/trash_teleop", help="LeRobotDataset repo id")
    parser.add_argument("--task", default="pick up the trash and drop it in the bin", help="natural-language task label")
    parser.add_argument("--root", default=None, help="dataset output dir (default: LeRobot cache)")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing dataset at the target path instead of erroring")
    parser.add_argument("--push-to-hub", action="store_true", help="upload to the HF Hub when done")

    # Loop shape.
    parser.add_argument("--episodes", type=int, default=5, help="max demos to record")
    parser.add_argument("--manual", action="store_true", help="press ENTER to start each episode and ENTER again to end it (recommended; ignores --episode-seconds/--reset-seconds)")
    parser.add_argument("--episode-seconds", type=float, default=20.0, help="recorded seconds per demo (timed mode only)")
    parser.add_argument("--reset-seconds", type=float, default=5.0, help="unrecorded reset time between demos (timed mode only)")
    parser.add_argument("--fps", type=int, default=30, help="control + record rate")

    parser.add_argument("--mock", action="store_true", help="use simulated arms (no hardware)")
    parser.add_argument("--synthetic-frames", action="store_true", help="synthesize any camera frame missing from the observation (auto-on with --mock)")
    parser.add_argument("--no-calibrate", action="store_true", help="skip calibration prompts (arms must already be calibrated)")
    parser.add_argument("--display", action="store_true", help="stream leader/follower joints to Rerun")
    # OAK-D-PRO (DepthAI) RGB -- recorded alongside the OpenCV --camera(s).
    parser.add_argument("--oak", action="store_true", help="also record the OAK-D-PRO RGB camera (DepthAI, non-UVC) simultaneously")
    parser.add_argument("--oak-name", default="oak", help="dataset image key for the OAK camera")
    parser.add_argument("--oak-width", type=int, default=640)
    parser.add_argument("--oak-height", type=int, default=400)
    args = parser.parse_args()

    if not args.mock and (not args.robot_port or not args.teleop_port):
        parser.error("--robot-port and --teleop-port are required unless --mock is given")

    cameras = dict(parse_camera_spec(spec) for spec in args.camera)
    # Under mock there are no real cameras; synthesize so the dataset still has images.
    synthetic_frames = args.synthetic_frames or args.mock

    watcher = _EnterWatcher() if args.manual else None
    if args.manual:
        episode_steps = None  # episodes end on ENTER, not a frame count
        reset_steps = 0
    else:
        episode_steps = max(1, round(args.episode_seconds * args.fps))
        reset_steps = max(0, round(args.reset_seconds * args.fps))

    # Fail fast on a dataset name clash -- before spawning Rerun or touching hardware.
    try:
        target = prepare_dataset_dir(args.repo_id, args.root, args.overwrite)
    except FileExistsError as e:
        print(f"\nERROR: {e}")
        raise SystemExit(1)
    print(f"Recording into: {target}\n")

    on_step, on_images = _make_rerun_logger() if args.display else (None, None)

    follower = make_arm(
        mock=args.mock, port=args.robot_port, arm_id=args.robot_id,
        calibrate=not args.no_calibrate, cameras=cameras or None,
    )
    teleop = make_teleop(
        mock=args.mock, port=args.teleop_port, teleop_id=args.teleop_id,
        calibrate=not args.no_calibrate,
    )

    # Extra (non-UVC) cameras captured alongside the OpenCV ones -- e.g. the OAK.
    extra_cameras = {}
    if args.oak:
        if args.oak_name in cameras:
            parser.error(f"--oak-name {args.oak_name!r} collides with an OpenCV --camera name")
        from coord_grasp.oak import OakCamera
        extra_cameras[args.oak_name] = OakCamera(size=(args.oak_width, args.oak_height))

    label = "MOCK arms" if args.mock else f"follower {args.robot_id}@{args.robot_port}, leader {args.teleop_id}@{args.teleop_port}"
    cam_label = ", ".join(list(cameras) + list(extra_cameras)) or "none"
    print(f"Connecting to {label} ... (cameras: {cam_label})")
    follower.connect()
    teleop.connect()
    for cam in extra_cameras.values():
        cam.connect()
    mode = "manual (ENTER starts/ends each episode)" if args.manual else f"{episode_steps} frames/episode"
    print(
        f"Connected. Up to {args.episodes} episode(s), {mode} @ {args.fps} fps into {args.repo_id!r}.\n"
        "Move the LEADER to demonstrate. Keep clear of the follower. Ctrl+C = stop/e-stop.\n"
    )

    interrupted = {"flag": False}

    def progress(done: int, total) -> None:
        bar = f"{done}/{total}" if total else str(done)
        print(f"\r  frame {bar}", end="", flush=True)

    def await_start(ep: int) -> bool:
        try:
            resp = input(f"\n=== Episode {ep + 1}/{args.episodes} === press ENTER to START "
                         "(or type 'q' then ENTER to finish): ")
        except EOFError:
            return False  # stdin closed -> finish the session cleanly
        if resp.strip().lower() == "q":
            return False
        watcher.arm()
        print("  recording... move the leader. Press ENTER to END this episode.")
        return True

    try:
        summary = record_teleop_dataset(
            follower, teleop,
            repo_id=args.repo_id, task=args.task,
            cameras=cameras, extra_cameras=extra_cameras,
            num_episodes=args.episodes, episode_steps=episode_steps,
            reset_steps=reset_steps, fps=args.fps, root=args.root, overwrite=args.overwrite,
            push_to_hub=args.push_to_hub, synthetic_frames=synthetic_frames,
            on_step=on_step, on_images=on_images, progress=progress,
            should_stop=lambda: interrupted["flag"],
            await_start=await_start if args.manual else None,
            end_episode=(lambda step: watcher.triggered) if args.manual else None,
        )
    except KeyboardInterrupt:
        # First Ctrl+C lands here only if it interrupts outside the loop's own
        # handling; flag it so any in-progress call also unwinds, then report.
        interrupted["flag"] = True
        print("\nInterrupted -- stopping.")
        summary = None
    finally:
        teleop.disconnect()
        follower.disconnect()
        for cam in extra_cameras.values():
            try:
                cam.close()
            except Exception:
                pass
        print("\nDisconnected.")

    if summary:
        print(
            f"\nDone. {summary['num_episodes']} episode(s), {summary['num_frames']} frames "
            f"-> {summary['root']}"
            + ("  (pushed to hub)" if summary["pushed"] else "")
            + ("  [stopped early]" if summary["stopped_early"] else "")
        )


def _make_rerun_logger():
    """Return (on_step, on_images) that stream joints + live camera feeds to Rerun.

    Returns (None, None) if rerun isn't importable.
    """
    try:
        import rerun as rr
    except ImportError:
        print("(rerun not installed -- skipping --display)")
        return None, None

    rr.init("so101_teleop_record", spawn=True)

    def on_step(state: dict, action: dict) -> None:
        for name in SO101_JOINTS:
            rr.log(f"follower/{name}", rr.Scalars(state[name]))
            rr.log(f"leader/{name}", rr.Scalars(action[name]))
        print("\r" + format_joint_line(state), end="", flush=True)

    def on_images(images: dict) -> None:
        for name, img in images.items():
            rr.log(f"cameras/{name}", rr.Image(img))

    return on_step, on_images


if __name__ == "__main__":
    main()
