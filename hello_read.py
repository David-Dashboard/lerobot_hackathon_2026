"""
Hello-world #1: READ ONLY (no motion).

Connects to one SO-101 arm, reads its joint positions in a loop, prints them,
and streams them to Rerun as live time-series plots. Sends NO commands, so the
arm will not move -- this just proves the USB/serial comms work end-to-end.

Usage (from the project folder, venv python):
    .\.venv\Scripts\python.exe hello_read.py --port COM3 --id my_follower

No arm handy? Run the whole thing against a simulated arm:
    .\.venv\Scripts\python.exe hello_read.py --mock

Move the arm by hand while this runs; you'll see the plots react in Rerun.
Press Ctrl+C to stop.
"""

import argparse
import time

import rerun as rr

from arm_utils import extract_joint_positions, format_joint_line


def make_robot(args):
    """Return a connected robot -- real SO-101 or the hardware-free mock."""
    if args.mock:
        from mock_robot import MockSO101Follower

        robot = MockSO101Follower(port="MOCK", robot_id=args.id)
        robot.connect()
        return robot

    from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

    cfg = SO101FollowerConfig(port=args.port, id=args.id, cameras={})
    robot = SO101Follower(cfg)
    robot.connect(calibrate=False)  # just read raw positions; don't calibrate
    return robot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="COM port, e.g. COM3 (omit with --mock)")
    parser.add_argument("--id", default="hello_follower", help="arm id / name")
    parser.add_argument("--hz", type=float, default=30.0, help="read rate")
    parser.add_argument("--mock", action="store_true", help="use a simulated arm (no hardware)")
    args = parser.parse_args()

    if not args.mock and not args.port:
        parser.error("--port is required unless --mock is given")

    rr.init("so101_hello_read", spawn=True)

    label = "MOCK arm" if args.mock else f"{args.id} on {args.port}"
    print(f"Connecting to {label} ...")
    robot = make_robot(args)
    print("Connected. Reading positions (move the arm by hand). Ctrl+C to stop.\n")

    period = 1.0 / args.hz
    try:
        while True:
            obs = robot.get_observation()
            joints = extract_joint_positions(obs)
            for name, value in joints.items():
                rr.log(f"joints/{name}", rr.Scalars(value))
            print("\r" + format_joint_line(joints), end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
