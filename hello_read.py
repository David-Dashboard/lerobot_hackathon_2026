"""
Hello-world #1: READ ONLY (no motion).

Connects to one SO-101 arm, reads its joint positions in a loop, prints them,
and streams them to Rerun as live time-series plots. Sends NO commands, so the
arm will not move -- this just proves the USB/serial comms work end-to-end.

Usage (from the project folder, venv python):
    .\.venv\Scripts\python.exe hello_read.py --port COM3 --id my_follower

Move the arm by hand while this runs; you'll see the plots react in Rerun.
Press Ctrl+C to stop.
"""

import argparse
import time

import rerun as rr

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="COM port, e.g. COM3")
    parser.add_argument("--id", default="hello_follower", help="arm id / name")
    parser.add_argument("--hz", type=float, default=30.0, help="read rate")
    args = parser.parse_args()

    # Start the Rerun viewer (opens a window and streams data to it).
    rr.init("so101_hello_read", spawn=True)

    cfg = SO101FollowerConfig(port=args.port, id=args.id, cameras={})
    robot = SO101Follower(cfg)

    # calibrate=False -> just read raw present positions, don't run calibration.
    print(f"Connecting to {args.id} on {args.port} ...")
    robot.connect(calibrate=False)
    print("Connected. Reading positions (move the arm by hand). Ctrl+C to stop.\n")

    period = 1.0 / args.hz
    try:
        while True:
            obs = robot.get_observation()
            # obs is a dict like {"shoulder_pan.pos": 12.3, ...} plus any cameras.
            joints = {k: v for k, v in obs.items() if k.endswith(".pos")}
            for name, value in joints.items():
                rr.log(f"joints/{name.removesuffix('.pos')}", rr.Scalars(float(value)))
            line = "  ".join(f"{k.removesuffix('.pos')}={v:7.2f}" for k, v in joints.items())
            print("\r" + line, end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        robot.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
