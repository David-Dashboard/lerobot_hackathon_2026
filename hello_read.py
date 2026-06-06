"""
Hello-world #1: READ ONLY (no motion).

Reads one SO-101 arm's joint positions in a loop, prints them, and streams them
to Rerun as live plots. Sends NO commands. Proves comms end-to-end.

    .\.venv\Scripts\python.exe hello_read.py --port COM3 --id my_follower
    .\.venv\Scripts\python.exe hello_read.py --mock        # no hardware

Move the arm by hand; the plots react. Ctrl+C to stop.
"""

import argparse
import time

import rerun as rr

from so101 import format_joint_line, make_arm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="COM port, e.g. COM3 (omit with --mock)")
    parser.add_argument("--id", default="hello_follower", help="arm id / name")
    parser.add_argument("--hz", type=float, default=30.0, help="read rate")
    parser.add_argument("--mock", action="store_true", help="use a simulated arm")
    args = parser.parse_args()

    if not args.mock and not args.port:
        parser.error("--port is required unless --mock is given")

    rr.init("so101_hello_read", spawn=True)

    # Read-only: no need to calibrate just to read raw positions.
    arm = make_arm(mock=args.mock, port=args.port, arm_id=args.id, calibrate=False)
    label = "MOCK arm" if args.mock else f"{args.id} on {args.port}"
    print(f"Connecting to {label} ...")
    arm.connect()
    print("Connected. Reading positions. Ctrl+C to stop.\n")

    period = 1.0 / args.hz
    try:
        while True:
            joints = arm.read_joints()
            for name, value in joints.items():
                rr.log(f"joints/{name}", rr.Scalars(value))
            print("\r" + format_joint_line(joints), end="", flush=True)
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        arm.disconnect()
        print("Disconnected.")


if __name__ == "__main__":
    main()
