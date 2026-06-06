"""
Run the SO-101 remote-control server -- "talk to the arm over the network".

No hardware:
    .\.venv\Scripts\python.exe serve.py --mock

Real arm:
    .\.venv\Scripts\python.exe serve.py --port COM3 --id my_follower

Then, from this or another machine:
    curl http://HOST:8000/health
    curl http://HOST:8000/joints
    curl -X POST http://HOST:8000/joints -H "content-type: application/json" \
         -d '{"positions": {"gripper": 10.0}}'

Or in Python:
    from so101.client import ArmClient
    ArmClient("http://HOST:8000").read_joints()
"""

import argparse

import uvicorn

from so101.factory import make_arm
from so101.server import create_app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", help="arm COM port (omit with --mock)")
    parser.add_argument("--id", default="follower", help="arm id / name")
    parser.add_argument("--mock", action="store_true", help="serve a simulated arm")
    parser.add_argument("--host", default="0.0.0.0", help="HTTP bind host")
    parser.add_argument("--http-port", type=int, default=8000, help="HTTP port")
    parser.add_argument(
        "--no-calibrate", action="store_true", help="skip arm calibration on connect"
    )
    args = parser.parse_args()

    if not args.mock and not args.port:
        parser.error("--port is required unless --mock is given")

    arm = make_arm(
        mock=args.mock, port=args.port, arm_id=args.id, calibrate=not args.no_calibrate
    )
    print(f"Connecting to {'MOCK' if args.mock else args.port} ...")
    arm.connect()
    app = create_app(arm)
    print(f"Serving on http://{args.host}:{args.http_port}  (Ctrl+C to stop)")
    try:
        uvicorn.run(app, host=args.host, port=args.http_port, log_level="info")
    finally:
        arm.disconnect()
        print("Arm disconnected.")


if __name__ == "__main__":
    main()
