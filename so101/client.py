"""Drive a remote arm from another machine.

    from so101.client import ArmClient
    arm = ArmClient("http://ROBOT_HOST:8000")
    print(arm.read_joints())
    arm.write_joints({"gripper": 10.0})

Note: this satisfies the same read/write shape as `RobotArm`, so code written
against the interface can talk to a local arm or a remote one interchangeably.
"""

from __future__ import annotations

import httpx


class ArmClient:
    def __init__(self, base_url: str, timeout: float = 5.0, transport=None):
        # `transport` lets tests point the client straight at the ASGI app (no network).
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"), timeout=timeout, transport=transport
        )

    def health(self) -> dict:
        return self._client.get("/health").json()

    @property
    def is_connected(self) -> bool:
        try:
            return bool(self.health().get("connected"))
        except httpx.HTTPError:
            return False

    def read_joints(self) -> dict[str, float]:
        r = self._client.get("/joints")
        r.raise_for_status()
        return r.json()["positions"]

    def write_joints(self, positions: dict[str, float]) -> None:
        r = self._client.post("/joints", json={"positions": positions})
        r.raise_for_status()

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "ArmClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
