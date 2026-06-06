"""A tiny HTTP API to read/command an arm over the network.

`create_app(arm)` wires routes onto any `RobotArm` -- mock or real -- without
caring which. The caller owns the arm's connect/disconnect lifecycle (see serve.py).
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from .interface import RobotArm


class JointsBody(BaseModel):
    positions: dict[str, float]


def create_app(arm: RobotArm) -> FastAPI:
    app = FastAPI(title="SO-101 Remote", version="0.1.0")

    @app.get("/health")
    def health() -> dict:
        return {"connected": arm.is_connected}

    @app.get("/joints")
    def get_joints() -> dict:
        return {"positions": arm.read_joints()}

    @app.post("/joints")
    def set_joints(body: JointsBody) -> dict:
        arm.write_joints(body.positions)
        return {"ok": True, "positions": body.positions}

    return app
