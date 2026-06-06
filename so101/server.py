"""HTTP API + dashboard UI for the SO-101.

`create_app(arm)` wires:
  - arm routes:    GET /health, GET /joints, POST /joints   (works on any RobotArm)
  - qualia routes: GET /qualia/{credits,models,jobs}, POST /qualia/finetune, ...
  - the UI:        GET /  -> static dashboard

Qualia routes degrade gracefully (return ok:false + message) when no token /
network, so the dashboard always loads. The arm and Qualia concerns stay
independent -- neither imports the other.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .interface import RobotArm

STATIC_DIR = Path(__file__).parent / "static"


class JointsBody(BaseModel):
    positions: dict[str, float]


class FinetuneBody(BaseModel):
    dataset_id: str
    vla_type: str = "smolvla"
    hours: float = 2.0
    model_id: str | None = "lerobot/smolvla_base"
    camera_mappings: dict[str, str] | None = None
    project_name: str = "SO-101 Hackathon"
    batch_size: int = 32


def create_app(arm: RobotArm, *, enable_qualia: bool = True) -> FastAPI:
    app = FastAPI(title="SO-101 Remote", version="0.2.0")

    # --- arm ---------------------------------------------------------------
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

    # --- qualia ------------------------------------------------------------
    if enable_qualia:
        _add_qualia_routes(app)

    # --- ui ----------------------------------------------------------------
    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC_DIR / "index.html")

    return app


def _add_qualia_routes(app: FastAPI) -> None:
    from . import qualia_client as q

    def safe(fn):
        """Run a Qualia call, turning any failure into ok:false JSON (UI stays alive)."""
        try:
            return {"ok": True, "data": fn()}
        except q.QualiaNotConfigured as e:
            return JSONResponse(
                {"ok": False, "configured": False, "error": str(e)}, status_code=200
            )
        except Exception as e:  # noqa: BLE001 - surface message to the dashboard
            return JSONResponse(
                {"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=200
            )

    @app.get("/qualia/credits")
    def qualia_credits():
        return safe(q.credits)

    @app.get("/qualia/models")
    def qualia_models():
        return safe(q.list_models)

    @app.get("/qualia/jobs")
    def qualia_jobs():
        return safe(lambda: q.list_jobs(limit=20))

    @app.get("/qualia/finetune/{job_id}")
    def qualia_job_status(job_id: str):
        return safe(lambda: q.job_status(job_id))

    @app.post("/qualia/finetune")
    def qualia_launch(body: FinetuneBody):
        return safe(lambda: q.launch_finetune(**body.model_dump()))
