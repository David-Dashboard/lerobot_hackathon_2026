"""HTTP API + dashboard for the SO-101 -- the whole workflow in one server.

`create_app(arm)` wires four concerns, all decoupled (none import each other):
  - arm      : GET /health, GET /joints, POST /joints
  - record   : POST /record  (gather data -> LeRobotDataset)         [background]
  - deploy   : POST /deploy, POST /job/stop  (run a policy on the arm)[background]
  - job      : GET /job  (status of the running record/deploy)
  - qualia   : GET /qualia/{credits,models,jobs}, POST /qualia/finetune, status
  - ui       : GET /  -> dashboard

record + deploy share one background slot (one arm => one activity at a time).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .interface import RobotArm
from .jobs import BackgroundJob

STATIC_DIR = Path(__file__).parent / "static"
RECORD_ROOT = Path("recorded")


class JointsBody(BaseModel):
    positions: dict[str, float]


class RecordBody(BaseModel):
    repo_id: str = "local/so101_demo"
    task: str = "pick up the cube and place it in the bowl"
    num_episodes: int = 2
    episode_steps: int = 30
    fps: int = 30
    push_to_hub: bool = False


class DeployBody(BaseModel):
    policy: str = "sine"   # 'sine'/'hold' or a trained model path
    steps: int = 300
    fps: int = 30


class FinetuneBody(BaseModel):
    dataset_id: str
    vla_type: str = "smolvla"
    hours: float = 2.0
    model_id: str | None = "lerobot/smolvla_base"
    camera_mappings: dict[str, str] | None = None
    project_name: str = "SO-101 Hackathon"
    batch_size: int = 32


def create_app(arm: RobotArm, *, enable_qualia: bool = True) -> FastAPI:
    app = FastAPI(title="SO-101 Control", version="0.3.0")
    job = BackgroundJob()

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

    # --- record (gather data -> dataset) -----------------------------------
    @app.post("/record")
    def record(body: RecordBody) -> dict:
        from .record import record_dataset

        root = RECORD_ROOT / body.repo_id.replace("/", "__")

        def target(j: BackgroundJob):
            shutil.rmtree(root, ignore_errors=True)  # allow re-record
            j.info = {"done": 0, "total": body.num_episodes * body.episode_steps}

            def progress(done, total):
                j.info = {"done": done, "total": total}

            summary = record_dataset(
                arm,
                repo_id=body.repo_id,
                task=body.task,
                num_episodes=body.num_episodes,
                episode_steps=body.episode_steps,
                fps=body.fps,
                root=root,
                push_to_hub=body.push_to_hub,
                progress=progress,
            )
            j.info = {**j.info, "summary": summary}

        try:
            job.start("record", target)
            return {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}

    # --- deploy (run a policy on the arm) ----------------------------------
    @app.post("/deploy")
    def deploy(body: DeployBody) -> dict:
        from .deploy import load_policy, run_policy

        policy = load_policy(body.policy)

        def target(j: BackgroundJob):
            def on_step(done, total):
                j.info = {"step": done, "total": total}

            run_policy(
                arm, policy, steps=body.steps, fps=body.fps,
                should_stop=lambda: j.should_stop, on_step=on_step,
            )

        try:
            job.start("deploy", target)
            return {"ok": True}
        except RuntimeError as e:
            return {"ok": False, "error": str(e)}

    @app.get("/job")
    def job_status() -> dict:
        return job.status()

    @app.post("/job/stop")
    def job_stop() -> dict:
        job.request_stop()
        return {"ok": True}

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
        try:
            return {"ok": True, "data": fn()}
        except q.QualiaNotConfigured as e:
            return JSONResponse({"ok": False, "configured": False, "error": str(e)}, status_code=200)
        except Exception as e:  # noqa: BLE001 - surface to dashboard
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=200)

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
