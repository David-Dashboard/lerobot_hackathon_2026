"""Thin wrapper over the Qualia SDK for VLA finetuning.

Returns plain JSON-serializable dicts so the server/UI never depend on Qualia's
model types. The SDK + token are loaded lazily, so importing this module (and the
server) does not require Qualia to be configured.

Token: read from QUALIA_TOKEN (this repo's .env) or QUALIA_API_KEY.
"""

from __future__ import annotations

import os
from typing import Any


class QualiaNotConfigured(RuntimeError):
    """Raised when no Qualia API token is available."""


def load_token() -> str:
    try:
        from dotenv import load_dotenv

        load_dotenv()  # harmless if there's no .env
    except ImportError:
        pass
    token = os.getenv("QUALIA_TOKEN") or os.getenv("QUALIA_API_KEY")
    if not token:
        raise QualiaNotConfigured(
            "No Qualia token. Set QUALIA_TOKEN (or QUALIA_API_KEY) in .env"
        )
    return token


def get_client():
    """Construct a Qualia client (lazy import; raises QualiaNotConfigured if no token)."""
    from qualia import Qualia

    return Qualia(api_key=load_token())


# --- serialization helpers ------------------------------------------------
def _to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):  # pydantic v2
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):  # pydantic v1
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    return {"value": str(obj)}


def _page_items(page: Any) -> list:
    for attr in ("items", "data", "results"):
        if hasattr(page, attr):
            return list(getattr(page, attr))
    try:
        return list(page)
    except TypeError:
        return [page]


# --- read-only (safe; no credits spent) -----------------------------------
def credits() -> dict:
    return _to_dict(get_client().credits.get())


def list_models() -> list[dict]:
    return [_to_dict(m) for m in get_client().models.list()]


def list_jobs(limit: int = 20) -> list[dict]:
    return [_to_dict(j) for j in _page_items(get_client().jobs.list(limit=limit))]


def job_status(job_id: str) -> dict:
    return _to_dict(get_client().finetune.get(job_id))


# --- launches a real job (spends credits) ---------------------------------
def launch_finetune(
    *,
    dataset_id: str,
    vla_type: str = "smolvla",
    hours: float = 2.0,
    model_id: str | None = "lerobot/smolvla_base",
    camera_mappings: dict[str, str] | None = None,
    project_name: str = "SO-101 Hackathon",
    batch_size: int = 32,
) -> dict:
    """Create a project + launch a finetune job. Returns {job_id, project_id, status}."""
    client = get_client()
    project = client.projects.create(name=project_name)
    project_id = getattr(project, "project_id", None) or getattr(project, "id")
    job = client.finetune.create(
        project_id=project_id,
        vla_type=vla_type,
        dataset_id=dataset_id,
        hours=hours,
        model_id=model_id,
        camera_mappings=camera_mappings,
        batch_size=batch_size,
    )
    job_id = getattr(job, "job_id", None) or getattr(job, "id")
    return {
        "job_id": str(job_id),
        "project_id": str(project_id),
        "status": _to_dict(job).get("status", "submitted"),
    }
