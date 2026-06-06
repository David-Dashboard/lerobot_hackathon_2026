"""Qualia wrapper + server routes -- fully mocked, so NO network and NO credits spent.

We never call the real SDK here; `get_client` and the wrapper functions are
monkeypatched. This verifies our glue (argument passing, dict mapping, graceful
"not configured" handling, UI serving) without touching Qualia.
"""

import types

from fastapi.testclient import TestClient

import so101.qualia_client as qc
from so101.mock import MockArm
from so101.server import create_app


# --------------------------------------------------------------------------
# wrapper logic (against a fake SDK client)
# --------------------------------------------------------------------------
class _FakeClient:
    def __init__(self):
        self.finetune_kwargs = None
        outer = self

        class Projects:
            def create(self, name, description=None):
                outer.project_name = name
                return types.SimpleNamespace(project_id="proj-123")

        class Finetune:
            def create(self, **kw):
                outer.finetune_kwargs = kw
                return types.SimpleNamespace(job_id="job-456", status="submitted")

            def get(self, job_id):
                return types.SimpleNamespace(status="running", current_phase="train")

        class Credits:
            def get(self):
                return types.SimpleNamespace(balance=400)

        self.projects = Projects()
        self.finetune = Finetune()
        self.credits = Credits()


def test_load_token_reads_env(monkeypatch):
    monkeypatch.setenv("QUALIA_TOKEN", "secret-123")
    assert qc.load_token() == "secret-123"


def test_launch_finetune_passes_args(monkeypatch):
    fake = _FakeClient()
    monkeypatch.setattr(qc, "get_client", lambda: fake)
    out = qc.launch_finetune(dataset_id="me/ds", vla_type="smolvla", hours=1.5)
    assert out == {"job_id": "job-456", "project_id": "proj-123", "status": "submitted"}
    assert fake.finetune_kwargs["dataset_id"] == "me/ds"
    assert fake.finetune_kwargs["vla_type"] == "smolvla"
    assert fake.finetune_kwargs["hours"] == 1.5
    assert fake.finetune_kwargs["project_id"] == "proj-123"


def test_job_status_maps_fields(monkeypatch):
    monkeypatch.setattr(qc, "get_client", lambda: _FakeClient())
    s = qc.job_status("job-456")
    assert s["status"] == "running"
    assert s["current_phase"] == "train"


def test_credits_maps(monkeypatch):
    monkeypatch.setattr(qc, "get_client", lambda: _FakeClient())
    assert qc.credits()["balance"] == 400


# --------------------------------------------------------------------------
# server routes (mock the wrapper functions; no SDK at all)
# --------------------------------------------------------------------------
def _client(monkeypatch, **patches):
    for name, fn in patches.items():
        monkeypatch.setattr(qc, name, fn)
    arm = MockArm()
    arm.connect()
    return TestClient(create_app(arm))


def test_route_credits_ok(monkeypatch):
    c = _client(monkeypatch, credits=lambda: {"balance": 400})
    body = c.get("/qualia/credits").json()
    assert body["ok"] is True
    assert body["data"]["balance"] == 400


def test_route_credits_not_configured(monkeypatch):
    def boom():
        raise qc.QualiaNotConfigured("no token")

    c = _client(monkeypatch, credits=boom)
    body = c.get("/qualia/credits").json()
    assert body["ok"] is False
    assert body["configured"] is False


def test_route_launch_finetune(monkeypatch):
    captured = {}

    def fake_launch(**kw):
        captured.update(kw)
        return {"job_id": "j1", "project_id": "p1", "status": "submitted"}

    c = _client(monkeypatch, launch_finetune=fake_launch)
    r = c.post("/qualia/finetune", json={"dataset_id": "me/ds", "vla_type": "act", "hours": 1.0})
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["job_id"] == "j1"
    assert captured["dataset_id"] == "me/ds"
    assert captured["vla_type"] == "act"


def test_dashboard_is_served(monkeypatch):
    c = _client(monkeypatch)
    r = c.get("/")
    assert r.status_code == 200
    assert "SO-101 Control" in r.text
