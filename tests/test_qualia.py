"""Qualia wrapper logic -- fully mocked, so NO network and NO credits spent.

We never call the real SDK; `get_client` is monkeypatched. This verifies the glue
auto_record relies on (argument passing, dict mapping, token handling).
"""

import types

import so101.qualia_client as qc


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
    out = qc.launch_finetune(dataset_id="me/ds", vla_type="act", model_id=None, hours=1.5)
    assert out == {"job_id": "job-456", "project_id": "proj-123", "status": "submitted"}
    assert fake.finetune_kwargs["dataset_id"] == "me/ds"
    assert fake.finetune_kwargs["vla_type"] == "act"
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
