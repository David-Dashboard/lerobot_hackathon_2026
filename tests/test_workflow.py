"""End-to-end click-through THROUGH THE SERVER, no hardware/network/credits.

Mirrors what a teammate does in the dashboard:
  start UI -> record data -> dataset created -> launch finetune -> deploy.

record_dataset and Qualia are mocked at the edges (no disk/network/credits);
the deploy loop runs for real against the mock arm.
"""

import time

from fastapi.testclient import TestClient

import so101.qualia_client as qc
import so101.record as rec
from so101 import SO101_JOINTS, MockArm
from so101.server import create_app


def _poll_job_until_idle(client, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get("/job").json()
        if not j["running"]:
            return j
        time.sleep(0.02)
    raise AssertionError("job did not finish in time")


def test_full_workflow(monkeypatch):
    arm = MockArm()
    arm.connect()

    # mock the heavy edges
    def fake_record(arm, repo_id, task, *, num_episodes, episode_steps, fps, root, push_to_hub, progress):
        progress(num_episodes * episode_steps, num_episodes * episode_steps)
        return {"repo_id": repo_id, "num_episodes": num_episodes,
                "num_frames": num_episodes * episode_steps, "root": "x", "pushed": push_to_hub}

    captured = {}

    def fake_launch(**kw):
        captured.update(kw)
        return {"job_id": "job-xyz", "project_id": "p1", "status": "submitted"}

    monkeypatch.setattr(rec, "record_dataset", fake_record)
    monkeypatch.setattr(qc, "launch_finetune", fake_launch)

    client = TestClient(create_app(arm))

    # 1. server up + dashboard served
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["connected"] is True

    # 2. gather data -> dataset
    assert client.post("/record", json={"repo_id": "me/demo", "num_episodes": 2, "episode_steps": 5}).json()["ok"]
    job = _poll_job_until_idle(client)
    assert job["summary"]["repo_id"] == "me/demo"
    assert job["summary"]["num_frames"] == 10

    # 3. finetune on the recorded dataset (Qualia)
    r = client.post("/qualia/finetune", json={"dataset_id": "me/demo", "vla_type": "smolvla", "hours": 1.0}).json()
    assert r["ok"] and r["data"]["job_id"] == "job-xyz"
    assert captured["dataset_id"] == "me/demo"

    # 4. deploy a policy on the arm
    assert client.post("/deploy", json={"policy": "hold", "steps": 4, "fps": 1000}).json()["ok"]
    _poll_job_until_idle(client)
    assert set(arm.last_command) == set(SO101_JOINTS)
