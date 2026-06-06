"""HTTP server tests using FastAPI's in-process TestClient + the mock arm.

No network, no hardware -- exercises the exact API a remote client will hit.
"""

from fastapi.testclient import TestClient

from so101 import SO101_JOINTS
from so101.mock import MockArm
from so101.server import create_app


def _client_and_arm():
    arm = MockArm()
    arm.connect()
    return TestClient(create_app(arm)), arm


def test_health():
    client, _ = _client_and_arm()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"connected": True}


def test_get_joints():
    client, _ = _client_and_arm()
    r = client.get("/joints")
    assert r.status_code == 200
    assert set(r.json()["positions"]) == set(SO101_JOINTS)


def test_post_joints_commands_arm():
    client, arm = _client_and_arm()
    body = {"positions": {j: 0.0 for j in SO101_JOINTS}}
    r = client.post("/joints", json=body)
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert arm.last_command == body["positions"]


def test_post_joints_rejects_bad_shape():
    client, _ = _client_and_arm()
    r = client.post("/joints", json={"nope": 1})
    assert r.status_code == 422  # pydantic validation
