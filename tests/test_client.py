"""End-to-end client <-> server test over REAL HTTP, no hardware.

Spins up the FastAPI server (mock arm) in a background uvicorn thread on a free
port, then drives it with ArmClient. Proves the remote-control round trip works.
"""

import socket
import threading
import time

import pytest
import uvicorn

from so101 import SO101_JOINTS
from so101.client import ArmClient
from so101.mock import MockArm
from so101.server import create_app


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def live_server():
    arm = MockArm()
    arm.connect()
    port = _free_port()
    config = uvicorn.Config(create_app(arm), host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    # wait until the server reports it's accepting connections
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    assert server.started, "uvicorn did not start in time"
    try:
        yield f"http://127.0.0.1:{port}", arm
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_client_reports_connected(live_server):
    base_url, _ = live_server
    with ArmClient(base_url) as client:
        assert client.is_connected is True


def test_client_reads_joints(live_server):
    base_url, _ = live_server
    with ArmClient(base_url) as client:
        assert set(client.read_joints()) == set(SO101_JOINTS)


def test_client_writes_joints(live_server):
    base_url, arm = live_server
    cmd = {j: 1.0 for j in SO101_JOINTS}
    with ArmClient(base_url) as client:
        client.write_joints(cmd)
    assert arm.last_command == cmd


def test_client_offline_is_not_connected():
    # Nothing listening -> is_connected swallows the error and returns False.
    with ArmClient("http://127.0.0.1:1", timeout=1.0) as client:
        assert client.is_connected is False
