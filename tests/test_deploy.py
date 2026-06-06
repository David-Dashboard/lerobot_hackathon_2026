"""Deploy loop + policies + the background job runner (no hardware)."""

import pytest

from so101 import SO101_JOINTS, MockArm
from so101.deploy import HoldPolicy, LeRobotPolicy, SinePolicy, load_policy, run_policy
from so101.jobs import BackgroundJob


def test_load_policy_resolves_stubs_and_paths():
    assert isinstance(load_policy("sine"), SinePolicy)
    assert isinstance(load_policy("hold"), HoldPolicy)
    assert isinstance(load_policy("davidbaofu/act_so101"), LeRobotPolicy)


def test_run_policy_commands_arm():
    arm = MockArm()
    arm.connect()
    result = run_policy(arm, HoldPolicy(), steps=5, fps=1000)
    assert result["steps"] == 5
    assert set(arm.last_command) == set(SO101_JOINTS)  # arm was commanded


def test_run_policy_honors_should_stop():
    arm = MockArm()
    arm.connect()
    result = run_policy(arm, SinePolicy(), steps=100, fps=1000, should_stop=lambda: True)
    assert result["steps"] == 0  # stopped before first command


def test_run_policy_reports_progress():
    arm = MockArm()
    arm.connect()
    steps_seen = []
    run_policy(arm, HoldPolicy(), steps=3, fps=1000, on_step=lambda d, t: steps_seen.append((d, t)))
    assert steps_seen == [(1, 3), (2, 3), (3, 3)]


# --- background job runner -------------------------------------------------
def _wait(job, timeout=3.0):
    job.join(timeout)


def test_job_runs_and_reports():
    job = BackgroundJob()
    job.start("demo", lambda j: j.__setattr__("info", {"done": 1}))
    _wait(job)
    assert job.running is False
    assert job.status()["done"] == 1


def test_job_captures_error():
    job = BackgroundJob()

    def boom(j):
        raise ValueError("nope")

    job.start("demo", boom)
    _wait(job)
    assert "ValueError" in (job.error or "")


def test_job_rejects_concurrent_start():
    import time

    job = BackgroundJob()
    job.start("a", lambda j: time.sleep(0.2))
    with pytest.raises(RuntimeError):
        job.start("b", lambda j: None)
    _wait(job)
