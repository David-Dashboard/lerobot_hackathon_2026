"""connect_with_retry + graceful_stop — no hardware."""

import pytest

from so101.reliability import connect_with_retry, graceful_stop


class FakeDevice:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0
        self.disconnects = 0
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ConnectionError("Failed to write 'Torque_Enable' ... There is no status packet!")
        self._connected = True

    def disconnect(self) -> None:
        self.disconnects += 1
        self._connected = False


def test_retry_then_succeed():
    d = FakeDevice(fail_times=2)
    connect_with_retry(d, "arm", attempts=3, delay=0)
    assert d.is_connected
    assert d.calls == 3
    assert d.disconnects == 2  # cleared half-open state between the 2 failures


def test_aborts_after_max_attempts():
    d = FakeDevice(fail_times=5)
    with pytest.raises(RuntimeError):
        connect_with_retry(d, "arm", attempts=2, delay=0)
    assert not d.is_connected


def test_already_connected_is_noop():
    d = FakeDevice(fail_times=0)
    d._connected = True
    connect_with_retry(d, "arm", attempts=1, delay=0)
    assert d.calls == 0  # never re-connects


def test_graceful_stop_predicate_and_handler_restore():
    import signal

    before = signal.getsignal(signal.SIGINT)
    with graceful_stop() as should_stop:
        assert should_stop() is False
    assert signal.getsignal(signal.SIGINT) is before  # handler restored on exit
