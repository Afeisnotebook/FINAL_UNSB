from __future__ import annotations

from operations.paper_aio_control_supervisor import child_state_decision
from operations.paper_aio_endpoint_reachability_watch import (
    FINAL_STATUS,
    STATE_SCHEMA,
    make_state,
    probe_tcp,
)


class _Connection:
    def __init__(self) -> None:
        self.closed = False

    def getpeername(self) -> tuple[str, int]:
        return ("203.0.113.9", 44804)

    def close(self) -> None:
        self.closed = True


def _reachable_connector(*_args: object, **_kwargs: object) -> _Connection:
    return _Connection()


def _timeout_connector(*_args: object, **_kwargs: object) -> _Connection:
    raise TimeoutError("not reachable")


def test_reachable_endpoint_requires_identity_review_not_training() -> None:
    result = probe_tcp("example.invalid", 44804, 0.1, connector=_reachable_connector)
    state = make_state(
        host="example.invalid",
        port=44804,
        attempt=3,
        result=result,
        started_unix_time=1.0,
        observed_unix_time=2.0,
    )
    assert state["status"] == FINAL_STATUS
    assert state["authentication_attempted"] is False
    assert state["gpu_identity_collected"] is False
    assert state["long_training_launched"] is False
    assert state["credentials_recorded"] is False
    assert child_state_decision("endpoint_reachability", state) == "COMPLETE"


def test_unreachable_endpoint_remains_a_safe_wait() -> None:
    result = probe_tcp("example.invalid", 44804, 0.1, connector=_timeout_connector)
    state = make_state(
        host="example.invalid",
        port=44804,
        attempt=1,
        result=result,
        started_unix_time=1.0,
        observed_unix_time=2.0,
    )
    assert state["schema"] == STATE_SCHEMA
    assert state["status"] == "WAITING_FOR_TCP_REACHABILITY"
    assert state["reachable"] is False
    assert state["performance_values_read"] is False
    assert state["paired_metric_control"] is False
    assert state["confirmation20_opened"] is False
    assert child_state_decision("endpoint_reachability", state) == "WAIT"
