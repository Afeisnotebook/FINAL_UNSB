from __future__ import annotations

import pytest

import operations.paper_aio_ddsb_source_watch_recovery as recovery
from operations.paper_aio_ddsb_source_watch_recovery import (
    initial_source_pid,
    source_state_decision,
    tracked_child_pid,
)


def _state(status: str) -> dict:
    return {
        "schema": "final-unsb-ddsb-source-watch-state-v1",
        "status": status,
        "training_authorized": False,
        "training_started": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "watcher_pid": 4242,
    }


def test_waiting_and_transient_network_states_are_recoverable() -> None:
    assert source_state_decision(_state("WAITING_FOR_AUTHORITATIVE_SOURCE")) == "WAIT"
    assert source_state_decision(_state("TRANSIENT_NETWORK_ERROR")) == "WAIT"


def test_repository_candidates_stop_for_manual_review() -> None:
    assert (
        source_state_decision(_state("AUTHORITATIVE_SOURCE_CANDIDATE_REVIEW_REQUIRED"))
        == "REVIEW"
    )
    assert (
        source_state_decision(_state("UNVERIFIED_SOURCE_CANDIDATE_REVIEW_REQUIRED"))
        == "REVIEW"
    )


def test_timeout_is_terminal_without_becoming_scientific_failure() -> None:
    assert source_state_decision(_state("TIMED_OUT_WITHOUT_AUTHORITY_DECISION")) == "TIMEOUT"


def test_boundary_violation_fails_closed() -> None:
    assert source_state_decision({**_state("WAITING_FOR_AUTHORITATIVE_SOURCE"), "training_authorized": True}) == "BLOCK"
    assert source_state_decision({**_state("WAITING_FOR_AUTHORITATIVE_SOURCE"), "training_started": True}) == "BLOCK"
    assert source_state_decision({**_state("WAITING_FOR_AUTHORITATIVE_SOURCE"), "paired_metric_control": True}) == "BLOCK"
    assert source_state_decision({**_state("WAITING_FOR_AUTHORITATIVE_SOURCE"), "confirmation20_opened": True}) == "BLOCK"
    assert source_state_decision(_state("FATAL")) == "BLOCK"


def test_new_contract_requires_exact_live_initial_child(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 4242)
    assert initial_source_pid(
        _state("WAITING_FOR_AUTHORITATIVE_SOURCE"),
        contract_created=True,
        required_initial_pid=4242,
    ) == 4242
    with pytest.raises(RuntimeError, match="uniquely adoptable"):
        initial_source_pid(
            _state("WAITING_FOR_AUTHORITATIVE_SOURCE"),
            contract_created=True,
            required_initial_pid=5151,
        )


def test_existing_contract_can_recover_after_child_pid_changes_or_dies(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: False)
    assert initial_source_pid(
        {**_state("WAITING_FOR_AUTHORITATIVE_SOURCE"), "watcher_pid": 5151},
        contract_created=False,
        required_initial_pid=4242,
    ) == 5151


def test_just_launched_child_is_tracked_until_it_publishes_state(monkeypatch) -> None:
    monkeypatch.setattr(recovery, "_pid_alive", lambda pid: pid == 5151)
    assert tracked_child_pid(state_pid=4242, launched_pid=5151) == (5151, True)
    assert tracked_child_pid(state_pid=5151, launched_pid=5151) == (5151, False)
