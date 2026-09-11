from __future__ import annotations

from operations.paper_aio_ddsb_source_watch_recovery import source_state_decision


def _state(status: str) -> dict:
    return {
        "schema": "final-unsb-ddsb-source-watch-state-v1",
        "status": status,
        "training_authorized": False,
        "training_started": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
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
