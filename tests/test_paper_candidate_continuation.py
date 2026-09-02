from operations.paper_aio_candidate_continuation import (
    ACTIVATION_READY,
    continuation_decision,
)


def test_candidate_continuation_waits_for_e1_probe() -> None:
    assert continuation_decision(
        activation_status="RUNNING_RUNTIME_GATE",
        parent_status="CHILD_RUNNING", mode="co_resident",
    ) == "WAIT_ACTIVATION"


def test_candidate_co_resident_mode_accepts_healthy_parent() -> None:
    assert continuation_decision(
        activation_status=ACTIVATION_READY,
        parent_status="CHILD_RUNNING", mode="co_resident",
    ) == "START"


def test_candidate_after_parent_mode_is_metric_blind() -> None:
    assert continuation_decision(
        activation_status=ACTIVATION_READY,
        parent_status="CHILD_RUNNING", mode="after_parent",
    ) == "WAIT_PARENT"
    assert continuation_decision(
        activation_status=ACTIVATION_READY,
        parent_status="COMPLETE_E200", mode="after_parent",
    ) == "START"


def test_candidate_continuation_fails_closed() -> None:
    assert continuation_decision(
        activation_status="BLOCKED_ACTIVATION_GATE_FAILURE",
        parent_status="CHILD_RUNNING", mode="co_resident",
    ) == "BLOCK_ACTIVATION"
    assert continuation_decision(
        activation_status=ACTIVATION_READY,
        parent_status="BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE",
        mode="after_parent",
    ) == "BLOCK_PARENT"
