from operations.paper_aio_local_evaluation_node_successor import mirror_decision


def _state(status: str) -> dict:
    return {
        "schema": "final-unsb-paper-local-data-mirror-state-v1",
        "status": status,
        "source_mutated": False,
        "confirmation20_evaluated": False,
    }


def test_local_mirror_successor_waits_then_allows_only_manifest_gate():
    assert mirror_decision(_state("TRANSFERRING_IDLE_IO_PRIORITY")) == "WAITING_FOR_MIRROR"
    assert (
        mirror_decision(_state("MIRROR_COMPLETE_AWAITING_MANIFEST_HASH_GATE"))
        == "READY_FOR_MANIFEST_GATE"
    )


def test_local_mirror_successor_fail_closes_failed_or_boundary_crossed_state():
    assert mirror_decision(_state("MIRROR_FAILED_REVIEW_REQUIRED")) == "BLOCKED_MIRROR_FAILED"
    crossed = _state("MIRROR_COMPLETE_AWAITING_MANIFEST_HASH_GATE")
    crossed["confirmation20_evaluated"] = True
    try:
        mirror_decision(crossed)
    except RuntimeError as error:
        assert "confirmation" in str(error)
    else:
        raise AssertionError("confirmation boundary must fail closed")
