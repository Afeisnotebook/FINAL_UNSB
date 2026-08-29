import pytest

from operations.local_route1_handoff_autocomplete import next_action


def test_handoff_autocomplete_waits_for_both_pause_gate_and_source():
    assert next_action(record_status=None, receipt_status=None, ready=False) == "WAIT_RECORD"
    assert next_action(record_status="ARMED", receipt_status=None, ready=True) == "WAIT_RECORD"
    assert next_action(
        record_status="EXECUTOR_PAUSED_FINAL_HJ_CHILD_RUNNING",
        receipt_status=None,
        ready=True,
    ) == "WAIT_RECORD"
    assert next_action(
        record_status="READY_FOR_INDEPENDENT_HNEK_IMPORT",
        receipt_status=None,
        ready=False,
    ) == "WAIT_SOURCE"
    assert next_action(
        record_status="READY_FOR_INDEPENDENT_HNEK_IMPORT",
        receipt_status=None,
        ready=True,
    ) == "IMPORT_AND_RESUME"


def test_handoff_autocomplete_recovers_import_before_resume_and_is_idempotent():
    assert next_action(
        record_status="READY_FOR_INDEPENDENT_HNEK_IMPORT",
        receipt_status="IMPORT_VERIFIED",
        ready=True,
    ) == "RESUME_ONLY"
    assert next_action(
        record_status="EXECUTOR_RESUMED_AFTER_VERIFIED_IMPORT",
        receipt_status="IMPORT_VERIFIED",
        ready=True,
    ) == "COMPLETE"


def test_handoff_autocomplete_rejects_incoherent_or_unknown_records():
    with pytest.raises(RuntimeError, match="without a resumable"):
        next_action(record_status="ARMED", receipt_status="IMPORT_VERIFIED", ready=True)
    with pytest.raises(RuntimeError, match="unexpected handoff status"):
        next_action(record_status="FAILED", receipt_status=None, ready=False)
