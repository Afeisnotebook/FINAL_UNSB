import pytest

from operations.paper_aio_incremental_audit_export import (
    AUDIT_EPOCHS,
    export_set,
    validate_existing_receipt,
)


def _contract() -> dict:
    return {
        "lane_id": "plain",
        "source_host_label": "hostA",
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "required_manifest_sha256": "c" * 64,
        "audit_epochs": list(AUDIT_EPOCHS),
    }


def _receipt(epoch: int) -> dict:
    return {
        "schema": "final-unsb-paper-checkpoint-export-v1",
        "status": "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT",
        "lane_id": "plain",
        "epoch": epoch,
        "updates": epoch * 8553,
        "source_host_label": "hostA",
        "training_git_commit": "a" * 40,
        "training_protocol_fingerprint": "b" * 64,
        "manifest_sha256": "c" * 64,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_incremental_export_set_is_partial_until_all_fixed_epochs() -> None:
    contract = _contract()
    one = [{"epoch": 100}]
    partial = export_set(contract, one)
    assert partial["status"] == "PARTIAL_INCREMENTAL_AUDIT_EXPORT_SET"
    assert partial["available_epochs"] == [100]
    complete = export_set(
        contract, [{"epoch": epoch} for epoch in AUDIT_EPOCHS],
    )
    assert complete["status"] == "COMPLETE_INCREMENTAL_AUDIT_EXPORT_SET"
    assert complete["available_epochs"] == list(AUDIT_EPOCHS)


def test_existing_incremental_receipt_binds_training_identity() -> None:
    validate_existing_receipt(_receipt(100), contract=_contract(), epoch=100)
    wrong = _receipt(100)
    wrong["training_protocol_fingerprint"] = "d" * 64
    with pytest.raises(RuntimeError, match="differs"):
        validate_existing_receipt(wrong, contract=_contract(), epoch=100)
    wrong = _receipt(100)
    wrong["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="differs"):
        validate_existing_receipt(wrong, contract=_contract(), epoch=100)
