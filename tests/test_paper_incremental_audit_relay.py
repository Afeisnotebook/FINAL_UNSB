import json
from types import SimpleNamespace

import pytest

from operations.paper_aio_incremental_audit_relay import (
    AUDIT_EPOCHS,
    IncrementalImportNotReady,
    _contract,
    incremental_import_lane_path,
    validate_incremental_import_lane,
    validate_source_set,
)
from operations.paper_aio_local_terminal_audit_successor import _ready_rows
from research.paper_aio.protocol import file_sha256


def _source_set(epochs=(100,)) -> dict:
    complete = tuple(epochs) == AUDIT_EPOCHS
    return {
        "schema": "final-unsb-paper-incremental-audit-export-set-v1",
        "status": (
            "COMPLETE_INCREMENTAL_AUDIT_EXPORT_SET"
            if complete else "PARTIAL_INCREMENTAL_AUDIT_EXPORT_SET"
        ),
        "lane_id": "plain",
        "source_host_label": "hostA",
        "required_epochs": list(AUDIT_EPOCHS),
        "available_epochs": list(epochs),
        "exports": [
            {
                "epoch": epoch,
                "receipt": f"/exports/plain/e{epoch:03d}.export.json",
                "receipt_sha256": "1" * 64,
            }
            for epoch in epochs
        ],
        "checkpoint_copy_performed": False,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_partial_source_set_accepts_only_ordered_audit_epoch_subset() -> None:
    rows = validate_source_set(
        _source_set(), lane_id="plain", source_host_label="hostA",
    )
    assert [row["epoch"] for row in rows] == [100]
    bad = _source_set((150, 100))
    with pytest.raises(RuntimeError, match="epochs"):
        validate_source_set(bad, lane_id="plain", source_host_label="hostA")
    bad = _source_set((125,))
    with pytest.raises(RuntimeError, match="epochs"):
        validate_source_set(bad, lane_id="plain", source_host_label="hostA")


def _write_incremental_import(root, *, confirmation=False):
    lane_root = root / "sources" / "hostA" / "plain"
    lane_root.mkdir(parents=True)
    checkpoint = lane_root / "e100.pt"
    sidecar = lane_root / "e100.pt.json"
    receipt = lane_root / "e100.export.json"
    checkpoint.write_bytes(b"checkpoint")
    sidecar.write_text("{}\n", encoding="utf-8")
    export = {
        "schema": "final-unsb-paper-checkpoint-export-v1",
        "status": "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT",
        "lane_id": "plain",
        "epoch": 100,
        "updates": 855300,
        "source_host_label": "hostA",
        "source_checkpoint": "/source/e100.pt",
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_sidecar": "/source/e100.pt.json",
        "sidecar_sha256": file_sha256(sidecar),
        "scientific_state_sha256": "d" * 64,
        "training_git_commit": "a" * 40,
        "training_protocol_fingerprint": "b" * 64,
        "manifest_sha256": "c" * 64,
        "paired_metric_control": False,
        "confirmation20_opened": confirmation,
    }
    receipt.write_text(json.dumps(export) + "\n", encoding="utf-8")
    lane = {
        "schema": "final-unsb-paper-incremental-audit-imported-lane-v1",
        "status": "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
        "source_host_label": "hostA",
        "lane_id": "plain",
        "required_epochs": list(AUDIT_EPOCHS),
        "available_epochs": [100],
        "source_export_set_sha256": "e" * 64,
        "training_git_commit": "a" * 40,
        "training_protocol_fingerprint": "b" * 64,
        "manifest_sha256": "c" * 64,
        "imports": [{
            "epoch": 100,
            "export_receipt": str(receipt.resolve()),
            "export_receipt_sha256": file_sha256(receipt),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": file_sha256(sidecar),
            "scientific_state_sha256": "d" * 64,
        }],
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    lane_path = incremental_import_lane_path(root, "plain", "hostA")
    lane_path.write_text(json.dumps(lane) + "\n", encoding="utf-8")
    operations = root / "operations"
    operations.mkdir()
    import_set = {
        "schema": "final-unsb-paper-incremental-audit-import-set-v1",
        "status": "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
        "relay_id": "test",
        "source_host_label": "hostA",
        "lane_id": "plain",
        "required_epochs": list(AUDIT_EPOCHS),
        "available_epochs": [100],
        "lane_import_receipt": str(lane_path.resolve()),
        "lane_import_receipt_sha256": file_sha256(lane_path),
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    (operations / "INCREMENTAL_IMPORT_SET_test.json").write_text(
        json.dumps(import_set) + "\n", encoding="utf-8",
    )
    return lane_path


def test_incremental_import_is_source_bound_and_terminal_ready(tmp_path) -> None:
    lane_path = _write_incremental_import(tmp_path)
    rows = validate_incremental_import_lane(
        lane_path, import_root=tmp_path, lane_id="plain", host_label="hostA",
    )
    assert [row["epoch"] for row in rows] == [100]
    ready = _ready_rows(
        {"import_root": str(tmp_path)},
        {"import_lane": "plain", "host_label": "hostA"},
    )
    assert list(ready) == [100]


def test_incremental_import_rejects_confirmation(tmp_path) -> None:
    lane_path = _write_incremental_import(tmp_path, confirmation=True)
    with pytest.raises(RuntimeError, match="checkpoint export"):
        validate_incremental_import_lane(
            lane_path, import_root=tmp_path,
            lane_id="plain", host_label="hostA",
        )


def test_partial_lane_before_relay_set_is_waiting_not_audit_failure(tmp_path) -> None:
    lane_path = _write_incremental_import(tmp_path)
    (tmp_path / "operations" / "INCREMENTAL_IMPORT_SET_test.json").unlink()
    with pytest.raises(IncrementalImportNotReady):
        validate_incremental_import_lane(
            lane_path, import_root=tmp_path,
            lane_id="plain", host_label="hostA",
        )
    assert _ready_rows(
        {"import_root": str(tmp_path)},
        {"import_lane": "plain", "host_label": "hostA"},
    ) == {}


def test_incremental_relay_contract_never_persists_password(tmp_path) -> None:
    args = SimpleNamespace(
        lane="plain", relay_id="plain5090C", source_host_label="5090C",
        password_env="FINAL_UNSB_INCREMENTAL_5090C_PASSWORD",
        expected_host_key_sha256="SHA256:fixed", poll_seconds=60,
        timeout_hours=720, source_host="example", source_port=36525,
        source_user="root", remote_export_root="/runs/exports",
        destination_root=tmp_path, required_training_git_commit="a" * 40,
        required_training_protocol_fingerprint="b" * 64,
        required_manifest_sha256="c" * 64,
    )
    contract = _contract(args)
    assert contract["password_persisted"] is False
    assert "secret-value" not in json.dumps(contract)
