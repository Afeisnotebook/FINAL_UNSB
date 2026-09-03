import json
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from operations.paper_aio_incremental_audit_export import (
    available_exports,
    export_set,
)
from operations.paper_aio_incremental_audit_relay import (
    AUDIT_EPOCHS,
    IncrementalImportNotReady,
    _contract,
    _import_available,
    incremental_import_lane_path,
    validate_incremental_import_lane,
    validate_source_set,
)
from operations.paper_aio_local_terminal_audit_successor import _ready_rows
from research.local_route1.runtime import full_state_hash
from research.paper_aio.protocol import FULL_STATE_SCHEMA, file_sha256, lane_spec


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


class _FixtureSftp:
    def __init__(self, files: dict[str, bytes]):
        self.files = files

    def open(self, path: str, mode: str = "rb"):
        if path not in self.files:
            raise FileNotFoundError(2, "missing", path)
        return BytesIO(self.files[path])

    def stat(self, path: str):
        if path not in self.files:
            raise FileNotFoundError(2, "missing", path)
        return SimpleNamespace(st_size=len(self.files[path]))


def test_first_e100_export_relay_and_terminal_readiness_integration(tmp_path) -> None:
    """Exercise the same partial-set shape used by tonight's first real e100."""
    source_output = tmp_path / "source_output"
    milestone = source_output / "lanes" / "plain" / "milestones"
    milestone.mkdir(parents=True)
    checkpoint = milestone / "e100.pt"
    payload = {
        "schema": FULL_STATE_SCHEMA,
        "lane": lane_spec("plain").to_dict(),
        "step": 855300,
        "target_steps": 1710600,
        "model": {"networks": {"G": {"weight": torch.tensor([1.0])}}},
        "rng": {"python": (3, (), None)},
        "samplers": {"primary": {}, "secondary": {}},
        "metadata": {
            "git_commit": "a" * 40,
            "protocol_fingerprint": "b" * 64,
            "manifest_sha256": "c" * 64,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    }
    torch.save(payload, checkpoint)
    sidecar = milestone / "e100.pt.json"
    sidecar.write_text(json.dumps({
        "schema": FULL_STATE_SCHEMA,
        "lane_id": "plain",
        "step": 855300,
        "physical_epoch_completed": 100,
        "full_state_sha256": file_sha256(checkpoint),
        "scientific_state_sha256": full_state_hash(payload),
    }) + "\n", encoding="utf-8")

    export_contract = {
        "source_output": str(source_output),
        "destination": str(tmp_path / "source_exports"),
        "lane_id": "plain",
        "source_host_label": "4090A",
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "required_manifest_sha256": "c" * 64,
        "audit_epochs": list(AUDIT_EPOCHS),
    }
    rows = available_exports(export_contract)
    assert [row["epoch"] for row in rows] == [100]

    local_receipt = tmp_path / "source_exports" / "plain" / "e100.export.json"
    receipt = json.loads(local_receipt.read_text(encoding="utf-8"))
    receipt["source_checkpoint"] = "/source/plain/e100.pt"
    receipt["source_sidecar"] = "/source/plain/e100.pt.json"
    receipt_bytes = (json.dumps(receipt) + "\n").encode()
    rows[0]["receipt"] = "/exports/plain/e100.export.json"
    import hashlib
    rows[0]["receipt_sha256"] = hashlib.sha256(receipt_bytes).hexdigest()
    source_set_bytes = (
        json.dumps(export_set(export_contract, rows)) + "\n"
    ).encode()
    sftp = _FixtureSftp({
        "/exports/plain/INCREMENTAL_AUDIT_EXPORT_SET.json": source_set_bytes,
        "/exports/plain/e100.export.json": receipt_bytes,
        "/source/plain/e100.pt": checkpoint.read_bytes(),
        "/source/plain/e100.pt.json": sidecar.read_bytes(),
    })
    import_root = tmp_path / "imports"
    relay_contract = {
        "lane_id": "plain",
        "relay_id": "incremental_4090A_plain",
        "source_host_label": "4090A",
        "remote_export_root": "/exports",
        "destination_root": str(import_root),
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "required_manifest_sha256": "c" * 64,
    }
    result = _import_available(sftp, relay_contract)
    assert result["status"] == "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT"
    assert result["available_epochs"] == [100]
    lane_path = incremental_import_lane_path(import_root, "plain", "4090A")
    imported = validate_incremental_import_lane(
        lane_path, import_root=import_root, lane_id="plain", host_label="4090A",
    )
    assert [row["epoch"] for row in imported] == [100]
    ready = _ready_rows(
        {"import_root": str(import_root)},
        {"import_lane": "plain", "host_label": "4090A"},
    )
    assert list(ready) == [100]
    assert ready[100]["checkpoint_sha256"] == file_sha256(checkpoint)


def test_incremental_export_waits_for_complete_atomic_milestone_pair(tmp_path) -> None:
    """A checkpoint or sidecar observed alone must never become export-visible."""
    source_output = tmp_path / "source_output"
    milestone = source_output / "lanes" / "plain" / "milestones"
    milestone.mkdir(parents=True)
    checkpoint = milestone / "e100.pt"
    sidecar = milestone / "e100.pt.json"
    contract = {
        "source_output": str(source_output),
        "destination": str(tmp_path / "source_exports"),
        "lane_id": "plain",
        "source_host_label": "4090A",
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "required_manifest_sha256": "c" * 64,
        "audit_epochs": list(AUDIT_EPOCHS),
    }

    checkpoint.write_bytes(b"not-yet-sidecar-bound")
    assert available_exports(contract) == []
    checkpoint.unlink()
    sidecar.write_text("{}\n", encoding="utf-8")
    assert available_exports(contract) == []


def test_incremental_export_rejects_milestone_sidecar_hash_mismatch(tmp_path) -> None:
    """A complete-looking but torn/corrupt pair must fail closed before publication."""
    source_output = tmp_path / "source_output"
    milestone = source_output / "lanes" / "plain" / "milestones"
    milestone.mkdir(parents=True)
    checkpoint = milestone / "e100.pt"
    payload = {
        "schema": FULL_STATE_SCHEMA,
        "lane": lane_spec("plain").to_dict(),
        "step": 855300,
        "target_steps": 1710600,
        "model": {"networks": {"G": {"weight": torch.tensor([1.0])}}},
        "rng": {"python": (3, (), None)},
        "samplers": {"primary": {}, "secondary": {}},
        "metadata": {
            "git_commit": "a" * 40,
            "protocol_fingerprint": "b" * 64,
            "manifest_sha256": "c" * 64,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    }
    torch.save(payload, checkpoint)
    sidecar = milestone / "e100.pt.json"
    sidecar.write_text(json.dumps({
        "schema": FULL_STATE_SCHEMA,
        "lane_id": "plain",
        "step": 855300,
        "physical_epoch_completed": 100,
        "full_state_sha256": "0" * 64,
        "scientific_state_sha256": full_state_hash(payload),
    }) + "\n", encoding="utf-8")
    contract = {
        "source_output": str(source_output),
        "destination": str(tmp_path / "source_exports"),
        "lane_id": "plain",
        "source_host_label": "4090A",
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "required_manifest_sha256": "c" * 64,
        "audit_epochs": list(AUDIT_EPOCHS),
    }

    with pytest.raises(RuntimeError, match="file hash mismatch"):
        available_exports(contract)
    assert not (tmp_path / "source_exports" / "plain" / "e100.export.json").exists()
