from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.paper_aio_unified_evaluation_successor import (
    COMPLETE_STATUS,
    IMPORT_LANE_SCHEMA,
    STATE_SCHEMA,
    UNIFIED_EPOCHS,
    import_lane_path,
    imports_ready,
    parse_lane_source,
    release_decision,
    validate_import_lane,
)
from operations.paper_aio_health_watch import _terminal


def _write(path: Path, value: dict | str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        path.write_text(json.dumps(value), encoding="utf-8")
    elif isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")
    return path


def test_lane_source_and_release_state_are_metric_blind(tmp_path: Path) -> None:
    assert STATE_SCHEMA == "final-unsb-paper-unified-evaluation-successor-state-v2"
    assert parse_lane_source("proposal=5090C") == ("proposal", "5090C")
    with pytest.raises(ValueError):
        parse_lane_source("../proposal=5090C")
    state = tmp_path / "SUPERVISOR_amtnc.json"
    assert release_decision(state, "COMPLETE_E200") == "WAIT"
    _write(state, {"status": "CHILD_RUNNING", "macro_psnr": 999})
    assert release_decision(state, "COMPLETE_E200") == "WAIT"
    _write(state, {"status": "COMPLETE_E200", "macro_psnr": -999})
    assert release_decision(state, "COMPLETE_E200") == "READY"
    _write(state, {"status": "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE"})
    assert release_decision(state, "COMPLETE_E200") == "BLOCKED"
    assert _terminal({"status": COMPLETE_STATUS})


def test_import_readiness_uses_only_fixed_completion_artifacts(tmp_path: Path) -> None:
    lanes = {
        "plain": "5090A", "proposal": "5090C",
        "cut": "5090B", "cyclegan": "5090B",
    }
    assert not imports_ready(tmp_path, lanes)
    for lane, host in lanes.items():
        receipt = _write(import_lane_path(tmp_path, lane, host), {})
        from research.paper_aio.protocol import file_sha256
        _write(tmp_path / "operations" / f"IMPORT_SET_{lane}.json", {
            "schema": "final-unsb-paper-import-set-v1",
            "status": "COMPLETE_VERIFIED_IMPORT_SET",
            "source_host_label": host,
            "lanes": [lane], "epochs": list(UNIFIED_EPOCHS),
            "lane_imports": {lane: {
                "receipt": str(receipt.resolve()),
                "receipt_sha256": file_sha256(receipt),
            }},
            "checkpoint_copy_performed": True,
            "source_checkpoint_mutation": False,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
    assert imports_ready(tmp_path, lanes)


def test_import_lane_validation_binds_every_source_file(tmp_path: Path) -> None:
    from research.paper_aio.protocol import file_sha256

    lane_root = tmp_path / "sources" / "5090A" / "plain"
    imports = []
    for epoch in UNIFIED_EPOCHS:
        receipt = _write(lane_root / f"e{epoch:03d}.export.json", "receipt")
        checkpoint = _write(lane_root / f"e{epoch:03d}.pt", b"checkpoint")
        sidecar = _write(lane_root / f"e{epoch:03d}.pt.json", "sidecar")
        imports.append({
            "epoch": epoch,
            "export_receipt": str(receipt.resolve()),
            "export_receipt_sha256": file_sha256(receipt),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": file_sha256(checkpoint),
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": file_sha256(sidecar),
        })
    receipt = _write(lane_root / "IMPORT_LANE.json", {
        "schema": IMPORT_LANE_SCHEMA,
        "status": "COMPLETE_VERIFIED_IMPORTED_LANE",
        "source_host_label": "5090A",
        "lane_id": "plain",
        "epochs": list(UNIFIED_EPOCHS),
        "imports": imports,
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "operations" / "IMPORT_SET_plain.json", {
        "schema": "final-unsb-paper-import-set-v1",
        "status": "COMPLETE_VERIFIED_IMPORT_SET",
        "source_host_label": "5090A", "lanes": ["plain"],
        "epochs": list(UNIFIED_EPOCHS),
        "lane_imports": {"plain": {
            "receipt": str(receipt.resolve()),
            "receipt_sha256": file_sha256(receipt),
        }},
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    rows = validate_import_lane(
        receipt, import_root=tmp_path, lane_id="plain", host_label="5090A",
    )
    assert [row["epoch"] for row in rows] == list(UNIFIED_EPOCHS)
    checkpoint = Path(rows[0]["checkpoint"])
    checkpoint.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_import_lane(
            receipt, import_root=tmp_path, lane_id="plain", host_label="5090A",
        )


def test_import_lane_cannot_escape_staging_root(tmp_path: Path) -> None:
    outside = _write(tmp_path.parent / "outside.pt", b"x")
    lane_root = tmp_path / "sources" / "5090A" / "plain"
    receipt = _write(lane_root / "IMPORT_LANE.json", {
        "schema": IMPORT_LANE_SCHEMA,
        "status": "COMPLETE_VERIFIED_IMPORTED_LANE",
        "source_host_label": "5090A",
        "lane_id": "plain",
        "epochs": list(UNIFIED_EPOCHS),
        "imports": [{
            "epoch": epoch,
            "export_receipt": str(outside), "export_receipt_sha256": "x",
            "checkpoint": str(outside), "checkpoint_sha256": "x",
            "sidecar": str(outside), "sidecar_sha256": "x",
        } for epoch in UNIFIED_EPOCHS],
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    from research.paper_aio.protocol import file_sha256
    _write(tmp_path / "operations" / "IMPORT_SET_plain.json", {
        "schema": "final-unsb-paper-import-set-v1",
        "status": "COMPLETE_VERIFIED_IMPORT_SET",
        "source_host_label": "5090A", "lanes": ["plain"],
        "epochs": list(UNIFIED_EPOCHS),
        "lane_imports": {"plain": {
            "receipt": str(receipt.resolve()),
            "receipt_sha256": file_sha256(receipt),
        }},
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    with pytest.raises(RuntimeError, match="escapes verified import root"):
        validate_import_lane(
            receipt, import_root=tmp_path, lane_id="plain", host_label="5090A",
        )
