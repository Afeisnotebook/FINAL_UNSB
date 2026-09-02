import json
from types import SimpleNamespace

import pytest

from operations.paper_aio_export_relay import (
    UNIFIED_EPOCHS,
    _contract,
    _remote_path,
    validate_export_receipt,
    validate_export_set,
)


def _receipt(lane: str, epoch: int, host: str = "hostA") -> dict:
    return {
        "schema": "final-unsb-paper-checkpoint-export-v1",
        "status": "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT",
        "lane_id": lane,
        "epoch": epoch,
        "updates": epoch * 8553,
        "source_host_label": host,
        "source_checkpoint": f"/runs/{lane}/e{epoch:03d}.pt",
        "checkpoint_sha256": "a" * 64,
        "source_sidecar": f"/runs/{lane}/e{epoch:03d}.pt.json",
        "sidecar_sha256": "b" * 64,
        "scientific_state_sha256": "c" * 64,
        "training_git_commit": "d" * 40,
        "training_protocol_fingerprint": "e" * 64,
        "manifest_sha256": "f" * 64,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _export_set(lane: str, host: str = "hostA") -> dict:
    return {
        "schema": "final-unsb-paper-source-export-set-v1",
        "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
        "lane_id": lane,
        "source_host_label": host,
        "epochs": list(UNIFIED_EPOCHS),
        "exports": [
            {
                "epoch": epoch,
                "receipt": f"/exports/{lane}/e{epoch:03d}.export.json",
                "receipt_sha256": "1" * 64,
            }
            for epoch in UNIFIED_EPOCHS
        ],
        "performance_values_read": False,
        "checkpoint_copy_performed": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_export_set_and_receipt_accept_exact_frozen_identity() -> None:
    rows = validate_export_set(
        _export_set("plain"), lane_id="plain", source_host_label="hostA",
    )
    assert [row["epoch"] for row in rows] == list(UNIFIED_EPOCHS)
    validate_export_receipt(
        _receipt("plain", 100), lane_id="plain", epoch=100,
        source_host_label="hostA",
    )


def test_export_set_rejects_confirmation_or_epoch_drift() -> None:
    payload = _export_set("plain")
    payload["confirmation20_opened"] = True
    with pytest.raises(RuntimeError):
        validate_export_set(payload, lane_id="plain", source_host_label="hostA")
    payload = _export_set("plain")
    payload["epochs"] = [100, 200]
    with pytest.raises(RuntimeError):
        validate_export_set(payload, lane_id="plain", source_host_label="hostA")


def test_export_receipt_rejects_lane_or_update_drift() -> None:
    payload = _receipt("plain", 100)
    payload["updates"] += 1
    with pytest.raises(RuntimeError):
        validate_export_receipt(
            payload, lane_id="plain", epoch=100, source_host_label="hostA",
        )


def test_remote_paths_must_be_absolute_and_cannot_escape() -> None:
    assert _remote_path("/safe/export", "test") == "/safe/export"
    with pytest.raises(RuntimeError):
        _remote_path("relative/export", "test")
    with pytest.raises(RuntimeError):
        _remote_path("/safe/../escape", "test")


def test_contract_never_persists_password(tmp_path, monkeypatch) -> None:
    import operations.paper_aio_export_relay as relay

    monkeypatch.setattr(relay, "__file__", str(tmp_path / "relay.py"))
    (tmp_path / "relay.py").write_text("frozen", encoding="utf-8")
    args = SimpleNamespace(
        lane=["cut", "cyclegan"], relay_id="external5090B",
        source_host_label="5090B",
        password_env="FINAL_UNSB_PAPER_RELAY_5090B_PASSWORD",
        expected_host_key_sha256="SHA256:fixed", poll_seconds=60,
        timeout_hours=480, source_host="example", source_port=44804,
        source_user="root", remote_export_root="/runs/exports",
        destination_root=tmp_path / "destination",
    )
    contract = _contract(args)
    encoded = json.dumps(contract)
    assert contract["password_persisted"] is False
    assert contract["relay_id"] == "external5090B"
    assert contract["password_env"] == "FINAL_UNSB_PAPER_RELAY_5090B_PASSWORD"
    assert "secret-value" not in encoded
