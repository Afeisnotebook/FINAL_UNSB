import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations.paper_aio_export_relay import (
    SourceExportNotReady,
    TransientRelayNetwork,
    UNIFIED_EPOCHS,
    _connect,
    _download_verified,
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


def _dclgan_receipt(epoch: int, host: str = "5090B") -> dict:
    value = _receipt("dclgan", epoch, host)
    value.update({
        "schema": "final-unsb-paper-dclgan-checkpoint-export-v1",
        "status": "ACCEPTED_SOURCE_BOUND_DCLGAN_CHECKPOINT",
        "upstream_commit": "1" * 40,
    })
    return value


def _dclgan_export_set(host: str = "5090B") -> dict:
    value = _export_set("dclgan", host)
    value["schema"] = "final-unsb-paper-dclgan-source-export-set-v1"
    return value


def test_export_set_and_receipt_accept_exact_frozen_identity() -> None:
    rows = validate_export_set(
        _export_set("plain"), lane_id="plain", source_host_label="hostA",
    )
    assert [row["epoch"] for row in rows] == list(UNIFIED_EPOCHS)
    validate_export_receipt(
        _receipt("plain", 100), lane_id="plain", epoch=100,
        source_host_label="hostA",
    )


def test_export_relay_accepts_source_bound_dclgan_profile() -> None:
    rows = validate_export_set(
        _dclgan_export_set(), lane_id="dclgan", source_host_label="5090B",
    )
    assert [row["epoch"] for row in rows] == list(UNIFIED_EPOCHS)
    validate_export_receipt(
        _dclgan_receipt(200), lane_id="dclgan", epoch=200,
        source_host_label="5090B",
    )


def test_export_relay_rejects_generic_schema_for_dclgan() -> None:
    with pytest.raises(RuntimeError):
        validate_export_set(
            _export_set("dclgan", "5090B"), lane_id="dclgan",
            source_host_label="5090B",
        )
    value = _dclgan_receipt(200)
    value.pop("upstream_commit")
    with pytest.raises(RuntimeError, match="upstream commit"):
        validate_export_receipt(
            value, lane_id="dclgan", epoch=200, source_host_label="5090B",
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


def test_download_missing_remote_is_waiting_not_local_io(monkeypatch, tmp_path) -> None:
    class MissingSftp:
        def stat(self, path):
            raise FileNotFoundError(2, "missing", path)

    with pytest.raises(SourceExportNotReady):
        _download_verified(
            MissingSftp(), "/not-ready/e100.pt", tmp_path / "e100.pt", "a" * 64,
        )


def test_connect_treats_gateway_auth_rejection_as_retryable(monkeypatch) -> None:
    class AuthenticationException(Exception):
        pass

    class SSHException(Exception):
        pass

    class Client:
        def set_missing_host_key_policy(self, policy):
            self.policy = policy

        def connect(self, **kwargs):
            raise AuthenticationException("gateway temporarily rejected auth")

        def close(self):
            pass

    fake_paramiko = SimpleNamespace(
        AuthenticationException=AuthenticationException,
        SSHException=SSHException,
        SSHClient=Client,
    )
    monkeypatch.setitem(__import__("sys").modules, "paramiko", fake_paramiko)
    monkeypatch.setenv("RELAY_PASSWORD", "not-persisted")
    with pytest.raises(TransientRelayNetwork, match="temporarily unavailable"):
        _connect({
            "password_env": "RELAY_PASSWORD",
            "expected_host_key_sha256": "SHA256:pinned",
            "source_host": "gateway",
            "source_port": 22,
            "source_user": "user",
        })


def test_json_write_retries_transient_windows_replace_denial(
    tmp_path, monkeypatch,
) -> None:
    import operations.paper_aio_export_relay as relay

    path = tmp_path / "relay-state.json"
    real_replace = Path.replace
    attempts = 0

    def flaky_replace(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError("transient reader lock")
        return real_replace(source, destination)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(relay.time, "sleep", lambda seconds: None)
    relay._write_json(path, {"status": "HEALTHY"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "HEALTHY"}
    assert attempts == 3
    assert not list(tmp_path.glob("relay-state.json.*.tmp"))


def test_existing_destination_hash_mismatch_fails_closed(tmp_path) -> None:
    destination = tmp_path / "e100.pt"
    destination.write_bytes(b"wrong")
    with pytest.raises(RuntimeError, match="existing imported file hash differs"):
        _download_verified(object(), "/remote/e100.pt", destination, "a" * 64)
