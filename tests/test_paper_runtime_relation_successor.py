from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import paper_aio_runtime_relation_successor as successor
from research.paper_aio.protocol import file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _runtime(host: str) -> dict:
    return {
        "schema": successor.RUNTIME_SCHEMA,
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": host,
        "updates": 2000,
        "e0_core_sha256": "e" * 64,
        "step_core_sha256": "s" * 64,
        "protocol_fingerprint": "p" * 64,
        "manifest_sha256": "m" * 64,
        "environment": {
            "python": "3.11", "torch": "2.8", "torch_cuda": "12.8",
            "cudnn": 91002, "gpu": "NVIDIA GeForce RTX 5090",
            "cublas_workspace_config": ":4096:8",
            "tf32_matmul": False, "tf32_cudnn": False,
        },
        "exact_runtime_equivalence": True,
        "differences": {},
        "confirmation20_opened": False,
    }


def _authorization(method_path: Path) -> dict:
    return {
        "schema": "final-unsb-paper-lane-authorization-v1",
        "status": "PASS",
        "lane_id": "proposal",
        "protocol_fingerprint": "p" * 64,
        "comparison": {"runtime_receipt_sha256": file_sha256(method_path)},
        "failures": [],
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_plain_receipt_rejects_metric_contamination() -> None:
    receipt = _runtime("5090B_MATCHED_PLAIN")
    successor.validate_plain_runtime_receipt(
        receipt, host_label="5090B_MATCHED_PLAIN",
        protocol_fingerprint="p" * 64, manifest_sha256="m" * 64,
    )
    receipt["psnr_delta"] = 1.0
    with pytest.raises(RuntimeError, match="exact sealed"):
        successor.validate_plain_runtime_receipt(
            receipt, host_label="5090B_MATCHED_PLAIN",
            protocol_fingerprint="p" * 64, manifest_sha256="m" * 64,
        )


def test_contract_persists_env_name_not_password(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    method = _write(tmp_path / "method.json", _runtime("5090C"))
    auth = _write(tmp_path / "auth.json", _authorization(method))
    script = tmp_path / "successor.py"
    script.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(successor, "__file__", str(script))
    monkeypatch.setattr(successor, "git_identity", lambda _: ("c" * 40, False))
    args = SimpleNamespace(
        repo=repo, required_control_git_commit="c" * 40,
        lane_id="proposal", method_source_host_label="5090C",
        plain_source_host_label="5090B_MATCHED_PLAIN",
        method_runtime_receipt=method,
        required_method_runtime_sha256=file_sha256(method),
        method_authorization_receipt=auth,
        required_method_authorization_sha256=file_sha256(auth),
        protocol_fingerprint="p" * 64, manifest_sha256="m" * 64,
        source_host="example", source_port=44804, source_user="root",
        expected_host_key_sha256="SHA256:pinned",
        password_env="FINAL_UNSB_5090B_PASSWORD",
        remote_plain_runtime_receipt="/runs/gates/RUNTIME_TWIN.json",
        destination_output=tmp_path / "out", poll_seconds=60,
        timeout_hours=720.0,
    )
    contract = successor.proposed_contract(args)
    encoded = json.dumps(contract)
    assert contract["password_env"] == "FINAL_UNSB_5090B_PASSWORD"
    assert contract["password_persisted"] is False
    assert "secret-value" not in encoded
    assert contract["registry_edited"] is False
    assert contract["comparison_authorized"] is False


def test_relation_candidate_is_idempotent_and_review_only(tmp_path: Path) -> None:
    method = _write(tmp_path / "method.json", _runtime("5090C"))
    plain = _write(tmp_path / "plain.json", _runtime("5090B_MATCHED_PLAIN"))
    auth = _write(tmp_path / "auth.json", _authorization(method))
    contract = {
        "lane_id": "proposal",
        "method_source_host_label": "5090C",
        "plain_source_host_label": "5090B_MATCHED_PLAIN",
        "method_runtime_receipt": str(method),
        "method_authorization_receipt": str(auth),
    }
    destination = tmp_path / "candidate.json"
    first = successor.publish_relation_candidate(
        contract=contract, plain_runtime_receipt=plain, destination=destination,
    )
    second = successor.publish_relation_candidate(
        contract=contract, plain_runtime_receipt=plain, destination=destination,
    )
    assert first == second
    assert first["performance_values_read"] is False
    assert first["paired_metric_control"] is False
    assert first["confirmation20_opened"] is False


def test_exact_byte_publication_rejects_existing_drift(tmp_path: Path) -> None:
    destination = tmp_path / "receipt.json"
    blob = b'{"exact":true}'
    expected = hashlib.sha256(blob).hexdigest()
    assert successor.publish_exact_bytes(destination, blob) == expected
    destination.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="immutable receipt destination changed"):
        successor.publish_exact_bytes(destination, blob)
