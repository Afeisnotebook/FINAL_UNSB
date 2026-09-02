from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.paper_aio.runtime_relation import (
    materialize_exact_runtime_relation,
    runtime_pair_passed,
    runtime_pair_status,
)
from research.paper_aio.run import parser


def _metric(host: str, fingerprint: str = "fp", manifest: str = "manifest") -> dict:
    return {
        "source_host_label": host,
        "training_protocol_fingerprint": fingerprint,
        "manifest_sha256": manifest,
        "confirmation20_opened": False,
    }


def _registry(path: Path) -> Path:
    value = {
        "schema": "final-unsb-paper-matched-runtime-relations-v1",
        "status": "ACTIVE_METRIC_BLIND_RELATIONS",
        "relations": {
            "proposal": {
                "status": "PASS_EXACT_RUNTIME_RELATION",
                "method_lane": "proposal",
                "method_source_host_label": "5090C",
                "plain_source_host_label": "5090A",
                "updates": 2000,
                "training_protocol_fingerprint": "fp",
                "manifest_sha256": "manifest",
                "e0_core_sha256": "e" * 64,
                "step_core_sha256": "s" * 64,
                "method_runtime_receipt_sha256": "m" * 64,
                "plain_runtime_receipt_sha256": "p" * 64,
                "method_authorization_receipt_sha256": "a" * 64,
                "differences": {},
                "performance_values_read": False,
                "paired_metric_control": False,
                "confirmation20_opened": False,
            }
        },
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_same_host_requires_same_training_identity(tmp_path: Path) -> None:
    result = runtime_pair_status(
        method=_metric("4090A"), plain=_metric("4090A"), lane_id="amtnc",
        relations_path=tmp_path / "unused.json",
    )
    assert runtime_pair_passed(result)
    mismatch = runtime_pair_status(
        method=_metric("4090A", "one"), plain=_metric("4090A", "two"),
        lane_id="amtnc", relations_path=tmp_path / "unused.json",
    )
    assert not runtime_pair_passed(mismatch)


def test_cross_host_requires_exact_metric_blind_relation(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "relations.json")
    result = runtime_pair_status(
        method=_metric("5090C"), plain=_metric("5090A"), lane_id="proposal",
        relations_path=registry,
    )
    assert result["status"] == "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"
    assert runtime_pair_passed(result)
    missing = runtime_pair_status(
        method=_metric("5090C"), plain=_metric("5090A"), lane_id="amtnc",
        relations_path=registry,
    )
    assert not runtime_pair_passed(missing)


def test_cross_host_selects_one_of_multiple_plain_relations(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "relations.json")
    value = json.loads(registry.read_text(encoding="utf-8"))
    original = value["relations"]["proposal"]
    alternate = {
        **original,
        "plain_source_host_label": "5090B_MATCHED_PLAIN",
        "plain_runtime_receipt_sha256": "q" * 64,
    }
    value["relations"]["proposal"] = [original, alternate]
    registry.write_text(json.dumps(value), encoding="utf-8")
    result = runtime_pair_status(
        method=_metric("5090C"), plain=_metric("5090B_MATCHED_PLAIN"),
        lane_id="proposal", relations_path=registry,
    )
    assert result["status"] == "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"
    assert runtime_pair_passed(result)


def _write(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _runtime(host: str) -> dict:
    return {
        "schema": "final-unsb-paper-runtime-twin-receipt-v1",
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": host,
        "updates": 2000,
        "e0_core_sha256": "e" * 64,
        "step_core_sha256": "s" * 64,
        "protocol_fingerprint": "fp",
        "manifest_sha256": "m" * 64,
        "environment": {
            "python": "3.11", "torch": "2.8", "torch_cuda": "12.8",
            "cudnn": 91002, "gpu": "RTX 5090",
            "cublas_workspace_config": ":4096:8",
            "tf32_matmul": False, "tf32_cudnn": False,
            "hostname": host,
        },
        "confirmation20_opened": False,
        "exact_runtime_equivalence": True,
        "differences": {},
    }


def test_relation_materializer_requires_primary_exact_receipts(tmp_path: Path) -> None:
    method_path = _write(tmp_path / "method.json", _runtime("5090C"))
    plain_path = _write(
        tmp_path / "plain.json", _runtime("5090B_MATCHED_PLAIN"),
    )
    from research.paper_aio.protocol import file_sha256

    authorization = {
        "schema": "final-unsb-paper-lane-authorization-v1",
        "status": "PASS",
        "lane_id": "proposal",
        "protocol_fingerprint": "fp",
        "comparison": {"runtime_receipt_sha256": file_sha256(method_path)},
        "failures": [],
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    authorization_path = _write(tmp_path / "authorization.json", authorization)
    destination = tmp_path / "relation.json"
    result = materialize_exact_runtime_relation(
        lane_id="proposal", method_source_host_label="5090C",
        plain_source_host_label="5090B_MATCHED_PLAIN",
        method_runtime_receipt=method_path,
        plain_runtime_receipt=plain_path,
        method_authorization_receipt=authorization_path,
        destination=destination,
    )
    assert result["differences"] == {}
    assert result["performance_values_read"] is False
    assert json.loads(destination.read_text(encoding="utf-8")) == result

    broken = _runtime("5090B_MATCHED_PLAIN")
    broken["step_core_sha256"] = "x" * 64
    _write(plain_path, broken)
    with pytest.raises(RuntimeError, match="do not prove an exact"):
        materialize_exact_runtime_relation(
            lane_id="proposal", method_source_host_label="5090C",
            plain_source_host_label="5090B_MATCHED_PLAIN",
            method_runtime_receipt=method_path,
            plain_runtime_receipt=plain_path,
            method_authorization_receipt=authorization_path,
            destination=tmp_path / "broken.json",
        )

    contaminated = _runtime("5090B_MATCHED_PLAIN")
    contaminated["psnr"] = 20.0
    _write(plain_path, contaminated)
    with pytest.raises(RuntimeError, match="performance fields"):
        materialize_exact_runtime_relation(
            lane_id="proposal", method_source_host_label="5090C",
            plain_source_host_label="5090B_MATCHED_PLAIN",
            method_runtime_receipt=method_path,
            plain_runtime_receipt=plain_path,
            method_authorization_receipt=authorization_path,
            destination=tmp_path / "contaminated.json",
        )


def test_runtime_relation_cli_requires_explicit_primary_receipts() -> None:
    args = parser().parse_args([
        "--stage", "runtime-relation", "--lane", "proposal",
        "--method-runtime-receipt", "method.json",
        "--plain-runtime-receipt", "plain.json",
        "--method-authorization-receipt", "authorization.json",
        "--method-source-host-label", "5090C",
        "--plain-source-host-label", "5090B_MATCHED_PLAIN",
        "--receipt-output", "candidate.json",
    ])
    assert args.stage == "runtime-relation"
    assert args.plain_source_host_label == "5090B_MATCHED_PLAIN"


def test_relation_registry_rejects_performance_contamination(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "relations.json")
    value = json.loads(registry.read_text(encoding="utf-8"))
    value["relations"]["proposal"]["psnr_delta"] = 1.0
    registry.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="metric-contaminated"):
        runtime_pair_status(
            method=_metric("5090C"), plain=_metric("5090A"), lane_id="proposal",
            relations_path=registry,
        )
