from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.paper_aio.runtime_relation import runtime_pair_passed, runtime_pair_status


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
