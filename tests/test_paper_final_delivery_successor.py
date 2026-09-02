from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operations import paper_aio_final_delivery_successor as final


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _entry(
    lane: str, gate: str = "PASS", *, method_host: str | None = None,
    plain_host: str | None = None, candidate_cross_code: bool = False,
) -> dict:
    value = {
        "lane_id": lane,
        "status": "COMPLETE_E200",
        "comparison_scope": "fixture",
        "terminal": {"macro_psnr": 1.0, "macro_ssim": 1.0, "macro_lpips": 0.0},
        "scientific_gate": {"status": gate},
    }
    if method_host is not None and plain_host is not None:
        value["late_trajectory"] = [
            {
                "epoch": epoch,
                "crn_exact": True,
                "runtime_relation": {
                    "status": (
                        "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION"
                        if candidate_cross_code else
                        "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"
                    ),
                    "method_source_host_label": method_host,
                    "plain_source_host_label": plain_host,
                    "runtime_twin_updates": 2000,
                    "e0_core_sha256": "e" * 64,
                    "step_core_sha256": "s" * 64,
                },
            }
            for epoch in (150, 175, 200)
        ]
    return value


def _disposition(tmp_path: Path, lane: str, gate: str = "PASS") -> tuple[Path, dict]:
    receipts = []
    for epoch in (100, 125, 150, 175, 200):
        path = _write(tmp_path / lane / f"e{epoch}.json", {"epoch": epoch})
        receipts.append({"epoch": epoch, "path": str(path), "sha256": _sha(path)})
    value = {
        "schema": final.DISPOSITION_SCHEMA,
        "status": "COMPLETE_POSTHOC_ALGORITHM_DISPOSITION",
        "method_lane": lane,
        "primary_epoch": 200,
        "fixed_epochs": [100, 125, 150, 175, 200],
        "entry": _entry(
            lane, gate, method_host="5090A", plain_host="5090B_MATCHED_PLAIN",
            candidate_cross_code=lane == final.STCGR_ID,
        ),
        "evaluation_receipts": receipts,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    return _write(tmp_path / f"{lane}.disposition.json", value), value


def _complexity(lane: str) -> dict:
    return {
        "lane_id": lane,
        "checkpoint_sha256": "a" * 64,
        "environment": {"gpu": "fixture"},
        "parameters": {"unique_parameters": 1},
        "inference": {"nfe": {}},
        "training_step": {"median_ms": 1.0},
        "flops": {"reported": False},
        "checkpoint_unchanged": True,
    }


def test_completion_decision_uses_only_fixed_status(tmp_path: Path) -> None:
    state = tmp_path / "state.json"
    assert final.completion_decision(state, "DONE") == "WAIT"
    _write(state, {"status": "RUNNING", "psnr": 99})
    assert final.completion_decision(state, "DONE") == "WAIT"
    _write(state, {"status": "DONE", "psnr": -99})
    assert final.completion_decision(state, "DONE") == "READY"
    _write(state, {"status": "FAIL_CLOSED"})
    assert final.completion_decision(state, "DONE") == "BLOCKED"


def test_disposition_validation_is_hash_and_boundary_bound(tmp_path: Path) -> None:
    path, value = _disposition(tmp_path, "amtnc")
    assert final.validate_disposition(path, "amtnc") == value
    value["best_checkpoint_selection"] = True
    _write(path, value)
    with pytest.raises(RuntimeError, match="invalid terminal"):
        final.validate_disposition(path, "amtnc")


def test_complexity_receipt_rejects_target_access_or_unaudited_flops(
    tmp_path: Path,
) -> None:
    checkpoint_hash = "a" * 64
    value = {
        "schema": final.COMPLEXITY_SCHEMA,
        "status": "PASS_TARGET_BLIND_CHECKPOINT_READ_ONLY_PROFILE",
        "lane_id": "amtnc",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_unchanged": True,
        "protocol_fingerprint": "b" * 64,
        "evaluation_bundle_fingerprint": final.FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "portable_candidate_authority_sha256": None,
        "source_input": {"target_path_read": False},
        "performance_metric_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "flops": {"reported": False},
    }
    path = _write(tmp_path / "complexity.json", value)
    assert final.validate_complexity_receipt(
        path, lane_id="amtnc", checkpoint_sha256=checkpoint_hash,
        expected_protocol_fingerprint="b" * 64,
    ) == value
    value["source_input"]["target_path_read"] = True
    _write(path, value)
    with pytest.raises(RuntimeError, match="invalid complexity"):
        final.validate_complexity_receipt(
            path, lane_id="amtnc", checkpoint_sha256=checkpoint_hash,
            expected_protocol_fingerprint="b" * 64,
        )


def test_portfolio_preserves_three_matched_relations_and_failure_scope(
    tmp_path: Path,
) -> None:
    _, amtnc = _disposition(tmp_path, "amtnc", "FAIL")
    _, stcgr = _disposition(tmp_path, final.STCGR_ID, "PASS")
    lanes = [
        _entry("input"), _entry("plain"),
        _entry(
            "proposal", method_host="5090C", plain_host="5090B_MATCHED_PLAIN",
        ),
        _entry("cut"), _entry("cyclegan"), _entry(final.STCGR_ID),
    ]
    first = {
        "schema": "final-unsb-paper-results-v1",
        "status": "FIRST_WAVE_COMPLETE",
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "lanes": lanes,
    }
    complexity = {lane: _complexity(lane) for lane in final.COMPLEXITY_LANES}
    portfolio = {
        "methods": {"hjcgr": {"status": "deferred", "mechanism_falsified": False}},
        "controls_and_external": {"ddsb": "reproduction_incomplete"},
    }
    value = final.build_portfolio(
        first_wave_results=first, amtnc_disposition=amtnc,
        stcgr_disposition=stcgr, complexity=complexity,
        source_hashes={"fixture": "f" * 64}, method_portfolio=portfolio,
        first_wave_lane_sources={
            "plain": "5090B_MATCHED_PLAIN", "proposal": "5090C",
            "cut": "5090B", "cyclegan": "5090B",
        },
        stcgr_source_host="5090A",
    )
    assert value["methods"]["proposal"]["matched_plain"] == "5090B_MATCHED_PLAIN/plain"
    assert value["methods"]["stcgr"]["matched_plain"] == "5090B_MATCHED_PLAIN/plain"
    assert value["methods"]["amtnc"]["matched_plain"] == "4090A/plain"
    assert "amtnc" in value["failed_current_implementation_and_protocol"]
    assert value["deferred_or_reproduction_incomplete"]["hjcgr"]["mechanism_falsified"] is False
    assert value["paper_claims_frozen"] is False
    assert value["confirmation20_opened"] is False


def test_portfolio_rejects_a_mismatched_control_host(tmp_path: Path) -> None:
    _, amtnc = _disposition(tmp_path, "amtnc", "PASS")
    _, stcgr = _disposition(tmp_path, final.STCGR_ID, "PASS")
    first = {
        "schema": "final-unsb-paper-results-v1",
        "status": "FIRST_WAVE_COMPLETE",
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "lanes": [
            _entry("input"), _entry("plain"),
            _entry("proposal", method_host="5090C", plain_host="5090A"),
            _entry("cut"), _entry("cyclegan"), _entry(final.STCGR_ID),
        ],
    }
    with pytest.raises(RuntimeError, match="frozen matched plain relation"):
        final.build_portfolio(
            first_wave_results=first, amtnc_disposition=amtnc,
            stcgr_disposition=stcgr,
            complexity={lane: _complexity(lane) for lane in final.COMPLEXITY_LANES},
            source_hashes={},
            method_portfolio={
                "methods": {"hjcgr": {"status": "deferred"}},
                "controls_and_external": {"ddsb": "reproduction_incomplete"},
            },
            first_wave_lane_sources={
                "plain": "5090B_MATCHED_PLAIN", "proposal": "5090C",
                "cut": "5090B", "cyclegan": "5090B",
            },
            stcgr_source_host="5090A",
        )
