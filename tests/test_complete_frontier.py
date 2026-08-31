from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from research.local_route1.complete_frontier import (
    ADAM_SYNTHESIS_SCHEMA,
    EUCLIDEAN_SYNTHESIS_SCHEMA,
    REPAIRED_RESULT_SCHEMA,
    STATUS,
    materialize_complete_4090_frontier,
)
from research.local_route1.pcnr_alternate_replay import (
    RESULT_SCHEMA as PCNR_ALTERNATE_RESULT_SCHEMA,
)
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.protocol import ROOT, file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipt(
    root: Path, candidate_id: str, late: float, *, positive: bool = True,
    suffix: str = "", plain: str = "plain",
) -> Path:
    status = POSITIVE_STATUS if positive else NEGATIVE_STATUS
    trajectory = _write(root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json", {
        "candidate_id": candidate_id,
        "status": status,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
    })
    path = root / "operations" / "terminal_receipts" / f"{candidate_id}{suffix}.json"
    _write(path, {
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": status,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "plain_e200_verification_sha256": plain,
        "training_git_commit": "a" * 40,
        "receipt_source_sha256": file_sha256(
            ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
        ),
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": file_sha256(trajectory),
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": late,
            "e200_macro_psnr_delta": late,
            "late_points_with_four_of_six_positive_domains": 2,
            "late_average_worst_domain_delta": -0.2,
            "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
            "late_mean_macro_ssim_delta": 0.01,
            "late_mean_macro_lpips_delta": -0.01,
        },
        "median_epoch_wall_seconds": 1.0,
        "terminal_integrity": {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(Path(str(path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    })
    return path


def _pre_frontier(root: Path, selected: str, old: str) -> None:
    _write(root / "final" / "CANDIDATE.json", {
        "candidate_id": selected,
        "confirmation20_opened": False,
        "paired_controller_access": False,
    })
    _write(root / "final" / "RESULTS.json", {
        "selected_candidate_id": selected,
        "ranking": [{"candidate_id": old}],
        "confirmation20_opened": False,
        "paired_controller_access": False,
        "cross_host_deltas_merged": False,
    })


def _terminal_results(
    root: Path, repair: Path, *, adam: Path | None = None,
    euclidean: Path | None = None, pcnr: Path | None = None,
) -> None:
    operations = root / "operations"
    _write(operations / "REPAIRED_PORTFOLIO_4090_RESULT.json", {
        "schema": REPAIRED_RESULT_SCHEMA,
        "status": "REPAIRED_PORTFOLIO_4090_REPLAYS_COMPLETE_E200",
        "candidate_results": [{
            "candidate_id": json.loads(repair.read_text())["candidate_id"],
            "receipt_path": str(repair.resolve()),
            "receipt_sha256": file_sha256(repair),
        }],
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    for name, schema, receipt in (
        ("RESIDUAL_SYNTHESIS_4090_RESULT.json", ADAM_SYNTHESIS_SCHEMA, adam),
        (
            "RESIDUAL_EUCLIDEAN_SYNTHESIS_4090_RESULT.json",
            EUCLIDEAN_SYNTHESIS_SCHEMA,
            euclidean,
        ),
    ):
        candidate_id = None if receipt is None else json.loads(receipt.read_text())["candidate_id"]
        value = {
            "schema": schema,
            "status": "SYNTHESIS_INAPPLICABLE" if receipt is None else "COMPLETE_E200",
            "candidate_id": candidate_id,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        if receipt is not None:
            value.update({
                "receipt_path": str(receipt.resolve()),
                "receipt_sha256": file_sha256(receipt),
            })
        _write(operations / name, value)
    pcnr_id = None if pcnr is None else json.loads(pcnr.read_text())["candidate_id"]
    pcnr_value = {
        "schema": PCNR_ALTERNATE_RESULT_SCHEMA,
        "status": (
            "PCNR_ALTERNATE_REPLAY_INAPPLICABLE"
            if pcnr is None else
            "PCNR_EVIDENCE_BACKED_ALTERNATE_4090_REPLAY_COMPLETE_E200"
        ),
        "candidate_id": pcnr_id,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if pcnr is not None:
        pcnr_value.update({
            "receipt_path": str(pcnr.resolve()),
            "receipt_sha256": file_sha256(pcnr),
        })
    _write(operations / "PCNR_ALTERNATE_4090_RESULT.json", pcnr_value)


def test_complete_frontier_ranks_all_same_host_candidates_and_keeps_base(tmp_path: Path):
    _receipt(tmp_path, "BASE", 0.2)
    _receipt(tmp_path, "OLD", -0.4, positive=False)
    repair = _receipt(tmp_path, "REPAIR", 0.3, suffix="_4090")
    g3 = _receipt(tmp_path, "G3", 0.5, suffix="_4090")
    _pre_frontier(tmp_path, "BASE", "OLD")
    _terminal_results(tmp_path, repair, adam=g3)

    result = materialize_complete_4090_frontier(tmp_path)
    assert result["status"] == STATUS
    assert result["action_priority_candidate_id"] == "G3"
    assert result["priority_alternate_candidate_ids"] == ["REPAIR", "BASE"]
    assert result["rankable_complete_e200_candidate_count"] == 4
    assert {row["candidate_id"] for row in result["ranking"]} == {
        "BASE", "OLD", "REPAIR", "G3",
    }
    assert result["pre_frontier_candidate_retained_in_ranking"] is True
    assert result["canonical_candidate_is_action_priority_only"] is True


def test_complete_frontier_rejects_cross_plain_authority(tmp_path: Path):
    _receipt(tmp_path, "BASE", 0.2)
    _receipt(tmp_path, "OLD", -0.4, positive=False)
    repair = _receipt(tmp_path, "REPAIR", 0.3, suffix="_4090", plain="other")
    _pre_frontier(tmp_path, "BASE", "OLD")
    _terminal_results(tmp_path, repair)
    with pytest.raises(RuntimeError, match="same-host/common-e0"):
        materialize_complete_4090_frontier(tmp_path)


def test_complete_frontier_keeps_evidence_backed_pcnr_replay(tmp_path: Path):
    _receipt(tmp_path, "BASE", 0.2)
    _receipt(tmp_path, "OLD", -0.4, positive=False)
    repair = _receipt(tmp_path, "REPAIR", -0.2, positive=False, suffix="_4090")
    pcnr = _receipt(tmp_path, "PCNR", 0.35, suffix="_4090")
    _pre_frontier(tmp_path, "BASE", "OLD")
    _terminal_results(tmp_path, repair, pcnr=pcnr)

    result = materialize_complete_4090_frontier(tmp_path)
    assert result["action_priority_candidate_id"] == "PCNR"
    assert result["rankable_complete_e200_candidate_count"] == 4
    pcnr_row = next(row for row in result["ranking"] if row["candidate_id"] == "PCNR")
    assert pcnr_row["source_role"] == (
        "pcnr_evidence_backed_alternate_4090_replay"
    )
    assert result["pcnr_alternate_replay"]["candidate_id"] == "PCNR"
