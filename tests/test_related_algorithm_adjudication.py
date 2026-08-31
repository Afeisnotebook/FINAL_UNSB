from __future__ import annotations

import json
from pathlib import Path

from research.local_route1.protocol import file_sha256
from research.local_route1.related_algorithm_adjudication import (
    COMBINED_SCHEMA,
    HOST_SCHEMA,
    adjudicate_related_host,
    combine_related_hosts,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(
    root: Path, candidate_id: str, algorithm: str, *, late: float, e200: float,
    lpips: float = -0.01,
) -> None:
    card = root / "derive" / "cards" / f"{candidate_id}.json"
    implementation = root / "derive" / "implementations" / f"{candidate_id}.json"
    trajectory = root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    receipt = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    _write(card, {"candidate_id": candidate_id})
    _write(implementation, {"candidate_id": candidate_id})
    _write(trajectory, {
        "candidate_id": candidate_id,
        "late_three_mean_macro_psnr_delta": late,
        "e200_macro_psnr_delta": e200,
        "late_points_with_four_of_six_positive_domains": 3,
        "late_average_worst_domain_delta": -0.2,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": lpips,
        "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
        "maximum_allowed_rolling_drawdown_db": 0.3,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    })
    _write(receipt, {
        "candidate_id": candidate_id,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": file_sha256(trajectory),
        "derivation_card_sha256": file_sha256(card),
        "implementation_sha256": file_sha256(implementation),
        "algorithm_fingerprint": algorithm,
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "training_git_commit": "a" * 40,
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "median_epoch_wall_seconds": 10.0,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })


def test_host_adjudication_preserves_multiple_strict_algorithms(tmp_path):
    _candidate(tmp_path, "A", "algorithm-a", late=0.5, e200=0.3)
    _candidate(tmp_path, "B", "algorithm-b", late=0.2, e200=0.4)
    output = tmp_path / "host.json"
    result = adjudicate_related_host(
        tmp_path, host_label="4090", candidate_ids=["A", "B"], output_path=output,
    )
    assert result["schema"] == HOST_SCHEMA
    assert result["strict_sustained_candidate_ids"] == ["A", "B"]
    assert result["action_priority_candidate_id"] == "A"
    assert result["action_priority_is_not_scientific_exclusivity"] is True
    assert result["algorithm_discovery_collapsed_to_single_candidate"] is False


def test_guardrail_failure_is_positive_but_fragile_not_erased(tmp_path):
    _candidate(tmp_path, "A", "algorithm-a", late=0.5, e200=0.3, lpips=0.02)
    result = adjudicate_related_host(
        tmp_path,
        host_label="5090",
        candidate_ids=["A"],
        output_path=tmp_path / "host.json",
    )
    assert result["positive_but_fragile_candidate_ids"] == ["A"]
    assert result["ranking"][0]["strict_checks"]["lpips_guardrail"] is False


def test_cross_host_combination_never_averages_deltas_or_collapses_algorithms(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _candidate(first_root, "A", "algorithm-a", late=0.5, e200=0.3)
    _candidate(first_root, "B", "algorithm-b", late=0.2, e200=0.1)
    _candidate(second_root, "A2", "algorithm-a", late=0.4, e200=0.2)
    first = tmp_path / "4090.json"
    second = tmp_path / "5090.json"
    adjudicate_related_host(
        first_root, host_label="4090", candidate_ids=["A", "B"], output_path=first,
    )
    adjudicate_related_host(
        second_root, host_label="5090", candidate_ids=["A2"], output_path=second,
    )
    result = combine_related_hosts(
        [first, second], output_path=tmp_path / "combined.json",
    )
    assert result["schema"] == COMBINED_SCHEMA
    assert result["status"] == "MULTIPLE_VIABLE_ALGORITHMS"
    assert result["viable_algorithm_count"] == 2
    assert result["cross_host_deltas_merged"] is False
    shared = next(
        row for row in result["algorithms"]
        if row["algorithm_fingerprint"] == "algorithm-a"
    )
    assert shared["cross_runtime_positive"] is True
    assert len(shared["host_results"]) == 2
    assert "mean_delta" not in shared

