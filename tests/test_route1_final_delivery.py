import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.local_route1.final_delivery import (
    POSITIVE_ADJUDICATION,
    materialize_final_delivery,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.seed_validation import MULTI_SEED_ADJUDICATION_SCHEMA


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _metric(psnr: float, protocol: str = "crn") -> dict:
    domains = {
        f"d{index}": {"psnr": psnr + index / 100.0, "ssim": 0.7, "lpips": 0.2}
        for index in range(6)
    }
    return {
        "schema": "final-unsb-route1-evaluation-v1",
        "count_per_domain": 70,
        "protocol_fingerprint": protocol,
        "evaluation_input_sha256": "inputs",
        "macro_psnr": sum(row["psnr"] for row in domains.values()) / 6,
        "macro_ssim": 0.7,
        "macro_lpips": 0.2,
        "domains": domains,
        "confirmation20_opened": False,
    }


def _trajectory(candidate_id: str, algorithm: str, delta: float) -> dict:
    return {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "status": "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": algorithm,
        "candidate_fingerprint": f"execution-{candidate_id}",
        "trajectory": [{"epoch": 200}],
        "late_three_mean_macro_psnr_delta": delta,
        "e200_macro_psnr_delta": delta,
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }


def test_final_delivery_refuses_incomplete_adjudication(tmp_path):
    with pytest.raises(RuntimeError, match="blocked until"):
        materialize_final_delivery(tmp_path)
    _write(tmp_path / "operations" / "GENERATION1_E200_ADJUDICATION.json", {
        "status": "WAITING_FOR_ALL_MATCHED_E200_TRAJECTORIES",
        "confirmation20_opened": False,
    })
    with pytest.raises(RuntimeError, match="refuses incomplete"):
        materialize_final_delivery(tmp_path)


def test_final_delivery_materializes_complete_positive_path(monkeypatch, tmp_path):
    ids = ("G1-FIRST", "G1-SECOND")
    algorithms = {ids[0]: "algorithm-first", ids[1]: "algorithm-second"}
    registrations = {}
    for candidate_id in ids:
        card = tmp_path / "cards" / f"{candidate_id}.json"
        implementation = tmp_path / "implementations" / f"{candidate_id}.json"
        _write(card, {
            "name": f"name-{candidate_id}", "formula": "operator formula",
            "unsb_object": "native UNSB object",
            "identity_or_unbiased_condition": "identity",
            "target_inaccessibility_proof": "unpaired only",
            "objective_change": False, "estimator_change": True,
            "coordinate_change": False, "endpoint_law_change": False,
            "algorithm_hyperparameters": {"fixed": 1},
            "ablation_definitions": {
                "proposal_only": "proposal", "observable_only": "observe",
                "projected_or_full": "full",
            },
            "parent_evidence": {"failure_type": "test"},
            "causal_matrix_sha256": "matrix",
            "reversal_atlas_sha256": "atlas",
            "compute_cost": "two views", "memory_cost": "one accumulator",
            "recovery_state_cost": "RNG",
        })
        _write(implementation, {
            "source_files": [], "model": "test_model", "method": {"fixed": 1},
        })
        registrations[candidate_id] = SimpleNamespace(
            algorithm_fingerprint=algorithms[candidate_id],
            candidate_fingerprint=f"execution-{candidate_id}",
            card_path=card, implementation_path=implementation,
        )
        trajectory_path = tmp_path / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        _write(trajectory_path, _trajectory(candidate_id, algorithms[candidate_id], 0.2))
        _write(tmp_path / "candidates" / candidate_id / "metrics" / "e200.json", _metric(20.2))
        _write(tmp_path / "anchors" / "plain" / "metrics" / "e200.json", _metric(20.0))
        (tmp_path / "candidates" / candidate_id / "TRAIN_TRACE.jsonl").write_text(
            json.dumps({"epoch_wall_seconds": 10.0}) + "\n", encoding="utf-8",
        )

    monkeypatch.setattr(
        "research.local_route1.final_delivery.load_candidate_registration",
        lambda output_root, candidate_id, require_gate: registrations[candidate_id],
    )
    ranking = []
    for rank, candidate_id in enumerate(ids, start=1):
        trajectory_path = tmp_path / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        ranking.append({
            "rank": rank, "candidate_id": candidate_id,
            "trajectory_sha256": file_sha256(trajectory_path),
            "trajectory_status": "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
        })
    _write(tmp_path / "operations" / "GENERATION1_E200_ADJUDICATION.json", {
        "status": POSITIVE_ADJUDICATION,
        "selected_candidate_id": ids[0], "winner_frozen_for_seed2027": True,
        "ranking": ranking,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "candidates" / ids[0] / "MULTI_SEED_ADJUDICATION.json", {
        "schema": MULTI_SEED_ADJUDICATION_SCHEMA,
        "status": "ROUTE1_SUSTAINED_LOCAL", "candidate_id": ids[0],
        "algorithm_fingerprint": algorithms[ids[0]], "included_seeds": [2026, 2027],
        "paired_metric_changed_algorithm": False, "confirmation20_opened": False,
    })
    _write(tmp_path / "seed_validation" / "seed2027" / "SEED_VALIDATION_SUMMARY.json", {
        "status": "COMPLETE", "trajectory": [{"epoch": 200}],
        "paired_metric_changed_algorithm": False, "confirmation20_opened": False,
    })
    _write(tmp_path / "seed_validation" / "seed2027" / "candidate" / "metrics" / "e200.json", _metric(20.3, "seed-crn"))
    _write(tmp_path / "seed_validation" / "seed2027" / "plain" / "metrics" / "e200.json", _metric(20.0, "seed-crn"))

    result = materialize_final_delivery(tmp_path)
    assert result["status"] == "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE"
    assert result["classification"] == "route1_sustained_local"
    assert result["candidate_id"] == ids[0]
    assert result["executable_configuration"]["model"] == "test_model"
    assert result["selected_fixed_checkpoint"] == {
        "data_epoch": 200, "best_checkpoint_selection": False,
    }
    assert (tmp_path / "final" / "CANDIDATE.json").is_file()
    assert (tmp_path / "final" / "FINAL_ROUTE1_REPORT.md").is_file()
    results = json.loads((tmp_path / "final" / "RESULTS.json").read_text())
    assert set(results["candidate_results"]) == set(ids)
    assert len(results["candidate_results"][ids[0]]["absolute_relative_domain_trajectory"][0]["domains"]) == 6
    alternates = json.loads((tmp_path / "final" / "ALTERNATES.json").read_text())
    assert len(alternates["alternates"]) == 2
