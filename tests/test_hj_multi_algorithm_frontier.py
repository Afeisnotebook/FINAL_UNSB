from __future__ import annotations

import json
from pathlib import Path

from research.local_route1.hj_multi_algorithm_frontier import (
    HJCGR_ID,
    build_hjcgr_implementation,
    select_hjcgr_parent_evidence,
)
from research.local_route1.multi_algorithm_frontier import PROPOSAL_ID
from research.local_route1.protocol import ROOT, file_sha256


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
    else:
        path.write_text(json.dumps(payload), encoding="utf-8")


def _fixture(tmp_path: Path) -> None:
    _write(tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json", {
        "status": "COMPLETE_CAUSAL_AUDIT",
        "probe_summaries": [{
            "probe": "hj",
            "case_counts": {"sustainable_on_both_states": 2},
            "next_batch_consensus_negative_rows": 22,
        }],
        "sampling_variance_summaries": [{
            "probe": "hj",
            "axes": {
                "independent_unpaired_batch": {
                    "rows": 22,
                    "variance_dominated_rows": 22,
                    "mean_variance_fraction": 0.872739,
                },
                "latent_time_bridge_rng": {
                    "rows": 22,
                    "variance_dominated_rows": 22,
                    "mean_variance_fraction": 0.872820,
                },
            },
        }],
    })
    _write(tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl", "{}\n")
    _write(tmp_path / "evidence" / "ANCHOR_TRAJECTORIES.json", {
        "schema": "local-route1-anchor-summary-v1",
        "summaries": [{
            "probe_id": "hj",
            "complete_e200": True,
            "late_three_mean_macro_psnr_delta": 0.649,
            "late_points_with_four_of_six_positive_domains": 3,
            "trajectory": [{
                "epoch": 200,
                "macro_psnr_delta": 0.159,
                "positive_domains": 4,
                "guardrails_pass": True,
            }],
        }],
    })
    card = tmp_path / "derive" / "cards" / f"{PROPOSAL_ID}.json"
    trajectory = tmp_path / "candidates" / PROPOSAL_ID / "CANDIDATE_TRAJECTORY.json"
    receipt = tmp_path / "operations" / "terminal_receipts" / f"{PROPOSAL_ID}.json"
    _write(card, {"candidate_id": PROPOSAL_ID})
    _write(trajectory, {
        "candidate_id": PROPOSAL_ID,
        "late_three_mean_macro_psnr_delta": 0.542,
        "e200_macro_psnr_delta": 0.451,
        "late_points_with_four_of_six_positive_domains": 3,
    })
    _write(receipt, {
        "candidate_id": PROPOSAL_ID,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "algorithm_fingerprint": "proposal",
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": file_sha256(trajectory),
        "derivation_card_sha256": file_sha256(card),
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })


def test_hjcgr_parent_gate_requires_positive_hj_and_dominated_variance(tmp_path):
    _fixture(tmp_path)
    evidence = select_hjcgr_parent_evidence(tmp_path)
    assert evidence["hj"]["e200_macro_psnr_delta"] == 0.159
    assert evidence["hj"]["next_batch_consensus_negative_rows"] == 22
    assert evidence["hj_sampling_variance"][
        "independent_batch_variance_dominated_rows"
    ] == 22
    assert evidence["pcrsmg_proposal_only"]["e200_macro_psnr_delta"] == 0.451


def test_hjcgr_implementation_binds_objective_estimator_and_gate_sources(tmp_path):
    card = tmp_path / "card.json"
    _write(card, {
        "algorithm_hyperparameters": {
            "route1_hjcgr_enable": True,
            "hjcgr_role": "full",
            "hj_enable": True,
        },
    })
    implementation = build_hjcgr_implementation(card)
    assert implementation["candidate_id"] == HJCGR_ID
    assert implementation["model"] == "route1_hjcgr"
    assert implementation["gate_hook"]["callable"] == "run_hjcgr_gate"
    paths = {row["path"] for row in implementation["source_files"]}
    assert "src/models/route1_hjcgr_model.py" in paths
    assert "src/models/hj/model.py" in paths
    assert "src/models/route1/pcrsmg_ablation.py" in paths
    assert "research/local_route1/generation1_gates.py" in paths
    assert all(file_sha256(ROOT / row["path"]) == row["sha256"] for row in implementation["source_files"])

