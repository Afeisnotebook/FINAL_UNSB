from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research.local_route1.multi_algorithm_frontier import (
    FRONTIER_SCHEMA,
    HPCGR_ID,
    HPCGR_INVALID_ID,
    PROPOSAL_ID,
    build_hpcgr_implementation,
    materialize_multi_algorithm_frontier,
    select_hpcgr_parent_evidence,
)
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
            "probe": "hnek",
            "case_counts": {"sustainable_on_both_states": 5},
        }],
        "ranked_failure_mechanisms": [{
            "failure_type": "sampling_variance",
            "candidate_generation_eligible": True,
            "evidence_rank": 3,
            "cross_probe_support": 3,
            "supporting_probes": ["dt", "hj", "hnek"],
        }],
    })
    _write(tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl", "{}\n")
    _write(tmp_path / "evidence" / "ANCHOR_TRAJECTORIES.json", {
        "schema": "local-route1-anchor-summary-v1",
        "summaries": [{
            "probe_id": "hnek",
            "complete_e200": True,
            "late_three_mean_macro_psnr_delta": 0.806,
            "late_points_with_four_of_six_positive_domains": 3,
            "trajectory": [{
                "epoch": 200,
                "macro_psnr_delta": 0.425,
                "positive_domains": 5,
                "guardrails_pass": True,
            }],
        }],
    })
    proposal_card = tmp_path / "derive" / "cards" / f"{PROPOSAL_ID}.json"
    trajectory = tmp_path / "candidates" / PROPOSAL_ID / "CANDIDATE_TRAJECTORY.json"
    receipt = (
        tmp_path / "operations" / "terminal_receipts" / f"{PROPOSAL_ID}.json"
    )
    _write(proposal_card, {"candidate_id": PROPOSAL_ID})
    _write(trajectory, {
        "candidate_id": PROPOSAL_ID,
        "late_three_mean_macro_psnr_delta": 0.542,
        "e200_macro_psnr_delta": 0.451,
        "late_points_with_four_of_six_positive_domains": 3,
    })
    _write(receipt, {
        "candidate_id": PROPOSAL_ID,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "algorithm_fingerprint": "proposal-algorithm",
        "trajectory_path": str(trajectory.resolve()),
        "trajectory_sha256": file_sha256(trajectory),
        "derivation_card_sha256": file_sha256(proposal_card),
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "derive" / "HYPOTHESIS_LEDGER.json", {
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [{
            "candidate_id": HPCGR_INVALID_ID,
            "generation": 3,
            "parent_evidence": {"parents": "hnek+proposal"},
            "construction_route": "evidence_qualified_nested_coordinate_estimator",
            "status": "FROZEN_FOR_GATES",
            "algorithm_fingerprint": (
                "cde008e63f69276a407fe5e97ca7defd0751946e123e37df26be9817010fd65e"
            ),
        }],
    })


def test_parent_selection_requires_two_distinct_positive_e200_mechanisms(tmp_path):
    _fixture(tmp_path)
    evidence = select_hpcgr_parent_evidence(tmp_path)
    assert evidence["hnek"]["e200_macro_psnr_delta"] == 0.425
    assert evidence["hnek"]["sustainable_on_both_states"] == 5
    assert evidence["pcrsmg_proposal_only"]["e200_macro_psnr_delta"] == 0.451
    assert evidence["sampling_variance"]["cross_probe_support"] == 3


def test_hpcgr_implementation_binds_all_nested_operator_sources(tmp_path):
    card = tmp_path / "card.json"
    _write(card, {"candidate_id": HPCGR_ID})
    implementation = build_hpcgr_implementation(card)
    assert implementation["model"] == "route1_hpcgr"
    assert implementation["method"]["hpcgr_role"] == "full"
    assert implementation["gate_hook"]["callable"] == "run_hpcgr_gate"
    paths = {row["path"] for row in implementation["source_files"]}
    assert "src/models/route1_hpcgr_model.py" in paths
    assert "src/models/hnek/hnek_search.py" in paths
    assert "src/models/route1/pcrsmg_ablation.py" in paths
    assert all(file_sha256(ROOT / row["path"]) == row["sha256"] for row in implementation["source_files"])


def test_materialization_preserves_a_multi_algorithm_frontier(monkeypatch, tmp_path):
    _fixture(tmp_path)

    def fake_freeze(output_root: Path, candidate_id: str):
        ledger_path = Path(output_root) / "derive" / "HYPOTHESIS_LEDGER.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        next(row for row in ledger["records"] if row["candidate_id"] == candidate_id)[
            "status"
        ] = "FROZEN_FOR_GATES"
        _write(ledger_path, ledger)
        return SimpleNamespace(
            algorithm_fingerprint="hpcgr-algorithm",
            to_dict=lambda: {"candidate_id": candidate_id},
        )

    monkeypatch.setattr(
        "research.local_route1.multi_algorithm_frontier.freeze_candidate_derivation",
        fake_freeze,
    )
    result = materialize_multi_algorithm_frontier(tmp_path)
    assert result["schema"] == FRONTIER_SCHEMA
    assert result["action_priority_candidate_id"] == HPCGR_ID
    assert result["action_priority_is_not_scientific_exclusivity"] is True
    assert len(result["frontier"]) == 6
    statuses = {row["id"]: row["status"] for row in result["frontier"]}
    assert statuses[PROPOSAL_ID] == "POSITIVE_E200_SOURCE_BOUND_PARENT"
    assert statuses[HPCGR_ID] == "FROZEN_FOR_TARGET_BLIND_GATE"
    assert statuses[HPCGR_INVALID_ID].startswith("IMPLEMENTATION_INVALID")
    assert statuses["HJ-CONDITIONAL-GF-RESAMPLING"].endswith("AUDIT_REQUIRED")
