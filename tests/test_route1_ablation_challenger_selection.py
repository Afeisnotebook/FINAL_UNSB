import json

import pytest

from operations.local_route1_ablation_challenger_successor import WORKSPACE_SCHEMA
from operations.local_route1_winner_ablation_adjudicate import SCHEMA as ABLATION_SCHEMA
from research.local_route1.ablation_challenger_selection import (
    SCHEMA,
    adjudicate_ablation_challenger_selection,
)
from research.local_route1.seed_validation import MULTI_SEED_ADJUDICATION_SCHEMA


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _multi(candidate_id, algorithm, *, sustained, psnr):
    return {
        "schema": MULTI_SEED_ADJUDICATION_SCHEMA,
        "status": "ROUTE1_SUSTAINED_LOCAL" if sustained else "MULTI_SEED_NOT_SUSTAINED",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": algorithm,
        "included_seeds": [2026, 2027],
        "combined_late_three_mean_macro_psnr_delta": psnr,
        "combined_late_average_positive_domains": 4.5,
        "combined_late_average_worst_domain_delta": -0.2,
        "algorithm_changes_after_seed2026_freeze": False,
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }


def _fixture(tmp_path, *, full_sustained=True, challenger_sustained=True):
    root = tmp_path / "run"
    workspace = root / "ablation_challenger_seed_validation" / "ABL-PROPOSAL"
    ablation = {
        "schema": ABLATION_SCHEMA,
        "status": "ABLATION_CHALLENGER_REQUIRES_FROZEN_SEED_VALIDATION",
        "roles": {
            "proposal_only": {
                "candidate_id": "ABL-PROPOSAL",
                "algorithm_fingerprint": "algorithm-proposal",
            },
            "projected_or_full": {
                "candidate_id": "G1-FULL",
                "algorithm_fingerprint": "algorithm-full",
            },
        },
        "proposal_only_out_ranks_full": True,
        "selection_change_blocked_pending_seed_validation": True,
        "selection_changed": False,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    _write(root / "operations" / "WINNER_ABLATION_ADJUDICATION.json", ablation)
    _write(workspace / "CHALLENGER_SEED_WORKSPACE.json", {
        "schema": WORKSPACE_SCHEMA,
        "source_root": str(root.resolve()),
        "workspace_root": str(workspace.resolve()),
        "candidate_id": "ABL-PROPOSAL",
        "full_winner_seed_namespace_reused": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(root / "candidates" / "G1-FULL" / "MULTI_SEED_ADJUDICATION.json",
           _multi("G1-FULL", "algorithm-full", sustained=full_sustained, psnr=0.2))
    _write(workspace / "candidates" / "ABL-PROPOSAL" / "MULTI_SEED_ADJUDICATION.json",
           _multi("ABL-PROPOSAL", "algorithm-proposal",
                  sustained=challenger_sustained, psnr=0.3))
    return root, workspace


def test_ablation_challenger_selection_prefers_sustained_then_late_gain(tmp_path):
    root, workspace = _fixture(tmp_path)
    result = adjudicate_ablation_challenger_selection(root, workspace)
    assert result["schema"] == SCHEMA
    assert result["status"] == "CHALLENGER_SELECTED_AFTER_FROZEN_SEEDS"
    assert result["selected_candidate_id"] == "ABL-PROPOSAL"
    assert result["selection_changed_before_both_seed_adjudications"] is False

    root, workspace = _fixture(
        tmp_path / "status-priority", full_sustained=True,
        challenger_sustained=False,
    )
    result = adjudicate_ablation_challenger_selection(root, workspace)
    assert result["status"] == "FULL_WINNER_RETAINED_AFTER_CHALLENGER_SEEDS"


def test_ablation_challenger_selection_rejects_algorithm_mutation(tmp_path):
    root, workspace = _fixture(tmp_path)
    path = workspace / "candidates" / "ABL-PROPOSAL" / "MULTI_SEED_ADJUDICATION.json"
    value = json.loads(path.read_text())
    value["algorithm_changes_after_seed2026_freeze"] = True
    _write(path, value)
    with pytest.raises(RuntimeError, match="algorithm changed"):
        adjudicate_ablation_challenger_selection(root, workspace)
