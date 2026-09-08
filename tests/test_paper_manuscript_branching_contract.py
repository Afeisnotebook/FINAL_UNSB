from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_manuscript_contract_binds_theory_baselines_and_novelty_boundary() -> None:
    contract = _load("configs/PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json")
    assert contract["schema"] == "final-unsb-paper-manuscript-branching-contract-v1"
    assert contract["status"] == "PRE_RESULT_STRUCTURE_FROZEN_NO_EMPIRICAL_CLAIM"
    for binding in contract["source_bindings"].values():
        assert _sha256(binding["path"]) == binding["sha256"]

    theory = _load("configs/PAPER_ALGORITHM_THEORY_BUNDLE.json")
    assert contract["method_keys"] == list(theory["methods"])
    baselines = _load("configs/PAPER_BASELINE_PORTFOLIO.json")
    assert contract["main_table_ids"] == [
        row["id"] for row in baselines["core_controlled_main_table"]
    ]


def test_manuscript_contract_covers_every_result_branch_exactly_once() -> None:
    contract = _load("configs/PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json")
    observed = {
        (row["proposal"], row["stcgr"], row["amtnc"])
        for row in contract["branch_matrix"]
    }
    expected = set(itertools.product((False, True), repeat=3))
    assert observed == expected
    assert len(contract["branch_matrix"]) == len(observed) == 8
    assert all(row["route"] for row in contract["branch_matrix"])


def test_manuscript_contract_cannot_predeclare_results_or_open_confirmation() -> None:
    contract = _load("configs/PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json")
    boundaries = contract["hard_boundaries"]
    assert boundaries == {
        "performance_values_read_while_creating_contract": False,
        "full_data_benefit_predeclared": False,
        "unique_winner_predeclared": False,
        "negative_operator_hidden": False,
        "best_checkpoint_selection": False,
        "paired_metrics_control_training_or_scheduling": False,
        "cross_non_equivalent_runtime_delta": False,
        "ddsb_missing_reproduction_called_negative_result": False,
        "hjcgr_deferred_called_mechanism_falsified": False,
        "terminal_singularity_repair_claimed": False,
        "multi_seed_stability_claimed": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    required = set(contract["required_post_result_artifacts"])
    assert "PAPER_ALGORITHM_BASELINE_CLAIM_FREEZE.json" in required
    assert "MANUSCRIPT_RESULT_BRANCH.json" in required
    assert "MANUSCRIPT_TABLES_RECEIPT.json" in required
    assert "MANUSCRIPT_FIGURES_RECEIPT.json" in required
    assert (ROOT / contract["canonical_outline"]).is_file()


def test_manuscript_contract_hash_binds_pre_result_limitations() -> None:
    contract = _load("configs/PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json")
    relative = contract["limitations_statement"]
    binding = contract["source_bindings"]["limitations_statement"]
    assert binding["path"] == relative
    assert _sha256(relative) == binding["sha256"]

    text = (ROOT / relative).read_text(encoding="utf-8").lower()
    required_boundaries = [
        "single",
        "pre-adam",
        "terminal",
        "ddsb",
        "confirmation20",
        "deleted-inode",
        "best-checkpoint",
    ]
    assert all(boundary in text for boundary in required_boundaries)
    assert "no empirical performance claim" in text


def test_live_status_branch_contract_commits_resolve_to_the_bound_blob() -> None:
    def walk(value: object):
        if isinstance(value, dict):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    found = 0
    for relative in (
        "PROJECT_STATE.json",
        "configs/FULL_DATA_METHOD_PORTFOLIO.json",
        "configs/PAPER_DELIVERY_COMPLETION_MATRIX.json",
    ):
        for row in walk(_load(relative)):
            if "branch_contract_commit" not in row:
                continue
            expected = row.get("branch_contract_source_sha256") or row.get(
                "branch_contract_sha256"
            )
            assert isinstance(expected, str)
            blob = subprocess.check_output(
                [
                    "git",
                    "show",
                    f"{row['branch_contract_commit']}:configs/PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json",
                ],
                cwd=ROOT,
            )
            assert hashlib.sha256(blob).hexdigest() == expected
            found += 1
    assert found == 9
