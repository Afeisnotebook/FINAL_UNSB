from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from operations import local_route1_cross_runtime_5090_successor as successor
from research.local_route1.complete_frontier import (
    SCHEMA as FRONTIER_SCHEMA,
    STATUS as FRONTIER_STATUS,
)
from research.local_route1.cross_runtime_portfolio import (
    AMTNC_ID,
    PCRSMG_PROPOSAL_ID,
    REPLAY_IDS,
    SCHEMA,
    SOURCE_PARENT_ID,
    STATUS,
    _canonical_json_sha256,
    validate_portable_cross_runtime_portfolio,
)
from research.local_route1.frontier_advancement import NEAR, STRICT
from research.local_route1.protocol import ROOT


def _authority() -> dict:
    parent = {"candidate_id": SOURCE_PARENT_ID}
    parent_sha = _canonical_json_sha256(parent)
    ranking = []
    evidence = []
    for candidate_id, classification in (
        (PCRSMG_PROPOSAL_ID, STRICT),
        (AMTNC_ID, NEAR),
    ):
        receipt = {
            "candidate_id": candidate_id,
            "algorithm_fingerprint": f"algorithm-{candidate_id}",
            "candidate_fingerprint": f"candidate-{candidate_id}",
        }
        receipt_sha = _canonical_json_sha256(receipt)
        card = {
            "candidate_id": candidate_id,
            **(
                {"parent_terminal_receipt_sha256": parent_sha}
                if candidate_id == PCRSMG_PROPOSAL_ID else {}
            ),
        }
        implementation = {
            "candidate_id": candidate_id,
            "training_target_access": "unpaired_only",
            "paired_controller_access": False,
        }
        trajectory = {"candidate_id": candidate_id}
        ledger = {
            "candidate_id": candidate_id,
            "status": "FROZEN_FOR_GATES",
        }
        ranking.append({
            "candidate_id": candidate_id,
            "classification": classification,
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "candidate_fingerprint": receipt["candidate_fingerprint"],
            "receipt_sha256": receipt_sha,
        })
        evidence.append({
            "candidate_id": candidate_id,
            "receipt": receipt,
            "receipt_sha256": receipt_sha,
            "trajectory": trajectory,
            "trajectory_sha256": _canonical_json_sha256(trajectory),
            "derivation_card": card,
            "derivation_card_sha256": _canonical_json_sha256(card),
            "implementation": implementation,
            "implementation_sha256": _canonical_json_sha256(implementation),
            "source_ledger_record": ledger,
            "source_ledger_record_sha256": _canonical_json_sha256(ledger),
        })
    frontier = {
        "schema": FRONTIER_SCHEMA,
        "status": FRONTIER_STATUS,
        "ranking": ranking,
        "evidence_preserved_candidate_ids": list(REPLAY_IDS),
    }
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "source_complete_4090_frontier": frontier,
        "source_frontier_sha256": _canonical_json_sha256(frontier),
        "candidate_evidence": evidence,
        "portable_dependencies": {
            "pcrsmg_parent_terminal_receipt": parent,
            "pcrsmg_parent_terminal_receipt_sha256": parent_sha,
        },
        "replay_candidate_ids": list(REPLAY_IDS),
        "maximum_parallel_replays": 2,
        "complete_source_e200_only": True,
        "checkpoint_transfer": False,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_portfolio_preserves_strict_and_independent_near_candidate() -> None:
    value = validate_portable_cross_runtime_portfolio(_authority())
    assert value["replay_candidate_ids"] == list(REPLAY_IDS)
    assert value["maximum_parallel_replays"] == 2


def test_portfolio_rejects_closed_amtnc() -> None:
    value = _authority()
    value["source_complete_4090_frontier"]["ranking"][1][
        "classification"
    ] = "closed_current_operator"
    value["source_frontier_sha256"] = _canonical_json_sha256(
        value["source_complete_4090_frontier"]
    )
    with pytest.raises(RuntimeError, match="AM-TNC"):
        validate_portable_cross_runtime_portfolio(value)


def test_5090_contract_keeps_two_parallel_batch1_replays(
    monkeypatch, tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.csv"
    environment = tmp_path / "environment.json"
    manifest.write_text("x", encoding="utf-8")
    environment.write_text("{}", encoding="utf-8")

    def fake_run_text(command, *, cwd):
        if command[:2] == ["git", "status"]:
            return ""
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        raise AssertionError(command)

    monkeypatch.setattr(successor.support, "run_text", fake_run_text)
    args = Namespace(
        repo=ROOT,
        pcrsmg_proposal_repo=ROOT,
        amtnc_repo=ROOT,
        run_root=tmp_path / "run",
        authority=tmp_path / "future_authority.json",
        train_view=tmp_path / "view",
        data_root=tmp_path / "data",
        manifest=manifest,
        python=Path(__import__("sys").executable),
        baseline_environment_record=environment,
        poll_seconds=60,
        timeout_seconds=1209600,
    )
    contract = successor.default_contract(args)
    successor.validate_contract(contract)
    assert set(contract["source_repos"]) == set(REPLAY_IDS)
    assert contract["maximum_parallel_replays"] == 2
    assert contract["batch_size"] == 1
    assert contract["target_data_epochs"] == 200
