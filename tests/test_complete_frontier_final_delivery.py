from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from operations import local_route1_complete_frontier_final_successor as successor
from research.local_route1 import complete_frontier_final_delivery as delivery
from research.local_route1.frontier_advancement import STRICT
from research.local_route1.protocol import ROOT, file_sha256


def _write(path: Path, value: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return path


def _frontiers() -> tuple[dict, dict]:
    frontier = {
        "ranking": [
            {
                "rank": 1,
                "candidate_id": delivery.PCRSMG_PROPOSAL,
                "classification": STRICT,
                "trajectory_status": "positive",
                "source_role": "pre_frontier_4090",
                "ranking_fields": {"late_three_mean_macro_psnr_delta": 0.5},
            },
            {
                "rank": 2,
                "candidate_id": delivery.RFAMMCRB,
                "classification": STRICT,
                "trajectory_status": "positive",
                "source_role": "repaired_4090_replay",
                "ranking_fields": {"late_three_mean_macro_psnr_delta": 0.4},
            },
            {
                "rank": 3,
                "candidate_id": "NEGATIVE-CURRENT-OPERATOR",
                "classification": "closed_current_protocol",
                "trajectory_status": "negative",
                "source_role": "pre_frontier_4090",
                "ranking_fields": {"late_three_mean_macro_psnr_delta": -0.2},
            },
        ],
        "evidence_preserved_candidate_ids": [
            delivery.PCRSMG_PROPOSAL, delivery.RFAMMCRB,
        ],
    }
    source = {
        "ranking": [
            {
                "rank": 1,
                "candidate_id": delivery.RFMCRB,
                "classification": "evidence_backed_alternate",
                "trajectory_status": "positive",
                "ranking_fields": {"late_three_mean_macro_psnr_delta": 0.1},
            },
        ],
        "evidence_preserved_candidate_ids": [delivery.RFMCRB],
        "observable_only_candidate_ids_excluded_from_ranking": ["OBS"],
        "parent_ablation_results": [{"parent_candidate_id": delivery.RFMCRB}],
    }
    return frontier, {
        "extended_adjudication": source,
        "candidate_evidence": [{"candidate_id": delivery.RFMCRB}],
    }


def test_research_frontier_keeps_action_priority_and_independent_directions():
    frontier, portable = _frontiers()
    value = delivery._research_frontier(
        frontier, portable, delivery.PCRSMG_PROPOSAL,
    )
    assert value["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert value["remote4090_advanceable_candidate_ids"] == [
        delivery.PCRSMG_PROPOSAL, delivery.RFAMMCRB,
    ]
    assert value["remote5090_mechanism_bearing_candidate_ids"] == [delivery.RFMCRB]
    assert value["remote5090_observable_negative_control_ids"] == ["OBS"]
    assert value["remote4090_same_host_frontier"][1][
        "frontier_disposition"
    ] == "co_leading_strict_frontier"


def test_mechanism_evidence_is_bound_to_selected_algorithm_family(tmp_path: Path):
    roles = {
        "proposal_only": {"candidate_id": delivery.PCRSMG_PROPOSAL},
        "observable_only": {"candidate_id": "OBS"},
        "projected_or_full": {"candidate_id": "FULL"},
    }
    ablation = _write(tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json", {
        "roles": roles,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    pc = delivery._mechanism_evidence(
        tmp_path, delivery.PCRSMG_PROPOSAL, {"ranking": []},
        {"extended_adjudication": {"parent_ablation_results": []}},
    )
    assert pc["same_host_as_selection"] is True
    assert pc["sha256"] == file_sha256(ablation)

    parent = {
        "parent_candidate_id": delivery.RFAMMCRB,
        "roles": {"proposal_only": {}, "observable_only": {}, "projected_or_full": {}},
    }
    portable = {"extended_adjudication": {"parent_ablation_results": [parent]}}
    rf = delivery._mechanism_evidence(
        tmp_path, delivery.RFAMMCRB, {"ranking": []}, portable,
    )
    assert rf["same_host_as_selection"] is False
    assert rf["used_for_4090_candidate_ranking"] is False

    frontier = {
        "ranking": [
            {"candidate_id": delivery.PCRSMG_PROPOSAL},
            {"candidate_id": delivery.RFAMMCRB},
            {"candidate_id": delivery.G3_ADAM},
        ],
    }
    g3 = delivery._mechanism_evidence(
        tmp_path, delivery.G3_ADAM, frontier, portable,
    )
    assert set(g3["components"]) == {
        "plain", "conditional_sampling_only",
        "residual_feasible_barrier_only", "combined_full",
    }
    assert g3["used_for_cross_host_delta_ranking"] is False

    with pytest.raises(RuntimeError, match="lacks a frozen mechanism-evidence route"):
        delivery._mechanism_evidence(tmp_path, "UNKNOWN", frontier, portable)


def test_complete_delivery_publishes_multi_candidate_frontier_atomically(
    monkeypatch, tmp_path: Path,
):
    operations = tmp_path / "operations"
    final = tmp_path / "final"
    for name in delivery.LEGACY_FINAL_FILES:
        _write(final / name, "old report" if name.endswith(".md") else {"old": name})

    receipt_path = _write(tmp_path / "receipt.json", {"candidate_id": delivery.PCRSMG_PROPOSAL})
    trajectory_path = _write(tmp_path / "trajectory.json", {"candidate_id": delivery.PCRSMG_PROPOSAL})
    card_path = _write(tmp_path / "derive" / "cards" / f"{delivery.PCRSMG_PROPOSAL}.json", {
        "candidate_id": delivery.PCRSMG_PROPOSAL,
        "name": "proposal",
        "formula": "x",
    })
    implementation = {"candidate_id": delivery.PCRSMG_PROPOSAL, "model": "m", "method": {}}
    implementation_path = _write(
        tmp_path / "derive" / "implementations" / f"{delivery.PCRSMG_PROPOSAL}.json",
        implementation,
    )
    executor_path = _write(operations / "executor.json", {"candidate_id": delivery.PCRSMG_PROPOSAL})
    frontier_path = _write(
        operations / "COMPLETE_FRONTIER_4090_ADJUDICATION.json",
        {"source": "4090"},
    )
    portable_path = _write(
        operations / "PORTABLE_EXTENDED_REPAIRED_FRONTIER_5090.json",
        {"source": "5090"},
    )
    frontier, portable = _frontiers()
    frontier.update({
        "action_priority_candidate_id": delivery.PCRSMG_PROPOSAL,
        "priority_alternate_candidate_ids": [delivery.RFAMMCRB, "NEGATIVE-CURRENT-OPERATOR"],
    })
    receipt = {
        "candidate_id": delivery.PCRSMG_PROPOSAL,
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "candidate",
        "candidate_training_core_fingerprint": "core",
        "training_git_commit": "a" * 40,
        "trajectory_status": "positive",
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": 0.5,
            "e200_macro_psnr_delta": 0.4,
        },
    }
    executor = {
        "candidate_git_commit": "a" * 40,
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "candidate",
    }
    monkeypatch.setattr(delivery, "_complete_frontier", lambda _root: (frontier, frontier_path))
    monkeypatch.setattr(delivery, "_portable_frontier", lambda _root: (portable, portable_path))
    monkeypatch.setattr(delivery, "_selected_source", lambda _root, _row: (
        receipt, receipt_path, {"candidate_id": delivery.PCRSMG_PROPOSAL},
        trajectory_path, {"candidate_id": delivery.PCRSMG_PROPOSAL, "name": "proposal"},
        card_path, implementation,
    ))
    monkeypatch.setattr(delivery, "_candidate_domain_trajectory", lambda *_args: [])
    monkeypatch.setattr(delivery, "_mechanism_evidence", lambda *_args: {"kind": "test"})
    monkeypatch.setattr(delivery, "_executor_contract", lambda *_args: (executor_path, executor))
    monkeypatch.setattr(delivery, "_median_epoch_seconds", lambda *_args: 1.0)

    pointer = delivery.materialize_complete_frontier_final_delivery(tmp_path)
    research = json.loads((final / "RESEARCH_FRONTIER.json").read_text(encoding="utf-8"))
    assert pointer["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert pointer["research_frontier_unique_candidate_count"] == 4
    assert pointer["research_frontier_host_scoped_row_count"] == 4
    assert research["remote4090_advanceable_candidate_ids"] == [
        delivery.PCRSMG_PROPOSAL, delivery.RFAMMCRB,
    ]
    assert set(pointer["final_file_sha256"]) == set(delivery.PUBLISHED_FILES)
    assert file_sha256(final / "RESEARCH_FRONTIER.json") == pointer[
        "final_file_sha256"
    ]["RESEARCH_FRONTIER.json"]
    assert delivery.materialize_complete_frontier_final_delivery(tmp_path) == pointer
    _write(frontier_path, {"source": "changed"})
    with pytest.raises(RuntimeError, match="final input changed"):
        delivery.materialize_complete_frontier_final_delivery(tmp_path)


def test_complete_final_successor_contract_preserves_full_frontier(
    monkeypatch, tmp_path: Path,
):
    def fake_run_text(command, *, cwd):
        if command[:2] == ["git", "status"]:
            return ""
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        raise AssertionError(command)

    monkeypatch.setattr(successor.support, "run_text", fake_run_text)
    contract = successor.default_contract(Namespace(
        repo=ROOT,
        run_root=tmp_path,
        poll_seconds=60,
        timeout_seconds=1209600,
    ))
    successor.validate_contract(contract)
    assert contract["requires_complete_same_host_4090_frontier"] is True
    assert contract["requires_complete_portable_5090_mechanism_frontier"] is True
    assert contract["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert contract["all_mechanism_bearing_candidates_preserved"] is True
    assert contract["cross_host_deltas_merged"] is False
