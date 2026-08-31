from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from operations import local_route1_repaired_portfolio_4090_successor as successor
from research.local_route1.protocol import ROOT
from research.local_route1.repaired_replay_portfolio import REPAIRED_IDS


def test_repaired_4090_contract_allows_two_parallel_algorithms(
    monkeypatch, tmp_path,
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
        rfammcrb_repo=ROOT,
        rfmcrb_repo=ROOT,
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
    assert set(contract["source_repos"]) == set(REPAIRED_IDS)
    assert contract["maximum_parallel_replays"] == 2
    assert contract["restart_from_destination_common_e0"] is True
    assert contract["action_priority_is_not_an_exclusivity_rule"] is True
    assert contract["paired_metric_scheduling"] is False


def test_empty_authority_does_not_start_a_candidate(tmp_path: Path) -> None:
    worker = object.__new__(successor.RepairedPortfolio4090Successor)
    worker.run_root = tmp_path
    worker.operations = tmp_path / "operations"
    worker.operations.mkdir()
    worker.wait_authority = lambda: (
        worker.operations / "authority.json",
        {"replay_candidates": []},
    )
    calls = []
    worker.prepare_candidate = lambda *_args: calls.append("prepare")
    states = []
    worker.state = lambda status, **fields: states.append((status, fields))
    assert worker.run() == 0
    assert calls == []
    assert states[-1][0] == "NO_STRICT_OR_NEAR_REPAIRED_4090_REPLAY"

