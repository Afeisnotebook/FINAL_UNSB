from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from operations import local_route1_repaired_replay_export_successor as successor
from research.local_route1.protocol import ROOT


def test_repaired_replay_export_contract_preserves_two_algorithm_portfolio(
    monkeypatch, tmp_path,
) -> None:
    def fake_run_text(command, *, cwd):
        if command[:2] == ["git", "status"]:
            return ""
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        raise AssertionError(command)

    monkeypatch.setattr(successor.support, "run_text", fake_run_text)
    args = Namespace(
        repo=ROOT,
        run_root=tmp_path / "run",
        poll_seconds=60,
        timeout_seconds=1209600,
    )
    contract = successor.default_contract(args)
    successor.validate_contract(contract)
    assert contract["maximum_4090_replays"] == 2
    assert contract["action_priority_is_not_an_exclusivity_rule"] is True
    assert contract["paired_metric_scheduling"] is False
    assert contract["cross_host_deltas_merged"] is False


def test_repaired_replay_export_runs_only_after_adjudication(
    monkeypatch, tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    operations = run_root / "operations"
    operations.mkdir(parents=True)
    adjudication = operations / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
    adjudication.write_text("{}", encoding="utf-8")
    worker = object.__new__(successor.RepairedReplayExportSuccessor)
    worker.run_root = run_root
    worker.operations = operations
    worker.state_path = operations / "state.json"
    worker.contract = {"timeout_seconds": 1209600, "poll_seconds": 60}
    worker.started = 0.0
    states = []
    worker.state = lambda status, **fields: states.append((status, fields))
    output = operations / "REPAIRED_4090_REPLAY_PORTFOLIO.json"

    def fake_export(*_args, **kwargs):
        kwargs["output_path"].write_text("{}", encoding="utf-8")
        return {"replay_candidates": [{"candidate_id": "A"}, {"candidate_id": "B"}]}

    monkeypatch.setattr(successor, "export_portable_authority", fake_export)
    monkeypatch.setattr(successor.support, "file_sha256", lambda path: "sha")
    assert worker.run() == 0
    assert output.is_file()
    assert states[-1][0] == "REPAIRED_4090_REPLAY_PORTFOLIO_READY"
    assert states[-1][1]["replay_candidate_ids"] == ["A", "B"]

