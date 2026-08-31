from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from operations import local_route1_repaired_followup_successor as successor
from research.local_route1.protocol import ROOT


def test_repaired_followup_contract_binds_two_parent_single_candidate_streams(
    monkeypatch, tmp_path,
):
    manifest = tmp_path / "manifest.json"
    environment = tmp_path / "environment.json"
    manifest.write_text("{}", encoding="utf-8")
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
        run_root=tmp_path / "run",
        train_view=tmp_path / "view",
        data_root=tmp_path / "data",
        manifest=manifest,
        python=Path(__import__("sys").executable),
        baseline_environment_record=environment,
        poll_seconds=30,
        timeout_seconds=1209600,
    )
    contract = successor.default_contract(args)
    successor.validate_contract(contract)
    assert contract["maximum_parallel_parent_streams"] == 2
    assert contract["maximum_active_candidates_per_parent_stream"] == 1
    assert contract["within_parent_order"] == [
        "proposal_only", "observable_only",
    ]
    assert contract["action_priority_is_not_an_exclusivity_rule"] is True
    assert contract["selection_seeds"] == [2026]
    assert contract["paired_metric_scheduling"] is False


def test_repaired_followup_source_contract_contains_both_repaired_families():
    source = set(successor.SOURCE_RELATIVES)
    for family in ("rfammcrb", "rfmcrb"):
        assert f"src/models/route1/{family}.py" in source
        assert f"src/models/route1/{family}_ablation.py" in source
        assert f"src/models/route1_{family}_ablation_model.py" in source
