from __future__ import annotations

from argparse import Namespace

from operations import local_route1_complete_frontier_4090_successor as successor
from research.local_route1.protocol import ROOT


def test_complete_frontier_successor_contract_waits_for_all_terminal_branches(
    monkeypatch, tmp_path,
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
    assert contract["requires_complete_repaired_portfolio"] is True
    assert contract[
        "requires_both_generation3_terminal_results_even_if_inapplicable"
    ] is True
    assert contract["canonical_candidate_is_action_priority_only"] is True
    assert contract["cross_host_deltas_merged"] is False
    assert contract["selection_seeds"] == [2026]
