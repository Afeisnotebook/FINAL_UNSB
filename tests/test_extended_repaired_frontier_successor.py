from __future__ import annotations

from argparse import Namespace

from operations import local_route1_extended_repaired_frontier_successor as successor
from research.local_route1.protocol import ROOT


def test_extended_repaired_successor_contract_keeps_full_multi_candidate_frontier(
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
    assert contract["observable_only_excluded_from_candidate_ranking"] is True
    assert contract["canonical_candidate_is_action_priority_only"] is True
    assert contract["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert contract["cross_host_deltas_merged"] is False
