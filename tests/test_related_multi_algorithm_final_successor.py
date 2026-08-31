from __future__ import annotations

import json
from pathlib import Path

from operations import local_route1_related_multi_algorithm_final_successor as successor


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_successor_requires_every_related_result(monkeypatch, tmp_path: Path):
    operations = tmp_path / "operations"
    for name in successor.REQUIRED_RESULTS:
        _write(operations / name, {"status": "complete"})
    contract = tmp_path / "contract.json"
    _write(contract, {
        "schema": successor.SCHEMA,
        "repo": str(tmp_path),
        "git_commit": "commit",
        "source_sha256": {},
        "run_root": str(tmp_path),
        "poll_seconds": 60,
        "timeout_seconds": 43200,
        "requires_all_related_e200_branches": True,
        "requires_host_separated_complete_frontiers": True,
        "requires_hj_specific_single_view_e200_control": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    monkeypatch.setattr(successor, "validate_contract", lambda _value: None)
    monkeypatch.setattr(
        successor, "materialize_complete_frontier_final_delivery",
        lambda _root: {"selected_candidate_id": "legacy"},
    )
    pointer = operations / successor.POINTER
    _write(pointer, {"status": "complete"})
    monkeypatch.setattr(
        successor, "materialize_related_multi_algorithm_final_delivery",
        lambda _root: {
            "status": "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_COMPLETE",
            "action_priority_candidate_id": "A",
            "algorithm_set_status": "MULTIPLE_VIABLE_ALGORITHMS",
            "strict_viable_candidate_count": 3,
        },
    )
    runner = successor.RelatedMultiAlgorithmFinalSuccessor(contract)
    assert runner.run() == 0
    state = json.loads(
        (operations / "RELATED_MULTI_ALGORITHM_FINAL_STATE.json").read_text()
    )
    assert state["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert state["strict_viable_candidate_count"] == 3
