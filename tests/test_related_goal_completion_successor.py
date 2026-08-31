from __future__ import annotations

import json
from pathlib import Path

from operations import local_route1_related_goal_completion_successor as successor


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_successor_waits_for_both_deliveries(monkeypatch, tmp_path: Path):
    compatibility = tmp_path / "compatibility"
    related = tmp_path / "related"
    _write(compatibility / "RELAY_MANIFEST.json", {"status": "complete"})
    _write(related / "RELAY_MANIFEST.json", {"status": "complete"})
    contract_path = tmp_path / "contract.json"
    output = tmp_path / "audit.json"
    state = tmp_path / "state.json"
    _write(contract_path, {
        "schema": successor.SCHEMA,
        "repo": str(tmp_path),
        "git_commit": "commit",
        "source_sha256": {},
        "compatibility_delivery": str(compatibility),
        "related_delivery": str(related),
        "output": str(output),
        "state": str(state),
        "poll_seconds": 60,
        "timeout_seconds": 43200,
        "requires_related_algorithm_set": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    monkeypatch.setattr(successor, "validate_contract", lambda _value: None)

    def materialize(_compatibility, _related, destination):
        _write(destination, {"status": "audit"})
        return {
            "status": "RELATED_ROUTE1_TERMINAL_ARTIFACTS_PROVEN_FINAL_GIT_COMMIT_REQUIRED",
            "action_priority_candidate_id": "A",
            "algorithm_set_status": "MULTIPLE_VIABLE_ALGORITHMS",
            "algorithm_member_count": 4,
        }

    monkeypatch.setattr(
        successor, "materialize_related_goal_completion_audit", materialize,
    )
    runner = successor.RelatedGoalCompletionSuccessor(contract_path)
    assert runner.run() == 0
    state_value = json.loads(state.read_text())
    assert state_value["algorithm_member_count"] == 4
    assert state_value["final_repository_commit_and_push_required"] is True

