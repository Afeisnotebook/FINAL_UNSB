from __future__ import annotations

import json
from pathlib import Path

from research.local_route1 import related_goal_completion_audit as audit
from research.local_route1.related_multi_algorithm_final_delivery import (
    POINTER,
    PUBLISHED_FILES,
)


def _write(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload) if isinstance(payload, dict) else payload,
        encoding="utf-8",
    )


def test_related_goal_audit_requires_multi_algorithm_semantics(monkeypatch, tmp_path: Path):
    related = tmp_path / "related"
    algorithm_set = {
        "schema": audit.ALGORITHM_SET_SCHEMA,
        "status": "MULTIPLE_VIABLE_ALGORITHMS",
        "action_priority_candidate_id": "A",
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "strict_viable_candidate_ids": ["A", "B"],
        "positive_but_fragile_candidate_ids": [],
        "members": [
            {
                "candidate_id": candidate_id,
                "mathematics": {"formula": f"formula-{candidate_id}"},
                "source_bound": {"receipt": candidate_id},
                "absolute_relative_domain_trajectory": [candidate_id],
            }
            for candidate_id in ("A", "B")
        ],
        "cross_runtime_related_evidence": {
            "algorithms": [{
                "host_results": [{
                    "trajectory_snapshot": {
                        "candidate_id": "A",
                        "confirmation20_opened": False,
                    },
                }],
            }],
            "cross_runtime_is_not_cross_seed": True,
            "cross_host_deltas_merged": False,
            "cross_seed_stability_claimed": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
        "cross_host_deltas_merged": False,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(related / "ALGORITHM_SET.json", algorithm_set)
    _write(related / "ACTION_PRIORITY.json", {
        "schema": audit.ACTION_SCHEMA,
        "status": "CURRENT_NEXT_ACTION_PRIORITY",
        "candidate_id": "A",
        "action_priority_is_not_scientific_exclusivity": True,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(related / "RELATED_RESULTS.json", {
        "schema": audit.RESULTS_SCHEMA,
        "status": "RELATED_MULTI_ALGORITHM_E200_COMPLETE",
        "cross_host_deltas_merged": False,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(related / "RELATED_FINAL_REPORT.md", "report")
    _write(related / POINTER, {"pointer": True})

    monkeypatch.setattr(
        audit, "audit_complete_delivery",
        lambda _path: {"status": "compatibility-proven"},
    )
    monkeypatch.setattr(
        audit, "validate_related_delivery",
        lambda _path: {
            "algorithm_set_status": "MULTIPLE_VIABLE_ALGORITHMS",
            "action_priority_candidate_id": "A",
            "strict_viable_candidate_count": 2,
        },
    )
    monkeypatch.setattr(
        audit, "_domain_trajectory",
        lambda rows, label: {"label": label, "rows": rows},
    )

    result = audit.audit_related_goal_completion(tmp_path / "compatibility", related)
    assert result["action_priority_candidate_id"] == "A"
    assert result["strict_viable_candidate_ids"] == ["A", "B"]
    assert result["action_priority_is_not_scientific_exclusivity"] is True
    assert result["completion_claim_allowed"] is False
    assert set(result["source_delivery_sha256"]) == {POINTER, *PUBLISHED_FILES}

