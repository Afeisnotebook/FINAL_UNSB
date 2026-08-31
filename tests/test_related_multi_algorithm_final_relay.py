from __future__ import annotations

import json
from pathlib import Path

from operations.local_route1_related_multi_algorithm_final_relay import (
    EXTRA_FILES,
    validate_local_delivery,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.related_multi_algorithm_final_delivery import (
    POINTER,
    POINTER_SCHEMA,
    PUBLISHED_FILES,
)


def _write(path: Path, payload: dict | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, dict):
        path.write_text(json.dumps(payload), encoding="utf-8")
    else:
        path.write_text(payload, encoding="utf-8")


def test_validate_local_related_delivery(tmp_path: Path):
    _write(tmp_path / "ALGORITHM_SET.json", {
        "action_priority_candidate_id": "A",
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "ACTION_PRIORITY.json", {"candidate_id": "A"})
    _write(tmp_path / "CANDIDATE.json", {"candidate_id": "A"})
    _write(tmp_path / "ALTERNATES.json", {
        "alternates": [{"candidate_id": "B"}, {"candidate_id": "C"}],
    })
    _write(tmp_path / "RELATED_RESULTS.json", {"status": "complete"})
    _write(tmp_path / "RELATED_FINAL_REPORT.md", "report")
    for name in EXTRA_FILES:
        _write(tmp_path / name, {"status": "complete", "name": name})
    pointer = {
        "schema": POINTER_SCHEMA,
        "status": "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_COMPLETE",
        "action_priority_candidate_id": "A",
        "algorithm_set_status": "MULTIPLE_VIABLE_ALGORITHMS",
        "strict_viable_candidate_count": 2,
        "final_file_sha256": {
            name: file_sha256(tmp_path / name) for name in PUBLISHED_FILES
        },
        "related_4090_host_adjudication_sha256": file_sha256(
            tmp_path / EXTRA_FILES[0]
        ),
        "related_5090_host_adjudication_sha256": file_sha256(
            tmp_path / EXTRA_FILES[1]
        ),
        "related_multi_host_adjudication_sha256": file_sha256(
            tmp_path / EXTRA_FILES[2]
        ),
        "hjpcnr_gain_source_receipt_sha256": file_sha256(
            tmp_path / EXTRA_FILES[3]
        ),
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(tmp_path / POINTER, pointer)
    assert validate_local_delivery(tmp_path)["action_priority_candidate_id"] == "A"
