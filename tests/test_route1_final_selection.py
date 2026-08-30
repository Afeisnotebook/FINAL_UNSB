from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
    CROSS_VERSION_NEGATIVE_STATUS,
)
from research.local_route1.final_selection import (
    FINAL_SELECTION_NAME,
    POSITIVE_STATUS,
    REVISION_SELECTION_NAME,
    resolve_e200_selection_path,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _selection(candidate: str, status: str) -> dict:
    return {
        "schema": "final-unsb-route1-cross-version-e200-adjudication-v1",
        "status": status,
        "ranking": [{
            "candidate_id": candidate,
            "algorithm_fingerprint": f"algorithm-{candidate}",
            "candidate_fingerprint": f"candidate-{candidate}",
            "training_git_commit": "a" * 40,
        }],
        "selected_candidate_id": candidate,
        "selected_algorithm_fingerprint": f"algorithm-{candidate}",
        "selected_candidate_fingerprint": f"candidate-{candidate}",
        "selected_training_git_commit": "a" * 40,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }


def test_negative_generation1_cannot_race_authorized_revision(tmp_path):
    operations = tmp_path / "operations"
    _write(
        operations / "CROSS_VERSION_E200_ADJUDICATION.json",
        _selection("G1", CROSS_VERSION_NEGATIVE_STATUS),
    )
    _write(operations / "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json", {
        "schema": CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
        "status": "REVISION_DERIVATION_REQUIRED",
    })
    with pytest.raises(RuntimeError, match="pending the authorized Generation-2"):
        resolve_e200_selection_path(tmp_path)

    revision = operations / REVISION_SELECTION_NAME
    _write(revision, _selection("G2", POSITIVE_STATUS))
    assert resolve_e200_selection_path(tmp_path) == revision.resolve()


def test_explicit_all_candidate_selection_has_highest_precedence(tmp_path):
    operations = tmp_path / "operations"
    _write(
        operations / "CROSS_VERSION_E200_ADJUDICATION.json",
        _selection("G1", POSITIVE_STATUS),
    )
    _write(operations / REVISION_SELECTION_NAME, _selection("G2", POSITIVE_STATUS))
    final = operations / FINAL_SELECTION_NAME
    _write(final, _selection("G3", CROSS_VERSION_NEGATIVE_STATUS))
    assert resolve_e200_selection_path(tmp_path) == final.resolve()

