import json
from pathlib import Path

import pytest

from operations.local_route1_hj_reversal_supplement import (
    EXPECTED_AUDIT_COMMIT,
    EXPECTED_TRAINING_CORE,
    verify_atlases,
)


def _row(row_id: str, *, epoch: int = 40) -> dict:
    return {
        "row_id": row_id,
        "probe": "hj",
        "data_epoch": epoch,
        "parent_state_sha256_before": "parent",
        "parent_state_sha256_after": "parent",
        "audit_identity": {
            "audit_git_commit": EXPECTED_AUDIT_COMMIT,
            "training_core_fingerprint": EXPECTED_TRAINING_CORE,
        },
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
        "finite_value": 0.25,
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_supplement_atlas_verifier_enforces_identity_isolation_and_access(tmp_path):
    audit = tmp_path / "audit"
    _write(audit / "LONG_REVERSAL_ATLAS.jsonl", [_row("reversal")])
    _write(audit / "SAMPLING_VARIANCE_ATLAS.jsonl", [_row("variance")])
    result = verify_atlases(
        tmp_path, expected=(1, 1), require_supplement=False,
    )
    assert result["reversal_rows"] == 1
    assert result["variance_rows"] == 1
    bad = _row("reversal")
    bad["confirmation20_opened"] = True
    _write(audit / "LONG_REVERSAL_ATLAS.jsonl", [bad])
    with pytest.raises(RuntimeError, match="confirmation20"):
        verify_atlases(tmp_path, expected=(1, 1), require_supplement=False)


def test_supplement_atlas_verifier_requires_all_three_hj_cells(tmp_path):
    audit = tmp_path / "audit"
    rows = [_row(f"e{epoch}", epoch=epoch) for epoch in (40, 60, 80)]
    _write(audit / "LONG_REVERSAL_ATLAS.jsonl", rows)
    _write(audit / "SAMPLING_VARIANCE_ATLAS.jsonl", [_row("variance")])
    result = verify_atlases(
        tmp_path, expected=(3, 1), require_supplement=True,
    )
    assert result["reversal_rows"] == 3
    _write(audit / "LONG_REVERSAL_ATLAS.jsonl", rows[:2])
    with pytest.raises(RuntimeError, match="atlas count mismatch"):
        verify_atlases(tmp_path, expected=(3, 1), require_supplement=True)
