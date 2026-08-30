import json
from pathlib import Path

import pytest

from operations.local_route1_final_reanalysis import (
    archive_collector_outputs,
    archive_retry_outputs,
    main,
    verify_queue,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_final_queue_requires_unique_repaired_hj_cells(tmp_path):
    jobs = [
        {"probe": "hj", "data_epoch": epoch}
        for epoch in (40, 60, 80)
    ] + [
        {"probe": f"probe-{index}", "data_epoch": index}
        for index in range(25)
    ]
    _write_json(
        tmp_path / "audit" / "AUDIT_QUEUE.json",
        {"jobs": jobs, "confirmation20_opened": False},
    )
    assert verify_queue(tmp_path)["jobs"] == 28
    jobs[-1] = {"probe": "hj", "data_epoch": 40}
    _write_json(
        tmp_path / "audit" / "AUDIT_QUEUE.json",
        {"jobs": jobs, "confirmation20_opened": False},
    )
    with pytest.raises(RuntimeError, match="28 unique"):
        verify_queue(tmp_path)


def test_collector_outputs_are_archived_without_deletion(tmp_path):
    matrix = tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json"
    _write_json(matrix, {"status": "old"})
    old_card = tmp_path / "derive" / "DERIVATION_QUEUE.json"
    _write_json(old_card, {"status": "old"})
    first = archive_collector_outputs(tmp_path, "a" * 40)
    archive = Path(first["archive"])
    assert not matrix.exists()
    assert not (tmp_path / "derive").exists()
    assert json.loads(
        (archive / "LONG_CAUSAL_MATRIX.json").read_text(encoding="utf-8")
    )["status"] == "old"
    assert (archive / "derive" / "DERIVATION_QUEUE.json").is_file()
    second = archive_collector_outputs(tmp_path, "a" * 40)
    assert second["intent_sha256"] == first["intent_sha256"]


def test_completed_collector_archive_allows_failed_reanalysis_retry(tmp_path):
    _write_json(
        tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json", {"status": "collector"},
    )
    _write_json(tmp_path / "derive" / "DERIVATION_QUEUE.json", {"old": True})
    archive_collector_outputs(tmp_path, "b" * 40)
    _write_json(
        tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json", {"status": "partial"},
    )
    _write_json(tmp_path / "derive" / "DERIVATION_QUEUE.json", {"new": True})
    archive_collector_outputs(tmp_path, "b" * 40)
    retry = archive_retry_outputs(tmp_path)
    assert retry is not None
    retry_root = Path(retry["archive"])
    assert json.loads(
        (retry_root / "LONG_CAUSAL_MATRIX.json").read_text(encoding="utf-8")
    )["status"] == "partial"
    assert not (tmp_path / "derive").exists()


def test_final_reanalysis_failure_is_persisted_for_watchers(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps({"schema": "bad", "run_root": str(tmp_path)}) + "\n",
        encoding="utf-8",
    )
    assert main(["--contract", str(contract)]) == 1
    state = json.loads(
        (tmp_path / "operations" / "FINAL_REANALYSIS_STATE.json").read_text(
            encoding="utf-8"
        )
    )
    assert state["status"] == "FAILED"
    assert state["paired_controller_access"] is False
