from __future__ import annotations

import json
from pathlib import Path

from operations.local_route1_auto_import import canonical_import_state, source_ready
from operations.local_route1_import_handoff import (
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def full_state_sidecar(root: Path, lane: str, epoch: int) -> None:
    write_json(root / "anchors" / lane / "full_state_latest.pt.json", {
        "schema": "final-unsb-local-route1-full-state-v1",
        "probe_id": lane,
        "step": epoch * 150,
        "physical_epoch_completed": epoch,
        "metadata": {
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        },
    })


def canonical_contract(root: Path) -> None:
    write_json(root / "operations" / "EXECUTOR_CONTRACT.json", {
        "schema": "final-unsb-route1-executor-contract-v1",
        "run_root": str(root.resolve()),
        "training_git_commit": EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    })
    write_json(root / "operations" / "EXECUTION_STATE.json", {
        "status": "CHUNK_RUNNING",
        "lane": "hj",
        "executor_pid": 123,
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    })


def test_source_ready_only_after_verified_result_and_e200(tmp_path):
    source = tmp_path / "source"
    assert source_ready(source) is False
    write_json(source / "operations" / "INDEPENDENT_PROBE_CONTRACT.json", {
        "status": "COMPLETE_MATCHED_BASELINE_VERIFIED"
    })
    write_json(source / "operations" / "INDEPENDENT_PROBE_RESULT.json", {
        "status": "COMPLETE_MATCHED_BASELINE_VERIFIED"
    })
    full_state_sidecar(source, "hnek", 199)
    assert source_ready(source) is False
    full_state_sidecar(source, "hnek", 200)
    assert source_ready(source) is True


def test_canonical_state_refuses_scheduled_or_unreceipted_hnek(tmp_path):
    root = (tmp_path / "canonical").resolve()
    canonical_contract(root)
    assert canonical_import_state(root) == "WAITING"
    state_path = root / "operations" / "EXECUTION_STATE.json"
    state = json.loads(state_path.read_text())
    state["lane"] = "hnek"
    write_json(state_path, state)
    assert canonical_import_state(root) == "CANONICAL_HNEK_ALREADY_SCHEDULED"
    state["lane"] = "hj"
    write_json(state_path, state)
    full_state_sidecar(root, "hnek", 200)
    assert canonical_import_state(root) == "CANONICAL_HNEK_EXISTS_WITHOUT_IMPORT_RECEIPT"
    write_json(root / "operations" / "INDEPENDENT_HNEK_IMPORT.json", {
        "status": "IMPORT_VERIFIED"
    })
    assert canonical_import_state(root) == "ALREADY_IMPORTED"
