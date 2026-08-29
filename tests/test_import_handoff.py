from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations.local_route1_import_handoff import (
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
    file_sha256,
    import_lane,
    tree_manifest,
    tree_sha256,
)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def sidecar(root: Path, lane: str, epoch: int, content: bytes = b"state") -> dict:
    checkpoint = root / "anchors" / lane / "full_state_latest.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(content)
    payload = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "probe_id": lane,
        "step": epoch * 150,
        "physical_epoch_completed": epoch,
        "full_state_sha256": file_sha256(checkpoint),
        "scientific_state_sha256": f"scientific-{lane}-{epoch}",
        "metadata": {
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        },
    }
    write_json(Path(str(checkpoint) + ".json"), payload)
    return payload


def canonical(root: Path) -> dict:
    plain = sidecar(root, "plain", 200, b"plain-e200")
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
    return plain


def independent(root: Path, canonical_root: Path, plain: dict) -> None:
    sidecar(root, "hnek", 200, b"hnek-e200")
    write_json(root / "anchors" / "hnek" / "RUN_STATE.json", {
        "status": "COMPLETE_E200", "final_data_epoch": 200,
    })
    write_json(root / "anchors" / "hnek" / "metrics" / "e200.json", {"epoch": 200})
    contract = {
        "status": "COMPLETE_MATCHED_BASELINE_VERIFIED",
        "lane": "hnek",
        "matched_plain_root": str(canonical_root.resolve()),
        "matched_plain_checkpoint_sha256": plain["full_state_sha256"],
        "matched_plain_scientific_state_sha256": plain["scientific_state_sha256"],
        "training_git_commit": EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "batch_size_changed": False,
        "training_update_changed": False,
        "cross_host_state_used": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(root / "operations" / "INDEPENDENT_PROBE_CONTRACT.json", contract)
    write_json(root / "operations" / "INDEPENDENT_PROBE_RESULT.json", {
        "status": "COMPLETE_MATCHED_BASELINE_VERIFIED"
    })


def test_import_is_exact_and_never_overwrites(tmp_path):
    canonical_root = (tmp_path / "canonical").resolve()
    source_root = (tmp_path / "source").resolve()
    plain = canonical(canonical_root)
    independent(source_root, canonical_root, plain)
    source_rows = tree_manifest(source_root / "anchors" / "hnek")
    assert import_lane(canonical_root, source_root, require_paused=False) == 0
    imported_rows = tree_manifest(canonical_root / "anchors" / "hnek")
    assert imported_rows == source_rows
    receipt = json.loads(
        (canonical_root / "operations" / "INDEPENDENT_HNEK_IMPORT.json").read_text()
    )
    assert receipt["status"] == "IMPORT_VERIFIED"
    assert receipt["tree_sha256"] == tree_sha256(source_rows)
    with pytest.raises(RuntimeError, match="already exists|never overwrites"):
        import_lane(canonical_root, source_root, require_paused=False)


def test_import_rejects_different_matched_plain(tmp_path):
    canonical_root = (tmp_path / "canonical").resolve()
    source_root = (tmp_path / "source").resolve()
    plain = canonical(canonical_root)
    independent(source_root, canonical_root, plain)
    contract_path = source_root / "operations" / "INDEPENDENT_PROBE_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    contract["matched_plain_checkpoint_sha256"] = "0" * 64
    write_json(contract_path, contract)
    with pytest.raises(RuntimeError, match="matched-plain checkpoint"):
        import_lane(canonical_root, source_root, require_paused=False)
