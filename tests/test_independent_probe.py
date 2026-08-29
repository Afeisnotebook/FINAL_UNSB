from __future__ import annotations

import json

import pytest

from operations.local_route1_independent_probe import (
    EXPECTED_E0_FILE,
    EXPECTED_E0_SCIENTIFIC,
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
    validate_inflight_plain_contract,
    validate_e0_sidecar,
    validate_plain_sidecar,
)


def test_independent_probe_accepts_only_exact_plain_e200_identity():
    sidecar = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "step": 30_000,
        "physical_epoch_completed": 200,
        "metadata": {
            "probe_id": "plain",
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        },
    }
    validate_plain_sidecar(sidecar)
    sidecar["step"] = 29_999
    with pytest.raises(RuntimeError, match="plain e200"):
        validate_plain_sidecar(sidecar)


def test_independent_probe_rejects_confirmation_unlock():
    sidecar = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "step": 30_000,
        "physical_epoch_completed": 200,
        "metadata": {
            "probe_id": "plain",
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": True,
        },
    }
    with pytest.raises(RuntimeError, match="confirmation20_opened"):
        validate_plain_sidecar(sidecar)


def test_independent_probe_requires_exact_shared_e0():
    sidecar = {
        "schema": "final-unsb-local-route1-shared-e0-v1",
        "checkpoint_sha256": EXPECTED_E0_FILE,
        "scientific_state_sha256": EXPECTED_E0_SCIENTIFIC,
        "metadata": {
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
        },
    }
    validate_e0_sidecar(sidecar)
    sidecar["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="file identity"):
        validate_e0_sidecar(sidecar)


def test_inflight_plain_is_quarantined_and_exact_path_bound(tmp_path):
    training_repo = (tmp_path / "training").resolve()
    run_root = (tmp_path / "run").resolve()
    train_view = (tmp_path / "view").resolve()
    data_root = (tmp_path / "data").resolve()
    manifest = (tmp_path / "manifest.csv").resolve()
    operations = run_root / "operations"
    operations.mkdir(parents=True)
    contract = {
        "schema": "final-unsb-route1-executor-contract-v1",
        "executor_repo": str(training_repo),
        "run_root": str(run_root),
        "train_view": str(train_view),
        "data_root": str(data_root),
        "manifest": str(manifest),
        "training_git_commit": EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    }
    state = {
        "status": "CHUNK_RUNNING",
        "lane": "plain",
        "current_data_epoch": 27,
        "executor_pid": 123,
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    }
    (operations / "EXECUTOR_CONTRACT.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    (operations / "EXECUTION_STATE.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    evidence = validate_inflight_plain_contract(
        matched_plain_root=run_root,
        training_repo=training_repo,
        train_view=train_view,
        data_root=data_root,
        manifest=manifest,
    )
    assert evidence["matched_plain_status"] == "INFLIGHT_QUARANTINED"
    assert evidence["matched_plain_current_data_epoch"] == 27
    contract["data_root"] = str(tmp_path / "different-data")
    (operations / "EXECUTOR_CONTRACT.json").write_text(
        json.dumps(contract), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="data_root"):
        validate_inflight_plain_contract(
            matched_plain_root=run_root,
            training_repo=training_repo,
            train_view=train_view,
            data_root=data_root,
            manifest=manifest,
        )
