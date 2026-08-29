import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import local_route1_audit_executor as audit_executor
from operations import local_route1_candidate_executor as candidate_executor
from operations import local_route1_seed_executor as seed_executor
from operations.local_route1_executor import (
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
    anchor_command,
    atomic_json,
    current_epoch,
    default_contract,
    latest_sidecar,
    validate_contract,
    validate_lane_sidecar,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_json_never_leaves_temporary(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "PASS"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not path.with_suffix(".json.tmp").exists()


def test_contract_locks_scripts_and_scientific_identity(tmp_path):
    supervisor = tmp_path / "local_route1_executor.py"
    backfill = tmp_path / "local_route1_metric_backfill.py"
    supervisor.write_text("supervisor", encoding="utf-8")
    backfill.write_text("backfill", encoding="utf-8")
    args = SimpleNamespace(
        main_repo=tmp_path,
        executor_repo=tmp_path,
        run_root=tmp_path,
        train_view=tmp_path,
        data_root=tmp_path,
        manifest=tmp_path / "manifest.csv",
        python=tmp_path / "python.exe",
    )
    # default_contract uses the real module paths; replace them with fixture files.
    contract = default_contract(args)
    contract.update(
        {
            "supervisor_script": str(supervisor),
            "supervisor_sha256": sha(supervisor),
            "backfill_script": str(backfill),
            "backfill_sha256": sha(backfill),
        }
    )
    validate_contract(contract)
    supervisor.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="supervisor script changed"):
        validate_contract(contract)


def test_checkpoint_sidecar_epoch_and_identity(tmp_path):
    lane_root = tmp_path / "anchors" / "plain"
    lane_root.mkdir(parents=True)
    sidecar = {
        "physical_epoch_completed": 100,
        "full_state_sha256": "checkpoint",
        "metadata": {
            "probe_id": "plain",
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        },
    }
    atomic_json(lane_root / "full_state_latest.pt.json", sidecar)
    assert current_epoch(tmp_path, "plain") == 100
    assert latest_sidecar(tmp_path, "plain") == sidecar
    identity = {
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
    }
    validate_lane_sidecar(sidecar, identity, "plain")
    sidecar["metadata"]["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="confirmation20_opened"):
        validate_lane_sidecar(sidecar, identity, "plain")


def test_anchor_command_explicitly_freezes_every_scientific_path(tmp_path):
    contract = {
        "python": str(tmp_path / "python.exe"),
        "run_root": str(tmp_path / "run"),
        "train_view": str(tmp_path / "view"),
        "data_root": str(tmp_path / "data"),
        "manifest": str(tmp_path / "frozen_manifest.csv"),
    }
    argv = anchor_command(contract, "plain", 105)
    for flag, expected in (
        ("--output", contract["run_root"]),
        ("--train-view", contract["train_view"]),
        ("--data-root", contract["data_root"]),
        ("--manifest", contract["manifest"]),
    ):
        assert argv[argv.index(flag) + 1] == expected
    assert argv[argv.index("--engineering-stop-after-epoch") + 1] == "105"


def test_audit_executor_contract_keeps_paired_and_confirmation_inputs_closed(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("frozen", encoding="utf-8")
    supervisor = tmp_path / "audit_executor.py"
    supervisor.write_text("# frozen", encoding="utf-8")
    monkeypatch.setattr(
        audit_executor, "file_sha256",
        lambda path: (
            audit_executor.EXPECTED_MANIFEST
            if Path(path) == manifest else "supervisor-hash"
        ),
    )
    contract = {
        "schema": "final-unsb-route1-audit-executor-contract-v1",
        "audit_repo": str(tmp_path),
        "training_repo": str(tmp_path),
        "run_root": str(tmp_path),
        "train_view": str(tmp_path),
        "data_root": str(tmp_path),
        "manifest": str(manifest),
        "python": str(tmp_path / "python.exe"),
        "audit_git_commit": "abc",
        "supervisor_script": str(supervisor),
        "supervisor_sha256": "supervisor-hash",
        "manifest_sha256": audit_executor.EXPECTED_MANIFEST,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    audit_executor.validate_contract(contract)
    contract["maximum_parallel_jobs"] = 2
    audit_executor.validate_contract(contract)
    contract["maximum_parallel_jobs"] = 3
    with pytest.raises(RuntimeError, match="maximum_parallel_jobs"):
        audit_executor.validate_contract(contract)
    contract["maximum_parallel_jobs"] = 2
    contract["paired_controller_access"] = True
    with pytest.raises(RuntimeError, match="paired controller"):
        audit_executor.validate_contract(contract)


def test_audit_executor_waits_for_an_explicit_anchor_terminal_state():
    assert audit_executor.ANCHOR_TERMINAL == {
        "PAUSED_PROXY_NOT_CALIBRATED", "ANCHOR_PHASE_COMPLETE"
    }
    assert audit_executor.AuditExecutor._job_key("hnek", 100) == "hnek:e100"
    assert audit_executor.post_audit_terminal_state("ANCHOR_PHASE_COMPLETE") == (
        "PHASE_C_COMPLETE_DERIVATION_REQUIRED"
    )
    assert audit_executor.post_audit_terminal_state("PAUSED_PROXY_NOT_CALIBRATED") == (
        "PHASE_C_COMPLETE_PROXY_ADJUDICATION_REQUIRED"
    )
    with pytest.raises(ValueError, match="unsupported terminal anchor status"):
        audit_executor.post_audit_terminal_state("CHUNK_RUNNING")


def test_audit_terminal_verification_scope_and_integrity():
    assert audit_executor.required_terminal_lanes("ANCHOR_PHASE_COMPLETE") == (
        "plain", "hj", "hnek", "dt"
    )
    assert audit_executor.required_terminal_lanes("PAUSED_PROXY_NOT_CALIBRATED") == (
        "plain", "hj", "hnek"
    )
    payload = {
        "schema": "final-unsb-route1-milestone-verification-v1",
        "status": "ACCEPTED_MILESTONE",
        "identity": {"probe_id": "dt", "data_epoch": 200},
        "integrity": {
            "checkpoint_file_hash_matches_sidecar": True,
            "scientific_state_hash_matches_sidecar": True,
            "metric_protocol_matches": True,
            "evaluation_bundle_matches_frozen_crn": True,
            "paired_metric_used_for_training_control": False,
            "confirmation20_opened": False,
        },
    }
    audit_executor.validate_terminal_verification(payload, "dt")
    payload["integrity"]["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="confirmation20"):
        audit_executor.validate_terminal_verification(payload, "dt")


def test_candidate_executor_command_has_no_selection_or_early_stop_inputs(tmp_path):
    contract = {
        "python": str(tmp_path / "python"),
        "candidate_id": "G1-01-SAFE",
        "run_root": str(tmp_path / "run"),
        "train_view": str(tmp_path / "view"),
        "data_root": str(tmp_path / "data"),
        "manifest": str(tmp_path / "manifest.csv"),
    }
    argv = candidate_executor.candidate_train_command(contract, 125)
    assert argv[argv.index("--candidate-id") + 1] == "G1-01-SAFE"
    assert argv[argv.index("--engineering-stop-after-epoch") + 1] == "125"
    assert "--resume" in argv
    assert not any("psnr" in value.lower() or "threshold" in value.lower() for value in argv)
    with pytest.raises(ValueError, match="unsafe candidate id"):
        candidate_executor.safe_candidate_id("../escape")


def test_candidate_executor_contract_closes_paired_control_and_locks_script(
    tmp_path, monkeypatch,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("frozen", encoding="utf-8")
    supervisor = tmp_path / "candidate_executor.py"
    supervisor.write_text("# frozen", encoding="utf-8")
    environment = tmp_path / "environment.json"
    environment_payload = {
        "python": "3.11 fixture", "platform": "fixture-os", "torch": "2.8",
        "torch_cuda": "12.8", "cudnn": 91002, "cuda_available": True,
        "gpu": "fixture-gpu",
    }
    environment.write_text(json.dumps(environment_payload), encoding="utf-8")
    verification = tmp_path / "plain_e200.json"
    verification.write_text(json.dumps({
        "schema": "final-unsb-route1-milestone-verification-v1",
        "status": "ACCEPTED_MILESTONE",
        "identity": {"probe_id": "plain", "data_epoch": 200},
        "checkpoint": {"file_sha256": "plain-file", "scientific_state_sha256": "plain-state"},
        "integrity": {
            "checkpoint_file_hash_matches_sidecar": True,
            "scientific_state_hash_matches_sidecar": True,
            "metric_protocol_matches": True,
            "evaluation_bundle_matches_frozen_crn": True,
            "paired_metric_used_for_training_control": False,
            "confirmation20_opened": False,
        },
    }), encoding="utf-8")
    monkeypatch.setattr(
        candidate_executor, "file_sha256",
        lambda path: (
            candidate_executor.EXPECTED_MANIFEST
            if Path(path) == manifest else
            "environment-hash" if Path(path) == environment else
            "plain-verification-hash" if Path(path) == verification else
            "supervisor-hash"
        ),
    )
    contract = {
        "schema": "final-unsb-route1-candidate-executor-contract-v1",
        "candidate_repo": str(tmp_path),
        "candidate_git_commit": "abc",
        "candidate_id": "G1-TEST",
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "fingerprint",
        "run_root": str(tmp_path),
        "train_view": str(tmp_path),
        "data_root": str(tmp_path),
        "manifest": str(manifest),
        "manifest_sha256": candidate_executor.EXPECTED_MANIFEST,
        "python": str(tmp_path / "python"),
        "supervisor_script": str(supervisor),
        "supervisor_sha256": "supervisor-hash",
        "baseline_environment_record": str(environment),
        "baseline_environment_record_sha256": "environment-hash",
        "baseline_environment": environment_payload,
        "runtime_environment_at_freeze": environment_payload,
        "plain_e200_verification": str(verification),
        "plain_e200_verification_sha256": "plain-verification-hash",
        "plain_e200_checkpoint_sha256": "plain-file",
        "plain_e200_scientific_state_sha256": "plain-state",
        "chunk_data_epochs_max": 5,
        "target_data_epochs": 200,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    candidate_executor.validate_contract(contract)
    contract["paired_metric_early_stop"] = True
    with pytest.raises(RuntimeError, match="paired_metric_early_stop=false"):
        candidate_executor.validate_contract(contract)
    contract["paired_metric_early_stop"] = False
    contract["runtime_environment_at_freeze"] = {**environment_payload, "gpu": "other"}
    with pytest.raises(RuntimeError, match="matched-plain environment"):
        candidate_executor.validate_contract(contract)


def test_candidate_executor_status_requires_evidence_gate_and_locks(tmp_path):
    ready = {
        "status": "READY_FOR_MATCHED_E200",
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "frozen",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    assert candidate_executor._parse_status(json.dumps(ready))["candidate_fingerprint"] == "frozen"
    ready["status"] = "READY_FOR_CANDIDATE_GATES"
    with pytest.raises(RuntimeError, match="not ready for e200"):
        candidate_executor._parse_status(json.dumps(ready))


def test_seed_executor_contract_forces_plain_then_candidate_and_no_control(
    tmp_path, monkeypatch,
):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("frozen", encoding="utf-8")
    supervisor = tmp_path / "seed_executor.py"
    support_script = tmp_path / "candidate_executor.py"
    supervisor.write_text("# frozen seed", encoding="utf-8")
    support_script.write_text("# frozen support", encoding="utf-8")
    monkeypatch.setattr(
        seed_executor.support, "file_sha256",
        lambda path: (
            seed_executor.EXPECTED_MANIFEST if Path(path) == manifest
            else "supervisor-hash" if Path(path) == supervisor
            else "support-hash"
        ),
    )
    contract = {
        "schema": seed_executor.CONTRACT_SCHEMA,
        "seed_repo": str(tmp_path),
        "seed_git_commit": "abc",
        "candidate_id": "G1-TEST",
        "validation_seed": 2027,
        "algorithm_fingerprint": "algorithm",
        "seed2026_candidate_fingerprint": "seed2026-execution",
        "seed_freeze_sha256": "freeze",
        "run_root": str(tmp_path),
        "train_view": str(tmp_path),
        "data_root": str(tmp_path),
        "manifest": str(manifest),
        "manifest_sha256": seed_executor.EXPECTED_MANIFEST,
        "python": str(tmp_path / "python"),
        "supervisor_script": str(supervisor),
        "supervisor_sha256": "supervisor-hash",
        "support_script": str(support_script),
        "support_sha256": "support-hash",
        "lane_order": ["plain", "candidate"],
        "chunk_data_epochs_max": 5,
        "target_data_epochs": 200,
        "algorithm_change_after_seed2026": False,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    seed_executor.validate_contract(contract)
    command = seed_executor.validation_train_command(contract, "plain", 5)
    assert "--resume" in command
    assert command[command.index("--validation-lane") + 1] == "plain"
    assert not any("psnr" in item.lower() or "threshold" in item.lower() for item in command)
    contract["lane_order"] = ["candidate", "plain"]
    with pytest.raises(RuntimeError, match="matched plain before candidate"):
        seed_executor.validate_contract(contract)


def test_seed_executor_status_locks_frozen_algorithm():
    ready = {
        "status": "READY_FOR_FROZEN_SEED_VALIDATION",
        "algorithm_fingerprint": "algorithm",
        "seed_freeze_sha256": "freeze",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    assert seed_executor._parse_status(json.dumps(ready))["algorithm_fingerprint"] == "algorithm"
    ready["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="lock confirmation20"):
        seed_executor._parse_status(json.dumps(ready))
