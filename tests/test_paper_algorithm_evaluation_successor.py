from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from operations import paper_aio_algorithm_evaluation_successor as successor
from operations import paper_aio_candidate_metadata_relay as metadata
from operations import paper_aio_export_relay as export_relay


CANDIDATE = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority(tmp_path: Path) -> tuple[Path, dict]:
    value = {
        "schema": metadata.AUTHORITY_SCHEMA,
        "status": "FROZEN_EVALUATION_ONLY_AUTHORITY",
        "candidate_id": CANDIDATE,
        "evaluation_only": True,
        "authorizes_training": False,
        "performance_metric_values_included": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "source_evidence": {
            "candidate_lock_sha256": "a" * 64,
            "runtime_gate_sha256": "b" * 64,
            "authorization_sha256": "c" * 64,
        },
    }
    return _write(tmp_path / "authority.json", value), value


def test_candidate_metadata_validators_bind_all_three_prior_artifacts(
    tmp_path: Path,
) -> None:
    authority_path, authority = _authority(tmp_path)
    assert metadata.validate_authority(authority_path, CANDIDATE) == authority
    lock = {
        "schema": "final-unsb-paper-candidate-lock-v1",
        "status": "PASS_FULL_DATA_CANDIDATE_LOCK",
        "candidate_id": CANDIDATE,
        "full_data_authorized": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    authorization = {
        "schema": "final-unsb-paper-candidate-authorization-v1",
        "status": "PASS_FULL_DATA_CANDIDATE_AUTHORIZATION",
        "candidate_id": CANDIDATE,
        "candidate_lock_sha256": "a" * 64,
        "candidate_runtime_gate_sha256": "b" * 64,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    runtime = {
        "schema": "final-unsb-paper-candidate-runtime-gate-v1",
        "status": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
        "candidate_id": CANDIDATE,
        "manifest_sha256": (
            "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
        ),
        "e0_scientific_core_exact": True,
        "plain_2000_transition_exact": True,
        "zero_intervention_identity_exact": True,
        "candidate_resume_exact": True,
        "candidate_evaluation_repeat_exact": True,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    metadata.validate_candidate_metadata(
        candidate_id=CANDIDATE, authority=authority, lock=lock,
        authorization=authorization, runtime_gate=runtime,
    )
    authorization["candidate_runtime_gate_sha256"] = "d" * 64
    with pytest.raises(RuntimeError, match="authorization"):
        metadata.validate_candidate_metadata(
            candidate_id=CANDIDATE, authority=authority, lock=lock,
            authorization=authorization, runtime_gate=runtime,
        )


def test_metadata_receipt_is_hash_bound_and_never_authorizes_training(
    tmp_path: Path,
) -> None:
    authority_path, _ = _authority(tmp_path)
    artifacts = {}
    for name in ("candidate_lock", "authorization", "runtime_gate"):
        path = tmp_path / f"{name}.json"
        path.write_text(name, encoding="utf-8")
        artifacts[name] = path
    receipt = {
        "schema": successor.METADATA_SCHEMA,
        "status": "COMPLETE_VERIFIED_CANDIDATE_METADATA_IMPORT",
        "candidate_id": CANDIDATE,
        "authority_sha256": _sha(authority_path),
        "candidate_lock": str(artifacts["candidate_lock"]),
        "candidate_lock_sha256": _sha(artifacts["candidate_lock"]),
        "authorization": str(artifacts["authorization"]),
        "authorization_sha256": _sha(artifacts["authorization"]),
        "runtime_gate": str(artifacts["runtime_gate"]),
        "runtime_gate_sha256": _sha(artifacts["runtime_gate"]),
        "training_authorized_or_scheduled": False,
        "paired_performance_used_for_training_or_scheduling": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    receipt_path = _write(tmp_path / "receipt.json", receipt)
    assert successor.validate_metadata_receipt(
        receipt_path, candidate_id=CANDIDATE, authority=authority_path,
    ) == receipt
    artifacts["runtime_gate"].write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        successor.validate_metadata_receipt(
            receipt_path, candidate_id=CANDIDATE, authority=authority_path,
        )


def test_local_export_lane_requires_every_fixed_checkpoint_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Export receipts are POSIX-absolute on deployment. Permit the Windows
    # pytest temporary path while retaining all hash/path-containment checks.
    monkeypatch.setattr(export_relay, "_remote_path", lambda value, _label: value)
    export_root = tmp_path / "exports"
    rows = []
    for epoch in successor.UNIFIED_EPOCHS:
        checkpoint = _write(tmp_path / "source" / f"e{epoch}.pt", {"epoch": epoch})
        sidecar = _write(tmp_path / "source" / f"e{epoch}.pt.json", {"epoch": epoch})
        receipt = {
            "schema": "final-unsb-paper-checkpoint-export-v1",
            "status": "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT",
            "lane_id": "amtnc",
            "epoch": epoch,
            "updates": epoch * 8553,
            "source_host_label": "4090A",
            "source_checkpoint": checkpoint.as_posix(),
            "source_sidecar": sidecar.as_posix(),
            "checkpoint_sha256": _sha(checkpoint),
            "sidecar_sha256": _sha(sidecar),
            "scientific_state_sha256": "1" * 64,
            "training_git_commit": "2" * 40,
            "training_protocol_fingerprint": "3" * 64,
            "manifest_sha256": "4" * 64,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        }
        receipt_path = _write(
            export_root / "amtnc" / f"e{epoch:03d}.export.json", receipt,
        )
        rows.append({
            "epoch": epoch,
            "receipt": receipt_path.as_posix(),
            "receipt_sha256": _sha(receipt_path),
        })
    _write(export_root / "amtnc" / "EXPORT_SET.json", {
        "schema": "final-unsb-paper-source-export-set-v1",
        "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
        "lane_id": "amtnc",
        "source_host_label": "4090A",
        "epochs": list(successor.UNIFIED_EPOCHS),
        "exports": rows,
        "performance_values_read": False,
        "checkpoint_copy_performed": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    validated = successor.validate_local_export_lane(
        export_root, lane_id="amtnc", source_host_label="4090A",
    )
    assert [row["epoch"] for row in validated] == list(successor.UNIFIED_EPOCHS)
    Path(validated[-1]["checkpoint"]).write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checkpoint changed"):
        successor.validate_local_export_lane(
            export_root, lane_id="amtnc", source_host_label="4090A",
        )


def test_cohort_dependency_fails_closed(tmp_path: Path) -> None:
    cohort = tmp_path / "cohort.json"
    assert successor.cohort_decision(cohort) == "WAIT"
    _write(cohort, {
        "schema": "final-unsb-paper-unified-evaluation-cohort-v1",
        "status": "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT",
        "cross_host_training_delta_merged": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    assert successor.cohort_decision(cohort) == "READY"
    value = json.loads(cohort.read_text(encoding="utf-8"))
    value["confirmation20_opened"] = True
    _write(cohort, value)
    assert successor.cohort_decision(cohort) == "BLOCKED"


def test_dynamic_readiness_requires_completed_import_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata_receipt = _write(tmp_path / "metadata.json", {})
    cohort = _write(tmp_path / "cohort.json", {
        "schema": "final-unsb-paper-unified-evaluation-cohort-v1",
        "status": "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT",
        "cross_host_training_delta_merged": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    observed = {}

    def complete_import(root: Path, lanes: dict[str, str]) -> bool:
        observed.update({"root": root, "lanes": lanes})
        return False

    monkeypatch.setattr(successor, "imports_ready", complete_import)
    contract = {
        "mode": "dynamic_candidate",
        "method_source_root": str(tmp_path / "imports"),
        "method_lane": CANDIDATE,
        "method_source_host": "5090A",
        "candidate_metadata_receipt": str(metadata_receipt),
        "first_wave_cohort": str(cohort),
    }
    ready, status = successor._ready(contract)
    assert ready is False
    assert status == "WAITING_FOR_CANDIDATE_IMPORT_METADATA_OR_COHORT"
    assert observed["lanes"] == {CANDIDATE: "5090A"}


def test_disposition_requires_fixed_e200_gate_and_binds_receipts(
    tmp_path: Path,
) -> None:
    lane = "amtnc"
    for epoch in successor.UNIFIED_EPOCHS:
        _write(
            tmp_path / "gates" / f"UNIFIED_EVALUATION_{lane}_e{epoch:03d}.json",
            {"epoch": epoch},
        )
    result = {"results": {"lanes": [{
        "lane_id": lane,
        "status": "COMPLETE_E200",
        "scientific_gate": {"status": "PASS"},
    }]}}
    value = successor._disposition(
        output=tmp_path, method_lane=lane, result=result,
    )
    assert value["status"] == "COMPLETE_POSTHOC_ALGORITHM_DISPOSITION"
    assert value["best_checkpoint_selection"] is False
    assert len(value["evaluation_receipts"]) == 5
    del result["results"]["lanes"][0]["scientific_gate"]
    with pytest.raises(RuntimeError, match="terminal adjudication"):
        successor._disposition(
            output=tmp_path, method_lane=lane, result=result,
        )
