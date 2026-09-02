from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.paper_aio.candidate_lock import materialize_candidate_lock
from research.paper_aio.candidate_runtime import (
    authorize_candidate,
    require_candidate_authorization,
)
from research.paper_aio.protocol import file_sha256, protocol_fingerprint


CANDIDATE_ID = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"
PARENT_COMMIT = "1" * 40
PARENT_PROTOCOL = "2" * 64
ALGORITHM = "3" * 64


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path | str]:
    card = _write(tmp_path / "card.json", {"candidate_id": CANDIDATE_ID})
    source = Path(__file__).resolve().parents[1] / "src" / "models" / "route1_stcgr_model.py"
    implementation = _write(tmp_path / "implementation.json", {
        "schema": "final-unsb-route1-candidate-implementation-v1",
        "candidate_id": CANDIDATE_ID,
        "status": "FROZEN_FOR_GATES",
        "model": "route1_stcgr",
        "method": {"route1_stcgr_enable": True},
        "zero_intervention": {"route1_stcgr_enable": False},
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "source_files": [{
            "path": "src/models/route1_stcgr_model.py",
            "sha256": file_sha256(source),
        }],
    })
    trajectory = _write(tmp_path / "trajectory.json", {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "candidate_id": CANDIDATE_ID,
        "algorithm_fingerprint": ALGORITHM,
        "status": "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
        "late_three_mean_macro_psnr_delta": 0.2,
        "e200_macro_psnr_delta": 0.1,
        "late_points_with_four_of_six_positive_domains": 2,
        "late_average_worst_domain_delta": -0.2,
        "late_mean_macro_ssim_delta": 0.001,
        "late_mean_macro_lpips_delta": -0.001,
        "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    })
    receipt = _write(tmp_path / "receipt.json", {
        "schema": "final-unsb-route1-candidate-terminal-receipt-v1",
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": CANDIDATE_ID,
        "trajectory_status": "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
        "algorithm_fingerprint": ALGORITHM,
        "candidate_fingerprint": "4" * 64,
        "candidate_training_core_fingerprint": "5" * 64,
        "base_e0_scientific_state_sha256": "6" * 64,
        "base_protocol_fingerprint": "7" * 64,
        "training_git_commit": "8" * 40,
        "trajectory_sha256": file_sha256(trajectory),
        "derivation_card_sha256": file_sha256(card),
        "implementation_sha256": file_sha256(implementation),
        "ranking_fields": {},
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })

    parent = tmp_path / "parent"
    _write(parent / "gates" / "PREFLIGHT.json", {
        "status": "PASS", "node_role": "training",
        "protocol_fingerprint": PARENT_PROTOCOL,
        "manifest": {"content_hashes_verified": True},
        "confirmation20_opened": False,
    })
    _write(parent / "gates" / "LANE_AUTHORIZATION_plain.json", {
        "status": "PASS", "lane_id": "plain",
        "protocol_fingerprint": PARENT_PROTOCOL,
        "confirmation20_opened": False,
    })
    _write(parent / "lanes" / "plain" / "RUN_STATE.json", {
        "status": "COMPLETE_E200", "final_updates": 1_710_600,
        "confirmation20_opened": False,
    })
    metadata = {
        "protocol_fingerprint": PARENT_PROTOCOL,
        "git_commit": PARENT_COMMIT,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(parent / "lanes" / "plain" / "full_state_latest.pt.json", {
        "step": 1_710_600,
        "scientific_state_sha256": "9" * 64,
        "metadata": metadata,
    })
    _write(parent / "shared_e0" / "unsb_common" / "e0.pt.json", {
        "scientific_state_sha256": "a" * 64,
        "metadata": {
            "protocol_fingerprint": PARENT_PROTOCOL,
            "git_commit": PARENT_COMMIT,
        },
    })
    runtime_gate = _write(tmp_path / "runtime_gate.json", {
        "schema": "final-unsb-paper-candidate-runtime-gate-v1",
        "status": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
        "candidate_id": CANDIDATE_ID,
        "algorithm_fingerprint": ALGORITHM,
        "parent_scientific_git_commit": PARENT_COMMIT,
        "parent_protocol_fingerprint": PARENT_PROTOCOL,
        "candidate_git_commit": "b" * 40,
        "candidate_protocol_fingerprint": "c" * 64,
        "e0_scientific_core_exact": True,
        "plain_2000_transition_exact": True,
        "zero_intervention_identity_exact": True,
        "candidate_resume_exact": True,
        "candidate_evaluation_repeat_exact": True,
        "parent_e0_scientific_core_sha256": "d" * 64,
        "candidate_e0_scientific_core_sha256": "d" * 64,
        "parent_plain_2000_transition_sha256": "e" * 64,
        "candidate_plain_2000_transition_sha256": "e" * 64,
        "plain_zero_transition_sha256": "f" * 64,
        "candidate_zero_transition_sha256": "f" * 64,
        "candidate_resume_continuous_sha256": "0" * 64,
        "candidate_resume_split_sha256": "0" * 64,
        "evaluation_first_sha256": "1" * 64,
        "evaluation_second_sha256": "1" * 64,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    return {
        "output_root": tmp_path / "paper_candidate",
        "candidate_id": CANDIDATE_ID,
        "terminal_receipt": receipt,
        "trajectory": trajectory,
        "derivation_card": card,
        "implementation": implementation,
        "runtime_gate": runtime_gate,
        "parent_output": parent,
        "parent_scientific_git_commit": PARENT_COMMIT,
        "parent_protocol_fingerprint": PARENT_PROTOCOL,
    }


def test_candidate_lock_binds_small25_parent_and_runtime_without_authorizing(tmp_path: Path) -> None:
    result = materialize_candidate_lock(**_fixture(tmp_path))
    assert result["status"] == "PASS_FULL_DATA_CANDIDATE_LOCK"
    assert result["full_data_authorized"] is False
    assert result["paired_metric_control"] is False
    assert result["confirmation20_opened"] is False
    assert result["parent_paper"]["scientific_git_commit"] == PARENT_COMMIT
    assert (
        tmp_path / "paper_candidate" / "candidate_locks" / CANDIDATE_ID
        / "CANDIDATE_LOCK.json"
    ).is_file()


def test_candidate_lock_rejects_negative_or_plain_collapse_trajectory(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    path = Path(kwargs["trajectory"])
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    trajectory["status"] = "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION"
    _write(path, trajectory)
    receipt_path = Path(kwargs["terminal_receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["trajectory_sha256"] = file_sha256(path)
    receipt["trajectory_status"] = trajectory["status"]
    _write(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="trajectory_status"):
        materialize_candidate_lock(**kwargs)


def test_candidate_lock_rejects_cross_code_runtime_mismatch(tmp_path: Path) -> None:
    kwargs = _fixture(tmp_path)
    path = Path(kwargs["runtime_gate"])
    gate = json.loads(path.read_text(encoding="utf-8"))
    gate["plain_2000_transition_exact"] = False
    _write(path, gate)
    with pytest.raises(RuntimeError, match="plain_2000_transition_exact"):
        materialize_candidate_lock(**kwargs)


def test_candidate_needs_separate_fresh_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _fixture(tmp_path)
    current_commit = "b" * 40
    current_fingerprint = protocol_fingerprint()
    runtime_path = Path(kwargs["runtime_gate"])
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["candidate_git_commit"] = current_commit
    runtime["candidate_protocol_fingerprint"] = current_fingerprint
    _write(runtime_path, runtime)
    lock = materialize_candidate_lock(**kwargs)
    output = Path(kwargs["output_root"])
    _write(output / "gates" / "PREFLIGHT.json", {
        "status": "PASS",
        "node_role": "training",
        "protocol_fingerprint": current_fingerprint,
        "manifest": {"content_hashes_verified": True},
    })
    monkeypatch.setattr(
        "research.paper_aio.candidate_runtime._require_clean_checkout",
        lambda: current_commit,
    )
    monkeypatch.setattr(
        "research.paper_aio.candidate_runtime.git_commit", lambda: current_commit,
    )
    with pytest.raises(RuntimeError, match="missing"):
        require_candidate_authorization(output, CANDIDATE_ID)
    authorization = authorize_candidate(output, CANDIDATE_ID)
    assert authorization["status"] == "PASS_FULL_DATA_CANDIDATE_AUTHORIZATION"
    assert authorization["candidate_lock_sha256"]
    assert lock["full_data_authorized"] is False
    assert require_candidate_authorization(output, CANDIDATE_ID) == authorization
