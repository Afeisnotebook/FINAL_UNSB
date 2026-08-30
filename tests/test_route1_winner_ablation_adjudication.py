from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from operations.local_route1_cross_version_adjudicate import SCHEMA as CROSS_SCHEMA
from operations.local_route1_winner_ablation_adjudicate import (
    SCHEMA,
    adjudicate,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import full_state_hash


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _receipt(
    root: Path, candidate_id: str, *, late: float, algorithm: str | None = None,
) -> Path:
    path = root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    payload = {
        "schema": RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": POSITIVE_STATUS,
        "algorithm_fingerprint": algorithm or f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "candidate_training_core_fingerprint": f"core-{candidate_id}",
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "training_git_commit": "a" * 40,
        "receipt_source_sha256": file_sha256(
            ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
        ),
        "trajectory_sha256": f"trajectory-{candidate_id}",
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": late,
            "e200_macro_psnr_delta": late,
            "late_points_with_four_of_six_positive_domains": 2,
            "late_average_worst_domain_delta": -0.2,
            "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
            "late_mean_macro_ssim_delta": 0.01,
            "late_mean_macro_lpips_delta": -0.01,
        },
        "median_epoch_wall_seconds": 1.0,
        "terminal_integrity": {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
        "plain_e200_verification_sha256": "plain",
        "evaluation_crn_matched_to_same_host_plain": True,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    _write(path, payload)
    _write(Path(str(path) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(path),
    })
    return path


def _observable_plain_identity(root: Path, candidate_id: str) -> None:
    payload = {
        "step": 30000,
        "physical_epoch_completed": 200,
        "target_steps": 30000,
        "model": {"networks": {"G": {"w": torch.tensor([1.0])}}, "optimizers": [], "schedulers": [], "method": {"search_global_step": 30000, "search_total_steps": 30000}},
        "rng": {"python": "rng"},
        "samplers": {"primary": {"cursor": 0}, "secondary": {"cursor": 0}},
        "probe": {"id": "plain"},
        "metadata": {"git": "plain"},
    }
    plain_checkpoint = root / "anchors" / "plain" / "full_state_latest.pt"
    candidate_checkpoint = root / "candidates" / candidate_id / "full_state_latest.pt"
    plain_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    candidate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, plain_checkpoint)
    candidate_payload = copy.deepcopy(payload)
    candidate_payload["probe"] = {"id": candidate_id}
    candidate_payload["metadata"] = {"git": "candidate"}
    candidate_payload["model"]["method"]["route1_observer"] = {
        "family": "test", "update_index": 30000,
    }
    torch.save(candidate_payload, candidate_checkpoint)
    _write(root / "anchors" / "plain" / "full_state_latest.pt.json", {
        "physical_epoch_completed": 200,
        "scientific_state_sha256": full_state_hash(payload),
    })
    _write(root / "candidates" / candidate_id / "full_state_latest.pt.json", {
        "physical_epoch_completed": 200,
        "scientific_state_sha256": full_state_hash(candidate_payload),
    })
    metric = {
        "macro_psnr": 20.0,
        "images": [{"stem": "a", "psnr": 20.0}],
        "confirmation20_opened": False,
        "probe_id": "plain",
    }
    _write(root / "anchors" / "plain" / "metrics" / "e200.json", metric)
    _write(root / "candidates" / candidate_id / "metrics" / "e200.json", {
        **metric, "probe_id": candidate_id,
    })


def _cross(root: Path, winner: str, algorithm: str) -> Path:
    path = root / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    _write(path, {
        "schema": CROSS_SCHEMA,
        "status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE",
        "selected_candidate_id": winner,
        "selected_algorithm_fingerprint": algorithm,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    return path


def test_winner_ablation_adjudication_accepts_matched_e200_receipts(tmp_path):
    full_id = "G1-FULL"
    observable_id = "ABL-OBSERVE"
    full = _receipt(tmp_path, full_id, late=0.3, algorithm="algorithm-full")
    proposal = _receipt(tmp_path, "ABL-PROPOSAL", late=0.1)
    observable = _receipt(tmp_path, observable_id, late=0.0)
    _observable_plain_identity(tmp_path, observable_id)
    cross = _cross(tmp_path, full_id, "algorithm-full")

    result = adjudicate(
        output_root=tmp_path,
        cross_adjudication_path=cross,
        proposal_receipt_path=proposal,
        observable_receipt_path=observable,
        full_receipt_path=full,
        output_path=tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json",
    )
    assert result["schema"] == SCHEMA
    assert result["status"] == "COMPLETE_NO_SELECTION_CHANGE"
    assert result["observable_only_identity"]["status"] == (
        "EXACT_PLAIN_E200_DYNAMICS_IDENTITY"
    )
    assert set(result["roles"]) == {
        "proposal_only", "observable_only", "projected_or_full",
    }


def test_winner_ablation_adjudication_refuses_observable_state_drift(tmp_path):
    full_id = "G1-FULL"
    observable_id = "ABL-OBSERVE"
    full = _receipt(tmp_path, full_id, late=0.3, algorithm="algorithm-full")
    proposal = _receipt(tmp_path, "ABL-PROPOSAL", late=0.1)
    observable = _receipt(tmp_path, observable_id, late=0.0)
    _observable_plain_identity(tmp_path, observable_id)
    checkpoint = tmp_path / "candidates" / observable_id / "full_state_latest.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"]["networks"]["G"]["w"] = torch.tensor([2.0])
    torch.save(payload, checkpoint)
    sidecar = tmp_path / "candidates" / observable_id / "full_state_latest.pt.json"
    _write(sidecar, {
        "physical_epoch_completed": 200,
        "scientific_state_sha256": full_state_hash(payload),
    })
    cross = _cross(tmp_path, full_id, "algorithm-full")
    with pytest.raises(RuntimeError, match="dynamics state differs from plain"):
        adjudicate(
            output_root=tmp_path,
            cross_adjudication_path=cross,
            proposal_receipt_path=proposal,
            observable_receipt_path=observable,
            full_receipt_path=full,
            output_path=tmp_path / "operations" / "WINNER_ABLATION_ADJUDICATION.json",
        )
