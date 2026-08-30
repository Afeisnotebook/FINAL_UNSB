from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.local_route1 import generation1_adjudication as adjudication
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
    adjudicate_generation1,
    trajectory_rank_key,
)


def _trajectory(candidate_id: str, *, late: float, final: float, cost_guard: float = 0.1):
    return {
        "candidate_id": candidate_id,
        "late_three_mean_macro_psnr_delta": late,
        "e200_macro_psnr_delta": final,
        "late_points_with_four_of_six_positive_domains": 2,
        "late_average_worst_domain_delta": -0.5,
        "candidate_best_to_terminal_three_point_rolling_drawdown": cost_guard,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
    }


def test_generation1_ranking_prioritizes_late_three_before_terminal_or_cost():
    sustained = _trajectory("sustained", late=0.2, final=0.1)
    terminal = _trajectory("terminal", late=0.1, final=0.8)
    assert trajectory_rank_key(sustained, measured_epoch_seconds=99.0) < trajectory_rank_key(
        terminal, measured_epoch_seconds=1.0
    )


def test_generation1_ranking_uses_terminal_then_registered_guards():
    stronger_terminal = _trajectory("a", late=0.2, final=0.2)
    weaker_terminal = _trajectory("b", late=0.2, final=0.1)
    assert trajectory_rank_key(stronger_terminal) < trajectory_rank_key(weaker_terminal)


def test_generation1_ranking_treats_missing_scientific_values_as_worst():
    complete = _trajectory("complete", late=0.0, final=0.0)
    missing = {"candidate_id": "missing"}
    assert trajectory_rank_key(complete) < trajectory_rank_key(
        missing, measured_epoch_seconds=math.inf
    )


def _registration(root: Path, candidate_id: str):
    card = root / "cards" / f"{candidate_id}.json"
    implementation = root / "implementations" / f"{candidate_id}.json"
    return SimpleNamespace(
        algorithm_fingerprint=f"algorithm-{candidate_id}",
        candidate_fingerprint=f"execution-{candidate_id}",
        base_protocol_fingerprint="base-protocol",
        card_path=card,
        implementation_path=implementation,
    )


def _complete_payload(candidate_id: str, *, late: float, status: str):
    return {
        "schema": adjudication.TRAJECTORY_SCHEMA,
        "candidate_id": candidate_id,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"execution-{candidate_id}",
        "status": status,
        "trajectory": [{"epoch": 150}, {"epoch": 175}, {"epoch": 200}],
        **_trajectory(candidate_id, late=late, final=late),
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }


def _terminal_artifacts(root: Path, candidate_id: str, registration) -> None:
    protocol = adjudication.load_protocol()
    target = int(protocol["local_view"]["target_updates_per_lane"])
    candidate_root = root / "candidates" / candidate_id
    metadata = {
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    payload = {"step": target, "value": torch.tensor([1.0])}
    latest = candidate_root / "full_state_latest.pt"
    latest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, latest)
    scientific_hash = adjudication.full_state_hash(payload)
    latest_sidecar = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "probe_id": candidate_id,
        "step": target,
        "physical_epoch_completed": 200,
        "target_steps": target,
        "full_state_sha256": adjudication.file_sha256(latest),
        "scientific_state_sha256": scientific_hash,
        "metadata": metadata,
    }
    Path(str(latest) + ".json").write_text(json.dumps(latest_sidecar), encoding="utf-8")

    images = [
        {"domain": f"domain-{index // 70}", "stem": str(index),
         "order": index % 70, "crn_bundle_sha256": f"crn-{index}"}
        for index in range(420)
    ]
    lpips_epochs = {int(value) for value in protocol["local_view"]["lpips_epochs"]}
    for epoch in (int(value) for value in protocol["local_view"]["trajectory_epochs"]):
        step = epoch * target // 200
        milestone = candidate_root / "milestones" / f"e{epoch:03d}.pt"
        milestone.parent.mkdir(parents=True, exist_ok=True)
        milestone.write_bytes(f"checkpoint-{epoch}".encode())
        sidecar = {
            "probe_id": candidate_id, "step": step,
            "physical_epoch_completed": epoch,
            "full_state_sha256": adjudication.file_sha256(milestone),
            "scientific_state_sha256": scientific_hash if epoch == 200 else f"state-{epoch}",
            "metadata": metadata,
        }
        Path(str(milestone) + ".json").write_text(json.dumps(sidecar), encoding="utf-8")
        base_metric = {
            "schema": "local-route1-discovery70-crn-single-rollout-v1",
            "split": "discovery", "count_per_domain": 70, "replicates": 1,
            "protocol_fingerprint": registration.base_protocol_fingerprint,
            "evaluation_input_sha256": "evaluation-input",
            "lpips_requested": epoch in lpips_epochs,
            "images": images,
        }
        plain_path = root / "anchors" / "plain" / "metrics" / f"e{epoch:03d}.json"
        plain_path.parent.mkdir(parents=True, exist_ok=True)
        plain_path.write_text(json.dumps(base_metric), encoding="utf-8")
        metric = {
            **base_metric,
            "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "candidate_fingerprint": registration.candidate_fingerprint,
            "epoch": epoch, "updates": step, "data_epoch": epoch,
        }
        metric_path = candidate_root / "metrics" / f"e{epoch:03d}.json"
        metric_path.parent.mkdir(parents=True, exist_ok=True)
        metric_path.write_text(json.dumps(metric), encoding="utf-8")


def test_terminal_artifact_gate_recomputes_hashes_and_crn_identity(tmp_path):
    registration = _registration(tmp_path, "candidate")
    _terminal_artifacts(tmp_path, "candidate", registration)
    result = adjudication._validate_terminal_artifacts(
        output_root=tmp_path, candidate_id="candidate", registration=registration,
    )
    assert result["status"] == "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"
    assert result["evaluation_crn_matched_to_plain"] is True
    (tmp_path / "candidates" / "candidate" / "milestones" / "e100.pt").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="checkpoint hash mismatch"):
        adjudication._validate_terminal_artifacts(
            output_root=tmp_path, candidate_id="candidate", registration=registration,
        )


def test_generation1_adjudication_waits_for_every_e200_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adjudication, "load_candidate_registration",
        lambda root, candidate_id, require_gate: _registration(root, candidate_id),
    )
    result = adjudicate_generation1(tmp_path, ("first", "second"))
    assert result["status"] == "WAITING_FOR_ALL_MATCHED_E200_TRAJECTORIES"
    assert result["ranking_performed"] is False
    assert {row["candidate_id"] for row in result["pending"]} == {"first", "second"}


def test_generation1_adjudication_freezes_only_ranked_positive_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adjudication, "load_candidate_registration",
        lambda root, candidate_id, require_gate: _registration(root, candidate_id),
    )
    frozen = []
    monkeypatch.setattr(
        adjudication, "freeze_for_seed_validation",
        lambda root, candidate_id: frozen.append(candidate_id) or {"candidate_id": candidate_id},
    )
    monkeypatch.setattr(
        adjudication, "_validate_terminal_artifacts",
        lambda **kwargs: {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
    )
    for candidate_id, late, status in (
        ("positive", 0.2, POSITIVE_STATUS),
        ("negative", 0.4, NEGATIVE_STATUS),
    ):
        root = tmp_path / "candidates" / candidate_id
        root.mkdir(parents=True)
        (root / "CANDIDATE_TRAJECTORY.json").write_text(
            json.dumps(_complete_payload(candidate_id, late=late, status=status)),
            encoding="utf-8",
        )
    result = adjudicate_generation1(
        tmp_path, ("positive", "negative"), freeze_winner=True,
    )
    assert result["status"] == "SEED2026_WINNER_READY_FOR_FROZEN_SEED2027"
    assert result["selected_candidate_id"] == "positive"
    assert result["ranking"][0]["candidate_id"] == "negative"
    assert result["ranking"][0]["terminal_integrity"]["status"] == (
        "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"
    )
    assert frozen == ["positive"]
