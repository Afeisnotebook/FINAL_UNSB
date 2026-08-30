"""Post-e200 adjudication for evidence-frozen Generation-1 candidates.

This module is deliberately downstream of complete trajectories.  It cannot
read a training controller, select a checkpoint, stop a run, or modify an
algorithm.  Its only optional mutation is the existing immutable seed-2027
freeze for the highest-ranked candidate that already passed every registered
seed-2026 numeric guard.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable

import torch

from .candidate_runner import freeze_for_seed_validation
from .candidates import load_candidate_registration, validate_candidate_id
from .protocol import file_sha256, load_protocol
from .runtime import full_state_hash, write_json


SCHEMA = "final-unsb-route1-generation1-e200-adjudication-v1"
TRAJECTORY_SCHEMA = "final-unsb-route1-candidate-trajectory-v1"
POSITIVE_STATUS = "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION"
NEGATIVE_STATUS = "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _number(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _metric_crn_identity(metric: dict[str, Any]) -> list[tuple[Any, ...]]:
    images = metric.get("images")
    if not isinstance(images, list) or len(images) != 420:
        raise RuntimeError("terminal metric must contain exactly 420 discovery images")
    return [
        (
            row.get("domain"), row.get("stem"), int(row.get("order", -1)),
            row.get("crn_bundle_sha256"),
        )
        for row in images if isinstance(row, dict)
    ]


def _validate_terminal_artifacts(
    *, output_root: Path, candidate_id: str, registration: Any,
) -> dict[str, Any]:
    """Independently accept e200 state and every registered metric post-hoc."""
    protocol = load_protocol()
    candidate_root = output_root / "candidates" / candidate_id
    target = int(protocol["local_view"]["target_updates_per_lane"])
    epochs = [int(value) for value in protocol["local_view"]["trajectory_epochs"]]
    lpips_epochs = {int(value) for value in protocol["local_view"]["lpips_epochs"]}

    latest = candidate_root / "full_state_latest.pt"
    latest_sidecar_path = Path(str(latest) + ".json")
    if not latest.is_file() or not latest_sidecar_path.is_file():
        raise RuntimeError(f"candidate terminal full state is missing: {candidate_id}")
    latest_sidecar = _read_json(latest_sidecar_path)
    required_latest = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "probe_id": candidate_id,
        "step": target,
        "physical_epoch_completed": 200,
        "target_steps": target,
    }
    for key, expected in required_latest.items():
        if latest_sidecar.get(key) != expected:
            raise RuntimeError(f"candidate terminal sidecar {key} mismatch: {candidate_id}")
    if file_sha256(latest) != latest_sidecar.get("full_state_sha256"):
        raise RuntimeError(f"candidate terminal checkpoint hash mismatch: {candidate_id}")
    metadata = latest_sidecar.get("metadata", {})
    for key, expected in {
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }.items():
        if metadata.get(key) != expected:
            raise RuntimeError(f"candidate terminal metadata {key} mismatch: {candidate_id}")
    payload = torch.load(latest, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != target:
        raise RuntimeError(f"candidate terminal payload is not e200: {candidate_id}")
    if full_state_hash(payload) != latest_sidecar.get("scientific_state_sha256"):
        raise RuntimeError(f"candidate terminal scientific state hash mismatch: {candidate_id}")

    metric_hashes: dict[str, str] = {}
    milestone_hashes: dict[str, str] = {}
    for epoch in epochs:
        milestone = candidate_root / "milestones" / f"e{epoch:03d}.pt"
        sidecar_path = Path(str(milestone) + ".json")
        metric_path = candidate_root / "metrics" / f"e{epoch:03d}.json"
        plain_path = output_root / "anchors" / "plain" / "metrics" / f"e{epoch:03d}.json"
        for path in (milestone, sidecar_path, metric_path, plain_path):
            if not path.is_file():
                raise RuntimeError(f"registered terminal evidence is missing: {path}")
        sidecar = _read_json(sidecar_path)
        expected_step = int(epoch * target // 200)
        if (
            sidecar.get("probe_id") != candidate_id
            or int(sidecar.get("step", -1)) != expected_step
            or int(sidecar.get("physical_epoch_completed", -1)) != epoch
            or sidecar.get("metadata", {}).get("algorithm_fingerprint")
            != registration.algorithm_fingerprint
            or sidecar.get("metadata", {}).get("candidate_fingerprint")
            != registration.candidate_fingerprint
        ):
            raise RuntimeError(f"candidate milestone identity mismatch: {candidate_id} e{epoch}")
        milestone_hash = file_sha256(milestone)
        if milestone_hash != sidecar.get("full_state_sha256"):
            raise RuntimeError(f"candidate milestone checkpoint hash mismatch: {candidate_id} e{epoch}")
        if epoch == 200 and sidecar.get("scientific_state_sha256") != latest_sidecar.get(
            "scientific_state_sha256"
        ):
            raise RuntimeError(f"candidate e200 latest/milestone scientific state differ: {candidate_id}")

        metric = _read_json(metric_path)
        plain = _read_json(plain_path)
        required_metric = {
            "schema": "local-route1-discovery70-crn-single-rollout-v1",
            "split": "discovery",
            "count_per_domain": 70,
            "replicates": 1,
            "protocol_fingerprint": registration.base_protocol_fingerprint,
            "evaluation_input_sha256": plain.get("evaluation_input_sha256"),
            "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "candidate_fingerprint": registration.candidate_fingerprint,
            "epoch": epoch,
            "updates": expected_step,
            "data_epoch": epoch,
            "lpips_requested": epoch in lpips_epochs,
        }
        for key, expected in required_metric.items():
            if metric.get(key) != expected:
                raise RuntimeError(f"candidate metric {key} mismatch: {candidate_id} e{epoch}")
        if _metric_crn_identity(metric) != _metric_crn_identity(plain):
            raise RuntimeError(f"candidate/plain CRN bundle mismatch: {candidate_id} e{epoch}")
        milestone_hashes[str(epoch)] = milestone_hash
        metric_hashes[str(epoch)] = file_sha256(metric_path)

    return {
        "status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET",
        "terminal_checkpoint_sha256": file_sha256(latest),
        "terminal_scientific_state_sha256": latest_sidecar["scientific_state_sha256"],
        "milestone_checkpoint_sha256": milestone_hashes,
        "metric_sha256": metric_hashes,
        "evaluation_crn_matched_to_plain": True,
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }


def trajectory_rank_key(
    trajectory: dict[str, Any], *, measured_epoch_seconds: float = math.inf,
) -> tuple[Any, ...]:
    """Return the frozen lexicographic ranking key (smaller is better)."""
    candidate_id = validate_candidate_id(str(trajectory.get("candidate_id", "")))
    return (
        -_number(trajectory.get("late_three_mean_macro_psnr_delta"), default=-math.inf),
        -_number(trajectory.get("e200_macro_psnr_delta"), default=-math.inf),
        -int(trajectory.get("late_points_with_four_of_six_positive_domains", -1)),
        -_number(trajectory.get("late_average_worst_domain_delta"), default=-math.inf),
        _number(
            trajectory.get("candidate_best_to_terminal_three_point_rolling_drawdown"),
            default=math.inf,
        ),
        -_number(trajectory.get("late_mean_macro_ssim_delta"), default=-math.inf),
        _number(trajectory.get("late_mean_macro_lpips_delta"), default=math.inf),
        _number(measured_epoch_seconds, default=math.inf),
        candidate_id,
    )


def _median_epoch_seconds(candidate_root: Path) -> float | None:
    trace = candidate_root / "TRAIN_TRACE.jsonl"
    if not trace.is_file():
        return None
    values = []
    for line in trace.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = _number(row.get("epoch_wall_seconds"), default=math.inf)
        if math.isfinite(value) and value > 0.0:
            values.append(value)
    return None if not values else float(statistics.median(values))


def _validate_trajectory(
    *, trajectory: dict[str, Any], candidate_id: str, algorithm_fingerprint: str,
) -> None:
    if trajectory.get("schema") != TRAJECTORY_SCHEMA:
        raise RuntimeError(f"candidate trajectory schema mismatch: {candidate_id}")
    if trajectory.get("candidate_id") != candidate_id:
        raise RuntimeError(f"candidate trajectory id mismatch: {candidate_id}")
    if trajectory.get("algorithm_fingerprint") != algorithm_fingerprint:
        raise RuntimeError(f"candidate trajectory algorithm changed: {candidate_id}")
    if trajectory.get("paired_metrics_used_for_training_or_gate") is not False:
        raise RuntimeError(f"candidate trajectory used paired training control: {candidate_id}")
    if trajectory.get("confirmation20_opened") is not False:
        raise RuntimeError(f"candidate trajectory opened confirmation20: {candidate_id}")
    rows = trajectory.get("trajectory")
    if not isinstance(rows, list):
        raise RuntimeError(f"candidate trajectory rows are missing: {candidate_id}")
    epochs = [int(row.get("epoch", -1)) for row in rows if isinstance(row, dict)]
    if trajectory.get("status") in (POSITIVE_STATUS, NEGATIVE_STATUS) and 200 not in epochs:
        raise RuntimeError(f"complete candidate trajectory has no e200 row: {candidate_id}")


def adjudicate_generation1(
    output_root: Path, candidate_ids: Iterable[str], *, freeze_winner: bool = False,
) -> dict[str, Any]:
    """Rank only after all candidates finish e200 and optionally freeze the winner."""
    output_root = Path(output_root).resolve()
    ids = tuple(validate_candidate_id(value) for value in candidate_ids)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("candidate_ids must be a non-empty unique sequence")

    complete: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for candidate_id in ids:
        registration = load_candidate_registration(
            output_root, candidate_id, require_gate=True,
        )
        candidate_root = output_root / "candidates" / candidate_id
        trajectory_path = candidate_root / "CANDIDATE_TRAJECTORY.json"
        if not trajectory_path.is_file():
            sidecar_path = candidate_root / "full_state_latest.pt.json"
            epoch = 0
            if sidecar_path.is_file():
                epoch = int(_read_json(sidecar_path).get("physical_epoch_completed", 0))
            pending.append({
                "candidate_id": candidate_id,
                "latest_data_epoch": epoch,
                "target_data_epoch": 200,
                "reason": "complete CANDIDATE_TRAJECTORY.json is not present",
            })
            continue
        trajectory = _read_json(trajectory_path)
        _validate_trajectory(
            trajectory=trajectory, candidate_id=candidate_id,
            algorithm_fingerprint=registration.algorithm_fingerprint,
        )
        if trajectory.get("status") not in (POSITIVE_STATUS, NEGATIVE_STATUS):
            pending.append({
                "candidate_id": candidate_id,
                "latest_data_epoch": max(
                    [int(row.get("epoch", 0)) for row in trajectory.get("trajectory", [])]
                    or [0]
                ),
                "target_data_epoch": 200,
                "reason": f"trajectory status is {trajectory.get('status')}",
            })
            continue
        terminal_integrity = _validate_terminal_artifacts(
            output_root=output_root, candidate_id=candidate_id,
            registration=registration,
        )
        median_seconds = _median_epoch_seconds(candidate_root)
        complete.append({
            "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "candidate_fingerprint": registration.candidate_fingerprint,
            "trajectory": trajectory,
            "trajectory_path": str(trajectory_path),
            "trajectory_sha256": file_sha256(trajectory_path),
            "median_epoch_wall_seconds": median_seconds,
            "derivation_card": str(registration.card_path),
            "implementation": str(registration.implementation_path),
            "terminal_integrity": terminal_integrity,
        })

    if pending:
        result = {
            "schema": SCHEMA,
            "status": "WAITING_FOR_ALL_MATCHED_E200_TRAJECTORIES",
            "candidate_ids": list(ids),
            "pending": pending,
            "complete_candidate_count": len(complete),
            "ranking_performed": False,
            "winner_frozen_for_seed2027": False,
            "paired_metrics_used_for_training_or_control": False,
            "confirmation20_opened": False,
        }
        write_json(output_root / "operations" / "GENERATION1_E200_ADJUDICATION.json", result)
        return result

    ranked = sorted(
        complete,
        key=lambda row: trajectory_rank_key(
            row["trajectory"],
            measured_epoch_seconds=(
                math.inf if row["median_epoch_wall_seconds"] is None
                else row["median_epoch_wall_seconds"]
            ),
        ),
    )
    ranking = []
    for index, row in enumerate(ranked, start=1):
        trajectory = row["trajectory"]
        ranking.append({
            "rank": index,
            "candidate_id": row["candidate_id"],
            "algorithm_fingerprint": row["algorithm_fingerprint"],
            "candidate_fingerprint": row["candidate_fingerprint"],
            "trajectory_sha256": row["trajectory_sha256"],
            "trajectory_status": trajectory["status"],
            "late_three_mean_macro_psnr_delta": trajectory.get(
                "late_three_mean_macro_psnr_delta"
            ),
            "e200_macro_psnr_delta": trajectory.get("e200_macro_psnr_delta"),
            "late_points_with_four_of_six_positive_domains": trajectory.get(
                "late_points_with_four_of_six_positive_domains"
            ),
            "late_average_worst_domain_delta": trajectory.get(
                "late_average_worst_domain_delta"
            ),
            "late_mean_macro_ssim_delta": trajectory.get("late_mean_macro_ssim_delta"),
            "late_mean_macro_lpips_delta": trajectory.get("late_mean_macro_lpips_delta"),
            "candidate_best_to_terminal_three_point_rolling_drawdown": trajectory.get(
                "candidate_best_to_terminal_three_point_rolling_drawdown"
            ),
            "median_epoch_wall_seconds": row["median_epoch_wall_seconds"],
            "derivation_card": row["derivation_card"],
            "implementation": row["implementation"],
            "terminal_integrity": row["terminal_integrity"],
        })

    eligible = [row for row in ranked if row["trajectory"]["status"] == POSITIVE_STATUS]
    winner = eligible[0] if eligible else ranked[0]
    seed_freeze = None
    if eligible and freeze_winner:
        seed_freeze = freeze_for_seed_validation(output_root, winner["candidate_id"])
    result = {
        "schema": SCHEMA,
        "status": (
            "SEED2026_WINNER_READY_FOR_FROZEN_SEED2027"
            if eligible else
            "NO_SEED2026_NUMERIC_GATE_PASS_CAUSAL_DEFECT_ADJUDICATION_REQUIRED"
        ),
        "candidate_ids": list(ids),
        "ranking_policy": [
            "late-three macro PSNR delta descending",
            "e200 macro PSNR delta descending",
            "late points with at least four positive domains descending",
            "late average worst-domain delta descending",
            "candidate terminal rolling drawdown ascending",
            "late macro SSIM delta descending",
            "late macro LPIPS delta ascending",
            "measured median epoch wall time ascending",
        ],
        "ranking": ranking,
        "selected_candidate_id": winner["candidate_id"],
        "selection_role": "seed2026_numeric_winner" if eligible else "current_best_fallback",
        "winner_frozen_for_seed2027": seed_freeze is not None,
        "seed_validation_freeze": seed_freeze,
        "negative_candidate_rule": (
            "A negative current implementation may use its one mathematical revision only "
            "after a target-blind defect measurement proves reduction and supplies a new "
            "causal failure reason; fixed windows, handoff and paired control remain forbidden."
        ),
        "paired_metrics_used_only_after_complete_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "operations" / "GENERATION1_E200_ADJUDICATION.json", result)
    return result
