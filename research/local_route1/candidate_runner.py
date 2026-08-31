"""Matched small25/e200 runner for algorithms discovered by the causal atlas."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .anchors import prepare_probe
from .candidates import CandidateRegistration, load_candidate_registration
from .evaluate import compare_to_plain, evaluate_model
from .protocol import (
    epoch_to_step,
    file_sha256,
    git_commit,
    load_protocol,
    milestone_steps,
    step_to_physical_epoch,
    steps_per_epoch,
)
from .runtime import (
    full_state_hash,
    load_full_state,
    save_full_state,
    write_json,
)
from .seed_validation import SEED_FREEZE_SCHEMA


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _candidate_root(output_root: Path, candidate_id: str) -> Path:
    return Path(output_root) / "candidates" / candidate_id


def validate_matched_plain_late_metrics(output_root: Path) -> dict[int, str]:
    """Require a real LPIPS authority before any new long candidate starts.

    Missing baseline LPIPS is an engineering-invalid comparison, not evidence
    that an algorithm violated a perceptual guardrail.  Hashing the accepted
    payloads also binds later candidate contracts to the repaired authority.
    """
    protocol = load_protocol()
    result = {}
    for epoch in (int(value) for value in protocol["local_view"]["late_epochs"]):
        path = Path(output_root) / "anchors" / "plain" / "metrics" / f"e{epoch:03d}.json"
        if not path.is_file():
            raise RuntimeError(f"matched plain late metric missing at e{epoch}")
        metric = _read_json(path)
        if (
            metric.get("lpips_requested") is not True
            or metric.get("lpips_available") is not True
            or metric.get("macro_lpips") is None
        ):
            raise RuntimeError(
                f"matched plain LPIPS authority is unavailable at e{epoch}"
            )
        images = metric.get("images", [])
        domains = metric.get("domains", {})
        if (
            len(images) != 420
            or len(domains) != 6
            or any(row.get("lpips") is None for row in images)
            or any(row.get("lpips") is None for row in domains.values())
        ):
            raise RuntimeError(f"matched plain LPIPS payload is incomplete at e{epoch}")
        result[epoch] = file_sha256(path)
    return result


def _assert_scientific_prerequisites(output_root: Path) -> None:
    protocol = load_protocol()
    target = int(protocol["local_view"]["target_updates_per_lane"])
    calibration_path = output_root / "evidence" / "PROXY_CALIBRATION.json"
    if not calibration_path.is_file():
        raise RuntimeError("candidate long run is blocked until proxy calibration exists")
    calibration = _read_json(calibration_path)
    if calibration.get("status") != "CALIBRATED":
        raise RuntimeError(
            f"candidate long run is blocked because proxy status is {calibration.get('status')}"
        )
    plain_path = output_root / "anchors" / "plain" / "full_state_latest.pt"
    if not plain_path.is_file():
        raise RuntimeError("candidate long run requires the matched plain e200 state")
    plain = torch.load(plain_path, map_location="cpu", weights_only=False)
    if int(plain.get("step", -1)) != target:
        raise RuntimeError("candidate long run requires plain at exactly e200")
    validate_matched_plain_late_metrics(output_root)


def _candidate_metadata(
    *, registration: CandidateRegistration, manifest_path: Path,
) -> dict:
    protocol = load_protocol()
    return {
        "project_id": protocol["project_id"],
        "probe_id": registration.candidate_id,
        "candidate_id": registration.candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed": int(protocol["seed"]),
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_fingerprint": registration.candidate_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "base_protocol_fingerprint": registration.base_protocol_fingerprint,
        "candidate_training_core_fingerprint": registration.candidate_training_core_fingerprint,
        "base_e0_scientific_state_sha256": registration.base_e0_scientific_state_sha256,
        "causal_matrix_sha256": registration.causal_matrix_sha256,
        "reversal_atlas_sha256": registration.reversal_atlas_sha256,
        "training_git_commit": git_commit(),
        "data_epochs_target": int(protocol["local_view"]["target_epochs"]),
        "steps_per_data_epoch": steps_per_epoch(protocol),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _write_metric(
    *, model, rows: list[dict], data_root: Path, registration: CandidateRegistration,
    candidate_root: Path, output_root: Path, epoch: int,
) -> dict:
    protocol = load_protocol()
    metric_path = candidate_root / "metrics" / f"e{epoch:03d}.json"
    if metric_path.is_file():
        return _read_json(metric_path)
    metrics = evaluate_model(
        model,
        rows=rows,
        data_root=data_root,
        # Reuse the anchor fingerprint as the CRN key.  Candidate identity is
        # recorded separately and must not change evaluation randomness.
        protocol_hash=registration.base_protocol_fingerprint,
        include_lpips=epoch in set(int(value) for value in protocol["local_view"]["lpips_epochs"]),
    )
    metrics.update({
        "candidate_id": registration.candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "epoch": int(epoch),
        "updates": epoch_to_step(epoch, protocol),
        "data_epoch": int(epoch),
    })
    plain_path = output_root / "anchors" / "plain" / "metrics" / f"e{epoch:03d}.json"
    if not plain_path.is_file():
        raise RuntimeError(f"matched plain metric missing at e{epoch}")
    delta = compare_to_plain(metrics, _read_json(plain_path), epoch=epoch)
    metrics["matched_plain_delta"] = delta
    write_json(metric_path, metrics)
    return metrics


def _backfill_current_milestone_if_needed(
    *, model, rows: list[dict], data_root: Path, registration: CandidateRegistration,
    candidate_root: Path, output_root: Path, start_step: int,
) -> None:
    protocol = load_protocol()
    if start_step not in set(milestone_steps(protocol)):
        return
    epoch = start_step // steps_per_epoch(protocol)
    _write_metric(
        model=model, rows=rows, data_root=data_root, registration=registration,
        candidate_root=candidate_root, output_root=output_root, epoch=epoch,
    )


def late_rolling_drawdown(trajectory: list[dict], *, window: int = 3) -> float | None:
    """Return best rolling absolute PSNR minus the terminal rolling PSNR."""
    values = [float(row["macro_psnr"]) for row in trajectory]
    if len(values) < int(window) or int(window) <= 0:
        return None
    rolling = [
        float(np.mean(values[index - window:index]))
        for index in range(window, len(values) + 1)
    ]
    return float(max(rolling) - rolling[-1])


def plain_collapse_adjudication(late: list[dict]) -> dict:
    """Reject a relative win when both method and plain collapse late.

    A stable candidate may remain admissible when plain declines, but a method
    that also loses more than 0.3 dB from e150 to e200 cannot claim that its
    relative advantage is an absolute-quality improvement.
    """
    if len(late) != 3 or [int(row["epoch"]) for row in late] != [150, 175, 200]:
        return {"status": "INCOMPLETE", "threshold_db": -0.3}
    candidate_change = float(late[-1]["macro_psnr"] - late[0]["macro_psnr"])
    plain_change = float(
        late[-1]["plain_macro_psnr"] - late[0]["plain_macro_psnr"]
    )
    both_collapse = plain_change < -0.3 and candidate_change < -0.3
    return {
        "status": (
            "FAIL_RELATIVE_ADVANTAGE_COINCIDES_WITH_BOTH_COLLAPSING"
            if both_collapse else "PASS_NOT_PLAIN_COLLAPSE"
        ),
        "threshold_db": -0.3,
        "candidate_e150_to_e200_change_db": candidate_change,
        "plain_e150_to_e200_change_db": plain_change,
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }


def summarize_candidate(output_root: Path, candidate_id: str) -> dict:
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    protocol = load_protocol()
    candidate_root = _candidate_root(output_root, candidate_id)
    trajectory = []
    for epoch in (int(value) for value in protocol["local_view"]["trajectory_epochs"]):
        metric_path = candidate_root / "metrics" / f"e{epoch:03d}.json"
        plain_path = output_root / "anchors" / "plain" / "metrics" / f"e{epoch:03d}.json"
        if metric_path.is_file() and plain_path.is_file():
            trajectory.append(compare_to_plain(
                _read_json(metric_path), _read_json(plain_path), epoch=epoch,
            ))
    late_epochs = {int(value) for value in protocol["local_view"]["late_epochs"]}
    late = [row for row in trajectory if row["epoch"] in late_epochs]
    complete = len(late) == 3 and late[-1]["epoch"] == 200
    late_mean = None if not complete else float(np.mean([row["macro_psnr_delta"] for row in late]))
    final_delta = None if not complete else float(late[-1]["macro_psnr_delta"])
    average_worst = None if not complete else float(np.mean([row["worst_domain_delta"] for row in late]))
    late_ssim = None if not complete else float(np.mean([row["macro_ssim_delta"] for row in late]))
    lpips_values = [] if not complete else [
        row["macro_lpips_delta"] for row in late if row["macro_lpips_delta"] is not None
    ]
    late_lpips = None if len(lpips_values) != 3 else float(np.mean(lpips_values))
    absolute_psnr = [float(row["macro_psnr"]) for row in trajectory]
    peak_to_terminal_drawdown = (
        None if not complete or not absolute_psnr
        else float(max(absolute_psnr) - absolute_psnr[-1])
    )
    rolling_drawdown = late_rolling_drawdown(trajectory)
    collapse = plain_collapse_adjudication(late)
    numeric_gate = bool(
        complete
        and late_mean is not None and late_mean > 0.0
        and final_delta is not None and final_delta > 0.0
        and sum(row["positive_domains"] >= 4 for row in late) >= 2
        and average_worst is not None and average_worst > -1.0
        and late_ssim is not None and late_ssim >= 0.0
        and late_lpips is not None and late_lpips <= 0.0
        and rolling_drawdown is not None and rolling_drawdown <= 0.3
        and collapse["status"] == "PASS_NOT_PLAIN_COLLAPSE"
    )
    result = {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "status": (
            "INCOMPLETE_E200" if not complete else
            "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION" if numeric_gate else
            "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION"
        ),
        "trajectory": trajectory,
        "late_three_mean_macro_psnr_delta": late_mean,
        "e200_macro_psnr_delta": final_delta,
        "late_points_with_four_of_six_positive_domains": sum(
            row["positive_domains"] >= 4 for row in late
        ),
        "late_average_worst_domain_delta": average_worst,
        "late_mean_macro_ssim_delta": late_ssim,
        "late_mean_macro_lpips_delta": late_lpips,
        "candidate_absolute_peak_to_e200_drawdown": peak_to_terminal_drawdown,
        "candidate_best_to_terminal_three_point_rolling_drawdown": rolling_drawdown,
        "maximum_allowed_rolling_drawdown_db": 0.3,
        "plain_collapse_adjudication": collapse,
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }
    write_json(candidate_root / "CANDIDATE_TRAJECTORY.json", result)
    return result


def freeze_for_seed_validation(output_root: Path, candidate_id: str) -> dict:
    """Freeze a positive seed2026 algorithm without changing its definition."""
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    trajectory = summarize_candidate(output_root, candidate_id)
    if trajectory.get("status") != "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION":
        raise RuntimeError("seed validation freeze requires the complete positive e200 gate")
    collapse = trajectory.get("plain_collapse_adjudication", {})
    if collapse.get("status") != "PASS_NOT_PLAIN_COLLAPSE":
        raise RuntimeError("candidate did not pass the plain-collapse adjudication")
    trajectory_path = _candidate_root(output_root, candidate_id) / "CANDIDATE_TRAJECTORY.json"
    freeze = {
        "schema": SEED_FREEZE_SCHEMA,
        "status": "FROZEN_FOR_SEED_VALIDATION",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2026_candidate_fingerprint": registration.candidate_fingerprint,
        "seed2026_trajectory_sha256": file_sha256(trajectory_path),
        "plain_collapse_adjudication": collapse["status"],
        "authorized_seeds": [2027, 2028],
        "seed2028_requires_seed2027_sign_inconsistency": True,
        "algorithm_changes_after_freeze_forbidden": True,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(_candidate_root(output_root, candidate_id) / "SEED_VALIDATION_FREEZE.json", freeze)
    return freeze


def run_candidate(
    *, output_root: Path, candidate_id: str, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, resume: bool,
    engineering_stop_after_epoch: int | None = None,
) -> dict:
    """Run a frozen candidate from the exact shared e0 without scientific early stop."""
    output_root = Path(output_root).resolve()
    _assert_scientific_prerequisites(output_root)
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    protocol = load_protocol()
    if file_sha256(Path(manifest_path).resolve()) != str(protocol["manifest"]["sha256"]):
        raise RuntimeError("candidate manifest hash differs from the frozen route-1 manifest")
    target_steps = int(protocol["local_view"]["target_updates_per_lane"])
    stop_steps = (
        target_steps if engineering_stop_after_epoch is None
        else epoch_to_step(int(engineering_stop_after_epoch), protocol)
    )
    if stop_steps <= 0 or stop_steps > target_steps:
        raise ValueError("engineering stop epoch must be in [1, 200]")

    candidate_root = _candidate_root(output_root, candidate_id)
    latest = candidate_root / "full_state_latest.pt"
    if latest.is_file() and not resume:
        raise RuntimeError(f"existing candidate state requires --resume: {latest}")
    e0_path = output_root / "shared_e0" / "e0.pt"
    e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    if full_state_hash(e0) != registration.base_e0_scientific_state_sha256:
        raise RuntimeError("shared e0 scientific state changed after candidate registration")
    model, primary, secondary, rows = prepare_probe(
        spec=registration.spec,
        output_root=output_root,
        train_view=Path(train_view).resolve(),
        manifest_path=Path(manifest_path).resolve(),
        gpu=gpu,
        e0=e0,
    )
    metadata = _candidate_metadata(
        registration=registration, manifest_path=Path(manifest_path).resolve(),
    )
    start_step = 0
    if resume and latest.is_file():
        restored = load_full_state(
            latest,
            model=model,
            spec=registration.spec,
            primary=primary,
            secondary=secondary,
            expected_metadata={
                "project_id": metadata["project_id"],
                "probe_id": candidate_id,
                "seed": metadata["seed"],
                "manifest_sha256": metadata["manifest_sha256"],
                "protocol_fingerprint": registration.candidate_fingerprint,
                "candidate_fingerprint": registration.candidate_fingerprint,
                "algorithm_fingerprint": registration.algorithm_fingerprint,
                "base_e0_scientific_state_sha256": registration.base_e0_scientific_state_sha256,
                "candidate_training_core_fingerprint": registration.candidate_training_core_fingerprint,
                "training_git_commit": metadata["training_git_commit"],
            },
        )
        start_step = int(restored["step"])
    _backfill_current_milestone_if_needed(
        model=model, rows=rows, data_root=Path(data_root).resolve(),
        registration=registration, candidate_root=candidate_root,
        output_root=output_root, start_step=start_step,
    )
    if start_step >= stop_steps:
        result = {
            "status": "ALREADY_AT_REQUESTED_STOP",
            "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "candidate_fingerprint": registration.candidate_fingerprint,
            "step": start_step,
            "data_epoch": start_step // steps_per_epoch(protocol),
        }
        if start_step == target_steps:
            result["evaluation"] = summarize_candidate(output_root, candidate_id)
        return result

    milestone_set = set(milestone_steps(protocol))
    started = time.time()
    epoch_started = time.time()
    last_losses = None
    for zero_step in range(start_step, stop_steps):
        physical_epoch = step_to_physical_epoch(zero_step, protocol)
        model.set_train_epoch(physical_epoch)
        model.set_search_step(zero_step, target_steps)
        model.set_input(primary.next(), secondary.next())
        model.optimize_parameters()
        completed = zero_step + 1
        last_losses = model.get_current_losses()
        if completed % steps_per_epoch(protocol) != 0:
            continue

        completed_epoch = completed // steps_per_epoch(protocol)
        model.update_learning_rate()
        sidecar = save_full_state(
            latest,
            model=model,
            spec=registration.spec,
            step=completed,
            target_steps=target_steps,
            primary=primary,
            secondary=secondary,
            metadata=metadata,
        )
        if completed in milestone_set:
            save_full_state(
                candidate_root / "milestones" / f"e{completed_epoch:03d}.pt",
                model=model,
                spec=registration.spec,
                step=completed,
                target_steps=target_steps,
                primary=primary,
                secondary=secondary,
                metadata=metadata,
            )
            _write_metric(
                model=model, rows=rows, data_root=Path(data_root).resolve(),
                registration=registration, candidate_root=candidate_root,
                output_root=output_root, epoch=completed_epoch,
            )
        trace_row = {
            "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "candidate_fingerprint": registration.candidate_fingerprint,
            "updates": completed,
            "data_epoch": completed_epoch,
            "epoch_wall_seconds": time.time() - epoch_started,
            "losses_last_update": last_losses,
            "latest_scientific_state_sha256": sidecar["scientific_state_sha256"],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        _trace(candidate_root / "TRAIN_TRACE.jsonl", trace_row)
        write_json(candidate_root / "HEARTBEAT.json", {
            **trace_row,
            "target_updates": target_steps,
            "target_data_epochs": 200,
            "wall_seconds_this_call": time.time() - started,
        })
        epoch_started = time.time()

    result = {
        "status": "COMPLETE_E200" if stop_steps == target_steps else "ENGINEERING_PAUSE",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "start_updates": start_step,
        "final_updates": stop_steps,
        "final_data_epoch": stop_steps // steps_per_epoch(protocol),
        "target_updates": target_steps,
        "target_data_epochs": 200,
        "wall_seconds_this_call": time.time() - started,
        "metadata": metadata,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if stop_steps == target_steps:
        result["evaluation"] = summarize_candidate(output_root, candidate_id)
    write_json(candidate_root / "RUN_STATE.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
