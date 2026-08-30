"""Frozen-algorithm matched seed validation for route-1 candidates.

Seed validation is intentionally a new execution identity: it changes e0 and
all RNG streams while preserving the algorithm fingerprint.  Plain and the
candidate always train from the same seed-specific e0 on the same host.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .candidates import CandidateRegistration, load_candidate_registration
from .evaluate import compare_to_plain, evaluate_model
from .protocol import (
    ProbeSpec,
    epoch_to_step,
    file_sha256,
    git_commit,
    load_protocol,
    milestone_steps,
    object_sha256,
    probe_spec,
    step_to_physical_epoch,
    steps_per_epoch,
)
from .runtime import (
    SerializableDataStream,
    atomic_torch_save,
    build_datasets,
    build_model,
    build_options,
    capture_rng,
    file_sha256 as runtime_file_sha256,
    full_state_hash,
    load_full_state,
    load_model_state,
    model_state,
    read_manifest,
    restore_rng,
    save_full_state,
    seed_everything,
    write_json,
)


SEED_E0_SCHEMA = "final-unsb-route1-seed-validation-e0-v1"
SEED_FREEZE_SCHEMA = "final-unsb-route1-seed-validation-freeze-v1"
SEED_SUMMARY_SCHEMA = "final-unsb-route1-seed-validation-summary-v2"
MULTI_SEED_ADJUDICATION_SCHEMA = "final-unsb-route1-multi-seed-adjudication-v1"
ALLOWED_VALIDATION_SEEDS = (2027, 2028)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _streams(opt, rows: list[dict], per_domain: int, seed: int):
    primary_data, secondary_data = build_datasets(opt, rows, per_domain)
    return (
        SerializableDataStream(primary_data, seed=seed + 101, label="primary"),
        SerializableDataStream(secondary_data, seed=seed + 202, label="secondary"),
    )


def _seed_root(output_root: Path, seed: int) -> Path:
    if int(seed) not in ALLOWED_VALIDATION_SEEDS:
        raise ValueError(f"seed validation is restricted to {ALLOWED_VALIDATION_SEEDS}")
    return Path(output_root) / "seed_validation" / f"seed{int(seed)}"


def _freeze_path(output_root: Path, candidate_id: str) -> Path:
    return Path(output_root) / "candidates" / candidate_id / "SEED_VALIDATION_FREEZE.json"


def validate_seed_freeze(
    output_root: Path, registration: CandidateRegistration, seed: int,
) -> dict:
    seed = int(seed)
    freeze_path = _freeze_path(output_root, registration.candidate_id)
    if not freeze_path.is_file():
        raise RuntimeError("seed validation is blocked until seed2026 candidate freeze")
    freeze = _read_json(freeze_path)
    if freeze.get("schema") != SEED_FREEZE_SCHEMA:
        raise RuntimeError("seed validation freeze schema mismatch")
    if freeze.get("status") != "FROZEN_FOR_SEED_VALIDATION":
        raise RuntimeError("candidate is not frozen for seed validation")
    if freeze.get("candidate_id") != registration.candidate_id:
        raise RuntimeError("seed validation freeze candidate mismatch")
    if freeze.get("algorithm_fingerprint") != registration.algorithm_fingerprint:
        raise RuntimeError("algorithm changed after seed2026 freeze")
    if freeze.get("seed2026_candidate_fingerprint") != registration.candidate_fingerprint:
        raise RuntimeError("seed2026 execution identity changed after freeze")
    trajectory_path = Path(output_root) / "candidates" / registration.candidate_id / "CANDIDATE_TRAJECTORY.json"
    if not trajectory_path.is_file():
        raise RuntimeError("seed2026 trajectory is missing")
    if freeze.get("seed2026_trajectory_sha256") != file_sha256(trajectory_path):
        raise RuntimeError("seed2026 trajectory changed after freeze")
    trajectory = _read_json(trajectory_path)
    if trajectory.get("status") != "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION":
        raise RuntimeError("seed validation is only allowed after a positive seed2026 numeric gate")
    if freeze.get("plain_collapse_adjudication") != "PASS_NOT_PLAIN_COLLAPSE":
        raise RuntimeError("seed2026 advantage has not passed the plain-collapse adjudication")
    if seed not in [int(value) for value in freeze.get("authorized_seeds", [])]:
        raise RuntimeError(f"seed {seed} is not authorized by the frozen record")
    if seed == 2028:
        authorization_path = _seed_root(output_root, 2027) / "SEED2028_AUTHORIZATION.json"
        if not authorization_path.is_file():
            raise RuntimeError("seed2028 requires a recorded seed2027 sign inconsistency")
        authorization = _read_json(authorization_path)
        if authorization.get("status") != "AUTHORIZED_SIGN_INCONSISTENCY":
            raise RuntimeError("seed2028 authorization is invalid")
        if authorization.get("algorithm_fingerprint") != registration.algorithm_fingerprint:
            raise RuntimeError("seed2028 authorization belongs to another algorithm")
        seed2027_summary = _seed_root(output_root, 2027) / "SEED_VALIDATION_SUMMARY.json"
        if not seed2027_summary.is_file():
            raise RuntimeError("seed2028 authorization has no seed2027 summary")
        if authorization.get("seed2027_summary_sha256") != file_sha256(seed2027_summary):
            raise RuntimeError("seed2028 authorization is stale")
        if authorization.get("seed2027_late_sign") != "nonpositive":
            raise RuntimeError("seed2028 requires a seed2027 sign inconsistency")
        if authorization.get("paired_metric_changed_algorithm") is not False:
            raise RuntimeError("seed2028 authorization cannot change the frozen algorithm")
        if authorization.get("confirmation20_opened") is not False:
            raise RuntimeError("confirmation20 must remain locked")
    if freeze.get("paired_controller_access") is not False:
        raise RuntimeError("seed freeze must deny paired controller access")
    if freeze.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 must remain locked")
    return freeze


def _e0_identity(
    *, seed: int, manifest_path: Path,
) -> dict:
    protocol = load_protocol()
    return {
        "project_id": protocol["project_id"],
        "purpose": "matched_seed_validation",
        "seed": int(seed),
        "manifest_sha256": file_sha256(manifest_path),
        "common_protocol_sha256": object_sha256(protocol["common"]),
        "local_view_sha256": object_sha256(protocol["local_view"]),
        "git_commit": git_commit(),
        "confirmation20_opened": False,
    }


def create_seed_e0(
    *, output_root: Path, seed: int,
    train_view: Path, manifest_path: Path, gpu: int,
) -> dict:
    protocol = load_protocol()
    seed_root = _seed_root(output_root, seed)
    e0_path = seed_root / "shared_e0" / "e0.pt"
    identity = _e0_identity(seed=seed, manifest_path=manifest_path)
    if e0_path.is_file():
        payload = torch.load(e0_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != SEED_E0_SCHEMA or payload.get("metadata") != identity:
            raise RuntimeError("seed validation e0 identity mismatch")
        sidecar = _read_json(Path(str(e0_path) + ".json"))
        if sidecar.get("checkpoint_sha256") != runtime_file_sha256(e0_path):
            raise RuntimeError("seed validation e0 file hash mismatch")
        if sidecar.get("scientific_state_sha256") != full_state_hash(payload):
            raise RuntimeError("seed validation e0 scientific hash mismatch")
        return payload

    rows = read_manifest(manifest_path)
    per_domain = int(protocol["local_view"]["train_per_domain"])
    plain_spec = probe_spec("plain", protocol)
    seed_everything(seed)
    opt = build_options(
        plain_spec, dataroot=train_view,
        option_root=seed_root / "option_records", seed=seed, gpu=gpu,
        diagnostic_root=None,
    )
    primary, secondary = _streams(opt, rows, per_domain, seed)
    model = build_model(opt, primary.next(), secondary.next())
    payload = {
        "schema": SEED_E0_SCHEMA,
        "metadata": identity,
        "model": model_state(model),
        "rng": capture_rng(),
        "samplers": {
            "primary": primary.state_dict(),
            "secondary": secondary.state_dict(),
        },
        "steps_per_epoch": steps_per_epoch(protocol),
        "target_steps": int(protocol["local_view"]["target_updates_per_lane"]),
    }
    atomic_torch_save(e0_path, payload)
    write_json(Path(str(e0_path) + ".json"), {
        "schema": SEED_E0_SCHEMA,
        "metadata": identity,
        "checkpoint_sha256": runtime_file_sha256(e0_path),
        "scientific_state_sha256": full_state_hash(payload),
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def _validation_spec(
    lane: str, registration: CandidateRegistration, seed: int,
) -> ProbeSpec:
    if lane == "plain":
        base = probe_spec("plain")
        return ProbeSpec(
            id=f"seed{seed}_plain", contract_id=f"SEED{seed}_MATCHED_PLAIN",
            model=base.model, role="seed_validation_plain", method={},
        )
    if lane != "candidate":
        raise ValueError("seed validation lane must be plain or candidate")
    return ProbeSpec(
        id=f"seed{seed}_{registration.candidate_id}",
        contract_id=f"SEED{seed}_{registration.candidate_id}",
        model=registration.spec.model,
        role="frozen_algorithm_seed_validation",
        method=dict(registration.spec.method),
    )


def _prepare_lane(
    *, spec: ProbeSpec, e0: dict, seed_root: Path, train_view: Path,
    manifest_path: Path, seed: int, gpu: int,
):
    protocol = load_protocol()
    rows = read_manifest(manifest_path)
    per_domain = int(protocol["local_view"]["train_per_domain"])
    seed_everything(seed)
    opt = build_options(
        spec, dataroot=train_view, option_root=seed_root / "option_records",
        seed=seed, gpu=gpu, diagnostic_root=None,
    )
    primary, secondary = _streams(opt, rows, per_domain, seed)
    model = build_model(opt, primary.next(), secondary.next())
    load_model_state(model, e0["model"], load_method=False)
    primary.load_state_dict(e0["samplers"]["primary"])
    secondary.load_state_dict(e0["samplers"]["secondary"])
    restore_rng(e0["rng"])
    model.set_search_step(0, int(protocol["local_view"]["target_updates_per_lane"]))
    return model, primary, secondary, rows


def _crn_fingerprint(registration: CandidateRegistration, seed: int) -> str:
    return object_sha256({
        "schema": "final-unsb-route1-seed-validation-crn-v1",
        "base_protocol_fingerprint": registration.base_protocol_fingerprint,
        "seed": int(seed),
    })


def _seed_late_rolling_drawdown(
    trajectory: list[dict], *, window: int = 3,
) -> float | None:
    """Seed-validation copy of the frozen candidate absolute-stability guard.

    This reporting-only module deliberately does not import candidate_runner:
    candidate_runner imports the seed-freeze schema from this module, and the
    candidate training fingerprint must not be changed while e200 jobs run.
    """
    values = [float(row["macro_psnr"]) for row in trajectory]
    if len(values) < int(window) or int(window) <= 0:
        return None
    rolling = [
        float(np.mean(values[index - window:index]))
        for index in range(window, len(values) + 1)
    ]
    return float(max(rolling) - rolling[-1])


def _seed_plain_collapse_adjudication(late: list[dict]) -> dict:
    """Apply the same e150-to-e200 absolute-quality rule to a new seed."""
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


def _execution_fingerprint(
    *, registration: CandidateRegistration, seed: int, lane: str, e0: dict,
) -> str:
    return object_sha256({
        "schema": "final-unsb-route1-seed-validation-execution-v1",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed": int(seed),
        "lane": lane,
        "e0_scientific_state_sha256": full_state_hash(e0),
        "crn_fingerprint": _crn_fingerprint(registration, seed),
        "git_commit": git_commit(),
    })


def _metric(
    *, model, rows: list[dict], data_root: Path, lane_root: Path,
    registration: CandidateRegistration, seed: int, lane: str, epoch: int,
) -> dict:
    path = lane_root / "metrics" / f"e{epoch:03d}.json"
    if path.is_file():
        return _read_json(path)
    protocol = load_protocol()
    payload = evaluate_model(
        model, rows=rows, data_root=data_root,
        protocol_hash=_crn_fingerprint(registration, seed),
        include_lpips=epoch in set(int(value) for value in protocol["local_view"]["lpips_epochs"]),
    )
    payload.update({
        "seed": int(seed), "lane": lane,
        "candidate_id": registration.candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "epoch": int(epoch), "updates": epoch_to_step(epoch),
        "data_epoch": int(epoch), "confirmation20_opened": False,
    })
    write_json(path, payload)
    return payload


def summarize_seed_validation(
    output_root: Path, candidate_id: str, seed: int,
) -> dict:
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    seed_root = _seed_root(output_root, seed)
    protocol = load_protocol()
    trajectory = []
    for epoch in (int(value) for value in protocol["local_view"]["trajectory_epochs"]):
        plain = seed_root / "plain" / "metrics" / f"e{epoch:03d}.json"
        candidate = seed_root / "candidate" / "metrics" / f"e{epoch:03d}.json"
        if plain.is_file() and candidate.is_file():
            trajectory.append(compare_to_plain(
                _read_json(candidate), _read_json(plain), epoch=epoch,
            ))
    late_epochs = {int(value) for value in protocol["local_view"]["late_epochs"]}
    late = [row for row in trajectory if row["epoch"] in late_epochs]
    complete = len(late) == 3 and late[-1]["epoch"] == 200
    late_mean = None if not complete else float(np.mean([row["macro_psnr_delta"] for row in late]))
    final_delta = None if not complete else float(late[-1]["macro_psnr_delta"])
    average_positive_domains = (
        None if not complete else float(np.mean([row["positive_domains"] for row in late]))
    )
    average_worst = (
        None if not complete else float(np.mean([row["worst_domain_delta"] for row in late]))
    )
    late_ssim = (
        None if not complete else float(np.mean([row["macro_ssim_delta"] for row in late]))
    )
    lpips_values = [] if not complete else [
        row["macro_lpips_delta"] for row in late if row["macro_lpips_delta"] is not None
    ]
    late_lpips = None if len(lpips_values) != 3 else float(np.mean(lpips_values))
    rolling_drawdown = _seed_late_rolling_drawdown(trajectory)
    collapse = _seed_plain_collapse_adjudication(late)
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
        "schema": SEED_SUMMARY_SCHEMA,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "seed": int(seed),
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "trajectory": trajectory,
        "late_three_mean_macro_psnr_delta": late_mean,
        "late_sign": None if late_mean is None else ("positive" if late_mean > 0.0 else "nonpositive"),
        "e200_macro_psnr_delta": final_delta,
        "late_points_with_four_of_six_positive_domains": sum(
            row["positive_domains"] >= 4 for row in late
        ),
        "late_average_positive_domains": average_positive_domains,
        "late_average_worst_domain_delta": average_worst,
        "late_mean_macro_ssim_delta": late_ssim,
        "late_mean_macro_lpips_delta": late_lpips,
        "candidate_best_to_terminal_three_point_rolling_drawdown": rolling_drawdown,
        "maximum_allowed_rolling_drawdown_db": 0.3,
        "plain_collapse_adjudication": collapse,
        "numeric_gate_pass": numeric_gate,
        "paired_metrics_used_only_after_complete_trajectory": complete,
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }
    summary_path = seed_root / "SEED_VALIDATION_SUMMARY.json"
    write_json(summary_path, result)
    if int(seed) == 2027 and complete and result["late_sign"] == "nonpositive":
        write_json(seed_root / "SEED2028_AUTHORIZATION.json", {
            "schema": "final-unsb-route1-seed2028-authorization-v1",
            "status": "AUTHORIZED_SIGN_INCONSISTENCY",
            "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "seed2027_summary_sha256": file_sha256(summary_path),
            "seed2027_late_sign": "nonpositive",
            "reason": "seed2026 passed while frozen algorithm is nonpositive on seed2027",
            "paired_metric_changed_algorithm": False,
            "confirmation20_opened": False,
        })
    return result


def _normalized_seed2026_row(trajectory: dict) -> dict:
    late = [
        row for row in trajectory.get("trajectory", [])
        if int(row.get("epoch", -1)) in (150, 175, 200)
    ]
    collapse = trajectory.get("plain_collapse_adjudication", {})
    return {
        "seed": 2026,
        "late_three_mean_macro_psnr_delta": trajectory.get(
            "late_three_mean_macro_psnr_delta"
        ),
        "late_sign": (
            "positive"
            if float(trajectory.get("late_three_mean_macro_psnr_delta", 0.0)) > 0.0
            else "nonpositive"
        ),
        "e200_macro_psnr_delta": trajectory.get("e200_macro_psnr_delta"),
        "late_average_positive_domains": (
            None if len(late) != 3
            else float(np.mean([row["positive_domains"] for row in late]))
        ),
        "late_average_worst_domain_delta": trajectory.get(
            "late_average_worst_domain_delta"
        ),
        "late_mean_macro_ssim_delta": trajectory.get("late_mean_macro_ssim_delta"),
        "late_mean_macro_lpips_delta": trajectory.get("late_mean_macro_lpips_delta"),
        "candidate_best_to_terminal_three_point_rolling_drawdown": trajectory.get(
            "candidate_best_to_terminal_three_point_rolling_drawdown"
        ),
        "plain_collapse_adjudication": collapse,
        "numeric_gate_pass": (
            trajectory.get("status")
            == "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION"
        ),
    }


def summarize_multi_seed_validation(output_root: Path, candidate_id: str) -> dict:
    """Fail-closed route1_sustained_local adjudication across frozen seeds."""
    output_root = Path(output_root).resolve()
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    freeze = validate_seed_freeze(output_root, registration, 2027)
    trajectory_path = output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    seed2026 = _read_json(trajectory_path)
    rows = [_normalized_seed2026_row(seed2026)]
    source_hashes = {"seed2026_trajectory_sha256": file_sha256(trajectory_path)}

    seed2027_path = _seed_root(output_root, 2027) / "SEED_VALIDATION_SUMMARY.json"
    if not seed2027_path.is_file():
        status = "WAITING_FOR_SEED2027"
    else:
        seed2027 = _read_json(seed2027_path)
        if seed2027.get("schema") != SEED_SUMMARY_SCHEMA:
            raise RuntimeError("seed2027 summary schema mismatch")
        if seed2027.get("status") != "COMPLETE":
            status = "WAITING_FOR_SEED2027"
        elif seed2027.get("algorithm_fingerprint") != registration.algorithm_fingerprint:
            raise RuntimeError("seed2027 algorithm differs from the frozen winner")
        else:
            rows.append(seed2027)
            source_hashes["seed2027_summary_sha256"] = file_sha256(seed2027_path)
            status = "READY_FOR_FINAL_ADJUDICATION"

    if status == "READY_FOR_FINAL_ADJUDICATION" and rows[-1]["late_sign"] == "nonpositive":
        seed2028_path = _seed_root(output_root, 2028) / "SEED_VALIDATION_SUMMARY.json"
        if not seed2028_path.is_file():
            status = "WAITING_FOR_AUTHORIZED_SEED2028"
        else:
            validate_seed_freeze(output_root, registration, 2028)
            seed2028 = _read_json(seed2028_path)
            if seed2028.get("schema") != SEED_SUMMARY_SCHEMA:
                raise RuntimeError("seed2028 summary schema mismatch")
            if seed2028.get("status") != "COMPLETE":
                status = "WAITING_FOR_AUTHORIZED_SEED2028"
            elif seed2028.get("algorithm_fingerprint") != registration.algorithm_fingerprint:
                raise RuntimeError("seed2028 algorithm differs from the frozen winner")
            else:
                rows.append(seed2028)
                source_hashes["seed2028_summary_sha256"] = file_sha256(seed2028_path)
                status = "READY_FOR_FINAL_ADJUDICATION"

    complete = status == "READY_FOR_FINAL_ADJUDICATION"
    combined_psnr = (
        None if not complete else float(np.mean([
            row["late_three_mean_macro_psnr_delta"] for row in rows
        ]))
    )
    combined_positive_domains = (
        None if not complete else float(np.mean([
            row["late_average_positive_domains"] for row in rows
        ]))
    )
    combined_worst = (
        None if not complete else float(np.mean([
            row["late_average_worst_domain_delta"] for row in rows
        ]))
    )
    all_signs_positive = complete and all(row["late_sign"] == "positive" for row in rows)
    all_numeric_gates = complete and all(bool(row["numeric_gate_pass"]) for row in rows)
    all_ssim_safe = complete and all(
        row["late_mean_macro_ssim_delta"] is not None
        and float(row["late_mean_macro_ssim_delta"]) >= 0.0
        for row in rows
    )
    all_lpips_safe = complete and all(
        row["late_mean_macro_lpips_delta"] is not None
        and float(row["late_mean_macro_lpips_delta"]) <= 0.0
        for row in rows
    )
    all_not_plain_collapse = complete and all(
        row["plain_collapse_adjudication"].get("status") == "PASS_NOT_PLAIN_COLLAPSE"
        for row in rows
    )
    sustained = bool(
        complete
        and all_signs_positive
        and all_numeric_gates
        and combined_psnr is not None and combined_psnr >= 0.15
        and combined_positive_domains is not None and combined_positive_domains >= 4.0
        and combined_worst is not None and combined_worst > -1.0
        and all_ssim_safe and all_lpips_safe and all_not_plain_collapse
    )
    failures = []
    checks = {
        "all_run_seed_late_signs_positive": all_signs_positive,
        "all_seed_numeric_gates_pass": all_numeric_gates,
        "combined_late_macro_psnr_at_least_0_15_db": (
            complete and combined_psnr is not None and combined_psnr >= 0.15
        ),
        "combined_average_positive_domains_at_least_4_of_6": (
            complete and combined_positive_domains is not None
            and combined_positive_domains >= 4.0
        ),
        "combined_average_worst_domain_above_minus_1_db": (
            complete and combined_worst is not None and combined_worst > -1.0
        ),
        "all_seed_ssim_guardrails_pass": all_ssim_safe,
        "all_seed_lpips_guardrails_pass": all_lpips_safe,
        "all_seed_absolute_trajectories_pass_plain_collapse_guard": all_not_plain_collapse,
    }
    if complete:
        failures = [name for name, passed in checks.items() if not passed]
        status = "ROUTE1_SUSTAINED_LOCAL" if sustained else "MULTI_SEED_NOT_SUSTAINED"
    result = {
        "schema": MULTI_SEED_ADJUDICATION_SCHEMA,
        "status": status,
        "classification": "route1_sustained_local" if sustained else None,
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2026_candidate_fingerprint": registration.candidate_fingerprint,
        "seed_validation_freeze_sha256": file_sha256(_freeze_path(output_root, candidate_id)),
        "included_seeds": [int(row["seed"]) for row in rows],
        "per_seed": rows,
        "combined_late_three_mean_macro_psnr_delta": combined_psnr,
        "combined_late_average_positive_domains": combined_positive_domains,
        "combined_late_average_worst_domain_delta": combined_worst,
        "checks": checks,
        "failed_checks": failures,
        "source_hashes": source_hashes,
        "seed2028_only_when_seed2027_sign_inconsistent": True,
        "algorithm_changes_after_seed2026_freeze": False,
        "paired_metrics_used_only_for_post_training_adjudication": True,
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }
    path = output_root / "candidates" / candidate_id / "MULTI_SEED_ADJUDICATION.json"
    write_json(path, result)
    return result


def seed_validation_status(
    output_root: Path, candidate_id: str, seed: int,
) -> dict:
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    freeze = validate_seed_freeze(output_root, registration, seed)
    seed_root = _seed_root(output_root, seed)

    def completed_epoch(lane: str) -> int:
        path = seed_root / lane / "full_state_latest.pt.json"
        if not path.is_file():
            return 0
        sidecar = _read_json(path)
        return int(sidecar.get("physical_epoch_completed", 0))

    return {
        "schema": "final-unsb-route1-seed-validation-status-v1",
        "status": "READY_FOR_FROZEN_SEED_VALIDATION",
        "candidate_id": candidate_id,
        "seed": int(seed),
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2026_candidate_fingerprint": registration.candidate_fingerprint,
        "seed_freeze_sha256": file_sha256(
            _freeze_path(output_root, registration.candidate_id)
        ),
        "authorized_seeds": freeze["authorized_seeds"],
        "plain_data_epoch": completed_epoch("plain"),
        "candidate_data_epoch": completed_epoch("candidate"),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def run_seed_validation_lane(
    *, output_root: Path, candidate_id: str, seed: int, lane: str,
    train_view: Path, data_root: Path, manifest_path: Path, gpu: int,
    resume: bool, engineering_stop_after_epoch: int | None = None,
) -> dict:
    output_root = Path(output_root).resolve()
    seed = int(seed)
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    validate_seed_freeze(output_root, registration, seed)
    protocol = load_protocol()
    if file_sha256(manifest_path) != str(protocol["manifest"]["sha256"]):
        raise RuntimeError("seed validation manifest differs from route-1 manifest")
    seed_root = _seed_root(output_root, seed)
    target_steps = int(protocol["local_view"]["target_updates_per_lane"])
    e0 = create_seed_e0(
        output_root=output_root, seed=seed,
        train_view=train_view, manifest_path=manifest_path, gpu=gpu,
    )
    if lane == "candidate":
        plain_latest = seed_root / "plain" / "full_state_latest.pt"
        plain_sidecar_path = Path(str(plain_latest) + ".json")
        if not plain_latest.is_file() or not plain_sidecar_path.is_file():
            raise RuntimeError("seed validation candidate is blocked until matched plain e200")
        plain_payload = torch.load(plain_latest, map_location="cpu", weights_only=False)
        plain_sidecar = _read_json(plain_sidecar_path)
        plain_metadata = plain_payload.get("metadata", {})
        if int(plain_payload.get("step", -1)) != target_steps:
            raise RuntimeError("seed validation matched plain is not at e200")
        if plain_sidecar.get("checkpoint_sha256") != runtime_file_sha256(plain_latest):
            raise RuntimeError("seed validation matched plain file hash mismatch")
        if plain_sidecar.get("scientific_state_sha256") != full_state_hash(plain_payload):
            raise RuntimeError("seed validation matched plain scientific hash mismatch")
        if plain_metadata.get("algorithm_fingerprint") != registration.algorithm_fingerprint:
            raise RuntimeError("seed validation plain belongs to another frozen algorithm")
        if plain_metadata.get("e0_scientific_state_sha256") != full_state_hash(e0):
            raise RuntimeError("seed validation candidate and plain do not share e0")
        if plain_metadata.get("crn_fingerprint") != _crn_fingerprint(registration, seed):
            raise RuntimeError("seed validation candidate and plain do not share CRN")
        if plain_metadata.get("confirmation20_opened") is not False:
            raise RuntimeError("confirmation20 must remain locked")
    spec = _validation_spec(lane, registration, seed)
    model, primary, secondary, rows = _prepare_lane(
        spec=spec, e0=e0, seed_root=seed_root, train_view=train_view,
        manifest_path=manifest_path, seed=seed, gpu=gpu,
    )
    lane_root = seed_root / lane
    latest = lane_root / "full_state_latest.pt"
    if latest.is_file() and not resume:
        raise RuntimeError(f"existing seed validation state requires --resume: {latest}")
    execution_fingerprint = _execution_fingerprint(
        registration=registration, seed=seed, lane=lane, e0=e0,
    )
    metadata = {
        "project_id": protocol["project_id"],
        "purpose": "matched_seed_validation",
        "probe_id": spec.id,
        "candidate_id": candidate_id,
        "lane": lane,
        "seed": seed,
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_fingerprint": execution_fingerprint,
        "execution_fingerprint": execution_fingerprint,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "e0_scientific_state_sha256": full_state_hash(e0),
        "crn_fingerprint": _crn_fingerprint(registration, seed),
        "training_git_commit": git_commit(),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    start_step = 0
    if resume and latest.is_file():
        restored = load_full_state(
            latest, model=model, spec=spec, primary=primary, secondary=secondary,
            expected_metadata={
                "project_id": metadata["project_id"],
                "probe_id": spec.id,
                "candidate_id": candidate_id,
                "lane": lane,
                "seed": seed,
                "manifest_sha256": metadata["manifest_sha256"],
                "execution_fingerprint": execution_fingerprint,
                "algorithm_fingerprint": registration.algorithm_fingerprint,
                "training_git_commit": metadata["training_git_commit"],
            },
        )
        start_step = int(restored["step"])
    stop_steps = (
        target_steps if engineering_stop_after_epoch is None
        else epoch_to_step(int(engineering_stop_after_epoch), protocol)
    )
    if stop_steps <= 0 or stop_steps > target_steps:
        raise ValueError("engineering stop epoch must be in [1, 200]")
    if start_step in set(milestone_steps(protocol)):
        _metric(
            model=model, rows=rows, data_root=data_root, lane_root=lane_root,
            registration=registration, seed=seed, lane=lane,
            epoch=start_step // steps_per_epoch(protocol),
        )
    if start_step >= stop_steps:
        result = {
            "status": "ALREADY_AT_REQUESTED_STOP", "seed": seed, "lane": lane,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "step": start_step, "data_epoch": start_step // steps_per_epoch(protocol),
        }
        if lane == "candidate" and start_step == target_steps:
            result["evaluation"] = summarize_seed_validation(output_root, candidate_id, seed)
        return result

    milestone_set = set(milestone_steps(protocol))
    started = time.time()
    epoch_started = time.time()
    for zero_step in range(start_step, stop_steps):
        physical_epoch = step_to_physical_epoch(zero_step, protocol)
        model.set_train_epoch(physical_epoch)
        model.set_search_step(zero_step, target_steps)
        model.set_input(primary.next(), secondary.next())
        model.optimize_parameters()
        completed = zero_step + 1
        if completed % steps_per_epoch(protocol) != 0:
            continue
        completed_epoch = completed // steps_per_epoch(protocol)
        model.update_learning_rate()
        sidecar = save_full_state(
            latest, model=model, spec=spec, step=completed,
            target_steps=target_steps, primary=primary, secondary=secondary,
            metadata=metadata,
        )
        if completed in milestone_set:
            save_full_state(
                lane_root / "milestones" / f"e{completed_epoch:03d}.pt",
                model=model, spec=spec, step=completed, target_steps=target_steps,
                primary=primary, secondary=secondary, metadata=metadata,
            )
            _metric(
                model=model, rows=rows, data_root=data_root, lane_root=lane_root,
                registration=registration, seed=seed, lane=lane, epoch=completed_epoch,
            )
        trace = {
            "seed": seed, "lane": lane, "candidate_id": candidate_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "updates": completed, "data_epoch": completed_epoch,
            "epoch_wall_seconds": time.time() - epoch_started,
            "latest_scientific_state_sha256": sidecar["scientific_state_sha256"],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        lane_root.mkdir(parents=True, exist_ok=True)
        with (lane_root / "TRAIN_TRACE.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")
        write_json(lane_root / "HEARTBEAT.json", {
            **trace, "target_updates": target_steps, "target_data_epochs": 200,
            "wall_seconds_this_call": time.time() - started,
        })
        epoch_started = time.time()

    result = {
        "status": "COMPLETE_E200" if stop_steps == target_steps else "ENGINEERING_PAUSE",
        "seed": seed, "lane": lane, "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "start_updates": start_step, "final_updates": stop_steps,
        "final_data_epoch": stop_steps // steps_per_epoch(protocol),
        "target_data_epochs": 200, "target_updates": target_steps,
        "wall_seconds_this_call": time.time() - started,
        "metadata": metadata, "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if lane == "candidate" and stop_steps == target_steps:
        result["evaluation"] = summarize_seed_validation(output_root, candidate_id, seed)
    write_json(lane_root / "RUN_STATE.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result
