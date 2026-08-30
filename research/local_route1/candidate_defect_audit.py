"""Target-blind e200 defect audit for negative Generation-1 candidates.

The audit determines whether the candidate actually reduced the mathematical
defect it was derived to fix.  It never uses paired quality to measure the
defect.  A revision is eligible only when this defect decreased while the
already-complete e200 trajectory remained negative.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from .anchors import prepare_probe
from .candidates import (
    DEFECT_ADJUDICATION_SCHEMA,
    load_candidate_registration,
    validate_candidate_id,
)
from .protocol import file_sha256, load_protocol
from .runtime import load_model_state, restore_rng, write_json


FINAL_OUTCOME_SCHEMA = "final-unsb-route1-final-revision-outcome-v1"
GENERATION1_NEGATIVE_STATUS = (
    "NO_SEED2026_NUMERIC_GATE_PASS_CAUSAL_DEFECT_ADJUDICATION_REQUIRED"
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _network_parameters(*networks) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter for network in networks for parameter in network.parameters()
        if parameter.requires_grad
    )


def _cpu_gradients(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor, ...]:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=True, allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter, device="cpu")
        if gradient is None else gradient.detach().cpu().float().clone()
        for parameter, gradient in zip(parameters, gradients)
    )


class _GradientTraceAccumulator:
    def __init__(self) -> None:
        self.count = 0
        self.sum: list[torch.Tensor] | None = None
        self.sum_squared_norm = 0.0

    def add(self, gradients: tuple[torch.Tensor, ...]) -> None:
        if self.sum is None:
            self.sum = [value.clone() for value in gradients]
        else:
            if len(self.sum) != len(gradients):
                raise RuntimeError("gradient structures differ across replicas")
            for total, value in zip(self.sum, gradients):
                total.add_(value)
        self.sum_squared_norm += sum(
            float(value.double().square().sum().item()) for value in gradients
        )
        self.count += 1

    def trace_variance(self) -> float:
        if self.count < 2 or self.sum is None:
            raise RuntimeError("trace variance requires at least two gradients")
        mean_correction = sum(
            float(value.double().square().sum().item()) for value in self.sum
        ) / float(self.count)
        numerator = max(self.sum_squared_norm - mean_correction, 0.0)
        return float(numerator / float(self.count - 1))


def _mean_gradients(
    first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    if len(first) != len(second):
        raise RuntimeError("paired gradient structures differ")
    return tuple((left + right) * 0.5 for left, right in zip(first, second))


def _prepare_candidate(
    *, output_root: Path, candidate_id: str, train_view: Path,
    manifest_path: Path, gpu: int,
):
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    trajectory_path = output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    if not trajectory_path.is_file():
        raise RuntimeError("candidate defect audit requires a complete e200 trajectory")
    trajectory = _read_json(trajectory_path)
    if not any(int(row.get("epoch", -1)) == 200 for row in trajectory.get("trajectory", [])):
        raise RuntimeError("candidate defect audit requires the e200 trajectory row")
    checkpoint = output_root / "candidates" / candidate_id / "full_state_latest.pt"
    if not checkpoint.is_file():
        raise RuntimeError("candidate defect audit requires the e200 full state")
    checkpoint_sha256 = file_sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    target = int(load_protocol()["local_view"]["target_updates_per_lane"])
    if int(payload.get("step", -1)) != target:
        raise RuntimeError("candidate defect audit checkpoint is not e200")
    e0 = torch.load(output_root / "shared_e0" / "e0.pt", map_location="cpu", weights_only=False)
    model, primary, secondary, _ = prepare_probe(
        spec=registration.spec,
        output_root=output_root / "derive" / "defect_audit_work" / candidate_id,
        train_view=train_view, manifest_path=manifest_path, gpu=gpu, e0=e0,
    )
    load_model_state(model, payload["model"], load_method=True)
    primary.load_state_dict(payload["samplers"]["primary"])
    secondary.load_state_dict(payload["samplers"]["secondary"])
    restore_rng(payload["rng"])
    model.set_train_epoch(200)
    model.set_search_step(target, target)
    model.set_input(primary.next(), secondary.next())
    return registration, trajectory, checkpoint, checkpoint_sha256, model


def _audit_bvcp(model, *, samples: int) -> dict[str, Any]:
    pre_excess = []
    post_excess = []
    eligible = 0
    with torch.no_grad():
        for _ in range(int(samples)):
            model.forward()
            row = dict(getattr(model, "_bvcp_last", {}))
            required = ("current_rms", "lagged_rms", "projected_rms")
            if any(key not in row for key in required):
                raise RuntimeError("BVCP forward did not expose target-blind velocity diagnostics")
            before = max(float(row["current_rms"]) - float(row["lagged_rms"]), 0.0)
            after = max(float(row["projected_rms"]) - float(row["lagged_rms"]), 0.0)
            pre_excess.append(before)
            post_excess.append(after)
            eligible += int(before > 0.0)
    reference = float(sum(pre_excess) / len(pre_excess))
    candidate = float(sum(post_excess) / len(post_excess))
    return {
        "observable": "mean_positive_current_minus_lagged_rollout_rms_excess",
        "reference_value": reference,
        "candidate_value": candidate,
        "desired_direction": "decrease",
        "samples": int(samples),
        "eligible_samples": eligible,
        "constraint_tolerance": 1e-7,
        "constraint_holds": all(after <= 1e-7 for after in post_excess),
    }


def _audit_rsmg(model, *, samples: int) -> dict[str, Any]:
    if int(samples) < 4 or int(samples) % 2:
        raise ValueError("RSMG defect audit requires an even sample count of at least four")
    players = {
        "D": _GradientTraceAccumulator(),
        "E": _GradientTraceAccumulator(),
        "GF": _GradientTraceAccumulator(),
    }
    paired = {name: _GradientTraceAccumulator() for name in players}
    pending: dict[str, tuple[torch.Tensor, ...]] = {}
    for index in range(int(samples)):
        model.forward()
        model.set_requires_grad(model.netD, True)
        d_parameters = _network_parameters(model.netD)
        d_gradient = _cpu_gradients(model.compute_D_loss(), d_parameters)

        model.set_requires_grad(model.netE, True)
        e_parameters = _network_parameters(model.netE)
        e_gradient = _cpu_gradients(model.compute_E_loss(), e_parameters)

        model.set_requires_grad(model.netD, False)
        model.set_requires_grad(model.netE, False)
        gf_parameters = _network_parameters(model.netG, model.netF)
        gf_gradient = _cpu_gradients(model.compute_G_loss(), gf_parameters)
        model.set_requires_grad(model.netD, True)
        model.set_requires_grad(model.netE, True)

        for name, gradient in (("D", d_gradient), ("E", e_gradient), ("GF", gf_gradient)):
            players[name].add(gradient)
            if index % 2 == 0:
                pending[name] = gradient
            else:
                paired[name].add(_mean_gradients(pending.pop(name), gradient))
    ratios = {}
    for name in players:
        native = players[name].trace_variance()
        replicated = paired[name].trace_variance()
        ratios[name] = {
            "native_trace_variance": native,
            "two_replica_trace_variance": replicated,
            "variance_ratio": None if native <= 0.0 else float(replicated / native),
        }
    finite_ratios = [
        row["variance_ratio"] for row in ratios.values()
        if row["variance_ratio"] is not None and math.isfinite(row["variance_ratio"])
    ]
    if len(finite_ratios) != 3:
        raise RuntimeError("RSMG target-blind variance audit produced a degenerate player")
    mean_ratio = float(sum(finite_ratios) / len(finite_ratios))
    return {
        "observable": "mean_per_player_two_replica_to_native_gradient_trace_variance_ratio",
        "reference_value": 1.0,
        "candidate_value": mean_ratio,
        "desired_direction": "decrease",
        "complete_native_views": int(samples),
        "paired_two_replica_estimates": int(samples) // 2,
        "players": ratios,
        "same_official_unpaired_batch": True,
        "patchnce_cross_replica_negatives": False,
    }


def audit_candidate_defect(
    *, output_root: Path, candidate_id: str, train_view: Path,
    manifest_path: Path, gpu: int, samples: int = 16,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    candidate_id = validate_candidate_id(candidate_id)
    registration, trajectory, checkpoint, before_hash, model = _prepare_candidate(
        output_root=output_root, candidate_id=candidate_id,
        train_view=train_view, manifest_path=manifest_path, gpu=gpu,
    )
    try:
        if registration.spec.model == "route1_bvcp":
            measurement = _audit_bvcp(model, samples=samples)
            failure_reason = (
                "The one-step radial rollout-speed excess was reduced, but controlling only the "
                "immediate residual norm did not preserve long-horizon transport direction or game state."
            )
        elif registration.spec.model == "route1_rsmg":
            measurement = _audit_rsmg(model, samples=samples)
            failure_reason = (
                "Conditional native gradient variance was reduced, but the lower-variance expectation "
                "still followed an unpaired objective direction that did not preserve long-horizon PSNR."
            )
        else:
            raise RuntimeError("unsupported Generation-1 defect audit model")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if file_sha256(checkpoint) != before_hash:
        raise RuntimeError("candidate defect audit mutated its parent checkpoint")
    reference = float(measurement["reference_value"])
    candidate = float(measurement["candidate_value"])
    reduced = math.isfinite(reference) and math.isfinite(candidate) and candidate < reference
    negative = trajectory.get("status") == "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION"
    result = {
        "schema": DEFECT_ADJUDICATION_SCHEMA,
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "data_epoch_adjudicated": 200,
        "source_candidate_trajectory_sha256": file_sha256(
            output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        ),
        "source_checkpoint_sha256": before_hash,
        "source_checkpoint_unchanged_after_audit": True,
        "target_blind_defect_reduced": bool(reduced),
        "long_horizon_benefit_reversed": bool(negative),
        "revision_applicable": bool(reduced and negative),
        "target_blind_defect_measurement": measurement,
        "new_causal_failure_reason": failure_reason if reduced and negative else None,
        "paired_target_used_to_compute_defect": False,
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    path = output_root / "candidates" / candidate_id / "TARGET_BLIND_DEFECT_ADJUDICATION.json"
    write_json(path, result)
    return result


def adjudicate_revision_need(output_root: Path, candidate_ids: list[str]) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    generation1_path = output_root / "operations" / "GENERATION1_E200_ADJUDICATION.json"
    generation1 = _read_json(generation1_path)
    if generation1.get("status") != GENERATION1_NEGATIVE_STATUS:
        raise RuntimeError("causal-revision need is only adjudicated after two negative e200 results")
    records = []
    for candidate_id in candidate_ids:
        candidate_id = validate_candidate_id(candidate_id)
        path = output_root / "candidates" / candidate_id / "TARGET_BLIND_DEFECT_ADJUDICATION.json"
        if not path.is_file():
            raise RuntimeError(f"target-blind candidate defect audit missing: {candidate_id}")
        row = _read_json(path)
        if row.get("candidate_id") != candidate_id or row.get("data_epoch_adjudicated") != 200:
            raise RuntimeError("candidate defect adjudication identity mismatch")
        if row.get("paired_target_used_to_compute_defect") is not False:
            raise RuntimeError("candidate defect audit used paired target")
        if row.get("confirmation20_opened") is not False:
            raise RuntimeError("candidate defect audit opened confirmation20")
        records.append({**row, "source_sha256": file_sha256(path)})
    applicable = [row for row in records if row.get("revision_applicable") is True]
    rank_order = [
        row.get("candidate_id") for row in generation1.get("ranking", [])
        if isinstance(row, dict)
    ]
    if applicable:
        applicable.sort(key=lambda row: (
            rank_order.index(row["candidate_id"])
            if row["candidate_id"] in rank_order else len(rank_order)
        ))
    result = {
        "schema": FINAL_OUTCOME_SCHEMA,
        "status": (
            "REVISION_DERIVATION_REQUIRED" if applicable
            else "NO_REVISION_APPLICABLE_FINAL_FALLBACK"
        ),
        "selected_candidate_id": (
            applicable[0]["candidate_id"] if applicable
            else generation1["selected_candidate_id"]
        ),
        "source_generation1_adjudication_sha256": file_sha256(generation1_path),
        "candidate_defect_adjudications": records,
        "revision_applicable_candidate_ids": [row["candidate_id"] for row in applicable],
        "automatic_revision_started": False,
        "fixed_window_or_handoff": False,
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "operations" / "FINAL_CAUSAL_REVISION_OUTCOME.json", result)
    return result
