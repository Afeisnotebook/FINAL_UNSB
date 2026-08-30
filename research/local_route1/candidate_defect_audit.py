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
CROSS_VERSION_FINAL_OUTCOME_SCHEMA = (
    "final-unsb-route1-cross-version-final-revision-outcome-v1"
)
GENERATION1_NEGATIVE_STATUS = (
    "NO_SEED2026_NUMERIC_GATE_PASS_CAUSAL_DEFECT_ADJUDICATION_REQUIRED"
)
CROSS_VERSION_NEGATIVE_STATUS = (
    "NO_SEED2026_NUMERIC_GATE_PASS_CAUSAL_ADJUDICATION_REQUIRED"
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


class _ReplicaGeometryAccumulator:
    """Accumulate the exact mean/disagreement geometry of iid gradient pairs."""

    def __init__(self) -> None:
        self.pairs = 0
        self.difference_energy = 0.0
        self.parallel_energy = 0.0
        self.orthogonal_energy = 0.0
        self.mean_energy = 0.0
        self.signed_cosine_sum = 0.0
        self.absolute_cosine_sum = 0.0
        self.cosine_count = 0

    def add(
        self, first: tuple[torch.Tensor, ...], second: tuple[torch.Tensor, ...],
        *, scales: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        if len(first) != len(second) or (
            scales is not None and len(scales) != len(first)
        ):
            raise RuntimeError("replica geometry structures differ")
        mean_sq = difference_sq = mean_difference = 0.0
        for index, (left, right) in enumerate(zip(first, second)):
            mean = (left.double() + right.double()) * 0.5
            difference = (left.double() - right.double()) * 0.5
            if scales is not None:
                scale = scales[index].double()
                mean = mean * scale
                difference = difference * scale
            mean_sq += float(mean.square().sum().item())
            difference_sq += float(difference.square().sum().item())
            mean_difference += float((mean * difference).sum().item())
        parallel = (
            mean_difference * mean_difference / mean_sq
            if mean_sq > 0.0 else 0.0
        )
        parallel = min(max(parallel, 0.0), max(difference_sq, 0.0))
        orthogonal = max(difference_sq - parallel, 0.0)
        self.pairs += 1
        self.mean_energy += mean_sq
        self.difference_energy += difference_sq
        self.parallel_energy += parallel
        self.orthogonal_energy += orthogonal
        if mean_sq > 0.0 and difference_sq > 0.0:
            cosine = mean_difference / math.sqrt(mean_sq * difference_sq)
            cosine = min(max(cosine, -1.0), 1.0)
            self.signed_cosine_sum += cosine
            self.absolute_cosine_sum += abs(cosine)
            self.cosine_count += 1

    def summary(self) -> dict[str, Any]:
        if self.pairs < 1 or not math.isfinite(self.difference_energy):
            raise RuntimeError("replica geometry requires finite gradient pairs")
        denominator = self.difference_energy
        return {
            "pairs": int(self.pairs),
            "mean_gradient_energy": float(self.mean_energy),
            "antisymmetric_difference_energy": float(denominator),
            "difference_parallel_to_mean_energy": float(self.parallel_energy),
            "difference_orthogonal_to_mean_energy": float(self.orthogonal_energy),
            "parallel_fraction_of_difference": (
                None if denominator <= 0.0 else
                float(self.parallel_energy / denominator)
            ),
            "orthogonal_fraction_of_difference": (
                None if denominator <= 0.0 else
                float(self.orthogonal_energy / denominator)
            ),
            "mean_signed_mean_difference_cosine": (
                None if self.cosine_count == 0 else
                float(self.signed_cosine_sum / self.cosine_count)
            ),
            "mean_absolute_mean_difference_cosine": (
                None if self.cosine_count == 0 else
                float(self.absolute_cosine_sum / self.cosine_count)
            ),
            "cosine_pairs": int(self.cosine_count),
        }


def _adam_update_space_scales(
    parameters: tuple[torch.nn.Parameter, ...],
    optimizers: tuple[torch.optim.Optimizer, ...],
) -> tuple[torch.Tensor, ...]:
    """Return the frozen pre-step Adam diagonal map from gradient to update."""
    records: dict[int, tuple[torch.optim.Optimizer, float]] = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            epsilon = float(group.get("eps", 1e-8))
            for parameter in group["params"]:
                key = id(parameter)
                if key in records:
                    raise RuntimeError("parameter appears in multiple audit optimizers")
                records[key] = (optimizer, epsilon)
    scales = []
    for parameter in parameters:
        record = records.get(id(parameter))
        if record is None:
            raise RuntimeError("audit parameter is missing from its Adam optimizer")
        optimizer, epsilon = record
        second_moment = optimizer.state.get(parameter, {}).get("exp_avg_sq")
        if second_moment is None:
            scale = torch.ones_like(parameter, device="cpu", dtype=torch.float32)
        else:
            scale = (
                second_moment.detach().cpu().float().sqrt().add(epsilon).reciprocal()
            )
        if not bool(torch.isfinite(scale).all().item()):
            raise RuntimeError("Adam update-space audit scale is nonfinite")
        scales.append(scale)
    return tuple(scales)


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
        "revision_eligibility_upper_bound": min(reference, 1e-7),
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
    model.netG.train()
    model.netE.train()
    model.netD.train()
    model.netF.train()
    model.set_requires_grad(model.netD, True)
    model.set_requires_grad(model.netE, True)
    parameters = {
        "D": _network_parameters(model.netD),
        "E": _network_parameters(model.netE),
        "GF": _network_parameters(model.netG, model.netF),
    }
    gf_optimizers = [model.optimizer_G]
    if getattr(model.opt, "netF", None) == "mlp_sample":
        gf_optimizers.append(model.optimizer_F)
    update_scales = {
        "D": _adam_update_space_scales(parameters["D"], (model.optimizer_D,)),
        "E": _adam_update_space_scales(parameters["E"], (model.optimizer_E,)),
        "GF": _adam_update_space_scales(parameters["GF"], tuple(gf_optimizers)),
    }
    geometry = {
        name: {
            "parameter_euclidean": _ReplicaGeometryAccumulator(),
            "pre_step_adam_update_space": _ReplicaGeometryAccumulator(),
        }
        for name in players
    }
    pending: dict[str, tuple[torch.Tensor, ...]] = {}
    for index in range(int(samples)):
        model.forward()
        model.netG.train()
        model.netE.train()
        model.netD.train()
        model.netF.train()
        model.set_requires_grad(model.netD, True)
        d_gradient = _cpu_gradients(model.compute_D_loss(), parameters["D"])

        model.set_requires_grad(model.netE, True)
        e_gradient = _cpu_gradients(model.compute_E_loss(), parameters["E"])

        model.set_requires_grad(model.netD, False)
        model.set_requires_grad(model.netE, False)
        gf_gradient = _cpu_gradients(model.compute_G_loss(), parameters["GF"])
        model.set_requires_grad(model.netD, True)
        model.set_requires_grad(model.netE, True)

        for name, gradient in (("D", d_gradient), ("E", e_gradient), ("GF", gf_gradient)):
            players[name].add(gradient)
            if index % 2 == 0:
                pending[name] = gradient
            else:
                first = pending.pop(name)
                paired[name].add(_mean_gradients(first, gradient))
                geometry[name]["parameter_euclidean"].add(first, gradient)
                geometry[name]["pre_step_adam_update_space"].add(
                    first, gradient, scales=update_scales[name],
                )
    ratios = {}
    for name in players:
        native = players[name].trace_variance()
        replicated = paired[name].trace_variance()
        ratios[name] = {
            "native_trace_variance": native,
            "two_replica_trace_variance": replicated,
            "variance_ratio": None if native <= 0.0 else float(replicated / native),
            "replica_disagreement_geometry": {
                metric: accumulator.summary()
                for metric, accumulator in geometry[name].items()
            },
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
        # Two independent replicas have an ideal conditional-variance ratio of
        # 1/2.  Requiring a material reduction avoids routing a mathematical
        # revision because of finite-sample noise around the native ratio.
        "revision_eligibility_upper_bound": 0.8,
        "desired_direction": "decrease",
        "complete_native_views": int(samples),
        "paired_two_replica_estimates": int(samples) // 2,
        "players": ratios,
        "geometry_role": (
            "Target-blind revision evidence only. Geometry does not alter the "
            "registered total-variance eligibility test or select an algorithm."
        ),
        "pre_step_adam_update_space_definition": (
            "Each gradient coordinate is divided by sqrt(exp_avg_sq)+eps from "
            "the unchanged e200 Adam state. Bias-correction scalars and the "
            "current gradient's future second-moment update are excluded."
        ),
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
        elif registration.spec.model in ("route1_rsmg", "route1_pcrsmg"):
            measurement = _audit_rsmg(model, samples=samples)
            failure_reason = (
                "Player-conditional native gradient variance was reduced, but the lower-variance "
                "expected unpaired game field did not preserve long-horizon PSNR; a revision must "
                "change the safe mathematical operator rather than tune replica count or a window."
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
    eligibility_bound = float(
        measurement.get("revision_eligibility_upper_bound", reference)
    )
    reduced = (
        math.isfinite(reference)
        and math.isfinite(candidate)
        and math.isfinite(eligibility_bound)
        and candidate < eligibility_bound
    )
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


def adjudicate_cross_version_revision_need(
    output_root: Path, candidate_ids: list[str],
) -> dict[str, Any]:
    """Route negative source-bound receipts without loading sibling code."""
    output_root = Path(output_root).resolve()
    cross_path = output_root / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    cross = _read_json(cross_path)
    if cross.get("status") != CROSS_VERSION_NEGATIVE_STATUS:
        raise RuntimeError(
            "cross-version causal revision is only routed after all e200 candidates are negative"
        )
    ranking = cross.get("ranking")
    if not isinstance(ranking, list) or len(ranking) < 2:
        raise RuntimeError("cross-version negative routing requires the complete ranking")
    ranked = {str(row["candidate_id"]): row for row in ranking}
    if set(candidate_ids) != set(ranked):
        raise RuntimeError("cross-version defect candidate set differs from the frozen ranking")
    records = []
    for raw_id in candidate_ids:
        candidate_id = validate_candidate_id(raw_id)
        path = (
            output_root / "candidates" / candidate_id
            / "TARGET_BLIND_DEFECT_ADJUDICATION.json"
        )
        if not path.is_file():
            raise RuntimeError(f"target-blind candidate defect audit missing: {candidate_id}")
        row = _read_json(path)
        expected = ranked[candidate_id]
        for key, value in (
            ("candidate_id", candidate_id),
            ("algorithm_fingerprint", expected["algorithm_fingerprint"]),
            ("data_epoch_adjudicated", 200),
            ("source_candidate_trajectory_sha256", expected["trajectory_sha256"]),
            ("source_checkpoint_unchanged_after_audit", True),
            ("paired_target_used_to_compute_defect", False),
            ("paired_metric_used_for_training_or_control", False),
            ("confirmation20_opened", False),
        ):
            if row.get(key) != value:
                raise RuntimeError(
                    f"cross-version defect adjudication mismatch for {candidate_id}: {key}"
                )
        records.append({**row, "source_sha256": file_sha256(path)})
    applicable = [row for row in records if row.get("revision_applicable") is True]
    rank_order = [str(row["candidate_id"]) for row in ranking]
    applicable.sort(key=lambda row: rank_order.index(row["candidate_id"]))
    result = {
        "schema": CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
        "status": (
            "REVISION_DERIVATION_REQUIRED" if applicable
            else "NO_REVISION_APPLICABLE_FINAL_FALLBACK"
        ),
        "selected_candidate_id": (
            applicable[0]["candidate_id"]
            if applicable else cross["selected_candidate_id"]
        ),
        "source_cross_version_adjudication_sha256": file_sha256(cross_path),
        "candidate_defect_adjudications": records,
        "revision_applicable_candidate_ids": [row["candidate_id"] for row in applicable],
        "automatic_revision_started": False,
        "fixed_window_or_handoff": False,
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    write_json(
        output_root / "operations" / "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json",
        result,
    )
    return result
