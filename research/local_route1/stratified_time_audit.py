"""Target-blind fixed-state audit for ST-CGR.

The audit estimates the within-time covariance and between-time conditional
mean covariance of the native post-D/E G/F gradient.  It then evaluates the
closed-form covariance trace of an iid Proposal pair and an ordered uniform
pair without replacement.  No optimizer transition is committed after the
one native D/E boundary used by Proposal, and no paired quality is addressable.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

import torch

from .anchors import prepare_probe
from .antithetic_gradient_audit import (
    _commit_fixed_de_state,
    _gf_gradient,
    _gradient_parameters,
)
from .candidate_defect_audit import _adam_update_space_scales
from .protocol import (
    ProbeSpec,
    file_sha256,
    step_to_physical_epoch,
    steps_per_epoch,
)
from .runtime import (
    capture_rng,
    full_state_hash,
    load_model_state,
    model_state,
    restore_rng,
    seed_everything,
    write_json,
)


SCHEMA = "final-unsb-route1-stratified-time-fixed-state-audit-v1"
DEFAULT_EPOCHS = (60, 100, 150, 200)
DEFAULT_REPLICATES = 8
MATERIAL_RATIO = 0.95
NONINFERIOR_TOLERANCE = 1e-9
SOURCE_CANDIDATES = {
    "proposal": "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
    "hjcgr": "G3-02-HJ-CONDITIONAL-GF-RESAMPLING",
}


def covariance_trace_prediction(
    *, within_trace: float, between_trace: float, time_strata: int,
) -> dict[str, float]:
    """Return iid and without-replacement pair-mean covariance traces."""
    if int(time_strata) < 2:
        raise ValueError("stratified-time covariance needs at least two strata")
    within = max(float(within_trace), 0.0)
    between = max(float(between_trace), 0.0)
    iid = 0.5 * within + 0.5 * between
    finite_population = float(time_strata - 2) / float(2 * (time_strata - 1))
    without_replacement = 0.5 * within + finite_population * between
    ratio = 1.0 if iid <= 0.0 else without_replacement / iid
    return {
        "within_time_covariance_trace": within,
        "between_time_mean_covariance_trace": between,
        "iid_pair_mean_covariance_trace": iid,
        "without_replacement_pair_mean_covariance_trace": without_replacement,
        "without_replacement_to_iid_trace_ratio": ratio,
        "trace_reduction_fraction": 1.0 - ratio,
    }


class _TimeMoment:
    """Online trace moments without retaining all replica gradients."""

    def __init__(self) -> None:
        self.count = 0
        self.sum: list[torch.Tensor] | None = None
        self.sum_squared_norm = 0.0

    def add(
        self, gradients: tuple[torch.Tensor, ...],
        *, scales: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        if scales is not None and len(scales) != len(gradients):
            raise RuntimeError("Adam audit scales differ from gradient structure")
        values = tuple(
            gradient if scales is None else gradient * scales[index]
            for index, gradient in enumerate(gradients)
        )
        if self.sum is None:
            self.sum = [value.clone() for value in values]
        else:
            if len(self.sum) != len(values):
                raise RuntimeError("time-stratum gradient structures differ")
            for total, value in zip(self.sum, values):
                total.add_(value)
        self.sum_squared_norm += sum(
            float(value.double().square().sum().item()) for value in values
        )
        self.count += 1

    def mean(self) -> tuple[torch.Tensor, ...]:
        if self.count < 1 or self.sum is None:
            raise RuntimeError("time-stratum mean requires observations")
        return tuple(value / float(self.count) for value in self.sum)

    def sample_trace_variance(self) -> float:
        if self.count < 2 or self.sum is None:
            raise RuntimeError("time-stratum variance requires two observations")
        correction = sum(
            float(value.double().square().sum().item()) for value in self.sum
        ) / float(self.count)
        return max(self.sum_squared_norm - correction, 0.0) / float(self.count - 1)


def _squared_distance(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...],
) -> float:
    if len(left) != len(right):
        raise RuntimeError("time mean structures differ")
    return float(sum(
        (x.double() - y.double()).square().sum().item()
        for x, y in zip(left, right)
    ))


def summarize_time_moments(moments: list[_TimeMoment]) -> dict[str, float]:
    """Estimate covariance terms with finite-replicate bias correction."""
    if len(moments) < 2:
        raise ValueError("time-moment summary needs at least two strata")
    counts = {moment.count for moment in moments}
    if len(counts) != 1 or next(iter(counts)) < 2:
        raise RuntimeError("time strata need equal replicate counts of at least two")
    replicates = next(iter(counts))
    means = [moment.mean() for moment in moments]
    grand = tuple(
        sum((mean[index] for mean in means), torch.zeros_like(means[0][index]))
        / float(len(means))
        for index in range(len(means[0]))
    )
    within = sum(moment.sample_trace_variance() for moment in moments) / len(moments)
    raw_between = sum(_squared_distance(mean, grand) for mean in means) / len(means)
    # Independent stratum replicas add (T-1)/T * Sigma_bar/K to the observed
    # dispersion of sample means.  Remove that finite-K contribution before
    # judging whether the conditional means differ materially.
    bias = (len(means) - 1) / len(means) * within / float(replicates)
    corrected_between = max(raw_between - bias, 0.0)
    result = covariance_trace_prediction(
        within_trace=within,
        between_trace=corrected_between,
        time_strata=len(means),
    )
    result.update({
        "raw_between_time_sample_mean_dispersion": raw_between,
        "finite_replicate_between_trace_bias": bias,
        "gradient_replicates_per_time": int(replicates),
        "time_strata": len(means),
    })
    return result


def _spec_from_checkpoint(payload: dict) -> ProbeSpec:
    row = payload.get("probe")
    if not isinstance(row, dict):
        raise RuntimeError("source checkpoint lacks its frozen probe identity")
    return ProbeSpec(
        id=str(row["id"]),
        contract_id=str(row["contract_id"]),
        model=str(row["model"]),
        role=str(row["role"]),
        method=dict(row.get("method", {})),
        historical_fact=row.get("historical_fact"),
    )


def _rng_seed(checkpoint_sha256: str, time_index: int, replicate: int) -> int:
    digest = hashlib.sha256(
        f"stcgr-audit-v1:{checkpoint_sha256}:{time_index}:{replicate}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "little") % (2**31 - 1)


def _audit_checkpoint(
    *, source_root: Path, candidate_id: str, epoch: int, train_view: Path,
    manifest_path: Path, work_root: Path, gpu: int, replicates: int,
) -> dict[str, Any]:
    checkpoint = (
        source_root / "candidates" / candidate_id / "milestones"
        / f"e{int(epoch):03d}.pt"
    )
    if not checkpoint.is_file():
        raise RuntimeError(f"missing fixed-state audit checkpoint: {checkpoint}")
    checkpoint_sha256 = file_sha256(checkpoint)
    parent = torch.load(checkpoint, map_location="cpu", weights_only=False)
    expected_step = int(epoch) * steps_per_epoch()
    if int(parent.get("step", -1)) != expected_step:
        raise RuntimeError("fixed-state audit checkpoint epoch mismatch")
    spec = _spec_from_checkpoint(parent)
    if spec.id != candidate_id:
        raise RuntimeError("fixed-state audit source candidate identity mismatch")
    e0_path = source_root / "shared_e0" / "e0.pt"
    if not e0_path.is_file():
        raise RuntimeError("fixed-state audit source shared e0 is missing")
    e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    model, primary, secondary, _ = prepare_probe(
        spec=spec,
        output_root=work_root / candidate_id / f"e{int(epoch):03d}",
        train_view=train_view,
        manifest_path=manifest_path,
        gpu=int(gpu),
        e0=e0,
    )
    load_model_state(model, copy.deepcopy(parent["model"]), load_method=True)
    primary.load_state_dict(copy.deepcopy(parent["samplers"]["primary"]))
    secondary.load_state_dict(copy.deepcopy(parent["samplers"]["secondary"]))
    restore_rng(copy.deepcopy(parent["rng"]))
    model.set_train_epoch(step_to_physical_epoch(expected_step))
    model.set_search_step(expected_step, int(parent.get("target_steps", 30_000)))
    model.set_input(primary.next(), secondary.next())
    _commit_fixed_de_state(model)

    post_de_model = copy.deepcopy(model_state(model))
    post_de_primary = copy.deepcopy(primary.state_dict())
    post_de_secondary = copy.deepcopy(secondary.state_dict())
    post_de_rng = copy.deepcopy(capture_rng())
    post_de_hash = full_state_hash({
        "model": post_de_model,
        "primary": post_de_primary,
        "secondary": post_de_secondary,
        "rng": post_de_rng,
    })
    method_state = copy.deepcopy(post_de_model.get("method", {}))
    parameters = _gradient_parameters(model)
    gf_optimizers = [model.optimizer_G]
    if getattr(model.opt, "netF", None) == "mlp_sample":
        gf_optimizers.append(model.optimizer_F)
    adam_scales = _adam_update_space_scales(parameters, tuple(gf_optimizers))
    original_sampler = model._sample_training_time_idx
    euclidean: list[_TimeMoment] = []
    adam_metric: list[_TimeMoment] = []
    try:
        for time_index in range(int(model.opt.num_timesteps)):
            native_moment = _TimeMoment()
            metric_moment = _TimeMoment()
            model._sample_training_time_idx = (
                lambda total, value=time_index: torch.full(
                    (1,), int(value), dtype=torch.long, device=model.device,
                )
            )
            for replicate in range(int(replicates)):
                seed_everything(_rng_seed(checkpoint_sha256, time_index, replicate))
                gradients = _gf_gradient(model, parameters, antithetic=False)
                native_moment.add(gradients)
                metric_moment.add(gradients, scales=adam_scales)
                model.load_extra_training_state(copy.deepcopy(method_state))
                del gradients
            euclidean.append(native_moment)
            adam_metric.append(metric_moment)
    finally:
        model._sample_training_time_idx = original_sampler
        load_model_state(model, post_de_model, load_method=True)
        primary.load_state_dict(post_de_primary)
        secondary.load_state_dict(post_de_secondary)
        restore_rng(post_de_rng)
        for optimizer in model.optimizers:
            optimizer.zero_grad(set_to_none=True)
    post_audit_hash = full_state_hash({
        "model": model_state(model),
        "primary": primary.state_dict(),
        "secondary": secondary.state_dict(),
        "rng": capture_rng(),
    })
    if post_de_hash != post_audit_hash:
        raise RuntimeError("stratified-time audit mutated the fixed post-D/E state")
    checkpoint_sha256_after = file_sha256(checkpoint)
    if checkpoint_sha256 != checkpoint_sha256_after:
        raise RuntimeError("stratified-time audit modified a source checkpoint")
    result = {
        "candidate_id": candidate_id,
        "data_epoch": int(epoch),
        "updates": expected_step,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256_before": checkpoint_sha256,
        "source_checkpoint_sha256_after": checkpoint_sha256_after,
        "fixed_post_de_state_sha256_before": post_de_hash,
        "fixed_post_de_state_sha256_after": post_audit_hash,
        "gradient_replicates_per_time": int(replicates),
        "time_strata": int(model.opt.num_timesteps),
        "euclidean": summarize_time_moments(euclidean),
        "adam_metric": summarize_time_moments(adam_metric),
        "paired_metric_computed": False,
        "confirmation20_opened": False,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _aggregate(rows: list[dict[str, Any]], geometry: str) -> dict[str, Any]:
    iid = sum(float(row[geometry]["iid_pair_mean_covariance_trace"]) for row in rows)
    wor = sum(
        float(row[geometry]["without_replacement_pair_mean_covariance_trace"])
        for row in rows
    )
    ratio = 1.0 if iid <= 0.0 else wor / iid
    individual = [
        float(row[geometry]["without_replacement_to_iid_trace_ratio"])
        for row in rows
    ]
    return {
        "pooled_iid_pair_mean_covariance_trace": iid,
        "pooled_without_replacement_pair_mean_covariance_trace": wor,
        "pooled_without_replacement_to_iid_trace_ratio": ratio,
        "individual_ratios": individual,
        "all_checkpoints_noninferior": all(
            value <= 1.0 + NONINFERIOR_TOLERANCE for value in individual
        ),
        "material_checkpoint_count": sum(value <= MATERIAL_RATIO for value in individual),
    }


def run_stratified_time_fixed_state_audit(
    *, source_root: Path, train_view: Path, manifest_path: Path, output: Path,
    gpu: int, lanes: tuple[str, ...] = ("proposal", "hjcgr"),
    epochs: tuple[int, ...] = DEFAULT_EPOCHS,
    replicates: int = DEFAULT_REPLICATES,
) -> dict[str, Any]:
    if int(replicates) < 2:
        raise ValueError("stratified-time audit needs at least two gradients per time")
    unknown = sorted(set(lanes) - set(SOURCE_CANDIDATES))
    if unknown:
        raise ValueError(f"unknown stratified-time parent lanes: {unknown}")
    source_root = Path(source_root).resolve()
    work_root = Path(output).resolve().parent / "STCGR_AUDIT_WORK"
    rows = [
        _audit_checkpoint(
            source_root=source_root,
            candidate_id=SOURCE_CANDIDATES[lane],
            epoch=int(epoch),
            train_view=Path(train_view).resolve(),
            manifest_path=Path(manifest_path).resolve(),
            work_root=work_root,
            gpu=int(gpu),
            replicates=int(replicates),
        )
        for lane in lanes for epoch in epochs
    ]
    by_lane = {}
    for lane in lanes:
        candidate_id = SOURCE_CANDIDATES[lane]
        lane_rows = [row for row in rows if row["candidate_id"] == candidate_id]
        euclidean = _aggregate(lane_rows, "euclidean")
        adam_metric = _aggregate(lane_rows, "adam_metric")
        lane_pass = (
            euclidean["pooled_without_replacement_to_iid_trace_ratio"] <= MATERIAL_RATIO
            and euclidean["all_checkpoints_noninferior"] is True
            and euclidean["material_checkpoint_count"] >= 2
        )
        by_lane[lane] = {
            "candidate_id": candidate_id,
            "euclidean": euclidean,
            "adam_metric_diagnostic": adam_metric,
            "material_gate_pass": lane_pass,
        }
    is_full_gate = (
        tuple(lanes) == ("proposal", "hjcgr")
        and tuple(int(value) for value in epochs) == DEFAULT_EPOCHS
        and int(replicates) >= DEFAULT_REPLICATES
    )
    authorized = is_full_gate and all(
        lane["material_gate_pass"] for lane in by_lane.values()
    )
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_TARGET_BLIND_FIXED_STATE_AUDIT",
        "source_root": str(source_root),
        "lanes": list(lanes),
        "epochs": [int(value) for value in epochs],
        "gradient_replicates_per_time": int(replicates),
        "state_boundary": "one native D/E commit, then frozen G/F audit",
        "bias_correction": "subtract (T-1)/T * mean_within_trace / replicates from sample-mean dispersion",
        "material_ratio_threshold": MATERIAL_RATIO,
        "minimum_material_checkpoints_per_lane": 2,
        "noninferiority_tolerance": NONINFERIOR_TOLERANCE,
        "lane_summaries": by_lane,
        "rows": rows,
        "full_preregistered_gate_executed": is_full_gate,
        "small25_e200_authorized": authorized,
        "decision": (
            "AUTHORIZE_SMALL25_E200"
            if authorized else
            "NO_LONG_RUN_AUTHORIZATION_FROM_THIS_AUDIT"
        ),
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output).resolve(), result)
    return result
