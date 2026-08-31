"""Target-blind fixed-state audit for an antithetic native G/F estimator.

The audited involution keeps the official batch, bridge-time index, PatchNCE
patch ids, and all non-Gaussian randomness fixed, while negating every Gaussian
draw used by ``forward`` and ``compute_G_loss``.  Gaussian symmetry makes the
second view marginally native.  Averaging gradients therefore preserves the
conditional native G/F mean without averaging endpoints or changing inference.

This module only measures fixed-state covariance/variance.  It cannot freeze
or promote an algorithm and never reads paired quality.
"""

from __future__ import annotations

import contextlib
import copy
import math
from pathlib import Path
from typing import Any, Iterator

import torch

from .anchors import prepare_probe
from .candidate_defect_audit import _GradientTraceAccumulator, _mean_gradients
from .protocol import ProbeSpec, step_to_physical_epoch, steps_per_epoch
from .runtime import (
    capture_rng,
    full_state_hash,
    load_model_state,
    model_state,
    restore_rng,
    write_json,
)


SCHEMA = "final-unsb-route1-antithetic-gradient-audit-v1"
DEFAULT_EPOCHS = (20, 100, 200)


@contextlib.contextmanager
def negated_gaussian_draws() -> Iterator[None]:
    """Apply the measure-preserving ``x -> -x`` map to torch Gaussians."""
    original_randn = torch.randn
    original_randn_like = torch.randn_like

    def randn(*args, **kwargs):
        return -original_randn(*args, **kwargs)

    def randn_like(*args, **kwargs):
        return -original_randn_like(*args, **kwargs)

    torch.randn = randn
    torch.randn_like = randn_like
    try:
        yield
    finally:
        torch.randn = original_randn
        torch.randn_like = original_randn_like


def _plain_spec(label: str) -> ProbeSpec:
    return ProbeSpec(
        id=label,
        contract_id=label,
        model="sb",
        role="target_blind_antithetic_gradient_audit",
        method={},
    )


def _rng_hash(state: dict[str, Any]) -> str:
    return full_state_hash({"rng": state})


def _gradient_parameters(model) -> tuple[torch.nn.Parameter, ...]:
    networks = [model.netG]
    if getattr(model.opt, "netF", None) == "mlp_sample":
        networks.append(model.netF)
    return tuple(
        parameter
        for network in networks
        for parameter in network.parameters()
        if parameter.requires_grad
    )


def _set_training_modes(model) -> None:
    model.netG.train()
    model.netF.train()
    model.netD.train()
    model.netE.train()


def _gf_gradient(
    model,
    parameters: tuple[torch.nn.Parameter, ...],
    *,
    antithetic: bool,
) -> tuple[torch.Tensor, ...]:
    manager = negated_gaussian_draws() if antithetic else contextlib.nullcontext()
    with manager:
        model.forward()
        _set_training_modes(model)
        loss = model.compute_G_loss()
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=False, allow_unused=True,
        )
    result = tuple(
        torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        if gradient is None else gradient.detach().cpu().float().clone()
        for parameter, gradient in zip(parameters, gradients)
    )
    # Loss attributes retain graphs in the original training model.  Detach
    # only these diagnostics; no parameter, optimizer, sampler, or RNG changes.
    for name in ("loss_G", "loss_G_GAN", "loss_SB", "loss_NCE", "loss_NCE_Y"):
        value = getattr(model, name, None)
        if torch.is_tensor(value):
            setattr(model, name, value.detach())
    return result


def _commit_fixed_de_state(model) -> None:
    """Commit one native D/E transition, leaving G/F untouched."""
    model.forward()
    _set_training_modes(model)
    model.set_requires_grad(model.netD, True)
    model.optimizer_D.zero_grad()
    model.loss_D = model.compute_D_loss()
    model.loss_D.backward()
    model.optimizer_D.step()

    model.set_requires_grad(model.netE, True)
    model.optimizer_E.zero_grad()
    model.loss_E = model.compute_E_loss()
    model.loss_E.backward()
    model.optimizer_E.step()
    model.set_requires_grad(model.netD, False)
    model.set_requires_grad(model.netE, False)


def _dot(
    left: tuple[torch.Tensor, ...], right: tuple[torch.Tensor, ...],
) -> float:
    if len(left) != len(right):
        raise RuntimeError("gradient structures differ")
    return float(sum(
        (x.double() * y.double()).sum().item() for x, y in zip(left, right)
    ))


def _sum_difference_l2(
    left: _GradientTraceAccumulator, right: _GradientTraceAccumulator,
) -> float:
    if left.sum is None or right.sum is None or left.count != right.count:
        raise RuntimeError("gradient marginal sums are incomplete")
    count = float(left.count)
    return math.sqrt(max(sum(
        ((x.double() - y.double()) / count).square().sum().item()
        for x, y in zip(left.sum, right.sum)
    ), 0.0))


def _trace_covariance(
    left: _GradientTraceAccumulator,
    right: _GradientTraceAccumulator,
    cross_dot_sum: float,
) -> float:
    if left.sum is None or right.sum is None or left.count != right.count:
        raise RuntimeError("gradient covariance structures are incomplete")
    if left.count < 2:
        raise RuntimeError("gradient covariance requires two pairs")
    centered = cross_dot_sum - _dot(tuple(left.sum), tuple(right.sum)) / left.count
    return float(centered / (left.count - 1))


def summarize_gradient_pairs(
    native: _GradientTraceAccumulator,
    antithetic_marginal: _GradientTraceAccumulator,
    independent_marginal: _GradientTraceAccumulator,
    antithetic_mean: _GradientTraceAccumulator,
    independent_mean: _GradientTraceAccumulator,
    *,
    native_antithetic_cross_dot_sum: float,
    native_independent_cross_dot_sum: float,
) -> dict[str, Any]:
    native_variance = native.trace_variance()
    anti_variance = antithetic_mean.trace_variance()
    iid_variance = independent_mean.trace_variance()
    if native_variance <= 0.0:
        raise RuntimeError("native fixed-state gradient variance is degenerate")
    return {
        "pair_count": native.count,
        "native_trace_variance": native_variance,
        "antithetic_pair_mean_trace_variance": anti_variance,
        "independent_pair_mean_trace_variance": iid_variance,
        "antithetic_to_native_variance_ratio": anti_variance / native_variance,
        "independent_to_native_variance_ratio": iid_variance / native_variance,
        "antithetic_to_independent_variance_ratio": (
            None if iid_variance <= 0.0 else anti_variance / iid_variance
        ),
        "native_antithetic_trace_covariance": _trace_covariance(
            native, antithetic_marginal, native_antithetic_cross_dot_sum,
        ),
        "native_independent_trace_covariance": _trace_covariance(
            native, independent_marginal, native_independent_cross_dot_sum,
        ),
        "native_antithetic_empirical_mean_difference_l2": _sum_difference_l2(
            native, antithetic_marginal,
        ),
        "native_independent_empirical_mean_difference_l2": _sum_difference_l2(
            native, independent_marginal,
        ),
    }


def _audit_epoch(
    *,
    output_root: Path,
    train_view: Path,
    manifest_path: Path,
    gpu: int,
    epoch: int,
    pairs: int,
) -> dict[str, Any]:
    checkpoint_path = (
        output_root / "anchors" / "plain" / "milestones" / f"e{epoch:03d}.pt"
    )
    if not checkpoint_path.is_file():
        raise RuntimeError(f"antithetic audit requires plain e{epoch}")
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if int(parent.get("step", -1)) != epoch * steps_per_epoch():
        raise RuntimeError("antithetic audit parent epoch mismatch")
    parent_hash = full_state_hash(parent)
    e0 = torch.load(
        output_root / "shared_e0" / "e0.pt",
        map_location="cpu",
        weights_only=False,
    )
    model, primary, secondary, _ = prepare_probe(
        spec=_plain_spec(f"antithetic_audit_e{epoch}"),
        output_root=output_root / "audit" / "antithetic_gradient_work",
        train_view=train_view,
        manifest_path=manifest_path,
        gpu=gpu,
        e0=e0,
    )
    load_model_state(model, copy.deepcopy(parent["model"]), load_method=False)
    primary.load_state_dict(copy.deepcopy(parent["samplers"]["primary"]))
    secondary.load_state_dict(copy.deepcopy(parent["samplers"]["secondary"]))
    restore_rng(copy.deepcopy(parent["rng"]))
    zero_step = int(parent["step"])
    model.set_train_epoch(step_to_physical_epoch(zero_step))
    model.set_search_step(zero_step, int(parent.get("target_steps", 30000)))
    model.set_input(primary.next(), secondary.next())
    _commit_fixed_de_state(model)
    parameters = _gradient_parameters(model)
    state_before = full_state_hash({"model": model_state(model)})

    native = _GradientTraceAccumulator()
    anti_marginal = _GradientTraceAccumulator()
    iid_marginal = _GradientTraceAccumulator()
    anti_mean = _GradientTraceAccumulator()
    iid_mean = _GradientTraceAccumulator()
    anti_cross = 0.0
    iid_cross = 0.0
    rng_pair_identity = True
    for _ in range(int(pairs)):
        before = capture_rng()
        first = _gf_gradient(model, parameters, antithetic=False)
        after_first = capture_rng()
        restore_rng(copy.deepcopy(before))
        opposite = _gf_gradient(model, parameters, antithetic=True)
        after_opposite = capture_rng()
        rng_pair_identity = rng_pair_identity and (
            _rng_hash(after_first) == _rng_hash(after_opposite)
        )
        restore_rng(copy.deepcopy(after_first))
        independent = _gf_gradient(model, parameters, antithetic=False)

        native.add(first)
        anti_marginal.add(opposite)
        iid_marginal.add(independent)
        anti_mean.add(_mean_gradients(first, opposite))
        iid_mean.add(_mean_gradients(first, independent))
        anti_cross += _dot(first, opposite)
        iid_cross += _dot(first, independent)

    state_after = full_state_hash({"model": model_state(model)})
    result = summarize_gradient_pairs(
        native,
        anti_marginal,
        iid_marginal,
        anti_mean,
        iid_mean,
        native_antithetic_cross_dot_sum=anti_cross,
        native_independent_cross_dot_sum=iid_cross,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "data_epoch": epoch,
        "updates": epoch * steps_per_epoch(),
        "parent_checkpoint": str(checkpoint_path),
        "parent_state_sha256": parent_hash,
        "parent_state_sha256_after_audit": full_state_hash(parent),
        "fixed_post_de_model_state_sha256_before": state_before,
        "fixed_post_de_model_state_sha256_after": state_after,
        "all_antithetic_pairs_advance_rng_exactly_like_native": rng_pair_identity,
        "paired_metric_computed": False,
        **result,
    }


def run_antithetic_gradient_audit(
    *,
    output_root: Path,
    train_view: Path,
    manifest_path: Path,
    gpu: int,
    epochs: tuple[int, ...] = DEFAULT_EPOCHS,
    pairs: int = 8,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    if int(pairs) < 4:
        raise ValueError("antithetic gradient audit requires at least four pairs")
    rows = [
        _audit_epoch(
            output_root=output_root,
            train_view=Path(train_view).resolve(),
            manifest_path=Path(manifest_path).resolve(),
            gpu=int(gpu),
            epoch=int(epoch),
            pairs=int(pairs),
        )
        for epoch in epochs
    ]
    for row in rows:
        if row["parent_state_sha256"] != row["parent_state_sha256_after_audit"]:
            raise RuntimeError("antithetic audit mutated a parent checkpoint")
        if row["fixed_post_de_model_state_sha256_before"] != row[
            "fixed_post_de_model_state_sha256_after"
        ]:
            raise RuntimeError("antithetic gradient observations mutated training state")
        if row["all_antithetic_pairs_advance_rng_exactly_like_native"] is not True:
            raise RuntimeError("antithetic Gaussian transform changed RNG consumption")
    clearer = [
        row for row in rows
        if row["antithetic_to_independent_variance_ratio"] is not None
        and row["antithetic_to_independent_variance_ratio"] <= 0.9
        and row["native_antithetic_trace_covariance"] < 0.0
    ]
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE_TARGET_BLIND_FIXED_STATE_AUDIT",
        "epochs": [int(epoch) for epoch in epochs],
        "pairs_per_epoch": int(pairs),
        "involution": (
            "same official batch/time/patch and non-Gaussian randomness; negate every "
            "Gaussian forward and G-loss draw; average gradients, never endpoints"
        ),
        "unbiased_identity": (
            "conditional Gaussian symmetry gives E[(g(xi)+g(Axi))/2|S,b,t,q]=E[g|S,b,t,q]"
        ),
        "clear_variance_advantage_rule": (
            "antithetic/iid pair-mean variance <=0.9 and negative trace covariance"
        ),
        "clear_variance_advantage_epochs": [row["data_epoch"] for row in clearer],
        "candidate_generation_supported": len(clearer) >= 2,
        "candidate_generation_is_not_performed_by_this_audit": True,
        "rows": rows,
        "paired_target_used": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(
        output_root / "audit" / "ANTITHETIC_GAUSSIAN_GRADIENT_AUDIT.json",
        result,
    )
    return result

