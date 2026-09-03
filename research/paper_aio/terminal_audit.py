"""Target-blind bridge-terminal spectrum and perturbation audit."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from production.metrics import bridge_times, build_rollout_bundle
from research.local_route1.runtime import (
    capture_rng,
    cpu_clone,
    full_state_hash,
    load_model_state,
    model_state,
    restore_rng,
)

from .evaluate import read_image
from .protocol import LaneSpec


def _gram_spectrum(samples: list[torch.Tensor]) -> dict:
    """Return the nonzero spectrum of the unbiased sample covariance.

    For flattened samples ``X`` with rows centred across stochastic rollouts,
    ``X.T @ X / (n - 1)`` is the feature-space sample covariance. Forming that
    matrix is prohibitive for images, so we use ``X @ X.T / (n - 1)``; the two
    matrices have exactly the same nonzero eigenvalues. Dividing by the feature
    dimension instead would describe a per-coordinate Gram matrix, not the
    covariance spectrum named by the paper protocol.
    """
    if not samples:
        raise ValueError("covariance spectrum requires at least one sample")
    matrix = torch.stack(
        [value.detach().double().cpu().reshape(-1) for value in samples]
    )
    matrix = matrix - matrix.mean(dim=0, keepdim=True)
    sample_count = int(matrix.shape[0])
    flattened_dimension = int(matrix.shape[1])
    gram = matrix @ matrix.T / max(1, sample_count - 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    total = float(eigenvalues.sum())
    squared = float(eigenvalues.square().sum())
    return {
        "top_eigenvalue": float(eigenvalues[0]) if eigenvalues.numel() else 0.0,
        "trace": total,
        "effective_rank": (total * total / squared) if squared > 0 else 0.0,
        "effective_rank_definition": "participation_ratio_trace_squared_over_frobenius_squared",
        "normalization": "unbiased_sample_covariance_nonzero_spectrum_n_minus_1",
        "sample_count": sample_count,
        "flattened_dimension": flattened_dimension,
        "eigenvalues": [float(value) for value in eigenvalues],
    }


def _cosine_to_mean(samples: list[torch.Tensor]) -> float:
    flat = torch.stack([value.detach().double().cpu().reshape(-1) for value in samples])
    mean = flat.mean(dim=0, keepdim=True)
    return float(F.cosine_similarity(flat, mean.expand_as(flat), dim=1).mean())


@torch.no_grad()
def rollout_trace(net_g, source: torch.Tensor, bundle: dict, *, tau: float) -> dict:
    times = bridge_times(5)
    state = source
    endpoints, states, increments = [], [], []
    endpoint = None
    for step in range(5):
        previous = state
        if step > 0:
            delta = float(times[step] - times[step - 1])
            denominator = float(times[-1] - times[step - 1])
            alpha = delta / denominator
            variance = delta * (1.0 - alpha)
            state = (
                (1.0 - alpha) * state
                + alpha * endpoint.detach()
                + math.sqrt(variance * tau) * bundle["noise"][step].to(source.device)
            )
        increments.append(state - previous)
        states.append(state)
        time_index = torch.full(
            (source.shape[0],), step, dtype=torch.long, device=source.device
        )
        endpoint = net_g(state, time_index, bundle["z"][step].to(source.device))
        endpoints.append(endpoint)
    return {"states": states, "increments": increments, "endpoints": endpoints}


@torch.no_grad()
def rollout_from_state(
    net_g,
    state: torch.Tensor,
    bundle: dict,
    *,
    start_step: int,
    tau: float,
) -> torch.Tensor:
    """Propagate a state perturbation through the remaining frozen operators."""
    times = bridge_times(5)
    endpoint = None
    for step in range(int(start_step), 5):
        if step > int(start_step):
            delta = float(times[step] - times[step - 1])
            denominator = float(times[-1] - times[step - 1])
            alpha = delta / denominator
            variance = delta * (1.0 - alpha)
            state = (
                (1.0 - alpha) * state
                + alpha * endpoint.detach()
                + math.sqrt(variance * tau) * bundle["noise"][step].to(state.device)
            )
        time_index = torch.full(
            (state.shape[0],),
            step,
            dtype=torch.long,
            device=state.device,
        )
        endpoint = net_g(state, time_index, bundle["z"][step].to(state.device))
    return endpoint


def _differentiable_rollout_from_state(
    net_g,
    state: torch.Tensor,
    bundle: dict,
    *,
    start_step: int,
    tau: float,
) -> torch.Tensor:
    """Return the numerical rollout map with its full state Jacobian intact.

    Production inference uses ``detach`` because it never differentiates the
    rollout.  The detach does not alter its numerical values, but retaining it
    here would incorrectly discard endpoint-mediated sensitivity in the audit.
    """
    times = bridge_times(len(bundle["z"]))
    endpoint = None
    for step in range(int(start_step), len(bundle["z"])):
        if step > int(start_step):
            delta = float(times[step] - times[step - 1])
            denominator = float(times[-1] - times[step - 1])
            alpha = delta / denominator
            variance = delta * (1.0 - alpha)
            state = (
                (1.0 - alpha) * state
                + alpha * endpoint
                + math.sqrt(variance * tau) * bundle["noise"][step].to(state.device)
            )
        time_index = torch.full(
            (state.shape[0],),
            step,
            dtype=torch.long,
            device=state.device,
        )
        endpoint = net_g(state, time_index, bundle["z"][step].to(state.device))
    return endpoint


@torch.no_grad()
def perturbation_gain_to_final(
    net_g,
    state: torch.Tensor,
    bundle: dict,
    *,
    start_step: int,
    tau: float,
    epsilon: float = 1e-3,
    direction: torch.Tensor | None = None,
) -> float:
    if direction is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(91_000 + int(start_step))
        direction = torch.randn(state.shape, generator=generator, dtype=torch.float32)
    else:
        direction = direction.detach().to(dtype=torch.float32, device="cpu")
    direction = direction.to(state.device)
    direction = direction / direction.norm().clamp_min(1e-12)
    baseline = rollout_from_state(
        net_g,
        state,
        bundle,
        start_step=start_step,
        tau=tau,
    )
    perturbed = rollout_from_state(
        net_g,
        state + float(epsilon) * direction,
        bundle,
        start_step=start_step,
        tau=tau,
    )
    return float((perturbed - baseline).norm().cpu() / float(epsilon))


def _top_singular_jvp(
    function,
    state: torch.Tensor,
    initial_direction: torch.Tensor,
    *,
    iterations: int = 2,
) -> float:
    """Estimate a map's top singular value from a fixed lane-blind direction."""
    x = state.detach().requires_grad_(True)
    vector = initial_direction.detach().to(device=x.device, dtype=x.dtype)
    vector = vector / vector.norm().clamp_min(1e-12)
    sigma = 0.0
    for _ in range(int(iterations)):
        _, jv = torch.autograd.functional.jvp(
            function,
            x,
            vector,
            create_graph=False,
            strict=False,
        )
        sigma = float(jv.norm().detach())
        if sigma <= 0:
            return 0.0
        output = function(x)
        jt = torch.autograd.grad(
            output,
            x,
            grad_outputs=jv.detach(),
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        vector = jt.detach() / jt.detach().norm().clamp_min(1e-12)
    return sigma


def _local_jvp_gain(
    net_g,
    state: torch.Tensor,
    time_index: int,
    z: torch.Tensor,
    initial_direction: torch.Tensor,
    *,
    iterations: int = 2,
) -> float:
    """Estimate the current generator call's local top singular value."""
    t = torch.full(
        (state.shape[0],),
        int(time_index),
        dtype=torch.long,
        device=state.device,
    )
    z = z.detach().to(state.device)
    return _top_singular_jvp(
        lambda value: net_g(value, t, z),
        state,
        initial_direction,
        iterations=iterations,
    )


def _rollout_jvp_gain(
    net_g,
    state: torch.Tensor,
    bundle: dict,
    *,
    start_step: int,
    tau: float,
    initial_direction: torch.Tensor,
    iterations: int = 2,
) -> float:
    """Estimate the complete X_t-to-final-endpoint rollout singular value."""
    return _top_singular_jvp(
        lambda value: _differentiable_rollout_from_state(
            net_g,
            value,
            bundle,
            start_step=start_step,
            tau=tau,
        ),
        state,
        initial_direction,
        iterations=iterations,
    )


def _gradient_tuple(loss, parameters, *, retain_graph: bool):
    if not torch.is_tensor(loss) or not loss.requires_grad:
        return tuple(torch.zeros_like(parameter) for parameter in parameters)
    values = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if value is None else value.detach()
        for parameter, value in zip(parameters, values)
    )


def _tuple_cosine(left, right) -> float:
    dot = sum((a.double() * b.double()).sum() for a, b in zip(left, right))
    norm_left = sum(a.double().square().sum() for a in left).sqrt()
    norm_right = sum(b.double().square().sum() for b in right).sqrt()
    denominator = norm_left * norm_right
    return float(dot / denominator) if float(denominator) > 0 else 0.0


def _adam_preconditioned_norm(parameters, gradients, optimizers) -> float:
    state_by_parameter = {}
    group_by_parameter = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state_by_parameter[parameter] = optimizer.state.get(parameter, {})
                group_by_parameter[parameter] = group
    total = torch.zeros((), dtype=torch.float64)
    for parameter, gradient in zip(parameters, gradients):
        state = state_by_parameter.get(parameter, {})
        group = group_by_parameter.get(parameter, {})
        variance = state.get("exp_avg_sq")
        if variance is None:
            scaled = gradient.double()
        else:
            step = state.get("step", 0)
            step = int(step.item()) if torch.is_tensor(step) else int(step)
            beta2 = float(group.get("betas", (0.5, 0.999))[1])
            correction = max(1e-30, 1.0 - beta2 ** max(1, step))
            denominator = (variance.detach().double() / correction).sqrt()
            denominator = denominator + float(group.get("eps", 1e-8))
            scaled = gradient.detach().double() / denominator
        total += scaled.square().sum().cpu()
    return float(total.sqrt())


def gradient_stratum_statistics(
    model,
    primary,
    secondary,
    *,
    replicates: int = 4,
) -> dict:
    """Measure forced-time native G/F gradients without an optimizer step.

    A and B are drawn from the checkpoint's independent training samplers.
    The routine cannot address paired discovery targets and restores every
    transition-defining state before returning.
    """
    if primary is None or secondary is None:
        return {"status": "NOT_REQUESTED_NO_TRAIN_STREAMS", "strata": []}
    if int(replicates) < 2:
        raise ValueError("gradient stratum covariance requires at least two replicates")
    saved_model = cpu_clone(model_state(model))
    saved_rng = capture_rng()
    primary_state = cpu_clone(primary.state_dict())
    secondary_state = cpu_clone(secondary.state_dict())
    sampler_was_instance_attribute = "_sample_training_time_idx" in model.__dict__
    original_instance_sampler = model.__dict__.get("_sample_training_time_idx")
    original_requires_grad = [
        (parameter, bool(parameter.requires_grad))
        for name in model.model_names
        for parameter in getattr(model, "net" + name).parameters()
    ]
    modes = {name: getattr(model, "net" + name).training for name in model.model_names}
    parameters = [
        parameter
        for network in (model.netG, model.netF)
        for parameter in network.parameters()
        if parameter.requires_grad
    ]
    strata = []
    try:
        for time_index in range(int(model.opt.num_timesteps)):
            # Every time stratum sees the same sequence of training images and
            # stochastic draws.  Only the forced bridge coordinate differs.
            load_model_state(model, saved_model)
            primary.load_state_dict(primary_state)
            secondary.load_state_dict(secondary_state)
            restore_rng(saved_rng)
            model._sample_training_time_idx = lambda total, value=time_index: (
                torch.full(
                    (1,),
                    value,
                    dtype=torch.long,
                    device=model.device,
                )
            )
            flattened = []
            metric_norms = []
            component_cosines = None
            for replicate in range(int(replicates)):
                model.set_input(primary.next(), secondary.next())
                for name in model.model_names:
                    getattr(model, "net" + name).train(True)
                model.forward()
                model.set_requires_grad(model.netD, False)
                model.set_requires_grad(model.netE, False)
                loss = model.compute_G_loss()
                if replicate == 0:
                    gan = _gradient_tuple(
                        model.loss_G_GAN, parameters, retain_graph=True
                    )
                    sb = _gradient_tuple(
                        float(model.opt.lambda_SB) * model.loss_SB,
                        parameters,
                        retain_graph=True,
                    )
                    residual_loss = (
                        loss
                        - model.loss_G_GAN
                        - float(model.opt.lambda_SB) * model.loss_SB
                    )
                    nce = _gradient_tuple(residual_loss, parameters, retain_graph=True)
                    component_cosines = {
                        "gan_sb": _tuple_cosine(gan, sb),
                        "gan_nce": _tuple_cosine(gan, nce),
                        "sb_nce": _tuple_cosine(sb, nce),
                    }
                    del gan, sb, nce
                gradients = _gradient_tuple(loss, parameters, retain_graph=False)
                flattened.append(
                    torch.cat([value.float().cpu().reshape(-1) for value in gradients])
                )
                metric_norms.append(
                    _adam_preconditioned_norm(
                        parameters,
                        gradients,
                        [model.optimizer_G, model.optimizer_F],
                    )
                )
                del gradients, loss
            stacked = torch.stack(flattened).double()
            mean = stacked.mean(dim=0)
            second_moment = stacked.square().sum(dim=1).mean()
            centered = stacked - mean
            variance_trace = centered.square().sum() / float(int(replicates) - 1)
            strata.append(
                {
                    "time_index": time_index,
                    "replicates": int(replicates),
                    "gradient_mean_norm": float(mean.norm()),
                    "gradient_variance_trace": float(variance_trace),
                    "gradient_variance_normalization": "unbiased_sample_covariance_trace_n_minus_1",
                    "gradient_second_moment": float(second_moment),
                    "adam_preconditioned_norm_mean": float(np.mean(metric_norms)),
                    "adam_preconditioned_norm_std": float(
                        np.std(metric_norms, ddof=1)
                    ),
                    "adam_preconditioned_norm_std_normalization": "sample_std_n_minus_1",
                    "loss_component_gradient_cosines_first_batch": component_cosines,
                }
            )
            del stacked, flattened, mean, centered
    finally:
        if sampler_was_instance_attribute:
            model.__dict__["_sample_training_time_idx"] = original_instance_sampler
        else:
            model.__dict__.pop("_sample_training_time_idx", None)
        load_model_state(model, saved_model)
        primary.load_state_dict(primary_state)
        secondary.load_state_dict(secondary_state)
        restore_rng(saved_rng)
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        for parameter, requires_grad in original_requires_grad:
            parameter.requires_grad_(requires_grad)
        for optimizer in model.optimizers:
            optimizer.zero_grad(set_to_none=True)
    return {
        "status": "TARGET_BLIND_NATIVE_OBJECTIVE_GRADIENT_AUDIT_COMPLETE",
        "definition": "forced-time native G/F objective gradient with cross-time common batches and RNG; no optimizer transition",
        "cross_time_common_sampler_state": True,
        "cross_time_common_rng_state": True,
        "forward_mode": "training_for_every_replicate",
        "parent_requires_grad_flags_restored": True,
        "strata": strata,
    }


def audit_model(
    *,
    model,
    spec: LaneSpec,
    rows: list[dict],
    data_root: Path,
    protocol_hash: str,
    checkpoint_label: str,
    replicates: int = 32,
    samples_per_domain: int = 1,
    primary=None,
    secondary=None,
    gradient_replicates: int = 4,
) -> dict:
    if spec.family != "unsb":
        raise RuntimeError("terminal bridge audit only applies to UNSB-family lanes")
    saved_rng = capture_rng()
    saved_rng_hash = full_state_hash(saved_rng)
    saved_model = cpu_clone(model_state(model))
    before = full_state_hash(saved_model)
    modes = {name: getattr(model, "net" + name).training for name in model.model_names}
    selected = []
    for domain in sorted({row["domain"] for row in rows}):
        candidates = [
            row
            for row in rows
            if row["domain"] == domain and row["split"] == "discovery"
        ]
        candidates.sort(key=lambda row: int(row["order"]))
        selected.extend(candidates[: int(samples_per_domain)])
    records = []
    gradient_audit = None
    try:
        model.eval()
        for row in selected:
            source = read_image(Path(data_root) / row["input_relpath"]).to(model.device)
            traces = []
            for replicate in range(int(replicates)):
                bundle = build_rollout_bundle(
                    protocol_hash=protocol_hash,
                    domain=row["domain"],
                    stem=row["stem"],
                    replicate=replicate,
                    latent_dim=4 * int(model.opt.ngf),
                    height=128,
                    width=128,
                    num_timesteps=5,
                )
                traces.append(
                    rollout_trace(model.netG, source, bundle, tau=float(model.opt.tau))
                )
            steps = []
            for step in range(5):
                increments = [trace["increments"][step] for trace in traces]
                endpoints = [trace["endpoints"][step] for trace in traces]
                transport_directions = [
                    trace["endpoints"][step] - trace["states"][step] for trace in traces
                ]
                reference_bundle = build_rollout_bundle(
                    protocol_hash=protocol_hash,
                    domain=row["domain"],
                    stem=row["stem"],
                    replicate=0,
                    latent_dim=4 * int(model.opt.ngf),
                    height=128,
                    width=128,
                    num_timesteps=5,
                )
                probe_direction = reference_bundle["noise"][step]
                local_jvp = _local_jvp_gain(
                    model.netG,
                    traces[0]["states"][step],
                    step,
                    reference_bundle["z"][step],
                    probe_direction,
                )
                rollout_jvp = _rollout_jvp_gain(
                    model.netG,
                    traces[0]["states"][step],
                    reference_bundle,
                    start_step=step,
                    tau=float(model.opt.tau),
                    initial_direction=probe_direction,
                )
                propagated_gain = perturbation_gain_to_final(
                    model.netG,
                    traces[0]["states"][step],
                    reference_bundle,
                    start_step=step,
                    tau=float(model.opt.tau),
                    direction=probe_direction,
                )
                steps.append(
                    {
                        "time_index": step,
                        "increment_spectrum": _gram_spectrum(increments),
                        "endpoint_spectrum": _gram_spectrum(endpoints),
                        "endpoint_direction_cosine_to_mean": _cosine_to_mean(
                            transport_directions,
                        ),
                        "endpoint_direction_definition": "endpoint_minus_bridge_state",
                        "local_jacobian_top_singular_proxy": local_jvp,
                        "rollout_jacobian_top_singular_proxy": rollout_jvp,
                        "jvp_initial_direction": "lane_blind_crn_bridge_noise_same_sample_time",
                        "perturbation_gain_to_final_output": propagated_gain,
                    }
                )
            nfe45 = [
                float(
                    (trace["endpoints"][4] - trace["endpoints"][3])
                    .square()
                    .mean()
                    .sqrt()
                    .cpu()
                )
                for trace in traces
            ]
            records.append(
                {
                    "domain": row["domain"],
                    "stem": row["stem"],
                    "steps": steps,
                    "nfe4_to_nfe5_output_rms_mean": float(np.mean(nfe45)),
                    "nfe4_to_nfe5_output_rms_std": float(np.std(nfe45)),
                }
            )
        gradient_audit = gradient_stratum_statistics(
            model,
            primary,
            secondary,
            replicates=gradient_replicates,
        )
    finally:
        load_model_state(model, saved_model)
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)
    after = full_state_hash(model_state(model))
    after_rng_hash = full_state_hash(capture_rng())
    if before != after:
        raise RuntimeError("terminal audit mutated parent model/optimizer state")
    if saved_rng_hash != after_rng_hash:
        raise RuntimeError("terminal audit mutated parent RNG state")
    result = {
        "schema": "final-unsb-paper-terminal-spectrum-audit-v1",
        "status": "TARGET_BLIND_AUDIT_COMPLETE",
        "lane_id": spec.id,
        "checkpoint_label": checkpoint_label,
        "replicates": int(replicates),
        "samples_per_domain": int(samples_per_domain),
        "records": records,
        "gradient_stratum_audit": gradient_audit,
        "rollout_jacobian_definition": (
            "full numerical frozen NFE5 map from X_t to final endpoint; "
            "includes endpoint-mediated subsequent bridge-state transitions"
        ),
        "parent_state_sha256_before": before,
        "parent_state_sha256_after": after,
        "parent_rng_sha256_before": saved_rng_hash,
        "parent_rng_sha256_after": after_rng_hash,
        "paired_labels_attached": False,
        "terminal_pathology_confirmed": False,
        "confirmation_rule": "requires posthoc cross-algorithm evidence in at least two methods and three domains",
        "confirmation20_opened": False,
    }
    return result


def append_audit(path: Path, payload: dict) -> None:
    """Append one row without exposing a torn authoritative JSONL file.

    Terminal audits can run for hours. An interruption during a direct append
    used to leave an existing but invalid file that the durable successor could
    neither trust nor replay. Preserve all complete rows, fsync a same-directory
    temporary file, and atomically replace the authority path. A pre-existing
    torn row remains fail-closed instead of being silently hidden by a retry.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    previous = path.read_bytes() if path.is_file() else b""
    if previous and not previous.endswith(b"\n"):
        raise RuntimeError(f"refusing to append to incomplete audit JSONL: {path}")
    row = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(previous)
            handle.write(row)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
