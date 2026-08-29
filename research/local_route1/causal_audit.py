"""Actual-update counterfactual audit for the frozen local route-1 anchors.

Every audit cell restores a complete historical state and serially evaluates
the native UNSB operator and one historical probe operator from the same
networks, optimizers, schedulers, samplers and RNG bundle.  Paired discovery70
quality is optional and is computed only after both branches have finished.
It is never part of :class:`StateObservation` or an intervention input.
"""

from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import types
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
import torch

from .evaluate import evaluate_model
from .interfaces import StateObservation
from .observations import (
    component_directional_derivatives,
    state_dict_delta_cosine,
    state_dict_update_geometry,
)
from .protocol import (
    FULL_STATE_SCHEMA,
    ROOT,
    canonical_json,
    file_sha256,
    git_commit,
    load_protocol,
    object_sha256,
    portable_source_sha256,
    probe_spec,
    steps_per_epoch,
)
from .runtime import (
    SerializableDataStream,
    build_datasets,
    build_model,
    build_options,
    capture_rng,
    cpu_clone,
    full_state_hash,
    inner,
    load_model_state,
    read_manifest,
    restore_rng,
    seed_everything,
    write_json,
)


ATLAS_SCHEMA = "final-unsb-local-route1-reversal-atlas-row-v1"
MATRIX_SCHEMA = "final-unsb-local-route1-causal-matrix-v1"
DEFAULT_HORIZONS = (1, 8, 32, 200)


@dataclass(frozen=True)
class AuditCell:
    probe: str
    data_epoch: int
    plain_state: Path
    method_state: Path

    @property
    def step(self) -> int:
        return int(self.data_epoch) * 150


@dataclass
class BranchResult:
    observation: StateObservation
    before_g: dict[str, torch.Tensor]
    after_g: dict[str, torch.Tensor]
    component_gradients: dict[str, dict[str, torch.Tensor]]
    metrics: dict | None
    scientific_state_after: str


class _NullDiagnostics:
    def log(self, **fields) -> None:
        del fields


def _cpu_state_dict(net) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in inner(net).state_dict().items()
    }


def _core_relative_files(root: Path) -> list[Path]:
    exact = [
        root / "configs" / "LOCAL_ROUTE1_PROBES.json",
        root / "production" / "metrics.py",
        root / "research" / "local_route1" / "protocol.py",
        root / "research" / "local_route1" / "runtime.py",
        root / "research" / "local_route1" / "anchors.py",
        root / "research" / "local_route1" / "evaluate.py",
    ]
    for directory in (root / "src" / "models", root / "src" / "data", root / "src" / "options"):
        if directory.is_dir():
            exact.extend(directory.rglob("*.py"))
    return sorted({path.resolve() for path in exact if path.is_file()}, key=lambda item: item.as_posix())


def training_core_fingerprint(root: Path) -> str:
    """Hash scientific runtime semantics without including the new auditor."""
    root = Path(root).resolve()
    rows = [
        (path.relative_to(root).as_posix(), portable_source_sha256(path))
        for path in _core_relative_files(root)
    ]
    return object_sha256(rows)


def audit_identity(training_root: Path | None = None) -> dict:
    training_root = Path(training_root or ROOT).resolve()
    current_core = training_core_fingerprint(ROOT)
    training_core = training_core_fingerprint(training_root)
    if current_core != training_core:
        raise RuntimeError(
            "audit/training core mismatch; do not interpret branches across different UNSB semantics"
        )
    audit_sources = [Path(__file__).resolve(), Path(__file__).with_name("observations.py")]
    return {
        "schema": "final-unsb-local-route1-audit-identity-v1",
        "training_root": str(training_root),
        "training_core_fingerprint": training_core,
        "audit_git_commit": git_commit(),
        "audit_source_fingerprint": object_sha256([
            (path.name, portable_source_sha256(path)) for path in audit_sources
        ]),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _load_checkpoint(path: Path, *, expected_probe: str) -> dict:
    path = Path(path).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FULL_STATE_SCHEMA:
        raise RuntimeError(f"unsupported full-state schema: {path}")
    if payload.get("probe", {}).get("id") != expected_probe:
        raise RuntimeError(f"checkpoint probe mismatch: expected {expected_probe}: {path}")
    if payload.get("metadata", {}).get("confirmation20_opened") is not False:
        raise RuntimeError("checkpoint does not prove confirmation20 remained closed")
    if int(payload.get("step", -1)) % 150 != 0:
        raise RuntimeError("audit source must be an exact data-epoch boundary")
    return payload


def validate_checkpoint_pair(cell: AuditCell) -> dict:
    plain = _load_checkpoint(cell.plain_state, expected_probe="plain")
    method = _load_checkpoint(cell.method_state, expected_probe=cell.probe)
    if int(plain["step"]) != cell.step or int(method["step"]) != cell.step:
        raise RuntimeError("checkpoint step does not match requested data epoch")
    keys = ("project_id", "seed", "manifest_sha256", "protocol_fingerprint", "git_commit")
    mismatches = [
        key for key in keys
        if plain.get("metadata", {}).get(key) != method.get("metadata", {}).get(key)
    ]
    if mismatches:
        raise RuntimeError(f"matched checkpoint identity mismatch: {mismatches}")
    return {
        "training_git_commit": plain["metadata"]["git_commit"],
        "training_protocol_fingerprint": plain["metadata"]["protocol_fingerprint"],
        "manifest_sha256": plain["metadata"]["manifest_sha256"],
        "seed": int(plain["metadata"]["seed"]),
        "plain_checkpoint_sha256": file_sha256(cell.plain_state),
        "method_checkpoint_sha256": file_sha256(cell.method_state),
        "plain_scientific_state_sha256": full_state_hash(plain),
        "method_scientific_state_sha256": full_state_hash(method),
        "confirmation20_opened": False,
    }


def _streams(opt, rows: list[dict], per_domain: int, seed: int):
    primary_data, secondary_data = build_datasets(opt, rows, per_domain)
    return (
        SerializableDataStream(primary_data, seed=seed + 101, label="primary"),
        SerializableDataStream(secondary_data, seed=seed + 202, label="secondary"),
    )


def _initialize_operator_costate(model, *, target_probe: str, source_label: str, source: dict) -> str:
    step = int(source["step"])
    target_steps = int(source.get("target_steps", 30_000))
    if target_probe != "plain" and source_label == target_probe:
        model.load_extra_training_state(copy.deepcopy(source["model"].get("method", {})))
        return "matched_historical_costate"
    model.load_extra_training_state({
        "search_global_step": step,
        "search_total_steps": target_steps,
    })
    return "reinitialized_from_source_state"


def _configure_operator(model, *, target_probe: str, operator_mode: str, step: int) -> None:
    if operator_mode not in {"registered", "forced_active_diagnostic"}:
        raise ValueError(f"unknown operator mode: {operator_mode}")
    model.set_search_step(step, max(int(getattr(model, "_search_total_steps", 0)), step + 1))
    if operator_mode == "registered" or target_probe == "plain":
        return
    if target_probe == "dt":
        model.opt.dtcov_lambda_schedule = "fixed"
        model.opt.dtcov_lambda = 0.001
        model.opt.dtcov_warmup_iters = 0
        model.dtcov.config.lambda_value = 0.001
        model.dtcov.config.warmup_iters = 0
    elif target_probe == "hj":
        model.opt.hj_enable = True
        model.hj_epoch = max(int(getattr(model.opt, "hj_start_epoch", 5)), 5)
    elif target_probe == "hnek":
        from models.hnek.hnek_search import set_hnek_search_active
        set_hnek_search_active(model, True)
    else:
        raise ValueError(f"forced-active mode is not registered for {target_probe}")


def _method_diagnostics(model, probe: str) -> dict[str, float]:
    result: dict[str, float] = {}
    if hasattr(model, "time_idx"):
        result["bridge_time_index"] = float(model.time_idx.reshape(-1)[0].item())
    if probe == "dt":
        result["dt_loss_u_match"] = float(getattr(model, "loss_U_match", 0.0))
        result["dt_lambda"] = float(model.dtcov.config.lambda_value)
        result["dt_chart_cells"] = float(len(model.dtcov.stats.store))
        result["dt_teacher_present"] = float(model.dtcov.teacher is not None)
    elif probe == "hj":
        for name in ("_hj_gate_sum", "_hj_risk_sum", "_hj_probe_sum", "_hj_risk_positive_sum"):
            result[name.removeprefix("_")] = float(getattr(model, name, 0.0))
        result["hj_active"] = float(bool(model._hj_active()))
    elif probe == "hnek":
        cfg = model._hnek_search_cfg
        result.update({
            "hnek_gamma": float(cfg.gamma),
            "hnek_physical_horizon": float(cfg.horizon_mode == "physical"),
            "hnek_residual_coordinate": float(cfg.coord == "residual"),
            "hnek_active": float(bool(getattr(model, "hnek_active", False))),
        })
    return result


def _native_state_diagnostics(model) -> dict[str, float]:
    result: dict[str, float] = {}

    def rms(value: torch.Tensor) -> float:
        return float(value.detach().float().square().mean().sqrt().item())

    with torch.no_grad():
        if all(hasattr(model, name) for name in ("fake_B", "real_A_noisy")):
            result["endpoint_residual_l2"] = rms(model.fake_B - model.real_A_noisy)
            horizon = 1.0
            if hasattr(model, "times") and hasattr(model, "time_idx"):
                index = model.time_idx.reshape(-1)[0]
                horizon = max(1.0 - float(model.times[index].detach().item()), 1e-8)
            result["physical_horizon"] = horizon
            result["rollout_velocity_l2"] = result["endpoint_residual_l2"] / horizon
        if all(hasattr(model, name) for name in ("real_A_noisy", "real_A")):
            result["rollout_input_displacement_l2"] = rms(model.real_A_noisy - model.real_A)
        if all(hasattr(model, name) for name in ("fake_B", "fake_B2")):
            result["independent_endpoint_separation_l2"] = rms(model.fake_B - model.fake_B2)

    gradient_sq = moment_sq = gradient_moment = 0.0
    optimizer = getattr(model, "optimizer_G", None)
    if optimizer is not None:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                gradient = parameter.grad
                moment = optimizer.state.get(parameter, {}).get("exp_avg")
                if gradient is None or moment is None:
                    continue
                grad = gradient.detach().double()
                avg = moment.detach().double()
                gradient_sq += float(torch.sum(grad * grad).item())
                moment_sq += float(torch.sum(avg * avg).item())
                gradient_moment += float(torch.sum(grad * avg).item())
    result["generator_grad_norm"] = gradient_sq ** 0.5
    result["adam_first_moment_norm"] = moment_sq ** 0.5
    denominator = (gradient_sq * moment_sq) ** 0.5
    result["adam_moment_gradient_cosine"] = gradient_moment / denominator if denominator else 0.0

    losses = model.get_current_losses()
    generator = max(abs(float(losses.get("G", 0.0))), 1e-12)
    discriminator = abs(float(losses.get("D_real", 0.0))) + abs(float(losses.get("D_fake", 0.0)))
    result["d_to_g_loss_ratio"] = discriminator / generator
    if hasattr(model, "loss_E"):
        result["e_to_g_loss_ratio"] = abs(float(model.loss_E.detach().item())) / generator
        result["bridge_kdd_critic_loss"] = float(model.loss_E.detach().item())
    return result


def _install_first_step_component_capture(model) -> dict[str, dict[str, torch.Tensor]]:
    """Capture component gradients without changing the committed update."""
    captured: dict[str, dict[str, torch.Tensor]] = {}
    original = model.compute_G_loss

    def wrapped(this):
        loss = original()
        if captured:
            return loss
        parameters = dict(inner(this.netG).named_parameters())
        names = list(parameters)
        values = [parameters[name] for name in names]
        components: dict[str, Any] = {
            "GAN": getattr(this, "loss_G_GAN", None),
            "SB": (
                getattr(this, "loss_SB", None) * float(this.opt.lambda_SB)
                if torch.is_tensor(getattr(this, "loss_SB", None)) else None
            ),
            "NCE": None,
            "TOTAL_NATIVE_REFERENCE": loss,
        }
        nce = getattr(this, "loss_NCE", None)
        if torch.is_tensor(nce):
            nce_y = getattr(this, "loss_NCE_Y", None)
            combined = (nce + nce_y) * 0.5 if torch.is_tensor(nce_y) else nce
            components["NCE"] = combined * float(this.opt.lambda_NCE)
        if hasattr(this, "loss_U_match") and torch.is_tensor(this.loss_U_match):
            components["METHOD_DT"] = this.loss_U_match * float(this.dtcov.config.lambda_value)
        for component, value in components.items():
            if not torch.is_tensor(value) or not value.requires_grad:
                continue
            gradients = torch.autograd.grad(
                value, values, retain_graph=True, allow_unused=True,
            )
            captured[component] = {
                name: gradient.detach().cpu().clone()
                for name, gradient in zip(names, gradients)
                if gradient is not None
            }
        return loss

    model.compute_G_loss = types.MethodType(wrapped, model)
    return captured


def _branch_snapshot(model, primary, secondary, *, step: int) -> str:
    payload = {
        "step": int(step),
        "model": {
            "networks": {
                name: cpu_clone(inner(getattr(model, "net" + name)).state_dict())
                for name in model.model_names
            },
            "optimizers": [cpu_clone(item.state_dict()) for item in model.optimizers],
            "schedulers": [copy.deepcopy(item.state_dict()) for item in model.schedulers],
            "method": cpu_clone(model.get_extra_training_state()),
        },
        "samplers": {"primary": primary.state_dict(), "secondary": secondary.state_dict()},
        "rng": capture_rng(),
    }
    return full_state_hash(payload)


def _run_branch(
    *,
    source: dict,
    target_probe: str,
    source_label: str,
    operator_mode: str,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    horizon: int,
    data_root: Path | None,
    evaluate_after: bool,
    capture_components: bool,
) -> tuple[BranchResult, str]:
    if int(horizon) <= 0:
        raise ValueError("branch horizon must be positive")
    protocol = load_protocol()
    per_domain = int(protocol["local_view"]["train_per_domain"])
    spec = probe_spec(target_probe, protocol)
    seed_everything(seed)
    with contextlib.redirect_stdout(io.StringIO()):
        opt = build_options(
            spec, dataroot=train_view, option_root=work_dir / "option_records",
            seed=seed, gpu=gpu, diagnostic_root=None,
        )
    primary, secondary = _streams(opt, rows, per_domain, seed)
    with contextlib.redirect_stdout(io.StringIO()):
        model = build_model(opt, primary.next(), secondary.next())
    load_model_state(model, source["model"], load_method=False)
    costate_policy = _initialize_operator_costate(
        model, target_probe=target_probe, source_label=source_label, source=source,
    )
    primary.load_state_dict(copy.deepcopy(source["samplers"]["primary"]))
    secondary.load_state_dict(copy.deepcopy(source["samplers"]["secondary"]))
    restore_rng(copy.deepcopy(source["rng"]))
    step = int(source["step"])
    target_steps = int(source.get("target_steps", 30_000))
    before = _cpu_state_dict(model.netG)
    captured = _install_first_step_component_capture(model) if capture_components else {}
    losses_sum: dict[str, float] = defaultdict(float)
    diagnostic_sum: dict[str, float] = defaultdict(float)
    domains_seen: dict[str, int] = defaultdict(int)
    times_seen: dict[int, int] = defaultdict(int)
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    previous_method = _method_diagnostics(model, target_probe)
    for offset in range(int(horizon)):
        current_step = step + offset
        physical_epoch = 1 + current_step // steps_per_epoch(protocol)
        model.set_train_epoch(physical_epoch)
        model.set_search_step(current_step, target_steps)
        _configure_operator(
            model, target_probe=target_probe,
            operator_mode=operator_mode, step=current_step,
        )
        batch_a, batch_b = primary.next(), secondary.next()
        paths = list(batch_a.get("A_paths", []))
        domain = "unknown"
        if paths:
            stem = Path(paths[0]).stem
            domain = stem.split("__", 1)[0] if "__" in stem else "unknown"
        domains_seen[domain] += 1
        model.set_input(batch_a, batch_b)
        model.optimize_parameters()
        for key, value in model.get_current_losses().items():
            losses_sum[key] += float(value)
        diagnostics = {**_method_diagnostics(model, target_probe), **_native_state_diagnostics(model)}
        if "bridge_time_index" in diagnostics:
            time_index = int(diagnostics["bridge_time_index"])
            times_seen[time_index] += 1
        for key, value in diagnostics.items():
            number = float(value)
            if key.startswith("hj_") and key.endswith("_sum"):
                number -= float(previous_method.get(key, 0.0))
            diagnostic_sum[key] += number
        for diagnostic_name in (
            "generator_grad_norm", "endpoint_residual_l2", "rollout_velocity_l2",
            "adam_moment_gradient_cosine", "independent_endpoint_separation_l2",
        ):
            if diagnostic_name not in diagnostics:
                continue
            grouped[("domain", domain, diagnostic_name)].append(float(diagnostics[diagnostic_name]))
            if "bridge_time_index" in diagnostics:
                grouped[("time", str(int(diagnostics["bridge_time_index"])), diagnostic_name)].append(float(diagnostics[diagnostic_name]))
        if (current_step + 1) % steps_per_epoch(protocol) == 0:
            model.update_learning_rate()
        previous_method = _method_diagnostics(model, target_probe)

    after = _cpu_state_dict(model.netG)
    averaged_diagnostics = {key: value / float(horizon) for key, value in diagnostic_sum.items()}
    for domain, count in sorted(domains_seen.items()):
        averaged_diagnostics[f"domain_count::{domain}"] = float(count)
    for index, count in sorted(times_seen.items()):
        averaged_diagnostics[f"time_count::{index}"] = float(count)
    for (kind, group_name, diagnostic_name), values in sorted(grouped.items()):
        averaged_diagnostics[f"{kind}_moment::{group_name}::{diagnostic_name}::mean"] = float(np.mean(values))
        averaged_diagnostics[f"{kind}_moment::{group_name}::{diagnostic_name}::variance"] = float(np.var(values))
    observation = StateObservation(
        step=step,
        physical_epoch=float(step) / float(steps_per_epoch(protocol)),
        gradient={
            "losses": {key: value / float(horizon) for key, value in losses_sum.items()},
            "diagnostics": {
                key: value for key, value in averaged_diagnostics.items()
                if key.startswith(("generator_", "adam_"))
            },
        },
        bridge={
            key: value for key, value in averaged_diagnostics.items()
            if key.startswith(("bridge_", "endpoint_", "rollout_", "independent_", "physical_"))
        },
        game_balance={
            key: value for key, value in averaged_diagnostics.items()
            if key.startswith(("d_to_g", "e_to_g"))
        },
        sampling={
            key: value for key, value in averaged_diagnostics.items()
            if key.startswith(("domain_", "time_"))
        },
        method_internal={
            "target_probe": target_probe,
            "source_state": source_label,
            "operator_mode": operator_mode,
            "costate_policy": costate_policy,
            **{
                key: value for key, value in averaged_diagnostics.items()
                if key.startswith(("dt_", "hj_", "hnek_"))
            },
        },
    )
    observation.validate_target_blind()
    metrics = None
    if evaluate_after:
        if data_root is None:
            raise ValueError("data_root is required for post-branch discovery labeling")
        metrics = evaluate_model(
            model, rows=rows, data_root=data_root,
            protocol_hash=str(source["metadata"]["protocol_fingerprint"]),
            include_lpips=False,
        )
    scientific_after = _branch_snapshot(
        model, primary, secondary, step=step + int(horizon),
    )
    result = BranchResult(
        observation=observation,
        before_g=before,
        after_g=after,
        component_gradients=captured,
        metrics=metrics,
        scientific_state_after=scientific_after,
    )
    del model, primary, secondary
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, costate_policy


def _post_branch_label(proposal: dict | None, reference: dict | None, *, step: int) -> dict | None:
    if proposal is None or reference is None:
        return None
    domains = {}
    for domain in sorted(reference["domains"]):
        domains[domain] = float(proposal["domains"][domain]["psnr"] - reference["domains"][domain]["psnr"])
    return {
        "future_step": int(step),
        "macro_psnr_delta": float(proposal["macro_psnr"] - reference["macro_psnr"]),
        "macro_ssim_delta": float(proposal["macro_ssim"] - reference["macro_ssim"]),
        "positive_domains": int(sum(value > 0.0 for value in domains.values())),
        "worst_domain_delta": float(min(domains.values())),
        "domain_psnr_delta": domains,
        "computed_after_both_branches": True,
        "available_to_controller": False,
        "split": "discovery70",
        "confirmation20_opened": False,
    }


def _operator_modes(probe: str, data_epoch: int) -> tuple[str, ...]:
    if probe == "dt" and (int(data_epoch) <= 20 or int(data_epoch) > 45):
        return ("registered", "forced_active_diagnostic")
    return ("registered",)


def audit_cell(
    cell: AuditCell,
    *,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    data_root: Path | None = None,
    label_horizons: Iterable[int] = (200,),
    training_root: Path | None = None,
    skip_row_ids: set[str] | None = None,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    """Run all source-state/operator/horizon rows for one matched cell."""
    if cell.probe not in {"dt", "hj", "hnek"}:
        raise ValueError("causal probes are dt, hj and hnek")
    pair_identity = validate_checkpoint_pair(cell)
    identity = audit_identity(training_root)
    plain_parent = _load_checkpoint(cell.plain_state, expected_probe="plain")
    method_parent = _load_checkpoint(cell.method_state, expected_probe=cell.probe)
    parent_hashes = {
        "plain": full_state_hash(plain_parent),
        cell.probe: full_state_hash(method_parent),
    }
    label_set = {int(value) for value in label_horizons}
    skip_row_ids = set(skip_row_ids or ())
    results: list[dict] = []
    for source_label, parent in (("plain", plain_parent), (cell.probe, method_parent)):
        for operator_mode in _operator_modes(cell.probe, cell.data_epoch):
            for horizon in sorted({int(value) for value in horizons}):
                row_key = {
                    "probe": cell.probe,
                    "data_epoch": int(cell.data_epoch),
                    "source_state": source_label,
                    "operator_mode": operator_mode,
                    "horizon": int(horizon),
                }
                row_id = object_sha256(row_key)
                if row_id in skip_row_ids:
                    continue
                evaluate_after = horizon in label_set
                reference, _ = _run_branch(
                    source=copy.deepcopy(parent), target_probe="plain",
                    source_label=source_label, operator_mode="registered",
                    rows=rows, train_view=train_view, work_dir=work_dir,
                    seed=seed, gpu=gpu, horizon=horizon, data_root=data_root,
                    evaluate_after=evaluate_after, capture_components=True,
                )
                proposal, costate_policy = _run_branch(
                    source=copy.deepcopy(parent), target_probe=cell.probe,
                    source_label=source_label, operator_mode=operator_mode,
                    rows=rows, train_view=train_view, work_dir=work_dir,
                    seed=seed, gpu=gpu, horizon=horizon, data_root=data_root,
                    evaluate_after=evaluate_after, capture_components=False,
                )
                if tuple(reference.before_g) != tuple(proposal.before_g):
                    raise RuntimeError("reference/proposal generator state identities differ")
                for key in reference.before_g:
                    if not torch.equal(reference.before_g[key], proposal.before_g[key]):
                        raise RuntimeError(f"reference/proposal start mismatch: {key}")
                geometry, blocks = state_dict_update_geometry(
                    reference.before_g, reference.after_g, proposal.after_g,
                )
                next_consensus = None
                if horizon == 1:
                    native_two, _ = _run_branch(
                        source=copy.deepcopy(parent), target_probe="plain",
                        source_label=source_label, operator_mode="registered",
                        rows=rows, train_view=train_view, work_dir=work_dir,
                        seed=seed, gpu=gpu, horizon=2, data_root=None,
                        evaluate_after=False, capture_components=False,
                    )
                    next_consensus = state_dict_delta_cosine(
                        reference.after_g, proposal.after_g,
                        reference.after_g, native_two.after_g,
                    )
                directional = component_directional_derivatives(
                    before=reference.before_g,
                    reference_after=reference.after_g,
                    proposal_after=proposal.after_g,
                    native_component_gradients=reference.component_gradients,
                )
                parent_after = full_state_hash(parent)
                if parent_after != parent_hashes[source_label]:
                    raise RuntimeError("virtual branch polluted parent full state")
                row = {
                    "schema": ATLAS_SCHEMA,
                    "row_id": row_id,
                    **row_key,
                    "step": int(cell.step),
                    "future_step": int(cell.step + horizon),
                    "time_unit": "data_epoch",
                    "operator_costate": costate_policy,
                    "reference_operator": "native_unsb",
                    "proposal_operator": cell.probe,
                    "reference_observation": asdict(reference.observation),
                    "proposal_observation": asdict(proposal.observation),
                    "update_geometry": geometry,
                    "block_geometry": blocks,
                    "next_independent_native_consensus": next_consensus,
                    "native_component_directional_derivatives": directional,
                    "post_branch_development_label": _post_branch_label(
                        proposal.metrics, reference.metrics,
                        step=cell.step + horizon,
                    ),
                    "parent_state_sha256_before": parent_hashes[source_label],
                    "parent_state_sha256_after": parent_after,
                    "reference_branch_state_sha256": reference.scientific_state_after,
                    "proposal_branch_state_sha256": proposal.scientific_state_after,
                    "checkpoint_identity": pair_identity,
                    "audit_identity": identity,
                    "paired_metrics_accessed_by_controller": False,
                    "paired_development_evaluated_after_branch": bool(evaluate_after),
                    "confirmation20_opened": False,
                }
                results.append(row)
                if on_row is not None:
                    on_row(row)
    return results


def append_unique_rows(path: Path, rows: Iterable[dict]) -> dict:
    path = Path(path)
    existing: dict[str, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                existing[row["row_id"]] = row
    added = 0
    for row in rows:
        if row["row_id"] in existing:
            if canonical_json(existing[row["row_id"]]) != canonical_json(row):
                raise RuntimeError(f"non-identical duplicate atlas row: {row['row_id']}")
            continue
        existing[row["row_id"]] = row
        added += 1
    ordered = sorted(
        existing.values(),
        key=lambda row: (
            row["probe"], row["data_epoch"], row["source_state"],
            row["operator_mode"], row["horizon"],
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return {"path": str(path.resolve()), "rows": len(ordered), "added": added}


def run_audit_job(
    *,
    output_root: Path,
    probe: str,
    epoch: int,
    train_view: Path,
    data_root: Path,
    manifest_path: Path,
    gpu: int,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    label_horizons: Iterable[int] = (200,),
    training_root: Path | None = None,
) -> dict:
    cell = AuditCell(
        probe=probe,
        data_epoch=int(epoch),
        plain_state=output_root / "anchors" / "plain" / "milestones" / f"e{int(epoch):03d}.pt",
        method_state=output_root / "anchors" / probe / "milestones" / f"e{int(epoch):03d}.pt",
    )
    if not cell.plain_state.is_file() or not cell.method_state.is_file():
        raise FileNotFoundError(f"matched milestone pair is incomplete for {probe} e{epoch}")
    rows = read_manifest(manifest_path)
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    existing_rows = _read_jsonl(atlas_path)
    existing_ids = {row["row_id"] for row in existing_rows}

    def persist_row(row: dict) -> None:
        append_unique_rows(atlas_path, [row])

    produced = audit_cell(
        cell, rows=rows, train_view=train_view,
        work_dir=output_root / "audit" / "work", seed=int(load_protocol()["seed"]),
        gpu=gpu, horizons=horizons, data_root=data_root,
        label_horizons=label_horizons, training_root=training_root,
        skip_row_ids=existing_ids, on_row=persist_row,
    )
    append_result = {
        "path": str(atlas_path.resolve()),
        "rows": len(_read_jsonl(atlas_path)),
        "added": len(produced),
    }
    matrix = build_causal_matrix(output_root)
    return {
        "schema": "final-unsb-local-route1-audit-job-result-v1",
        "cell": {
            "probe": cell.probe,
            "data_epoch": cell.data_epoch,
            "plain_state": str(cell.plain_state.resolve()),
            "method_state": str(cell.method_state.resolve()),
        },
        "horizons": sorted({int(value) for value in horizons}),
        "produced_rows": len(produced),
        "atlas": append_result,
        "matrix_status": matrix["status"],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expected_row_keys(queue: dict) -> set[tuple[str, int, str, str, int]]:
    expected: set[tuple[str, int, str, str, int]] = set()
    for job in queue.get("jobs", []):
        probe = str(job["probe"])
        epoch = int(job["data_epoch"])
        for source in ("plain", probe):
            for mode in _operator_modes(probe, epoch):
                for horizon in DEFAULT_HORIZONS:
                    expected.add((probe, epoch, source, mode, horizon))
    return expected


def _classify_probe(rows: list[dict], probe: str) -> dict:
    probe_rows = [row for row in rows if row["probe"] == probe]
    preferred = [
        row for row in probe_rows
        if row["operator_mode"] == ("forced_active_diagnostic" if probe == "dt" else "registered")
    ]
    horizon200 = [row for row in preferred if row["horizon"] == 200 and row.get("post_branch_development_label")]
    by_epoch: dict[int, dict[str, float]] = defaultdict(dict)
    for row in horizon200:
        by_epoch[int(row["data_epoch"])][row["source_state"]] = float(
            row["post_branch_development_label"]["macro_psnr_delta"]
        )
    state_matrix = [
        {
            "data_epoch": epoch,
            "ui_on_plain_minus_u0": values.get("plain"),
            "ui_on_method_minus_u0": values.get(probe),
        }
        for epoch, values in sorted(by_epoch.items())
    ]
    one_step = [row for row in preferred if row["horizon"] == 1]
    consensus = [
        float(row["next_independent_native_consensus"]["cosine"])
        for row in one_step if row.get("next_independent_native_consensus")
    ]
    correction_ratios = [
        float(row["update_geometry"]["correction_norm"])
        / max(float(row["update_geometry"]["reference_norm"]), 1e-20)
        for row in one_step
    ]
    cases = defaultdict(int)
    for values in by_epoch.values():
        if "plain" not in values or probe not in values:
            continue
        on_plain, on_self = values["plain"], values[probe]
        if on_plain > 0.0 and on_self > 0.0:
            cases["sustainable_on_both_states"] += 1
        elif on_plain > 0.0 and on_self <= 0.0:
            cases["plain_state_help_self_state_harm"] += 1
        elif on_plain <= 0.0 and on_self <= 0.0:
            cases["harmful_on_both_states"] += 1
        else:
            cases["self_state_only_help"] += 1
    return {
        "probe": probe,
        "state_operator_matrix": state_matrix,
        "case_counts": dict(cases),
        "next_batch_consensus_mean": None if not consensus else float(np.mean(consensus)),
        "next_batch_consensus_min": None if not consensus else float(np.min(consensus)),
        "correction_to_native_norm_ratio_mean": None if not correction_ratios else float(np.mean(correction_ratios)),
        "rows": len(probe_rows),
    }


def _rank_failure_mechanisms(probe_summaries: list[dict]) -> list[dict]:
    mechanisms: list[dict] = []
    sign_support = [
        row["probe"] for row in probe_summaries
        if row["next_batch_consensus_mean"] is not None
        and row["next_batch_consensus_mean"] < 0.0
    ]
    if sign_support:
        mechanisms.append({
            "failure_type": "correction_sign_reversal",
            "supporting_probes": sign_support,
            "cross_probe_support": len(sign_support),
            "observable": "correction cosine with next independent native update",
            "construction_route": "future_batch_consensus_or_one_sided_constraint",
        })
    state_support = [
        row["probe"] for row in probe_summaries
        if row["case_counts"].get("plain_state_help_self_state_harm", 0) > 0
    ]
    if state_support:
        mechanisms.append({
            "failure_type": "state_feedback_missing",
            "supporting_probes": state_support,
            "cross_probe_support": len(state_support),
            "observable": "same operator helps S_plain but harms its own induced state",
            "construction_route": "state_conditional_self_null_intervention",
        })
    amplitude_support = [
        row["probe"] for row in probe_summaries
        if row["correction_to_native_norm_ratio_mean"] is not None
        and row["correction_to_native_norm_ratio_mean"] > 1.0
        and (row["next_batch_consensus_mean"] or 0.0) >= 0.0
    ]
    if amplitude_support:
        mechanisms.append({
            "failure_type": "correct_direction_unstable_magnitude",
            "supporting_probes": amplitude_support,
            "cross_probe_support": len(amplitude_support),
            "observable": "correction/native Adam displacement ratio",
            "construction_route": "adam_metric_trust_region",
        })
    return sorted(
        mechanisms,
        key=lambda row: (-int(row["cross_probe_support"]), row["failure_type"]),
    )


def build_causal_matrix(output_root: Path) -> dict:
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    queue_path = output_root / "audit" / "AUDIT_QUEUE.json"
    rows = _read_jsonl(atlas_path)
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.is_file() else {"jobs": []}
    actual = {
        (
            row["probe"], int(row["data_epoch"]), row["source_state"],
            row["operator_mode"], int(row["horizon"]),
        )
        for row in rows
    }
    expected = _expected_row_keys(queue)
    missing = sorted(expected - actual)
    probes = sorted({row["probe"] for row in rows})
    summaries = [_classify_probe(rows, probe) for probe in probes]
    complete = bool(expected) and not missing
    matrix = {
        "schema": MATRIX_SCHEMA,
        "status": "COMPLETE_CAUSAL_AUDIT" if complete else "PARTIAL_CAUSAL_AUDIT",
        "rows": len(rows),
        "expected_rows": len(expected),
        "missing_rows": [
            {
                "probe": item[0], "data_epoch": item[1], "source_state": item[2],
                "operator_mode": item[3], "horizon": item[4],
            }
            for item in missing
        ],
        "probe_summaries": summaries,
        "ranked_failure_mechanisms": _rank_failure_mechanisms(summaries) if complete else [],
        "paired_labels_joined_only_after_branches": True,
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "audit" / "LONG_CAUSAL_MATRIX.json", matrix)
    return matrix
