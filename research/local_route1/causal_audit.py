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
import math
import os
import random
import threading
import time
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
DEFAULT_VARIANCE_REPLICATES = 8
TERMINAL_LOCAL_HORIZONS = (1, 8, 32)


_ATLAS_THREAD_LOCK = threading.RLock()


@contextlib.contextmanager
def _exclusive_path_lock(path: Path, *, timeout_seconds: float = 300.0):
    """Serialize cross-process read/merge/replace operations for one artifact."""
    lock_path = Path(str(Path(path)) + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with _ATLAS_THREAD_LOCK, lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + float(timeout_seconds)
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out acquiring artifact lock: {lock_path}")
                    time.sleep(0.05)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


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


def _disable_operator(model, target_probe: str) -> None:
    """Return a method model to the exact registered zero-intervention path."""
    if target_probe == "plain":
        return
    if target_probe == "dt":
        model.opt.dtcov_lambda = 0.0
        model.dtcov.config.lambda_value = 0.0
    elif target_probe == "hj":
        model.opt.hj_enable = False
    elif target_probe == "hnek":
        from models.hnek.hnek_search import set_hnek_search_active
        set_hnek_search_active(model, False)
    else:
        raise ValueError(f"zero-intervention path is not registered for {target_probe}")


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


def _restore_terminal_base_lrs(model) -> tuple[float, ...]:
    """Restore only step scale for an e200 local vector-field diagnostic.

    The registered e200 checkpoint has already taken its final scheduler step,
    leaving every optimizer LR at zero.  Keeping zero would make every probe
    appear to self-null.  Adam moments and scheduler state remain untouched;
    only the immutable base LR is restored inside the disposable branch.
    """
    if len(model.optimizers) != len(model.schedulers):
        raise RuntimeError("terminal LR audit requires optimizer/scheduler identity")
    restored: list[float] = []
    for optimizer, scheduler in zip(model.optimizers, model.schedulers):
        base_lrs = tuple(float(value) for value in scheduler.base_lrs)
        if len(optimizer.param_groups) != len(base_lrs):
            raise RuntimeError("terminal LR audit param-group identity mismatch")
        if not base_lrs or any(value <= 0.0 for value in base_lrs):
            raise RuntimeError("terminal LR audit requires positive frozen base LRs")
        for group, value in zip(optimizer.param_groups, base_lrs):
            group["lr"] = value
            restored.append(value)
    return tuple(restored)


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
    intervention_steps: int | None = None,
    batch_skip: int = 0,
    rng_seed_override: int | None = None,
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
    for _ in range(int(batch_skip)):
        primary.next()
        secondary.next()
    if rng_seed_override is not None:
        random.seed(int(rng_seed_override))
        np.random.seed(int(rng_seed_override))
        torch.manual_seed(int(rng_seed_override))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(rng_seed_override))
    step = int(source["step"])
    target_steps = int(source.get("target_steps", 30_000))
    terminal_extension = step >= target_steps
    restored_base_lrs: tuple[float, ...] = ()
    if terminal_extension:
        if int(horizon) not in TERMINAL_LOCAL_HORIZONS:
            raise RuntimeError(
                "terminal vector-field audit is local-only and may not cross a scheduler boundary"
            )
        restored_base_lrs = _restore_terminal_base_lrs(model)
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
        intervention_active = (
            intervention_steps is None or offset < int(intervention_steps)
        )
        if intervention_active:
            _configure_operator(
                model, target_probe=target_probe,
                operator_mode=operator_mode, step=current_step,
            )
        else:
            _disable_operator(model, target_probe)
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
            "intervention_steps": (
                "continuous" if intervention_steps is None else int(intervention_steps)
            ),
            "branch_semantics": (
                "terminal_base_lr_vector_field"
                if terminal_extension else "registered_training_continuation"
            ),
            "terminal_restored_base_lrs": list(restored_base_lrs),
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


def _audit_regimes(
    horizons: Iterable[int], *, start_step: int | None = None,
    target_steps: int = 30_000,
) -> tuple[tuple[str, int, int | None], ...]:
    """Registered continuous branches plus diagnostic pulse propagation.

    Pulse branches are causal diagnostics only.  They do not authorize a
    finite-window candidate, handoff, or exit policy.
    """
    values = sorted({int(value) for value in horizons})
    if start_step is not None and int(start_step) >= int(target_steps):
        values = [value for value in values if value in TERMINAL_LOCAL_HORIZONS]
    regimes: list[tuple[str, int, int | None]] = [
        ("continuous_intervention", horizon, None) for horizon in values
    ]
    for horizon in values:
        if horizon in (8, 32):
            regimes.append(("one_step_pulse_then_native", horizon, 1))
        if horizon == 200:
            regimes.append(("eight_step_pulse_then_native", horizon, 8))
    return tuple(regimes)


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
            for branch_regime, horizon, intervention_steps in _audit_regimes(
                horizons, start_step=cell.step,
                target_steps=int(parent.get("target_steps", 30_000)),
            ):
                row_key = {
                    "probe": cell.probe,
                    "data_epoch": int(cell.data_epoch),
                    "source_state": source_label,
                    "operator_mode": operator_mode,
                    "branch_regime": branch_regime,
                    "intervention_steps": intervention_steps,
                    "horizon": int(horizon),
                }
                row_id = object_sha256(row_key)
                if row_id in skip_row_ids:
                    continue
                terminal_extension = cell.step >= int(parent.get("target_steps", 30_000))
                evaluate_after = horizon in label_set and not terminal_extension
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
                    intervention_steps=intervention_steps,
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
                if horizon == 1 and branch_regime == "continuous_intervention":
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
                    "diagnostic_scope": (
                        "terminal base-LR local vector field; no post-training future label"
                        if terminal_extension else
                        "continuous operator validity"
                        if intervention_steps is None else
                        "pulse propagation under later native UNSB; not a route-2 policy"
                    ),
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


def _dict_displacement(
    start: Mapping[str, torch.Tensor], end: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if tuple(start) != tuple(end):
        raise RuntimeError("sampling audit state identity mismatch")
    return {
        key: end[key].detach().cpu() - start[key].detach().cpu()
        for key in start if torch.is_floating_point(start[key])
    }


def _dict_difference(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if tuple(left) != tuple(right):
        raise RuntimeError("sampling audit displacement identity mismatch")
    return {key: left[key] - right[key] for key in left}


def _dict_norm_sq(value: Mapping[str, torch.Tensor]) -> float:
    return float(sum(torch.sum(item.detach().double().square()).item() for item in value.values()))


def _dict_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if tuple(left) != tuple(right):
        raise RuntimeError("sampling audit vector identity mismatch")
    return float(sum(
        torch.sum(left[key].detach().double() * right[key].detach().double()).item()
        for key in left
    ))


def _dict_cosine(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    denominator = (_dict_norm_sq(left) * _dict_norm_sq(right)) ** 0.5
    if denominator == 0.0:
        return 0.0
    value = _dict_dot(left, right) / denominator
    return 0.0 if not math.isfinite(value) else max(-1.0, min(1.0, value))


def _sampling_group(observation: StateObservation, prefix: str) -> str:
    matches = [
        key.split("::", 1)[1]
        for key, value in observation.sampling.items()
        if key.startswith(prefix + "_count::") and float(value) > 0.0
    ]
    return matches[0] if len(matches) == 1 else "mixed"


def _group_scalar_records(records: list[dict], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[field])].append(float(record["correction_norm"]))
    return {
        key: {
            "n": float(len(values)),
            "correction_norm_mean": float(np.mean(values)),
            "correction_norm_variance": float(np.var(values)),
        }
        for key, values in sorted(grouped.items())
    }


def _sampling_variance_row(
    *,
    cell: AuditCell,
    parent: dict,
    source_label: str,
    operator_mode: str,
    axis: str,
    replicates: int,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    pair_identity: dict,
    identity: dict,
) -> dict:
    if axis not in {"independent_unpaired_batch", "latent_time_bridge_rng"}:
        raise ValueError(f"unknown sampling-variance axis: {axis}")
    if int(replicates) < 2:
        raise ValueError("sampling-variance audit requires at least two replicates")
    parent_hash = full_state_hash(parent)
    sum_correction: dict[str, torch.Tensor] | None = None
    sum_native: dict[str, torch.Tensor] | None = None
    correction_norm_sq_sum = 0.0
    native_norm_sq_sum = 0.0
    block_correction_norm_sq_sum: dict[str, float] = defaultdict(float)
    same_batch_cosines: list[float] = []
    next_batch_cosines: list[float] = []
    replicate_records: list[dict] = []
    exact_zero = 0
    pending_correction: dict[str, torch.Tensor] | None = None
    costate_policy = "unknown"
    total_references = int(replicates) + (1 if axis == "independent_unpaired_batch" else 0)
    seed_base = int(parent_hash[:12], 16) % 2_000_000_000
    for replicate in range(total_references):
        batch_skip = replicate if axis == "independent_unpaired_batch" else 0
        rng_override = (
            None if axis == "independent_unpaired_batch"
            else (seed_base + 104_729 * (replicate + 1)) % 2_147_483_647
        )
        reference, _ = _run_branch(
            source=copy.deepcopy(parent), target_probe="plain",
            source_label=source_label, operator_mode="registered",
            rows=rows, train_view=train_view, work_dir=work_dir,
            seed=seed, gpu=gpu, horizon=1, data_root=None,
            evaluate_after=False, capture_components=False,
            batch_skip=batch_skip, rng_seed_override=rng_override,
        )
        native = _dict_displacement(reference.before_g, reference.after_g)
        if pending_correction is not None:
            next_batch_cosines.append(_dict_cosine(pending_correction, native))
            pending_correction = None
        if replicate >= int(replicates):
            del reference, native
            continue
        proposal, costate_policy = _run_branch(
            source=copy.deepcopy(parent), target_probe=cell.probe,
            source_label=source_label, operator_mode=operator_mode,
            rows=rows, train_view=train_view, work_dir=work_dir,
            seed=seed, gpu=gpu, horizon=1, data_root=None,
            evaluate_after=False, capture_components=False,
            batch_skip=batch_skip, rng_seed_override=rng_override,
        )
        correction = _dict_difference(
            _dict_displacement(proposal.before_g, proposal.after_g), native,
        )
        correction_norm_sq = _dict_norm_sq(correction)
        native_norm_sq = _dict_norm_sq(native)
        correction_norm_sq_sum += correction_norm_sq
        native_norm_sq_sum += native_norm_sq
        for key, value in correction.items():
            block_correction_norm_sq_sum[key.split(".", 1)[0]] += float(
                torch.sum(value.detach().double().square()).item()
            )
        same_cosine = _dict_cosine(correction, native)
        same_batch_cosines.append(same_cosine)
        exact_zero += int(correction_norm_sq == 0.0)
        if sum_correction is None:
            sum_correction = {key: value.clone() for key, value in correction.items()}
            sum_native = {key: value.clone() for key, value in native.items()}
        else:
            for key in correction:
                sum_correction[key].add_(correction[key])
                assert sum_native is not None
                sum_native[key].add_(native[key])
        replicate_records.append({
            "replicate": replicate,
            "batch_skip": batch_skip,
            "rng_seed_override": rng_override,
            "domain": _sampling_group(reference.observation, "domain"),
            "bridge_time": _sampling_group(reference.observation, "time"),
            "correction_norm": correction_norm_sq ** 0.5,
            "native_norm": native_norm_sq ** 0.5,
            "same_batch_native_cosine": same_cosine,
        })
        if axis == "independent_unpaired_batch":
            pending_correction = {key: value.clone() for key, value in correction.items()}
        del reference, proposal, native, correction
        gc.collect()
    assert sum_correction is not None and sum_native is not None
    count = float(replicates)
    mean_correction = {key: value / count for key, value in sum_correction.items()}
    mean_native = {key: value / count for key, value in sum_native.items()}
    mean_correction_norm_sq = _dict_norm_sq(mean_correction)
    expected_correction_norm_sq = correction_norm_sq_sum / count
    variance_energy = max(0.0, expected_correction_norm_sq - mean_correction_norm_sq)
    block_keys: dict[str, list[str]] = defaultdict(list)
    for key in mean_correction:
        block_keys[key.split(".", 1)[0]].append(key)
    block_variance = {}
    for block, keys in sorted(block_keys.items()):
        mean_energy = float(sum(torch.sum(mean_correction[key].double().square()).item() for key in keys))
        expected_energy = block_correction_norm_sq_sum[block] / count
        variance = max(0.0, expected_energy - mean_energy)
        block_variance[block] = {
            "mean_correction_energy": mean_energy,
            "expected_correction_energy": expected_energy,
            "variance_energy": variance,
            "variance_fraction": variance / max(expected_energy, 1e-30),
        }
    row_key = {
        "probe": cell.probe,
        "data_epoch": int(cell.data_epoch),
        "source_state": source_label,
        "operator_mode": operator_mode,
        "axis": axis,
        "replicates": int(replicates),
    }
    return {
        "schema": "final-unsb-local-route1-sampling-variance-row-v1",
        "row_id": object_sha256(row_key),
        **row_key,
        "step": int(cell.step),
        "mean_correction_norm": mean_correction_norm_sq ** 0.5,
        "expected_correction_norm_sq": expected_correction_norm_sq,
        "correction_variance_energy": variance_energy,
        "correction_variance_fraction": (
            variance_energy / max(expected_correction_norm_sq, 1e-30)
        ),
        "mean_native_norm": _dict_norm_sq(mean_native) ** 0.5,
        "mean_correction_native_cosine": _dict_cosine(mean_correction, mean_native),
        "same_batch_native_cosine_mean": float(np.mean(same_batch_cosines)),
        "same_batch_native_cosine_min": float(np.min(same_batch_cosines)),
        "next_independent_batch_native_cosine_mean": (
            None if not next_batch_cosines else float(np.mean(next_batch_cosines))
        ),
        "next_independent_batch_native_cosine_min": (
            None if not next_batch_cosines else float(np.min(next_batch_cosines))
        ),
        "exact_zero_corrections": exact_zero,
        "domain_summary": _group_scalar_records(replicate_records, "domain"),
        "bridge_time_summary": _group_scalar_records(replicate_records, "bridge_time"),
        "replicate_records": replicate_records,
        "block_stable_mean_energy": block_variance,
        "operator_costate": costate_policy,
        "parent_state_sha256_before": parent_hash,
        "parent_state_sha256_after": full_state_hash(parent),
        "checkpoint_identity": pair_identity,
        "audit_identity": identity,
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    }


def audit_sampling_variance(
    cell: AuditCell,
    *,
    rows: list[dict],
    train_view: Path,
    work_dir: Path,
    seed: int,
    gpu: int,
    replicates: int = DEFAULT_VARIANCE_REPLICATES,
    training_root: Path | None = None,
    skip_row_ids: set[str] | None = None,
    on_row: Callable[[dict], None] | None = None,
) -> list[dict]:
    pair_identity = validate_checkpoint_pair(cell)
    identity = audit_identity(training_root)
    parents = {
        "plain": _load_checkpoint(cell.plain_state, expected_probe="plain"),
        cell.probe: _load_checkpoint(cell.method_state, expected_probe=cell.probe),
    }
    skipped = set(skip_row_ids or ())
    result = []
    for source_label, parent in parents.items():
        for operator_mode in _operator_modes(cell.probe, cell.data_epoch):
            for axis in ("independent_unpaired_batch", "latent_time_bridge_rng"):
                key = {
                    "probe": cell.probe, "data_epoch": int(cell.data_epoch),
                    "source_state": source_label, "operator_mode": operator_mode,
                    "axis": axis, "replicates": int(replicates),
                }
                row_id = object_sha256(key)
                if row_id in skipped:
                    continue
                row = _sampling_variance_row(
                    cell=cell, parent=parent, source_label=source_label,
                    operator_mode=operator_mode, axis=axis,
                    replicates=replicates, rows=rows, train_view=train_view,
                    work_dir=work_dir, seed=seed, gpu=gpu,
                    pair_identity=pair_identity, identity=identity,
                )
                if row["parent_state_sha256_before"] != row["parent_state_sha256_after"]:
                    raise RuntimeError("sampling audit polluted parent full state")
                result.append(row)
                if on_row is not None:
                    on_row(row)
    return result


def _append_unique_rows_unlocked(path: Path, rows: Iterable[dict]) -> dict:
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
            row["operator_mode"], row.get("branch_regime", row.get("axis", "")),
            row.get("horizon", 0),
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)
    return {"path": str(path.resolve()), "rows": len(ordered), "added": added}


def append_unique_rows(path: Path, rows: Iterable[dict]) -> dict:
    """Atomically merge rows when independent audit cells finish together."""
    path = Path(path)
    with _exclusive_path_lock(path):
        return _append_unique_rows_unlocked(path, rows)


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
    variance_replicates: int = DEFAULT_VARIANCE_REPLICATES,
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

    cell_work_dir = output_root / "audit" / "work" / f"{probe}_e{int(epoch):03d}"
    produced = audit_cell(
        cell, rows=rows, train_view=train_view,
        work_dir=cell_work_dir, seed=int(load_protocol()["seed"]),
        gpu=gpu, horizons=horizons, data_root=data_root,
        label_horizons=label_horizons, training_root=training_root,
        skip_row_ids=existing_ids, on_row=persist_row,
    )
    variance_path = output_root / "audit" / "SAMPLING_VARIANCE_ATLAS.jsonl"
    existing_variance = _read_jsonl(variance_path)

    def persist_variance(row: dict) -> None:
        append_unique_rows(variance_path, [row])

    produced_variance = audit_sampling_variance(
        cell, rows=rows, train_view=train_view,
        work_dir=cell_work_dir, seed=int(load_protocol()["seed"]),
        gpu=gpu, replicates=variance_replicates, training_root=training_root,
        skip_row_ids={row["row_id"] for row in existing_variance},
        on_row=persist_variance,
    )
    append_result = {
        "path": str(atlas_path.resolve()),
        "rows": len(_read_jsonl(atlas_path)),
        "added": len(produced),
    }
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    with _exclusive_path_lock(matrix_path):
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
        "produced_sampling_variance_rows": len(produced_variance),
        "atlas": append_result,
        "sampling_variance_atlas": {
            "path": str(variance_path.resolve()),
            "rows": len(_read_jsonl(variance_path)),
            "added": len(produced_variance),
            "replicates": int(variance_replicates),
        },
        "matrix_status": matrix["status"],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _expected_row_keys(queue: dict) -> set[tuple[str, int, str, str, str, int | None, int]]:
    expected: set[tuple[str, int, str, str, str, int | None, int]] = set()
    for job in queue.get("jobs", []):
        probe = str(job["probe"])
        epoch = int(job["data_epoch"])
        for source in ("plain", probe):
            for mode in _operator_modes(probe, epoch):
                for regime, horizon, intervention_steps in _audit_regimes(
                    DEFAULT_HORIZONS, start_step=epoch * 150,
                ):
                    expected.add((probe, epoch, source, mode, regime, intervention_steps, horizon))
    return expected


def _expected_variance_keys(
    queue: dict, replicates: int = DEFAULT_VARIANCE_REPLICATES,
) -> set[tuple[str, int, str, str, str, int]]:
    expected = set()
    for job in queue.get("jobs", []):
        probe = str(job["probe"])
        epoch = int(job["data_epoch"])
        for source in ("plain", probe):
            for mode in _operator_modes(probe, epoch):
                for axis in ("independent_unpaired_batch", "latent_time_bridge_rng"):
                    expected.add((probe, epoch, source, mode, axis, int(replicates)))
    return expected


def _preferred_operator_mode(probe: str, data_epoch: int) -> str:
    """Select the scientific DT operator without dropping its active-age evidence.

    DT's registered correction is scientifically informative inside its active
    source-state interval.  At the pre-support e20 state and after the finite
    support has ended, the registered path is zero or warmup-limited, so causal
    mechanism analysis uses the separately recorded forced-active diagnostic.
    This is an operator-validity diagnostic, not a candidate exit policy.
    """
    if probe != "dt":
        return "registered"
    return (
        "registered"
        if 21 <= int(data_epoch) <= 45
        else "forced_active_diagnostic"
    )


def _classify_probe(rows: list[dict], probe: str) -> dict:
    probe_rows = [row for row in rows if row["probe"] == probe]
    preferred = [
        row for row in probe_rows
        if row["operator_mode"] == _preferred_operator_mode(
            probe, int(row["data_epoch"])
        )
    ]
    horizon200 = [
        row for row in preferred
        if row["branch_regime"] == "continuous_intervention"
        and row["horizon"] == 200 and row.get("post_branch_development_label")
    ]
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
    one_step = [
        row for row in preferred
        if row["branch_regime"] == "continuous_intervention" and row["horizon"] == 1
        and float(row["update_geometry"]["correction_norm"]) > 1e-20
    ]
    consensus = [
        float(row["next_independent_native_consensus"]["cosine"])
        for row in one_step if row.get("next_independent_native_consensus")
    ]
    consensus_records = [
        {
            "data_epoch": int(row["data_epoch"]),
            "source_state": row["source_state"],
            "cosine": float(row["next_independent_native_consensus"]["cosine"]),
        }
        for row in one_step if row.get("next_independent_native_consensus")
    ]
    consensus_sign_changes = []
    for source_state in sorted({row["source_state"] for row in consensus_records}):
        state_rows = sorted(
            [row for row in consensus_records if row["source_state"] == source_state],
            key=lambda row: row["data_epoch"],
        )
        for previous, current in zip(state_rows, state_rows[1:]):
            if (previous["cosine"] > 0.0) != (current["cosine"] > 0.0):
                consensus_sign_changes.append({
                    "source_state": source_state,
                    "previous_data_epoch": previous["data_epoch"],
                    "data_epoch": current["data_epoch"],
                    "previous_cosine": previous["cosine"],
                    "cosine": current["cosine"],
                })
    correction_ratios = [
        float(row["update_geometry"]["correction_norm"])
        / max(float(row["update_geometry"]["reference_norm"]), 1e-20)
        for row in one_step
    ]
    correct_direction_overscale_rows = sum(
        (
            float(row["update_geometry"]["correction_norm"])
            / max(float(row["update_geometry"]["reference_norm"]), 1e-20)
        ) > 1.0
        and row.get("next_independent_native_consensus") is not None
        and float(row["next_independent_native_consensus"]["cosine"]) >= 0.0
        for row in one_step
    )
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
    propagation = [
        {
            "data_epoch": int(row["data_epoch"]),
            "source_state": row["source_state"],
            "eight_step_pulse_then_192_native_macro_psnr_delta": float(
                row["post_branch_development_label"]["macro_psnr_delta"]
            ),
            "final_parameter_gap_norm": float(row["update_geometry"]["correction_norm"]),
        }
        for row in preferred
        if row["branch_regime"] == "eight_step_pulse_then_native"
        and row["horizon"] == 200 and row.get("post_branch_development_label")
    ]
    return {
        "probe": probe,
        "state_operator_matrix": state_matrix,
        "case_counts": dict(cases),
        "next_batch_consensus_mean": None if not consensus else float(np.mean(consensus)),
        "next_batch_consensus_min": None if not consensus else float(np.min(consensus)),
        "next_batch_consensus_positive_rows": sum(value > 0.0 for value in consensus),
        "next_batch_consensus_negative_rows": sum(value < 0.0 for value in consensus),
        "next_batch_consensus_zero_rows": sum(value == 0.0 for value in consensus),
        "next_batch_consensus_sign_changes": consensus_sign_changes,
        "correction_to_native_norm_ratio_mean": None if not correction_ratios else float(np.mean(correction_ratios)),
        "correct_direction_overscale_rows": int(correct_direction_overscale_rows),
        "pulse_propagation": propagation,
        "rows": len(probe_rows),
    }


def _variance_summary(rows: list[dict], probe: str) -> dict:
    selected = [
        row for row in rows
        if row["probe"] == probe
        and row["operator_mode"] == _preferred_operator_mode(
            probe, int(row["data_epoch"])
        )
    ]
    axes = {}
    for axis in ("independent_unpaired_batch", "latent_time_bridge_rng"):
        recorded_rows = [row for row in selected if row["axis"] == axis]
        axis_rows = [
            row for row in recorded_rows
            if (
                float(row.get("expected_correction_norm_sq", 0.0)) > 1e-30
                or float(row.get("mean_correction_norm", 0.0)) > 1e-15
            )
        ]
        fractions = [float(row["correction_variance_fraction"]) for row in axis_rows]
        axes[axis] = {
            "rows": len(axis_rows),
            "recorded_rows": len(recorded_rows),
            "inactive_zero_correction_rows": len(recorded_rows) - len(axis_rows),
            "mean_variance_fraction": None if not fractions else float(np.mean(fractions)),
            "variance_dominated_rows": sum(value >= 0.75 for value in fractions),
            "mean_correction_norm": (
                None if not axis_rows else float(np.mean([row["mean_correction_norm"] for row in axis_rows]))
            ),
        }
    return {"probe": probe, "axes": axes, "rows": len(selected)}


def _average_ranks(values: list[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = np.zeros(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _spearman(first: list[float], second: list[float]) -> float:
    if len(first) < 2 or len(first) != len(second):
        return 0.0
    left, right = _average_ranks(first), _average_ranks(second)
    if float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _balanced_accuracy(labels: list[bool], predictions: list[bool]) -> float:
    scores = []
    for value in (False, True):
        indices = [index for index, label in enumerate(labels) if label is value]
        if indices:
            scores.append(float(np.mean([predictions[index] is value for index in indices])))
    return 0.0 if not scores else float(np.mean(scores))


def _signal_records(rows: list[dict], variance_rows: list[dict]) -> dict[str, list[dict]]:
    labels = {}
    one_step = {}
    for row in rows:
        preferred = _preferred_operator_mode(
            str(row["probe"]), int(row["data_epoch"])
        )
        if row["operator_mode"] != preferred or row["branch_regime"] != "continuous_intervention":
            continue
        key = (row["probe"], int(row["data_epoch"]), row["source_state"], row["operator_mode"])
        if int(row["horizon"]) == 200 and row.get("post_branch_development_label"):
            labels[key] = row["post_branch_development_label"]
        if int(row["horizon"]) == 1:
            one_step[key] = row
    variance_index = {
        (
            row["probe"], int(row["data_epoch"]), row["source_state"],
            row["operator_mode"], row["axis"],
        ): row
        for row in variance_rows
    }
    result: dict[str, list[dict]] = defaultdict(list)
    temporal_rollout: list[dict] = []
    for key, row in one_step.items():
        label = labels.get(key)
        if label is None:
            continue
        correction_norm = float(row["update_geometry"]["correction_norm"])
        if correction_norm <= 1e-20:
            continue
        features: dict[str, float] = {
            "correction_native_cosine": float(row["update_geometry"]["correction_reference_cosine"]),
            "correction_within_native_scale_margin": 1.0 - (
                correction_norm
                / max(float(row["update_geometry"]["reference_norm"]), 1e-20)
            ),
        }
        block_cosines = [
            float(values["correction_reference_cosine"])
            for values in row.get("block_geometry", {}).values()
            if values.get("correction_reference_cosine") is not None
            and float(values.get("correction_norm", 0.0)) > 1e-20
        ]
        if block_cosines:
            features["minimum_block_correction_native_cosine"] = float(
                np.min(block_cosines)
            )
        consensus = row.get("next_independent_native_consensus")
        if consensus is not None:
            features["correction_next_native_cosine"] = float(consensus["cosine"])
        reference_bridge = row.get("reference_observation", {}).get("bridge", {})
        proposal_bridge = row.get("proposal_observation", {}).get("bridge", {})
        reference_velocity = reference_bridge.get("rollout_velocity_l2")
        proposal_velocity = proposal_bridge.get("rollout_velocity_l2")
        if reference_velocity is not None and proposal_velocity is not None:
            features["rollout_speed_stability_margin"] = 1.0 - (
                float(proposal_velocity) / max(float(reference_velocity), 1e-20)
            )
        if reference_velocity is not None:
            temporal_rollout.append({
                "probe": key[0], "data_epoch": key[1], "source_state": key[2],
                "operator_mode": key[3], "native_rollout_velocity": float(reference_velocity),
                "label": label,
            })
        for field, feature in (
            ("independent_endpoint_separation_l2", "endpoint_dispersion_stability_margin"),
            ("bridge_kdd_critic_loss", "bridge_kdd_magnitude_stability_margin"),
        ):
            reference_value = reference_bridge.get(field)
            proposal_value = proposal_bridge.get(field)
            if (
                reference_value is not None and proposal_value is not None
                and abs(float(reference_value)) > 1e-20
            ):
                features[feature] = 1.0 - (
                    abs(float(proposal_value)) / abs(float(reference_value))
                )
        reference_gradient = row.get("reference_observation", {}).get("gradient", {}).get(
            "diagnostics", {}
        )
        proposal_gradient = row.get("proposal_observation", {}).get("gradient", {}).get(
            "diagnostics", {}
        )
        reference_grad_norm = reference_gradient.get("generator_grad_norm")
        proposal_grad_norm = proposal_gradient.get("generator_grad_norm")
        if (
            reference_grad_norm is not None and proposal_grad_norm is not None
            and float(reference_grad_norm) > 1e-20
        ):
            features["generator_gradient_scale_margin"] = 1.0 - (
                float(proposal_grad_norm) / float(reference_grad_norm)
            )
        proposal_adam_alignment = proposal_gradient.get("adam_moment_gradient_cosine")
        if proposal_adam_alignment is not None:
            features["adam_moment_gradient_alignment"] = float(proposal_adam_alignment)
        reference_balance = row.get("reference_observation", {}).get("game_balance", {})
        proposal_balance = row.get("proposal_observation", {}).get("game_balance", {})
        for field, feature in (
            ("d_to_g_loss_ratio", "d_to_g_balance_stability_margin"),
            ("e_to_g_loss_ratio", "e_to_g_balance_stability_margin"),
        ):
            reference_value = reference_balance.get(field)
            proposal_value = proposal_balance.get(field)
            if (
                reference_value is not None and proposal_value is not None
                and float(reference_value) > 1e-20
            ):
                features[feature] = 1.0 - (
                    float(proposal_value) / float(reference_value)
                )
        for component in ("GAN", "SB", "NCE", "TOTAL_NATIVE_REFERENCE"):
            value = row.get("native_component_directional_derivatives", {}).get(component)
            if value is not None:
                # A negative native-loss directional derivative is the safe
                # mathematical direction, hence the sign inversion.
                features[f"native_{component.lower()}_descent_score"] = -float(
                    value["gradient_correction_cosine"]
                )
        batch = variance_index.get((*key, "independent_unpaired_batch"))
        latent = variance_index.get((*key, "latent_time_bridge_rng"))
        batch_has_correction = bool(
            batch is not None
            and (
                float(batch.get("expected_correction_norm_sq", 0.0)) > 1e-30
                or float(batch.get("mean_correction_norm", 0.0)) > 1e-15
            )
        )
        if batch_has_correction:
            assert batch is not None
            next_mean = batch.get("next_independent_batch_native_cosine_mean")
            if next_mean is not None:
                features["replicated_next_batch_consensus"] = float(next_mean)
            features["low_batch_variance_margin"] = 0.75 - float(
                batch["correction_variance_fraction"]
            )
            block_variances = [
                float(values["variance_fraction"])
                for values in batch.get("block_stable_mean_energy", {}).values()
            ]
            if block_variances:
                features["low_max_block_variance_margin"] = 0.75 - float(
                    np.max(block_variances)
                )
            domain_cosines: dict[str, list[float]] = defaultdict(list)
            for replicate in batch.get("replicate_records", []):
                if float(replicate.get("correction_norm", 0.0)) > 1e-20:
                    domain_cosines[str(replicate["domain"])].append(
                        float(replicate["same_batch_native_cosine"])
                    )
            if len(domain_cosines) >= 2:
                features["minimum_domain_correction_native_cosine"] = float(
                    min(np.mean(values) for values in domain_cosines.values())
                )
        latent_has_correction = bool(
            latent is not None
            and (
                float(latent.get("expected_correction_norm_sq", 0.0)) > 1e-30
                or float(latent.get("mean_correction_norm", 0.0)) > 1e-15
            )
        )
        if latent_has_correction:
            assert latent is not None
            features["low_latent_time_variance_margin"] = 0.75 - float(
                latent["correction_variance_fraction"]
            )
            time_means = [
                float(values["correction_norm_mean"])
                for values in latent.get("bridge_time_summary", {}).values()
                if float(values.get("n", 0.0)) > 0.0
            ]
            if len(time_means) >= 2 and float(np.mean(time_means)) > 0.0:
                coefficient = float(np.std(time_means) / np.mean(time_means))
                features["low_time_conditioning_spread_margin"] = 1.0 - coefficient
            time_cosines: dict[str, list[float]] = defaultdict(list)
            for replicate in latent.get("replicate_records", []):
                if float(replicate.get("correction_norm", 0.0)) > 1e-20:
                    time_cosines[str(replicate["bridge_time"])].append(
                        float(replicate["same_batch_native_cosine"])
                    )
            if len(time_cosines) >= 2:
                features["minimum_time_correction_native_cosine"] = float(
                    min(np.mean(values) for values in time_cosines.values())
                )
        for feature, score in features.items():
            result[feature].append({
                "probe": key[0], "data_epoch": key[1], "source_state": key[2],
                "score": score,
                "future_macro_psnr_delta": float(label["macro_psnr_delta"]),
                "future_positive": bool(float(label["macro_psnr_delta"]) > 0.0),
                "domain_psnr_delta": dict(label["domain_psnr_delta"]),
                "paired_label_available_to_controller": False,
            })
    temporal_groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in temporal_rollout:
        temporal_groups[
            (record["probe"], record["source_state"])
        ].append(record)
    for records in temporal_groups.values():
        records.sort(key=lambda item: item["data_epoch"])
        for previous, current in zip(records, records[1:]):
            previous_velocity = float(previous["native_rollout_velocity"])
            current_velocity = float(current["native_rollout_velocity"])
            score = (previous_velocity - current_velocity) / max(previous_velocity, 1e-20)
            label = current["label"]
            result["rollout_velocity_growth_margin"].append({
                "probe": current["probe"],
                "data_epoch": current["data_epoch"],
                "source_state": current["source_state"],
                "score": score,
                "future_macro_psnr_delta": float(label["macro_psnr_delta"]),
                "future_positive": bool(float(label["macro_psnr_delta"]) > 0.0),
                "domain_psnr_delta": dict(label["domain_psnr_delta"]),
                "paired_label_available_to_controller": False,
            })
    return result


def _precursor_lead_fraction(records: list[dict]) -> tuple[float, list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["probe"], record["source_state"])].append(record)
    outcomes = []
    for key, values in grouped.items():
        values = sorted(values, key=lambda item: item["data_epoch"])
        seen_positive = False
        reversal_epoch = None
        for value in values:
            if value["future_positive"]:
                seen_positive = True
            elif seen_positive:
                reversal_epoch = int(value["data_epoch"])
                break
        if reversal_epoch is None:
            continue
        negative_signal_epochs = [
            int(value["data_epoch"]) for value in values
            if float(value["score"]) <= 0.0 and int(value["data_epoch"]) <= reversal_epoch
        ]
        outcomes.append({
            "probe": key[0], "source_state": key[1],
            "reversal_epoch": reversal_epoch,
            "first_nonpositive_signal_epoch": (
                None if not negative_signal_epochs else min(negative_signal_epochs)
            ),
            "precedes_or_matches": bool(negative_signal_epochs),
        })
    fraction = 0.0 if not outcomes else float(np.mean([row["precedes_or_matches"] for row in outcomes]))
    return fraction, outcomes


def _signal_performance(records: list[dict]) -> dict:
    labels = [bool(row["future_positive"]) for row in records]
    predictions = [float(row["score"]) > 0.0 for row in records]
    domain_counts = [
        sum(
            (delta > 0.0) == prediction
            for delta in record["domain_psnr_delta"].values()
        )
        for record, prediction in zip(records, predictions)
    ]
    lead_fraction, lead_rows = _precursor_lead_fraction(records)
    correlation = _spearman(
        [float(row["score"]) for row in records],
        [float(row["future_macro_psnr_delta"]) for row in records],
    )
    return {
        "records": len(records),
        "future_sign_accuracy": (
            0.0 if not records else float(np.mean([
                prediction == label
                for prediction, label in zip(predictions, labels)
            ]))
        ),
        "balanced_future_sign_accuracy": _balanced_accuracy(labels, predictions),
        "future_200_step_delta_spearman": correlation,
        "mean_domain_sign_agreement_of_six": (
            0.0 if not domain_counts else float(np.mean(domain_counts))
        ),
        "rows_with_at_least_four_domains_agreeing_fraction": (
            0.0 if not domain_counts else float(np.mean([
                value >= 4 for value in domain_counts
            ]))
        ),
        "reversal_precursor_lead_fraction": lead_fraction,
        "reversal_precursor_cases": lead_rows,
    }


def target_blind_signal_screen(rows: list[dict], variance_rows: list[dict]) -> dict:
    """Join post-branch labels offline and screen interpretable observables."""
    feature_records = _signal_records(rows, variance_rows)
    signals = []
    for feature, records in sorted(feature_records.items()):
        global_metrics = _signal_performance(records)
        per_method = {}
        for probe in sorted({row["probe"] for row in records}):
            selected = [row for row in records if row["probe"] == probe]
            per_method[probe] = _signal_performance(selected)
        heldout = {
            probe: values["future_sign_accuracy"]
            for probe, values in per_method.items()
        }
        lomo_mean = 0.0 if not heldout else float(np.mean(list(heldout.values())))
        lomo_minimum = 0.0 if not heldout else float(np.min(list(heldout.values())))
        shared_passes = (
            len(records) >= 6
            and len(heldout) >= 2
            and lomo_minimum >= 0.65
            and global_metrics["future_200_step_delta_spearman"] >= 0.30
            and global_metrics["mean_domain_sign_agreement_of_six"] >= 4.0
            and (
                not global_metrics["reversal_precursor_cases"]
                or global_metrics["reversal_precursor_lead_fraction"] >= 0.5
            )
        )
        method_specific_passes = sorted([
            probe for probe, values in per_method.items()
            if values["records"] >= 4
            and values["future_sign_accuracy"] >= 0.65
            and values["future_200_step_delta_spearman"] >= 0.30
            and values["mean_domain_sign_agreement_of_six"] >= 4.0
            and (
                not values["reversal_precursor_cases"]
                or values["reversal_precursor_lead_fraction"] >= 0.5
            )
        ])
        signals.append({
            "feature": feature,
            "records": len(records),
            "probes": sorted(heldout),
            "leave_one_method_out_future_sign_accuracy": lomo_minimum,
            "mean_per_method_future_sign_accuracy": lomo_mean,
            "heldout_accuracy_by_method": heldout,
            "per_method_performance": per_method,
            **{
                key: value for key, value in global_metrics.items()
                if key != "records"
            },
            "shared_driver_eligible": shared_passes,
            "method_specific_driver_eligible_for": method_specific_passes,
            "driver_eligible": bool(shared_passes or method_specific_passes),
            "threshold": "mathematical zero; no paired threshold fitting",
            "paired_label_available_to_controller": False,
        })
    shared_eligible = [
        row["feature"] for row in signals if row["shared_driver_eligible"]
    ]
    method_specific: dict[str, list[str]] = defaultdict(list)
    for row in signals:
        for probe in row["method_specific_driver_eligible_for"]:
            method_specific[probe].append(row["feature"])
    method_specific = {
        probe: sorted(features) for probe, features in sorted(method_specific.items())
    }
    eligible = sorted(set(shared_eligible).union(*(
        set(features) for features in method_specific.values()
    ))) if method_specific else sorted(set(shared_eligible))
    return {
        "schema": "final-unsb-local-route1-target-blind-signal-screen-v1",
        "status": (
            "ELIGIBLE_SIGNALS_FOUND"
            if eligible else "NO_ELIGIBLE_TARGET_BLIND_SIGNAL"
        ),
        "criteria": {
            "minimum_records": 6,
            "minimum_methods": 2,
            "minimum_method_specific_records": 4,
            "leave_one_method_out_future_sign_accuracy": 0.65,
            "method_specific_future_sign_accuracy": 0.65,
            "future_200_step_delta_spearman": 0.30,
            "mean_domain_sign_agreement_of_six": 4.0,
            "minimum_reversal_lead_fraction_when_observed": 0.5,
        },
        "signals": signals,
        "eligible_shared_driver_signals": shared_eligible,
        "eligible_method_specific_driver_signals": method_specific,
        "eligible_driver_signals": eligible,
        "paired_metrics_used_only_as_offline_post_branch_labels": True,
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    }


_MINIMUM_ROUTE_COMPLEXITY = {
    "correction_sign_reversal": {
        "operator_components": 1, "extra_gradient_or_forward_passes": 1,
        "persistent_model_copies": 0,
    },
    "correct_direction_unstable_magnitude": {
        "operator_components": 1, "extra_gradient_or_forward_passes": 0,
        "persistent_model_copies": 0,
    },
    "sampling_variance": {
        "operator_components": 1, "extra_gradient_or_forward_passes": 1,
        "persistent_model_copies": 0,
    },
    "coordinate_horizon_imbalance": {
        "operator_components": 1, "extra_gradient_or_forward_passes": 0,
        "persistent_model_copies": 0,
    },
    "rollout_distribution_speed": {
        "operator_components": 2, "extra_gradient_or_forward_passes": 1,
        "persistent_model_copies": 1,
    },
    "state_feedback_missing": {
        "operator_components": 2, "extra_gradient_or_forward_passes": 0,
        "persistent_model_copies": 0,
    },
    "state_independent_late_bias": {
        "operator_components": 2, "extra_gradient_or_forward_passes": 0,
        "persistent_model_copies": 0,
    },
    "endpoint_dispersion_instability": {
        "operator_components": 1, "extra_gradient_or_forward_passes": 1,
        "persistent_model_copies": 0,
    },
    "game_balance_instability": {
        "operator_components": 2, "extra_gradient_or_forward_passes": 0,
        "persistent_model_copies": 0,
    },
}


def _rank_mechanisms_by_discovery_evidence(
    mechanisms: list[dict], signal_screen: dict, rows: list[dict],
) -> list[dict]:
    """Attach the preregistered discovery evidence and rank lexicographically.

    Paired branch deltas are read only after both branches are frozen.  They rank
    hypotheses offline and are never exposed to a training-time observable.
    Complexity values are lower bounds for the construction family, not a
    prewritten candidate formula; the derivation card must replace them with the
    exact cost of the derived operator.
    """
    signal_index = {
        signal["feature"]: signal
        for signal in signal_screen.get("signals", [])
    }
    ranked = []
    for raw in mechanisms:
        mechanism = dict(raw)
        signal_metrics = []
        for feature in mechanism.get("eligible_target_blind_driver_signals", []):
            signal = signal_index.get(feature)
            if signal is not None:
                signal_metrics.append({
                    "scope": "shared",
                    "feature": feature,
                    "probe": None,
                    "reversal_precursor_lead_fraction": float(
                        signal.get("reversal_precursor_lead_fraction", 0.0)
                    ),
                    "reversal_precursor_cases": len(
                        signal.get("reversal_precursor_cases", [])
                    ),
                    "mean_domain_sign_agreement_of_six": float(
                        signal.get("mean_domain_sign_agreement_of_six", 0.0)
                    ),
                })
        for probe, features in mechanism.get(
            "eligible_method_specific_driver_signals_by_probe", {}
        ).items():
            for feature in features:
                signal = signal_index.get(feature)
                performance = (
                    signal.get("per_method_performance", {}).get(probe)
                    if signal is not None else None
                )
                if performance is not None:
                    signal_metrics.append({
                        "scope": "method_specific",
                        "feature": feature,
                        "probe": probe,
                        "reversal_precursor_lead_fraction": float(
                            performance.get("reversal_precursor_lead_fraction", 0.0)
                        ),
                        "reversal_precursor_cases": len(
                            performance.get("reversal_precursor_cases", [])
                        ),
                        "mean_domain_sign_agreement_of_six": float(
                            performance.get("mean_domain_sign_agreement_of_six", 0.0)
                        ),
                    })
        observed_leads = [
            item["reversal_precursor_lead_fraction"] for item in signal_metrics
            if item["reversal_precursor_cases"] > 0
        ]
        precursor_score = (
            0.0 if not observed_leads else float(np.mean(observed_leads))
        )
        domain_score = (
            0.0 if not signal_metrics else float(np.mean([
                item["mean_domain_sign_agreement_of_six"]
                for item in signal_metrics
            ]))
        )
        supporting_probes = set(mechanism.get("supporting_probes", []))
        short_labels = [
            float(row["post_branch_development_label"]["macro_psnr_delta"])
            for row in rows
            if row.get("probe") in supporting_probes
            and row.get("operator_mode") == _preferred_operator_mode(
                str(row["probe"]), int(row["data_epoch"])
            )
            and row.get("branch_regime") == "continuous_intervention"
            and int(row.get("horizon", 0)) == 200
            and float(row.get("update_geometry", {}).get("correction_norm", 0.0)) > 1e-20
            and row.get("post_branch_development_label")
        ]
        short_positive_fraction = (
            0.0 if not short_labels else float(np.mean([
                value > 0.0 for value in short_labels
            ]))
        )
        short_mean_delta = (
            0.0 if not short_labels else float(np.mean(short_labels))
        )
        complexity = dict(_MINIMUM_ROUTE_COMPLEXITY.get(
            str(mechanism.get("failure_type")),
            {
                "operator_components": 3,
                "extra_gradient_or_forward_passes": 2,
                "persistent_model_copies": 1,
            },
        ))
        simplicity_score = 1.0 / max(float(complexity["operator_components"]), 1.0)
        cost_score = 1.0 / (
            1.0
            + float(complexity["extra_gradient_or_forward_passes"])
            + float(complexity["persistent_model_copies"])
        )
        mechanism["discovery_ranking_evidence"] = {
            "cross_probe_support": int(mechanism.get("cross_probe_support", 0)),
            "target_blind_precursor_lead_fraction": precursor_score,
            "target_blind_domain_sign_agreement_of_six": domain_score,
            "eligible_signal_evidence": signal_metrics,
            "short_counterfactual": {
                "horizons_updates": [200],
                "data_epochs_at_small25": 200.0 / 150.0,
                "records": len(short_labels),
                "positive_fraction": short_positive_fraction,
                "mean_macro_psnr_delta": short_mean_delta,
                "paired_label_available_to_controller": False,
            },
            "minimum_route_complexity_prior": complexity,
            "mathematical_simplicity_score": simplicity_score,
            "compute_and_recovery_cost_score": cost_score,
            "ranking_policy": (
                "lexicographic: cross-probe support, observed precursor lead, "
                "domain consistency, short counterfactual benefit, mathematical "
                "simplicity, minimum compute/recovery cost"
            ),
        }
        mechanism["_ranking_key"] = (
            -int(mechanism.get("cross_probe_support", 0)),
            -precursor_score,
            -domain_score,
            -short_positive_fraction,
            -short_mean_delta,
            -simplicity_score,
            -cost_score,
            str(mechanism.get("failure_type")),
        )
        ranked.append(mechanism)
    ranked.sort(key=lambda item: item["_ranking_key"])
    for rank, mechanism in enumerate(ranked, 1):
        mechanism["evidence_rank"] = rank
        del mechanism["_ranking_key"]
    return ranked


def _rank_failure_mechanisms(
    probe_summaries: list[dict], variance_summaries: list[dict],
    signal_screen: dict, rows: list[dict], variance_rows: list[dict],
) -> list[dict]:
    mechanisms: list[dict] = []
    shared_eligible = set(signal_screen.get(
        "eligible_shared_driver_signals",
        signal_screen.get("eligible_driver_signals", []),
    ))
    method_specific_eligible = {
        probe: set(features)
        for probe, features in signal_screen.get(
            "eligible_method_specific_driver_signals", {}
        ).items()
    }

    def driver_evidence(
        candidate_features: set[str], supporting_probes: list[str],
    ) -> tuple[list[str], dict[str, list[str]]]:
        shared = sorted(candidate_features & shared_eligible)
        by_probe = {
            probe: sorted(candidate_features & method_specific_eligible.get(probe, set()))
            for probe in supporting_probes
            if candidate_features & method_specific_eligible.get(probe, set())
        }
        return shared, by_probe

    sign_support = [
        row["probe"] for row in probe_summaries
        if (
            int(row.get("next_batch_consensus_negative_rows", 0)) > 0
            or (
                row.get("next_batch_consensus_negative_rows") is None
                and row.get("next_batch_consensus_mean") is not None
                and row["next_batch_consensus_mean"] < 0.0
            )
        )
    ]
    if sign_support:
        shared_drivers, method_drivers = driver_evidence(
            {
                "correction_next_native_cosine",
                "replicated_next_batch_consensus",
                "minimum_block_correction_native_cosine",
                "minimum_domain_correction_native_cosine",
                "minimum_time_correction_native_cosine",
            },
            sign_support,
        )
        mechanisms.append({
            "failure_type": "correction_sign_reversal",
            "supporting_probes": sign_support,
            "cross_probe_support": len(sign_support),
            "observable": "correction cosine with next independent native update",
            "construction_route": "future_batch_consensus_or_one_sided_constraint",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
        })
    state_support = [
        row["probe"] for row in probe_summaries
        if row["case_counts"].get("plain_state_help_self_state_harm", 0) > 0
    ]
    if state_support:
        shared_drivers, method_drivers = driver_evidence(
            set(signal_screen.get("eligible_driver_signals", [])), state_support,
        )
        mechanisms.append({
            "failure_type": "state_feedback_missing",
            "supporting_probes": state_support,
            "cross_probe_support": len(state_support),
            "observable": "same operator helps S_plain but harms its own induced state",
            "construction_route": "state_conditional_self_null_intervention",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
        })
    late_bias_support = [
        row["probe"] for row in probe_summaries
        if row["case_counts"].get("harmful_on_both_states", 0) > 0
    ]
    if late_bias_support:
        shared_drivers, method_drivers = driver_evidence(
            set(signal_screen.get("eligible_driver_signals", [])),
            late_bias_support,
        )
        mechanisms.append({
            "failure_type": "state_independent_late_bias",
            "supporting_probes": late_bias_support,
            "cross_probe_support": len(late_bias_support),
            "observable": "the registered correction harms both native and self-induced states",
            "construction_route": "current_state_rate_or_curvature_reformulation",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
            "fixed_exit_or_handoff_forbidden": True,
            "fixed_annealing_forbidden": True,
            "ineligible_reason": (
                None if shared_drivers or method_drivers else
                "late state-independent bias has no eligible target-blind reformulation signal"
            ),
        })
    amplitude_support = [
        row["probe"] for row in probe_summaries
        if (
            int(row.get("correct_direction_overscale_rows", 0)) > 0
            or (
                row.get("correct_direction_overscale_rows") is None
                and row.get("correction_to_native_norm_ratio_mean") is not None
                and row["correction_to_native_norm_ratio_mean"] > 1.0
                and (row["next_batch_consensus_mean"] or 0.0) >= 0.0
            )
        )
    ]
    if amplitude_support:
        shared_drivers, method_drivers = driver_evidence(
            {
                "correction_within_native_scale_margin",
                "generator_gradient_scale_margin",
                "adam_moment_gradient_alignment",
            }, amplitude_support,
        )
        mechanisms.append({
            "failure_type": "correct_direction_unstable_magnitude",
            "supporting_probes": amplitude_support,
            "cross_probe_support": len(amplitude_support),
            "observable": "correction/native Adam displacement ratio",
            "construction_route": "adam_metric_trust_region",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
            "ineligible_reason": (
                None if shared_drivers or method_drivers else
                "the target-blind native-scale margin did not pass the signal screen"
            ),
        })
    variance_support = []
    for row in variance_summaries:
        supported_axes = [
            axis for axis, values in row["axes"].items()
            if values["rows"] > 0 and values["variance_dominated_rows"] >= max(1, values["rows"] // 2)
        ]
        if supported_axes:
            variance_support.append({"probe": row["probe"], "axes": supported_axes})
    if variance_support:
        mechanisms.append({
            "failure_type": "sampling_variance",
            "supporting_probes": [row["probe"] for row in variance_support],
            "supporting_axes": variance_support,
            "cross_probe_support": len(variance_support),
            "observable": "actual correction-field variance across independent unpaired batches and latent/time/bridge RNG",
            "construction_route": "unbiased_stratified_or_antithetic_estimator",
            "candidate_generation_eligible": True,
            "eligibility_basis": "unbiased estimator route does not require a paired-fitted controller",
        })
    probes = [summary["probe"] for summary in probe_summaries]
    labels = {
        (row["probe"], int(row["data_epoch"]), row["source_state"], row["operator_mode"]):
        float(row["post_branch_development_label"]["macro_psnr_delta"])
        for row in rows
        if row.get("branch_regime") == "continuous_intervention"
        and int(row.get("horizon", 0)) == 200
        and row.get("post_branch_development_label")
    }
    rollout_support = []
    for probe in probes:
        cases = []
        temporal: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            if row.get("probe") != probe or row.get("operator_mode") != _preferred_operator_mode(
                probe, int(row["data_epoch"])
            ):
                continue
            if row.get("branch_regime") != "continuous_intervention" or int(row.get("horizon", 0)) != 1:
                continue
            if float(row.get("update_geometry", {}).get("correction_norm", 0.0)) <= 1e-20:
                continue
            key = (
                probe,
                int(row["data_epoch"]),
                row["source_state"],
                _preferred_operator_mode(probe, int(row["data_epoch"])),
            )
            label = labels.get(key)
            reference = row.get("reference_observation", {}).get("bridge", {}).get(
                "rollout_velocity_l2"
            )
            proposal = row.get("proposal_observation", {}).get("bridge", {}).get(
                "rollout_velocity_l2"
            )
            if label is None or reference is None or proposal is None:
                continue
            ratio = float(proposal) / max(float(reference), 1e-20)
            temporal[str(row["source_state"])].append({
                "data_epoch": int(row["data_epoch"]),
                "native_rollout_velocity": float(reference),
                "future_200_step_macro_psnr_delta": label,
            })
            if ratio > 1.0 and label <= 0.0:
                cases.append({
                    "failure_mode": "proposal_speed_excess",
                    "data_epoch": int(row["data_epoch"]),
                    "source_state": row["source_state"],
                    "proposal_reference_velocity_ratio": ratio,
                    "future_200_step_macro_psnr_delta": label,
                })
            reference_kdd = row.get("reference_observation", {}).get(
                "bridge", {}
            ).get("bridge_kdd_critic_loss")
            proposal_kdd = row.get("proposal_observation", {}).get(
                "bridge", {}
            ).get("bridge_kdd_critic_loss")
            if (
                reference_kdd is not None and proposal_kdd is not None
                and abs(float(reference_kdd)) > 1e-20
                and abs(float(proposal_kdd)) > abs(float(reference_kdd))
                and label <= 0.0
            ):
                cases.append({
                    "failure_mode": "proposal_bridge_kdd_magnitude_excess",
                    "data_epoch": int(row["data_epoch"]),
                    "source_state": row["source_state"],
                    "proposal_reference_kdd_magnitude_ratio": (
                        abs(float(proposal_kdd)) / abs(float(reference_kdd))
                    ),
                    "future_200_step_macro_psnr_delta": label,
                })
        for source_state, state_rows in temporal.items():
            state_rows.sort(key=lambda item: item["data_epoch"])
            for previous, current in zip(state_rows, state_rows[1:]):
                previous_velocity = float(previous["native_rollout_velocity"])
                current_velocity = float(current["native_rollout_velocity"])
                if (
                    current_velocity > previous_velocity
                    and float(current["future_200_step_macro_psnr_delta"]) <= 0.0
                ):
                    cases.append({
                        "failure_mode": "native_rollout_velocity_growth",
                        "source_state": source_state,
                        "previous_data_epoch": int(previous["data_epoch"]),
                        "data_epoch": int(current["data_epoch"]),
                        "velocity_growth_ratio": (
                            (current_velocity - previous_velocity)
                            / max(previous_velocity, 1e-20)
                        ),
                        "future_200_step_macro_psnr_delta": float(
                            current["future_200_step_macro_psnr_delta"]
                        ),
                    })
        if cases:
            rollout_support.append({"probe": probe, "cases": cases})
    if rollout_support:
        shared_drivers, method_drivers = driver_evidence({
            "rollout_speed_stability_margin", "rollout_velocity_growth_margin",
            "bridge_kdd_magnitude_stability_margin",
        }, [row["probe"] for row in rollout_support])
        mechanisms.append({
            "failure_type": "rollout_distribution_speed",
            "supporting_probes": [row["probe"] for row in rollout_support],
            "supporting_cases": rollout_support,
            "cross_probe_support": len(rollout_support),
            "observable": "proposal/reference rollout velocity ratio at the current unpaired state",
            "construction_route": "bridge_gap_constrained_adaptive_teacher",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
            "ineligible_reason": (
                None if shared_drivers or method_drivers else
                "rollout speed preceded harm but did not pass the shared or method-specific signal screen"
            ),
        })
    endpoint_support = []
    game_support = []
    for probe in probes:
        endpoint_cases = []
        game_cases = []
        for row in rows:
            if (
                row.get("probe") != probe
                or row.get("operator_mode") != _preferred_operator_mode(
                    probe, int(row["data_epoch"])
                )
                or row.get("branch_regime") != "continuous_intervention"
                or int(row.get("horizon", 0)) != 1
                or float(row.get("update_geometry", {}).get("correction_norm", 0.0)) <= 1e-20
            ):
                continue
            key = (
                probe, int(row["data_epoch"]), row["source_state"],
                _preferred_operator_mode(probe, int(row["data_epoch"])),
            )
            label = labels.get(key)
            if label is None or label > 0.0:
                continue
            reference_bridge = row.get("reference_observation", {}).get("bridge", {})
            proposal_bridge = row.get("proposal_observation", {}).get("bridge", {})
            reference_endpoint = reference_bridge.get("independent_endpoint_separation_l2")
            proposal_endpoint = proposal_bridge.get("independent_endpoint_separation_l2")
            if (
                reference_endpoint is not None and proposal_endpoint is not None
                and float(reference_endpoint) > 1e-20
                and float(proposal_endpoint) > float(reference_endpoint)
            ):
                endpoint_cases.append({
                    "data_epoch": int(row["data_epoch"]),
                    "source_state": row["source_state"],
                    "proposal_reference_endpoint_dispersion_ratio": (
                        float(proposal_endpoint) / float(reference_endpoint)
                    ),
                    "future_200_step_macro_psnr_delta": label,
                })
            reference_balance = row.get("reference_observation", {}).get(
                "game_balance", {}
            )
            proposal_balance = row.get("proposal_observation", {}).get(
                "game_balance", {}
            )
            growing = {}
            for field in ("d_to_g_loss_ratio", "e_to_g_loss_ratio"):
                reference_value = reference_balance.get(field)
                proposal_value = proposal_balance.get(field)
                if (
                    reference_value is not None and proposal_value is not None
                    and float(reference_value) > 1e-20
                    and float(proposal_value) > float(reference_value)
                ):
                    growing[field] = float(proposal_value) / float(reference_value)
            adam_alignment = row.get("proposal_observation", {}).get(
                "gradient", {}
            ).get("diagnostics", {}).get("adam_moment_gradient_cosine")
            if growing or (adam_alignment is not None and float(adam_alignment) < 0.0):
                game_cases.append({
                    "data_epoch": int(row["data_epoch"]),
                    "source_state": row["source_state"],
                    "growing_game_balance_ratios": growing,
                    "adam_moment_gradient_cosine": (
                        None if adam_alignment is None else float(adam_alignment)
                    ),
                    "future_200_step_macro_psnr_delta": label,
                })
        if endpoint_cases:
            endpoint_support.append({"probe": probe, "cases": endpoint_cases})
        if game_cases:
            game_support.append({"probe": probe, "cases": game_cases})
    if endpoint_support:
        shared_drivers, method_drivers = driver_evidence(
            {"endpoint_dispersion_stability_margin"},
            [row["probe"] for row in endpoint_support],
        )
        mechanisms.append({
            "failure_type": "endpoint_dispersion_instability",
            "supporting_probes": [row["probe"] for row in endpoint_support],
            "supporting_cases": endpoint_support,
            "cross_probe_support": len(endpoint_support),
            "observable": "latent endpoint separation under proposal versus native operator",
            "construction_route": "endpoint_law_preserving_variance_or_constraint",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
            "endpoint_law_change_forbidden": True,
            "ineligible_reason": (
                None if shared_drivers or method_drivers else
                "endpoint dispersion increased before harm but no safe target-blind driver passed"
            ),
        })
    if game_support:
        shared_drivers, method_drivers = driver_evidence({
            "d_to_g_balance_stability_margin",
            "e_to_g_balance_stability_margin",
            "adam_moment_gradient_alignment",
            "generator_gradient_scale_margin",
        }, [row["probe"] for row in game_support])
        mechanisms.append({
            "failure_type": "game_balance_instability",
            "supporting_probes": [row["probe"] for row in game_support],
            "supporting_cases": game_support,
            "cross_probe_support": len(game_support),
            "observable": "D/G, E/G and Adam-moment geometry under proposal versus native operator",
            "construction_route": "state_conditional_game_metric_constraint",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
            "ineligible_reason": (
                None if shared_drivers or method_drivers else
                "game imbalance preceded harm but no safe target-blind driver passed"
            ),
        })
    coordinate_support = []
    for probe in probes:
        cases = []
        for row in variance_rows:
            if row.get("probe") != probe or row.get("operator_mode") != _preferred_operator_mode(
                probe, int(row["data_epoch"])
            ):
                continue
            if row.get("axis") != "latent_time_bridge_rng":
                continue
            time_means = [
                float(values["correction_norm_mean"])
                for values in row.get("bridge_time_summary", {}).values()
                if float(values.get("n", 0.0)) > 0.0
            ]
            if len(time_means) < 2 or float(np.mean(time_means)) <= 0.0:
                continue
            coefficient = float(np.std(time_means) / np.mean(time_means))
            if coefficient > 1.0:
                cases.append({
                    "data_epoch": int(row["data_epoch"]),
                    "source_state": row["source_state"],
                    "time_conditioning_coefficient_of_variation": coefficient,
                })
        if cases:
            coordinate_support.append({"probe": probe, "cases": cases})
    if coordinate_support:
        driver = "low_time_conditioning_spread_margin"
        shared_drivers, method_drivers = driver_evidence(
            {driver}, [row["probe"] for row in coordinate_support],
        )
        mechanisms.append({
            "failure_type": "coordinate_horizon_imbalance",
            "supporting_probes": [row["probe"] for row in coordinate_support],
            "supporting_cases": coordinate_support,
            "cross_probe_support": len(coordinate_support),
            "observable": "bridge-time correction-norm coefficient of variation",
            "construction_route": "identity_adaptive_coordinate",
            "candidate_generation_eligible": bool(shared_drivers or method_drivers),
            "eligible_target_blind_driver_signals": shared_drivers,
            "eligible_method_specific_driver_signals_by_probe": method_drivers,
            "ineligible_reason": (
                None if shared_drivers or method_drivers else
                "time conditioning was imbalanced but no target-blind safe driver passed"
            ),
        })
    return _rank_mechanisms_by_discovery_evidence(
        mechanisms, signal_screen, rows,
    )


def build_causal_matrix(output_root: Path) -> dict:
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    queue_path = output_root / "audit" / "AUDIT_QUEUE.json"
    rows = _read_jsonl(atlas_path)
    variance_rows = _read_jsonl(output_root / "audit" / "SAMPLING_VARIANCE_ATLAS.jsonl")
    queue = json.loads(queue_path.read_text(encoding="utf-8")) if queue_path.is_file() else {"jobs": []}
    actual = {
        (
            row["probe"], int(row["data_epoch"]), row["source_state"],
            row["operator_mode"], row["branch_regime"],
            row.get("intervention_steps"), int(row["horizon"]),
        )
        for row in rows
    }
    expected = _expected_row_keys(queue)
    missing = sorted(expected - actual)
    actual_variance = {
        (
            row["probe"], int(row["data_epoch"]), row["source_state"],
            row["operator_mode"], row["axis"], int(row["replicates"]),
        )
        for row in variance_rows
    }
    expected_variance = _expected_variance_keys(queue)
    missing_variance = sorted(expected_variance - actual_variance)
    probes = sorted({row["probe"] for row in rows})
    summaries = [_classify_probe(rows, probe) for probe in probes]
    variance_summaries = [_variance_summary(variance_rows, probe) for probe in probes]
    complete = bool(expected) and not missing and not missing_variance
    signal_screen = target_blind_signal_screen(rows, variance_rows) if complete else {
        "schema": "final-unsb-local-route1-target-blind-signal-screen-v1",
        "status": "BLOCKED_CAUSAL_ATLAS_INCOMPLETE",
        "eligible_driver_signals": [],
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    }
    matrix = {
        "schema": MATRIX_SCHEMA,
        "status": "COMPLETE_CAUSAL_AUDIT" if complete else "PARTIAL_CAUSAL_AUDIT",
        "analysis_identity": {
            "analysis_git_commit": git_commit(),
            "analysis_source_fingerprint": object_sha256([
                (Path(__file__).name, portable_source_sha256(Path(__file__))),
            ]),
            "reversal_atlas_sha256": (
                file_sha256(atlas_path) if atlas_path.is_file() else None
            ),
            "sampling_variance_atlas_sha256": (
                file_sha256(output_root / "audit" / "SAMPLING_VARIANCE_ATLAS.jsonl")
                if (output_root / "audit" / "SAMPLING_VARIANCE_ATLAS.jsonl").is_file()
                else None
            ),
            "audit_queue_sha256": file_sha256(queue_path) if queue_path.is_file() else None,
            "branch_rows_modified_by_analysis": False,
            "paired_metrics_accessed_by_controller": False,
            "confirmation20_opened": False,
        },
        "rows": len(rows),
        "expected_rows": len(expected),
        "missing_rows": [
            {
                "probe": item[0], "data_epoch": item[1], "source_state": item[2],
                "operator_mode": item[3], "branch_regime": item[4],
                "intervention_steps": item[5], "horizon": item[6],
            }
            for item in missing
        ],
        "sampling_variance_rows": len(variance_rows),
        "expected_sampling_variance_rows": len(expected_variance),
        "missing_sampling_variance_rows": [
            {
                "probe": item[0], "data_epoch": item[1], "source_state": item[2],
                "operator_mode": item[3], "axis": item[4], "replicates": item[5],
            }
            for item in missing_variance
        ],
        "probe_summaries": summaries,
        "sampling_variance_summaries": variance_summaries,
        "target_blind_signal_screen": signal_screen,
        "ranked_failure_mechanisms": (
            _rank_failure_mechanisms(
                summaries, variance_summaries, signal_screen, rows, variance_rows,
            )
            if complete else []
        ),
        "pulse_branches_are_diagnostics_not_exit_policies": True,
        "paired_labels_joined_only_after_branches": True,
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "audit" / "LONG_CAUSAL_MATRIX.json", matrix)
    return matrix
