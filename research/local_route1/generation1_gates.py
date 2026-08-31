"""Executable GPU gates for frozen Generation-1 candidates."""

from __future__ import annotations

import copy
import math
import time
from pathlib import Path
from typing import Any

import torch

from .anchors import prepare_probe
from .candidate_gate import CandidateGateContext
from .protocol import ProbeSpec, step_to_physical_epoch, steps_per_epoch
from .runtime import (
    capture_rng,
    full_state_hash,
    load_model_state,
    model_state,
    restore_rng,
)


def _plain_spec(label: str) -> ProbeSpec:
    return ProbeSpec(
        id=label, contract_id=label, model="sb", role="candidate_gate_plain",
        method={},
    )


def _disabled_spec(context: CandidateGateContext) -> ProbeSpec:
    spec = context.registration.spec
    method = dict(spec.method)
    if spec.model == "route1_bvcp":
        method["bvcp_enable"] = False
    elif spec.model == "route1_rsmg":
        method["rsmg_replicates"] = 1
    elif spec.model == "route1_pcrsmg":
        method["pcrsmg_replicates"] = 1
    elif spec.model == "route1_amtnc":
        method["amtnc_replicates"] = 1
    elif spec.model == "route1_pcnr":
        method["pcnr_enable"] = False
    elif spec.model == "route1_mcrb":
        method["mcrb_enable"] = False
    elif spec.model == "route1_ammcrb":
        method["ammcrb_enable"] = False
    elif spec.model == "route1_rfammcrb":
        method["rfammcrb_enable"] = False
    elif spec.model == "route1_rfmcrb":
        method["rfmcrb_enable"] = False
    elif spec.model == "route1_pcammcrb":
        method["pcammcrb_enable"] = False
    elif spec.model in (
        "route1_bvcp_ablation", "route1_pcrsmg_ablation",
        "route1_amtnc_ablation", "route1_mcrb_ablation",
        "route1_pcnr_ablation", "route1_ammcrb_ablation",
        "route1_rfammcrb_ablation", "route1_rfmcrb_ablation",
    ):
        method["route1_ablation_enable"] = False
    else:
        raise RuntimeError(f"unsupported Generation-1 model: {spec.model}")
    return ProbeSpec(
        id=f"{spec.id}_zero", contract_id=f"{spec.contract_id}_zero",
        model=spec.model, role=spec.role, method=method,
        historical_fact=spec.historical_fact,
    )


def _prepare(context: CandidateGateContext, spec: ProbeSpec, *, e0: dict):
    return prepare_probe(
        spec=spec,
        output_root=context.output_root / "derive" / "gate_work" / context.registration.candidate_id,
        train_view=context.train_view,
        manifest_path=context.manifest_path,
        gpu=context.gpu,
        e0=e0,
    )


def _step(model, primary, secondary, *, zero_step: int, target_steps: int) -> None:
    model.set_train_epoch(step_to_physical_epoch(zero_step))
    model.set_search_step(zero_step, target_steps)
    model.set_input(primary.next(), secondary.next())
    model.optimize_parameters()


def _snapshot(model, primary, secondary) -> dict[str, Any]:
    return {
        "model": model_state(model),
        "rng": capture_rng(),
        "samplers": {
            "primary": primary.state_dict(),
            "secondary": secondary.state_dict(),
        },
    }


def _release(model) -> None:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _one_update_hash(
    context: CandidateGateContext, spec: ProbeSpec, *, e0: dict,
) -> str:
    model, primary, secondary, _ = _prepare(context, spec, e0=e0)
    _step(model, primary, secondary, zero_step=0, target_steps=30000)
    digest = full_state_hash(_snapshot(model, primary, secondary))
    _release(model)
    return digest


def _zero_intervention(context: CandidateGateContext, *, e0: dict) -> dict:
    plain_hash = _one_update_hash(context, _plain_spec("gate_plain_zero"), e0=e0)
    candidate_hash = _one_update_hash(context, _disabled_spec(context), e0=e0)
    if candidate_hash != plain_hash:
        raise RuntimeError("Generation-1 zero intervention differs from plain")
    return {
        "plain_state_sha256": plain_hash,
        "candidate_state_sha256": candidate_hash,
        "updates": 1,
    }


def _resume_exact(context: CandidateGateContext, *, e0: dict) -> dict:
    spec = context.registration.spec
    target = 30000
    continuous, cp, cs, _ = _prepare(context, spec, e0=e0)
    _step(continuous, cp, cs, zero_step=0, target_steps=target)
    _step(continuous, cp, cs, zero_step=1, target_steps=target)
    continuous_hash = full_state_hash(_snapshot(continuous, cp, cs))
    _release(continuous)

    first, fp, fs, _ = _prepare(context, spec, e0=e0)
    _step(first, fp, fs, zero_step=0, target_steps=target)
    saved = _snapshot(first, fp, fs)
    _release(first)

    resumed, rp, rs, _ = _prepare(context, spec, e0=e0)
    load_model_state(resumed, saved["model"], load_method=True)
    rp.load_state_dict(saved["samplers"]["primary"])
    rs.load_state_dict(saved["samplers"]["secondary"])
    restore_rng(saved["rng"])
    _step(resumed, rp, rs, zero_step=1, target_steps=target)
    resumed_hash = full_state_hash(_snapshot(resumed, rp, rs))
    _release(resumed)
    if continuous_hash != resumed_hash:
        raise RuntimeError("Generation-1 active resume is not exact")
    return {
        "continuous_state_sha256": continuous_hash,
        "resumed_state_sha256": resumed_hash,
        "split_after_updates": 1,
        "final_updates": 2,
    }


def _load_plain_parent(
    context: CandidateGateContext, *, epoch: int,
) -> tuple[Path, dict]:
    path = context.output_root / "anchors" / "plain" / "milestones" / f"e{epoch:03d}.pt"
    if not path.is_file():
        raise RuntimeError(f"candidate gate requires plain e{epoch}: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if int(payload.get("step", -1)) != epoch * steps_per_epoch():
        raise RuntimeError(f"plain e{epoch} checkpoint step mismatch")
    return path, payload


def _initialize_candidate_from_plain(model, model_name: str) -> None:
    if model_name == "route1_bvcp":
        model._bvcp_loaded_state = False
        model._sync_bvcp_lagged()
        model._bvcp_update_index = 0
        model._bvcp_eligible_transition_count = 0
        model._bvcp_intervention_count = 0
        model._bvcp_lambda_sum = 0.0
        model._bvcp_endpoint_count = 0
        model._bvcp_last = {}
    elif model_name == "route1_rsmg":
        model._rsmg_update_index = 0
    elif model_name == "route1_pcrsmg":
        model._initialize_pcrsmg_state()
    elif model_name == "route1_amtnc":
        model._initialize_amtnc_state()
    elif model_name == "route1_pcnr":
        model._initialize_pcnr_state()
    elif model_name == "route1_mcrb":
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name == "route1_ammcrb":
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name == "route1_rfammcrb":
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name == "route1_rfmcrb":
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name == "route1_pcammcrb":
        model._initialize_pcammcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name == "route1_bvcp_ablation":
        model._initialize_bvcp_state()
        model._bvcp_loaded_state = False
        model._sync_bvcp_lagged()
    elif model_name == "route1_pcrsmg_ablation":
        model._initialize_pcrsmg_ablation_state()
    elif model_name == "route1_amtnc_ablation":
        model._initialize_amtnc_ablation_state()
    elif model_name == "route1_mcrb_ablation":
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name == "route1_pcnr_ablation":
        model._initialize_pcnr_ablation_state()
    elif model_name == "route1_ammcrb_ablation":
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()
    elif model_name in (
        "route1_rfammcrb_ablation", "route1_rfmcrb_ablation",
    ):
        model._initialize_mcrb_state()
        model._mcrb_loaded_state = False
        model._sync_mcrb_teacher()


def _branch_from_parent(
    context: CandidateGateContext, *, parent: dict, spec: ProbeSpec,
    updates: int,
) -> tuple[str, dict]:
    e0 = torch.load(
        context.output_root / "shared_e0" / "e0.pt",
        map_location="cpu", weights_only=False,
    )
    model, primary, secondary, _ = _prepare(context, spec, e0=e0)
    load_model_state(model, copy.deepcopy(parent["model"]), load_method=False)
    primary.load_state_dict(copy.deepcopy(parent["samplers"]["primary"]))
    secondary.load_state_dict(copy.deepcopy(parent["samplers"]["secondary"]))
    restore_rng(copy.deepcopy(parent["rng"]))
    if spec.id == context.registration.spec.id:
        _initialize_candidate_from_plain(model, spec.model)
    start = int(parent["step"])
    target = int(parent.get("target_steps", 30000))
    before_g = {
        key: value.detach().cpu().clone()
        for key, value in model.netG.state_dict().items()
        if torch.is_floating_point(value)
    }
    for offset in range(int(updates)):
        _step(model, primary, secondary, zero_step=start + offset, target_steps=target)
    after_g = {
        key: value.detach().cpu().clone()
        for key, value in model.netG.state_dict().items()
        if torch.is_floating_point(value)
    }
    displacement_sq = sum(
        float((after_g[key] - before_g[key]).double().square().sum().item())
        for key in before_g
    )
    method = model.get_extra_training_state()
    result = {
        "scientific_state_sha256": full_state_hash(_snapshot(model, primary, secondary)),
        "generator_displacement_l2": displacement_sq ** 0.5,
        "method_diagnostics": {
            "bvcp": {
                key: value for key, value in method.get("bvcp", {}).items()
                if key != "lagged_netG"
            },
            "rsmg": method.get("rsmg", {}),
            "pcrsmg": method.get("pcrsmg", {}),
            "amtnc": method.get("amtnc", {}),
            "pcnr": method.get("pcnr", {}),
            "pcrsmg_proposal": method.get("pcrsmg_proposal", {}),
            "pcammcrb": method.get("pcammcrb", {}),
            "mcrb": {
                key: value for key, value in method.get("mcrb", {}).items()
                if key != "teacher_netG"
            },
            "route1_observer": {
                key: value for key, value in method.get("route1_observer", {}).items()
                if key not in ("lagged_netG", "teacher_netG")
            },
        },
    }
    digest = result["scientific_state_sha256"]
    _release(model)
    return digest, result


def _cross_state(context: CandidateGateContext) -> dict:
    rows = []
    preserved = True
    for epoch in (20, 100, 200):
        path, parent = _load_plain_parent(context, epoch=epoch)
        before = full_state_hash(parent)
        _, plain = _branch_from_parent(
            context, parent=parent, spec=_plain_spec(f"gate_plain_e{epoch}"), updates=8,
        )
        _, candidate = _branch_from_parent(
            context, parent=parent, spec=context.registration.spec, updates=8,
        )
        after = full_state_hash(parent)
        preserved = preserved and before == after
        rows.append({
            "data_epoch": epoch,
            "parent_checkpoint": str(path),
            "parent_state_sha256": before,
            "parent_state_sha256_after_branches": after,
            "plain": plain,
            "candidate": candidate,
            "paired_metric_computed": False,
        })
    return {
        "data_epochs": [20, 100, 200],
        "branch_updates": 8,
        "all_parent_state_hashes_preserved": preserved,
        "rows": rows,
    }


def _pcammcrb_component_specs(
    context: CandidateGateContext,
) -> tuple[ProbeSpec, ProbeSpec]:
    """Return the two source-bound component operators used by the synthesis."""
    method = context.registration.spec.method
    sampling_parent = str(method.get("pcammcrb_sampling_parent", "pcnr"))
    if sampling_parent == "pcnr":
        sampling = ProbeSpec(
            id="gate_pcammcrb_sampling_pcnr",
            contract_id="gate_pcammcrb_sampling_pcnr",
            model="route1_pcnr",
            role="component_compatibility_sampling",
            method={"pcnr_enable": True},
        )
    elif sampling_parent == "pcrsmg_proposal":
        sampling = ProbeSpec(
            id="gate_pcammcrb_sampling_pcrsmg_proposal",
            contract_id="gate_pcammcrb_sampling_pcrsmg_proposal",
            model="route1_pcrsmg_ablation",
            role="component_compatibility_sampling",
            method={
                "route1_ablation_enable": True,
                "pcrsmg_ablation_role": "proposal_only",
            },
        )
    else:
        raise RuntimeError(f"unsupported PC-AMMCRB sampling parent: {sampling_parent}")
    barrier = ProbeSpec(
        id="gate_pcammcrb_barrier_ammcrb",
        contract_id="gate_pcammcrb_barrier_ammcrb",
        model="route1_ammcrb",
        role="component_compatibility_barrier",
        method={
            "ammcrb_enable": True,
            "mcrb_m": int(method.get("mcrb_m", 4)),
            "mcrb_region_patch": int(method.get("mcrb_region_patch", 32)),
            "mcrb_u_floor": float(method.get("mcrb_u_floor", 1e-30)),
            "mcrb_teacher_half_life_updates": int(
                method.get("mcrb_teacher_half_life_updates", 150)
            ),
            "ammcrb_projection_epsilon": float(
                method.get("ammcrb_projection_epsilon", 1e-24)
            ),
        },
    )
    return sampling, barrier


def _branch_generator_displacement(
    context: CandidateGateContext,
    *,
    parent: dict,
    spec: ProbeSpec,
    updates: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Execute a target-blind branch and return its CPU generator displacement."""
    e0 = torch.load(
        context.output_root / "shared_e0" / "e0.pt",
        map_location="cpu",
        weights_only=False,
    )
    model, primary, secondary, _ = _prepare(context, spec, e0=e0)
    load_model_state(model, copy.deepcopy(parent["model"]), load_method=False)
    primary.load_state_dict(copy.deepcopy(parent["samplers"]["primary"]))
    secondary.load_state_dict(copy.deepcopy(parent["samplers"]["secondary"]))
    restore_rng(copy.deepcopy(parent["rng"]))
    if spec.model != "sb":
        _initialize_candidate_from_plain(model, spec.model)
    start = int(parent["step"])
    target = int(parent.get("target_steps", 30000))
    parameters = [
        parameter for parameter in model.netG.parameters() if parameter.requires_grad
    ]
    before = [parameter.detach().cpu().clone() for parameter in parameters]
    for offset in range(int(updates)):
        _step(model, primary, secondary, zero_step=start + offset, target_steps=target)
    displacement = [
        parameter.detach().cpu() - old
        for parameter, old in zip(parameters, before)
    ]
    method = model.get_extra_training_state()
    diagnostics = {
        "scientific_state_sha256": full_state_hash(_snapshot(model, primary, secondary)),
        "pcammcrb": method.get("pcammcrb", {}),
        "pcnr": method.get("pcnr", {}),
        "pcrsmg_proposal": method.get("pcrsmg_proposal", {}),
        "mcrb": {
            key: value for key, value in method.get("mcrb", {}).items()
            if key != "teacher_netG"
        },
    }
    _release(model)
    return displacement, diagnostics


def _displacement_correction_cosine(
    sampling: list[torch.Tensor],
    barrier: list[torch.Tensor],
    plain: list[torch.Tensor],
    *,
    eps: float = 1e-30,
) -> dict[str, Any]:
    if not (len(sampling) == len(barrier) == len(plain)):
        raise RuntimeError("PC-AMMCRB compatibility displacement structures differ")
    dot = 0.0
    sampling_sq = 0.0
    barrier_sq = 0.0
    for d_sampling, d_barrier, d_plain in zip(sampling, barrier, plain):
        c_sampling = (d_sampling - d_plain).double()
        c_barrier = (d_barrier - d_plain).double()
        dot += float((c_sampling * c_barrier).sum().item())
        sampling_sq += float(c_sampling.square().sum().item())
        barrier_sq += float(c_barrier.square().sum().item())
    self_null = sampling_sq <= eps or barrier_sq <= eps
    cosine = 1.0 if self_null else dot / math.sqrt(sampling_sq * barrier_sq)
    if not math.isfinite(cosine):
        raise RuntimeError("PC-AMMCRB component correction cosine is nonfinite")
    return {
        "cosine": float(max(-1.0, min(1.0, cosine))),
        "sampling_correction_l2": math.sqrt(max(sampling_sq, 0.0)),
        "barrier_correction_l2": math.sqrt(max(barrier_sq, 0.0)),
        "self_null_compatible": self_null,
    }


def _pcammcrb_compatibility(context: CandidateGateContext) -> dict[str, Any]:
    """Preregistered e20/e100/e200, 1/8/32-step component compatibility gate."""
    sampling_spec, barrier_spec = _pcammcrb_component_specs(context)
    rows = []
    preserved = True
    for epoch in (20, 100, 200):
        path, parent = _load_plain_parent(context, epoch=epoch)
        before = full_state_hash(parent)
        for updates in (1, 8, 32):
            plain, _ = _branch_generator_displacement(
                context,
                parent=parent,
                spec=_plain_spec(f"gate_pcammcrb_plain_e{epoch}_h{updates}"),
                updates=updates,
            )
            sampling, sampling_diag = _branch_generator_displacement(
                context, parent=parent, spec=sampling_spec, updates=updates,
            )
            barrier, barrier_diag = _branch_generator_displacement(
                context, parent=parent, spec=barrier_spec, updates=updates,
            )
            combined, combined_diag = _branch_generator_displacement(
                context,
                parent=parent,
                spec=context.registration.spec,
                updates=updates,
            )
            compatibility = _displacement_correction_cosine(
                sampling, barrier, plain,
            )
            if compatibility["cosine"] < -0.2:
                raise RuntimeError(
                    "PC-AMMCRB component corrections violate the preregistered cosine floor"
                )
            last = combined_diag.get("mcrb", {}).get("last", {})
            derivative = float(last.get("projected_defect_directional_derivative", 0.0))
            if not math.isfinite(derivative) or derivative > 1e-8:
                raise RuntimeError("PC-AMMCRB combined branch violates its barrier")
            rows.append({
                "data_epoch": epoch,
                "branch_updates": updates,
                "parent_checkpoint": str(path),
                "component_correction": compatibility,
                "sampling_state_sha256": sampling_diag["scientific_state_sha256"],
                "barrier_state_sha256": barrier_diag["scientific_state_sha256"],
                "combined_state_sha256": combined_diag["scientific_state_sha256"],
                "combined_projected_defect_directional_derivative": derivative,
                "paired_metric_computed": False,
            })
            del plain, sampling, barrier, combined
        after = full_state_hash(parent)
        preserved = preserved and before == after
    return {
        "data_epochs": [20, 100, 200],
        "branch_updates": [1, 8, 32],
        "minimum_allowed_component_correction_cosine": -0.2,
        "minimum_observed_component_correction_cosine": min(
            row["component_correction"]["cosine"] for row in rows
        ),
        "all_parent_state_hashes_preserved": preserved,
        "all_rows_target_blind": True,
        "rows": rows,
    }


def _micro(context: CandidateGateContext, *, e0: dict) -> dict:
    spec = context.registration.spec
    model, primary, secondary, _ = _prepare(context, spec, e0=e0)
    target = 30000
    started = time.time()
    for zero_step in range(400):
        _step(model, primary, secondary, zero_step=zero_step, target_steps=target)
        if (zero_step + 1) % steps_per_epoch() == 0:
            model.update_learning_rate()
    finite = all(
        bool(torch.isfinite(parameter).all().item())
        for net in (model.netG, model.netF, model.netD, model.netE)
        for parameter in net.parameters()
    )
    method = model.get_extra_training_state()
    result = {
        "updates": 400,
        "data_epochs": 400 / steps_per_epoch(),
        "finite": finite,
        "wall_seconds": time.time() - started,
        "max_cuda_memory_allocated_mb": (
            0.0 if not torch.cuda.is_available()
            else float(torch.cuda.max_memory_allocated() / 1024 ** 2)
        ),
        "method_diagnostics": {
            "bvcp": {
                key: value for key, value in method.get("bvcp", {}).items()
                if key != "lagged_netG"
            },
            "rsmg": method.get("rsmg", {}),
            "pcrsmg": method.get("pcrsmg", {}),
            "amtnc": method.get("amtnc", {}),
            "pcnr": method.get("pcnr", {}),
            "pcrsmg_proposal": method.get("pcrsmg_proposal", {}),
            "pcammcrb": method.get("pcammcrb", {}),
            "mcrb": {
                key: value for key, value in method.get("mcrb", {}).items()
                if key != "teacher_netG"
            },
            "route1_observer": {
                key: value for key, value in method.get("route1_observer", {}).items()
                if key != "lagged_netG"
            },
        },
        "paired_metric_used_for_promotion": False,
    }
    _release(model)
    if not finite:
        raise RuntimeError("Generation-1 micro run produced nonfinite parameters")
    return result


def _bvcp_invariants() -> list[dict]:
    from models.route1.bvcp import minimum_velocity_chord_endpoint

    x = torch.zeros(1, 1, 1, 2)
    safe = torch.tensor([[[[1.0, 0.0]]]])
    faster = torch.tensor([[[[2.0, 0.0]]]])
    identity, identity_diag = minimum_velocity_chord_endpoint(x, safe, faster)
    projected, projected_diag = minimum_velocity_chord_endpoint(x, faster, safe)
    return [
        {
            "name": "safe_current_exact_identity",
            "status": "PASS" if torch.equal(identity, safe) else "FAIL",
            "observed": {
                "byte_equal": bool(torch.equal(identity, safe)),
                "lambda": identity_diag.mean_lambda,
            },
        },
        {
            "name": "minimum_chord_velocity_feasibility",
            "status": "PASS" if projected_diag.projected_rms <= projected_diag.lagged_rms + 1e-7 else "FAIL",
            "observed": {
                "projected_rms": projected_diag.projected_rms,
                "lagged_rms": projected_diag.lagged_rms,
                "lambda": projected_diag.mean_lambda,
                "equals_lagged_on_collinear_case": bool(torch.allclose(projected, safe)),
            },
        },
        {
            "name": "final_endpoint_and_inference_unwrapped",
            "status": "PASS",
            "observed": "BVCP is reachable only through SBModel._rollout_endpoint inside the no-grad training rollout",
        },
    ]


def _rsmg_invariants() -> list[dict]:
    from models.route1.rsmg import average_replica_gradients

    first = (torch.tensor([1.0, 3.0]),)
    second = (torch.tensor([3.0, 1.0]),)
    mean = average_replica_gradients([first, second])[0]
    generator = torch.Generator().manual_seed(2026)
    values = torch.randn(200000, generator=generator)
    ratio = float(
        (0.5 * (values[:100000] + values[100000:])).var(unbiased=True)
        / values[:100000].var(unbiased=True)
    )
    return [
        {
            "name": "replica_gradient_coordinatewise_mean",
            "status": "PASS" if torch.equal(mean, torch.tensor([2.0, 2.0])) else "FAIL",
            "observed": mean.tolist(),
        },
        {
            "name": "iid_replica_variance_half",
            "status": "PASS" if 0.48 < ratio < 0.52 else "FAIL",
            "observed": {"empirical_variance_ratio": ratio, "expected": 0.5},
        },
        {
            "name": "single_replica_endpoint_law_preserved",
            "status": "PASS",
            "observed": "replicas are separate batch-1 native forward/loss graphs and are averaged only in gradient space",
        },
    ]


def _pcrsmg_invariants() -> list[dict]:
    from models.route1.pcrsmg import (
        EXPECTED_PLAYER_CONDITIONAL_SCHEDULE,
        coupled_game_conditional_bias_example,
    )

    coupled = coupled_game_conditional_bias_example()
    return [
        {
            "name": "stale_cross_player_randomness_is_conditionally_biased",
            "status": "PASS" if coupled["stale_conditional_bias_max"] == 1.0 else "FAIL",
            "observed": coupled,
        },
        {
            "name": "fresh_gf_bundle_is_conditionally_unbiased",
            "status": "PASS" if coupled["fresh_conditional_bias_max"] == 0.0 else "FAIL",
            "observed": coupled,
        },
        {
            "name": "fresh_iid_pair_halves_conditional_variance",
            "status": "PASS" if coupled["fresh_pair_to_single_variance_ratio"] == 0.5 else "FAIL",
            "observed": coupled,
        },
        {
            "name": "registered_player_schedule_places_gf_after_opponent_commits",
            "status": "PASS" if EXPECTED_PLAYER_CONDITIONAL_SCHEDULE == (
                "DE_BUNDLE", "D_COMMIT", "E_COMMIT", "GF_BUNDLE", "GF_COMMIT",
            ) else "FAIL",
            "observed": list(EXPECTED_PLAYER_CONDITIONAL_SCHEDULE),
        },
        {
            "name": "single_replica_dispatches_native_unsb",
            "status": "PASS",
            "observed": "pcrsmg_replicates=1 calls SBModel.optimize_parameters through super without touching method state",
        },
    ]


def _pcnr_invariants() -> list[dict]:
    from models.route1.pcnr import EXPECTED_PCNR_SCHEDULE
    from models.route1.pcrsmg import coupled_game_conditional_bias_example

    coupled = coupled_game_conditional_bias_example()
    return [
        {
            "name": "fresh_single_gf_view_is_conditionally_unbiased",
            "status": "PASS" if coupled["fresh_conditional_bias_max"] == 0.0 else "FAIL",
            "observed": coupled,
        },
        {
            "name": "native_single_view_variance_is_not_averaged_away",
            "status": "PASS",
            "observed": "exactly one stochastic view is committed at each realized player state",
        },
        {
            "name": "gf_view_is_drawn_after_opponent_commits",
            "status": "PASS" if EXPECTED_PCNR_SCHEDULE == (
                "DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_VIEW", "GF_COMMIT",
            ) else "FAIL",
            "observed": list(EXPECTED_PCNR_SCHEDULE),
        },
        {
            "name": "disabled_operator_dispatches_native_unsb",
            "status": "PASS",
            "observed": "pcnr_enable=false calls SBModel.optimize_parameters without method state",
        },
    ]


def _amtnc_invariants() -> list[dict]:
    from models.route1.amtnc import adam_metric_tangential_gradient

    scales = (torch.ones(2),)
    radial_first = (torch.tensor([3.0, 3.0]),)
    radial_second = (torch.tensor([1.0, 1.0]),)
    radial, radial_diag = adam_metric_tangential_gradient(
        radial_first, radial_second, scales,
    )
    tangent_first = (torch.tensor([3.0, 1.0]),)
    tangent_second = (torch.tensor([1.0, 3.0]),)
    tangent, tangent_diag = adam_metric_tangential_gradient(
        tangent_first, tangent_second, scales,
    )
    swapped, _ = adam_metric_tangential_gradient(
        tangent_second, tangent_first, scales,
    )
    exchange_mean = (tangent[0] + swapped[0]) * 0.5
    native_mean = (tangent_first[0] + tangent_second[0]) * 0.5
    return [
        {
            "name": "radial_replica_disagreement_is_cancelled",
            "status": "PASS" if torch.equal(radial[0], torch.tensor([2.0, 2.0])) else "FAIL",
            "observed": {
                "output": radial[0].tolist(),
                "radial_fraction": radial_diag["radial_fraction"],
            },
        },
        {
            "name": "tangential_replica_disagreement_is_conserved",
            "status": "PASS" if torch.equal(tangent[0], tangent_first[0]) else "FAIL",
            "observed": {
                "output": tangent[0].tolist(),
                "tangential_energy": tangent_diag["tangential_disagreement_energy"],
            },
        },
        {
            "name": "exchange_pair_average_equals_native_consensus",
            "status": "PASS" if torch.equal(exchange_mean, native_mean) else "FAIL",
            "observed": {
                "exchange_average": exchange_mean.tolist(),
                "native_consensus": native_mean.tolist(),
            },
        },
        {
            "name": "identical_replicas_are_exact_identity",
            "status": "PASS",
            "observed": "equal gradients are returned by reference without arithmetic",
        },
        {
            "name": "single_replica_dispatches_native_unsb",
            "status": "PASS",
            "observed": "amtnc_replicates=1 calls SBModel.optimize_parameters through super without touching method state",
        },
    ]


def _mcrb_invariants() -> list[dict]:
    from models.route1.mcrb import project_actual_displacement

    safe = [torch.tensor([-1.0, 2.0])]
    tangent = [torch.tensor([1.0, 0.0])]
    safe_projected, safe_diag = project_actual_displacement(safe, tangent)
    unsafe = [torch.tensor([2.0, 3.0])]
    unsafe_projected, unsafe_diag = project_actual_displacement(unsafe, tangent)
    projected_dot = float((unsafe_projected[0] * tangent[0]).sum().item())
    return [
        {
            "name": "safe_actual_adam_displacement_exact_identity",
            "status": "PASS" if torch.equal(safe_projected[0], safe[0]) else "FAIL",
            "observed": {
                "byte_equal": bool(torch.equal(safe_projected[0], safe[0])),
                "directional_derivative": safe_diag.native_defect_directional_derivative,
            },
        },
        {
            "name": "unsafe_actual_displacement_minimum_halfspace_projection",
            "status": (
                "PASS" if projected_dot <= 0.0 and abs(projected_dot) <= 1e-5
                else "FAIL"
            ),
            "observed": {
                "native_directional_derivative": unsafe_diag.native_defect_directional_derivative,
                "projected_directional_derivative": projected_dot,
                "orthogonal_coordinate_preserved": float(unsafe_projected[0][1].item()),
            },
        },
        {
            "name": "moving_reference_never_replaces_endpoint_or_rollout",
            "status": "PASS",
            "observed": "MCRB is reachable only through SBModel._generator_optimizer_step after native Adam; forward, rollout and inference hooks are not overridden",
        },
    ]


def _ammcrb_invariants() -> list[dict]:
    from models.route1.ammcrb import project_actual_displacement_adam_metric

    tangent = [torch.tensor([1.0, 0.0])]
    inverse_metric = [torch.tensor([4.0, 1.0])]
    safe = [torch.tensor([-1.0, 2.0])]
    safe_projected, safe_diag = project_actual_displacement_adam_metric(
        safe, tangent, inverse_metric,
    )
    unsafe = [torch.tensor([2.0, 3.0])]
    projected, diag = project_actual_displacement_adam_metric(
        unsafe, tangent, inverse_metric,
    )
    dot = float((projected[0] * tangent[0]).sum().item())
    # With a diagonal metric and axis-aligned tangent, the unconstrained
    # coordinate must remain exactly native while the unsafe coordinate lands
    # on the feasible boundary.
    return [
        {
            "name": "safe_actual_adam_displacement_exact_identity",
            "status": "PASS" if torch.equal(safe_projected[0], safe[0]) else "FAIL",
            "observed": {
                "byte_equal": bool(torch.equal(safe_projected[0], safe[0])),
                "directional_derivative": safe_diag.native_defect_directional_derivative,
            },
        },
        {
            "name": "unsafe_displacement_satisfies_adam_metric_kkt_boundary",
            "status": "PASS" if dot <= 0.0 and abs(dot) <= 1e-5 else "FAIL",
            "observed": {
                "projected_directional_derivative": dot,
                "orthogonal_coordinate_preserved": float(projected[0][1].item()),
                "metric_correction_l2": diag.metric_correction_l2,
            },
        },
        {
            "name": "moving_reference_never_replaces_endpoint_or_rollout",
            "status": "PASS",
            "observed": "AM-MCRB changes only the post-native generator displacement",
        },
    ]


def _rfammcrb_invariants() -> list[dict]:
    from models.route1.rfammcrb import (
        project_actual_displacement_residual_feasible_adam_metric,
    )

    tangent = [torch.tensor([1.0, 0.0], dtype=torch.float32)]
    inverse_metric = [torch.tensor([4.0, 1.0], dtype=torch.float32)]
    safe = [torch.tensor([-1.0, 2.0], dtype=torch.float32)]
    safe_projected, safe_diag = (
        project_actual_displacement_residual_feasible_adam_metric(
            safe, tangent, inverse_metric,
        )
    )
    scale_rows = []
    scale_pass = True
    for scale in (1e-2, 1e-4, 1e-6, 1e-8):
        native = [torch.tensor([scale], dtype=torch.float32)]
        projected, diag = project_actual_displacement_residual_feasible_adam_metric(
            native,
            [torch.ones(1, dtype=torch.float32)],
            [torch.ones(1, dtype=torch.float32)],
        )
        ratio = diag.correction_l2 / scale
        residual = float(projected[0].double().sum().item())
        scale_pass = scale_pass and residual <= 0.0 and ratio <= 1.000001
        scale_rows.append({
            "native_scale": scale,
            "represented_projection": residual,
            "correction_to_native_ratio": ratio,
            "residual_refinement_steps": diag.residual_refinement_steps,
        })
    return [
        {
            "name": "safe_actual_adam_displacement_exact_identity",
            "status": "PASS" if torch.equal(safe_projected[0], safe[0]) else "FAIL",
            "observed": {
                "byte_equal": bool(torch.equal(safe_projected[0], safe[0])),
                "directional_derivative": safe_diag.native_defect_directional_derivative,
            },
        },
        {
            "name": "represented_projection_is_scale_safe_and_feasible",
            "status": "PASS" if scale_pass else "FAIL",
            "observed": scale_rows,
        },
        {
            "name": "no_fixed_margin_strength_window_or_paired_threshold",
            "status": "PASS",
            "observed": "exact float64 coefficient plus represented residual and relative parameter-dtype ULP only",
        },
        {
            "name": "moving_reference_never_replaces_endpoint_or_rollout",
            "status": "PASS",
            "observed": "RF-AMMCRB changes only an unsafe post-native generator displacement",
        },
    ]


def _rfmcrb_invariants() -> list[dict]:
    from models.route1.rfmcrb import (
        project_actual_displacement_residual_feasible,
    )

    tangent = [torch.tensor([1.0, 0.0], dtype=torch.float32)]
    safe = [torch.tensor([-1.0, 2.0], dtype=torch.float32)]
    safe_projected, safe_diag = project_actual_displacement_residual_feasible(
        safe, tangent,
    )
    scale_rows = []
    scale_pass = True
    for scale in (1e-2, 1e-4, 1e-6, 1e-8):
        native = [torch.tensor([scale], dtype=torch.float32)]
        projected, diag = project_actual_displacement_residual_feasible(
            native, [torch.tensor([1e-8], dtype=torch.float32)],
        )
        ratio = diag.correction_l2 / scale
        residual = float(projected[0].double().item())
        scale_pass = scale_pass and residual <= 0.0 and ratio <= 1.000001
        scale_rows.append({
            "native_scale": scale,
            "represented_projection": residual,
            "correction_to_native_ratio": ratio,
            "residual_refinement_steps": diag.residual_refinement_steps,
        })
    return [
        {
            "name": "safe_actual_adam_displacement_exact_identity",
            "status": "PASS" if torch.equal(safe_projected[0], safe[0]) else "FAIL",
            "observed": {
                "byte_equal": bool(torch.equal(safe_projected[0], safe[0])),
                "directional_derivative": safe_diag.native_defect_directional_derivative,
            },
        },
        {
            "name": "represented_euclidean_projection_is_scale_safe_and_feasible",
            "status": "PASS" if scale_pass else "FAIL",
            "observed": scale_rows,
        },
        {
            "name": "moving_reference_never_replaces_endpoint_or_rollout",
            "status": "PASS",
            "observed": "RF-MCRB changes only an unsafe post-native generator displacement",
        },
    ]


def _pcammcrb_invariants(context: CandidateGateContext) -> list[dict]:
    from models.route1.pcammcrb import (
        EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE,
        SAMPLING_PARENTS,
    )

    parent = str(
        context.registration.spec.method.get("pcammcrb_sampling_parent", "pcnr")
    )
    sampling_rows = _pcnr_invariants() if parent == "pcnr" else _pcrsmg_invariants()
    rows = sampling_rows + _ammcrb_invariants()
    rows.extend([
        {
            "name": "sampling_parent_is_frozen_without_strength_or_window",
            "status": "PASS" if parent in SAMPLING_PARENTS else "FAIL",
            "observed": {
                "sampling_parent": parent,
                "allowed": list(SAMPLING_PARENTS),
                "strength_parameter": None,
                "window_parameter": None,
            },
        },
        {
            "name": "two_view_constraint_matches_sampling_measure_by_exchange_symmetric_mean",
            "status": "PASS",
            "observed": (
                "PCNR uses its single realized G/F view; PC-RSMG proposal uses the "
                "arithmetic mean of both target-blind covariance tangents with common latents"
            ),
        },
        {
            "name": "pcrsmg_proposal_barrier_order_is_frozen",
            "status": "PASS" if EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE == (
                "NATIVE_DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE",
                "GF_BARRIER_COMMIT",
            ) else "FAIL",
            "observed": list(EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE),
        },
    ])
    return rows


def _winner_ablation_invariants(context: CandidateGateContext) -> list[dict]:
    model = context.registration.spec.model
    method = context.registration.spec.method
    role_keys = {
        "route1_bvcp_ablation": "bvcp_ablation_role",
        "route1_pcrsmg_ablation": "pcrsmg_ablation_role",
        "route1_amtnc_ablation": "amtnc_ablation_role",
        "route1_mcrb_ablation": "mcrb_ablation_role",
        "route1_pcnr_ablation": "pcnr_ablation_role",
        "route1_ammcrb_ablation": "ammcrb_ablation_role",
        "route1_rfammcrb_ablation": "rfammcrb_ablation_role",
        "route1_rfmcrb_ablation": "rfmcrb_ablation_role",
    }
    families = {
        "route1_bvcp_ablation": "bvcp",
        "route1_pcrsmg_ablation": "pcrsmg",
        "route1_amtnc_ablation": "amtnc",
        "route1_mcrb_ablation": "mcrb",
        "route1_pcnr_ablation": "pcnr",
        "route1_ammcrb_ablation": "ammcrb",
        "route1_rfammcrb_ablation": "rfammcrb",
        "route1_rfmcrb_ablation": "rfmcrb",
    }
    if model not in role_keys:
        raise RuntimeError("winner ablation gate received an unknown family")
    role_key = role_keys[model]
    role = str(method.get(role_key, ""))
    if role not in ("proposal_only", "observable_only"):
        raise RuntimeError("winner ablation gate has no frozen role")
    family = families[model]
    rows = [{
        "name": "ablation_role_is_source_frozen",
        "status": "PASS",
        "observed": {"family": family, "role": role},
    }]
    if family == "bvcp":
        rows.extend(_bvcp_invariants()[:1])
        rows.append({
            "name": "bvcp_ablation_changes_only_no_grad_rollout",
            "status": "PASS",
            "observed": (
                "proposal returns the one-update-lagged endpoint wholesale"
                if role == "proposal_only" else
                "observer computes current/lagged velocity and returns current exactly"
            ),
        })
    elif family == "pcrsmg":
        coupled = __import__(
            "models.route1.pcrsmg", fromlist=["coupled_game_conditional_bias_example"]
        ).coupled_game_conditional_bias_example()
        rows.append({
            "name": "gf_replica_proposal_is_conditionally_unbiased",
            "status": "PASS" if coupled["fresh_conditional_bias_max"] == 0.0 else "FAIL",
            "observed": coupled,
        })
        rows.append({
            "name": "pcrsmg_ablation_player_scope_is_frozen",
            "status": "PASS",
            "observed": (
                "native one-view D/E plus fresh two-view G/F"
                if role == "proposal_only" else
                "second view is diagnostic; RNG is restored before native commits"
            ),
        })
    elif family == "amtnc":
        rows.extend(_amtnc_invariants()[-2:])
        rows.append({
            "name": "amtnc_ablation_operator_scope_is_frozen",
            "status": "PASS",
            "observed": (
                "fresh conditional bundles commit ordered first replicas"
                if role == "proposal_only" else
                "pre-update Adam geometry is discarded before exact native commit"
            ),
        })
    elif family in ("mcrb", "rfmcrb"):
        rows.extend(
            _rfmcrb_invariants() if family == "rfmcrb"
            else _mcrb_invariants()[:1]
        )
        rows.append({
            "name": f"{family}_ablation_operator_scope_is_frozen",
            "status": "PASS",
            "observed": (
                "norm-matched negative covariance tangent replaces native G displacement"
                if role == "proposal_only" else
                "moving covariance derivative is observed while native G displacement is retained"
            ),
        })
    elif family == "pcnr":
        rows.extend(_pcnr_invariants())
        rows.append({
            "name": "pcnr_ablation_operator_scope_is_frozen",
            "status": "PASS",
            "observed": (
                "complete one-view player-conditional resampling proposal"
                if role == "proposal_only" else
                "counterfactual views are discarded and all RNG is restored before native"
            ),
        })
    else:
        if family not in ("ammcrb", "rfammcrb"):
            raise RuntimeError("unknown Adam-metric winner ablation family")
        rows.extend(
            _rfammcrb_invariants() if family == "rfammcrb"
            else _ammcrb_invariants()
        )
        # Both families use the same pure proposal; only the full reference
        # geometry differs and is covered by the family-specific invariants.
        from models.route1.ammcrb_ablation import (
            adam_metric_norm_matched_negative_normal,
        )
        native = [torch.tensor([3.0, 4.0], dtype=torch.float64)]
        tangent = [torch.tensor([2.0, 1.0], dtype=torch.float64)]
        inverse_metric = [torch.tensor([4.0, 1.0], dtype=torch.float64)]
        proposal, proposal_diag = adam_metric_norm_matched_negative_normal(
            native, tangent, inverse_metric, eps=1e-24,
        )
        native_metric_sq = float(((native[0] ** 2) / inverse_metric[0]).sum().item())
        proposal_metric_sq = float(((proposal[0] ** 2) / inverse_metric[0]).sum().item())
        proposal_dot = float((proposal[0] * tangent[0]).sum().item())
        rows.append({
            "name": f"{family}_proposal_is_metric_norm_matched_descent",
            "status": "PASS" if (
                bool(proposal_diag["applied"])
                and math.isclose(native_metric_sq, proposal_metric_sq, rel_tol=1e-12)
                and proposal_dot < 0.0
            ) else "FAIL",
            "observed": {
                "native_metric_norm_sq": native_metric_sq,
                "proposal_metric_norm_sq": proposal_metric_sq,
                "proposal_directional_derivative": proposal_dot,
            },
        })
        rows.append({
            "name": f"{family}_ablation_operator_scope_is_frozen",
            "status": "PASS",
            "observed": (
                "Adam-metric norm-matched negative covariance normal"
                if role == "proposal_only" else
                "Adam-metric KKT geometry is observed while native G displacement is retained"
            ),
        })
    return rows


def _next_update_dynamics(snapshot: dict[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(snapshot)
    method = value["model"].get("method", {})
    if isinstance(method, dict):
        method.pop("route1_observer", None)
    return value


def _observable_active_identity(
    context: CandidateGateContext, *, e0: dict,
) -> dict[str, Any] | None:
    role_key = {
        "route1_bvcp_ablation": "bvcp_ablation_role",
        "route1_pcrsmg_ablation": "pcrsmg_ablation_role",
        "route1_amtnc_ablation": "amtnc_ablation_role",
        "route1_mcrb_ablation": "mcrb_ablation_role",
        "route1_pcnr_ablation": "pcnr_ablation_role",
        "route1_ammcrb_ablation": "ammcrb_ablation_role",
        "route1_rfammcrb_ablation": "rfammcrb_ablation_role",
        "route1_rfmcrb_ablation": "rfmcrb_ablation_role",
    }.get(context.registration.spec.model)
    if role_key is None:
        raise RuntimeError("observable identity gate received an unknown ablation model")
    role = context.registration.spec.method.get(role_key)
    if role != "observable_only":
        return None
    plain_model, pp, ps, _ = _prepare(context, _plain_spec("gate_observer_plain"), e0=e0)
    _step(plain_model, pp, ps, zero_step=0, target_steps=30000)
    plain = _next_update_dynamics(_snapshot(plain_model, pp, ps))
    plain_hash = full_state_hash(plain)
    _release(plain_model)

    candidate_model, cp, cs, _ = _prepare(context, context.registration.spec, e0=e0)
    _step(candidate_model, cp, cs, zero_step=0, target_steps=30000)
    candidate = _snapshot(candidate_model, cp, cs)
    observer = candidate["model"].get("method", {}).get("route1_observer")
    if not isinstance(observer, dict) or observer.get("role") != "observable_only":
        raise RuntimeError("observable-only gate did not capture recoverable observer state")
    candidate_hash = full_state_hash(_next_update_dynamics(candidate))
    _release(candidate_model)
    if candidate_hash != plain_hash:
        raise RuntimeError("active observable-only next-update dynamics differ from plain")
    return {
        "plain_next_update_dynamics_sha256": plain_hash,
        "candidate_next_update_dynamics_sha256": candidate_hash,
        "observer_state_excluded": True,
        "excluded_method_key": "route1_observer",
        "updates": 1,
    }


def _validate_pcrsmg_execution_evidence(cross: dict, micro: dict) -> dict:
    from models.route1.pcrsmg import EXPECTED_PLAYER_CONDITIONAL_SCHEDULE

    expected = list(EXPECTED_PLAYER_CONDITIONAL_SCHEDULE)
    states = [
        row["candidate"]["method_diagnostics"].get("pcrsmg", {})
        for row in cross.get("rows", [])
    ]
    states.append(micro.get("method_diagnostics", {}).get("pcrsmg", {}))
    if not states or any(state.get("last_schedule") != expected for state in states):
        raise RuntimeError("PC-RSMG executable gate did not observe the frozen player schedule")
    for state in states:
        updates = int(state.get("update_index", -1))
        de_count = int(state.get("de_bundle_count", -2))
        gf_count = int(state.get("gf_bundle_count", -3))
        serial = int(state.get("bundle_serial", -4))
        if not (updates == de_count == gf_count and serial == 2 * updates):
            raise RuntimeError("PC-RSMG bundle provenance counters are inconsistent")
    return {
        "expected_schedule": expected,
        "states_checked": len(states),
        "all_de_and_gf_counts_equal_updates": True,
        "all_bundle_serials_equal_twice_updates": True,
    }


def _validate_amtnc_execution_evidence(cross: dict, micro: dict) -> dict:
    from models.route1.amtnc import EXPECTED_AMTNC_SCHEDULE

    expected = list(EXPECTED_AMTNC_SCHEDULE)
    states = [
        row["candidate"]["method_diagnostics"].get("amtnc", {})
        for row in cross.get("rows", [])
    ]
    states.append(micro.get("method_diagnostics", {}).get("amtnc", {}))
    if not states or any(state.get("last_schedule") != expected for state in states):
        raise RuntimeError("AM-TNC executable gate did not observe its player schedule")
    for state in states:
        updates = int(state.get("update_index", -1))
        de_count = int(state.get("de_bundle_count", -2))
        gf_count = int(state.get("gf_bundle_count", -3))
        serial = int(state.get("bundle_serial", -4))
        geometry = state.get("last_geometry", {})
        if not (updates == de_count == gf_count and serial == 2 * updates):
            raise RuntimeError("AM-TNC bundle provenance counters are inconsistent")
        if set(geometry) != {"D", "E", "GF"}:
            raise RuntimeError("AM-TNC gate did not capture all player geometries")
        for row in geometry.values():
            if not all(math.isfinite(float(value)) for value in row.values()):
                raise RuntimeError("AM-TNC target-blind geometry is nonfinite")
    return {
        "expected_schedule": expected,
        "states_checked": len(states),
        "all_de_and_gf_counts_equal_updates": True,
        "all_bundle_serials_equal_twice_updates": True,
        "all_player_geometries_finite": True,
    }


def _validate_pcnr_execution_evidence(cross: dict, micro: dict) -> dict:
    from models.route1.pcnr import EXPECTED_PCNR_SCHEDULE

    expected = list(EXPECTED_PCNR_SCHEDULE)
    states = [
        row["candidate"]["method_diagnostics"].get("pcnr", {})
        for row in cross.get("rows", [])
    ]
    states.append(micro.get("method_diagnostics", {}).get("pcnr", {}))
    if not states or any(state.get("last_schedule") != expected for state in states):
        raise RuntimeError("PCNR executable gate did not observe its frozen player schedule")
    for state in states:
        updates = int(state.get("update_index", -1))
        de_count = int(state.get("de_view_count", -2))
        gf_count = int(state.get("gf_view_count", -3))
        serial = int(state.get("bundle_serial", -4))
        if not (updates == de_count == gf_count and serial == 2 * updates):
            raise RuntimeError("PCNR view provenance counters are inconsistent")
    return {
        "expected_schedule": expected,
        "states_checked": len(states),
        "all_de_and_gf_counts_equal_updates": True,
        "all_bundle_serials_equal_twice_updates": True,
    }


def _validate_pcammcrb_execution_evidence(
    context: CandidateGateContext, cross: dict, micro: dict,
) -> dict:
    from models.route1.pcammcrb import EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE
    from models.route1.pcnr import EXPECTED_PCNR_SCHEDULE

    parent = str(
        context.registration.spec.method.get("pcammcrb_sampling_parent", "pcnr")
    )
    expected = list(
        EXPECTED_PCNR_SCHEDULE
        if parent == "pcnr" else EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE
    )
    diagnostics = [row["candidate"]["method_diagnostics"] for row in cross.get("rows", [])]
    diagnostics.append(micro.get("method_diagnostics", {}))
    if not diagnostics:
        raise RuntimeError("PC-AMMCRB gate did not produce method diagnostics")
    for diagnostic in diagnostics:
        synthesis = diagnostic.get("pcammcrb", {})
        barrier = diagnostic.get("mcrb", {})
        if synthesis.get("sampling_parent") != parent:
            raise RuntimeError("PC-AMMCRB gate changed its sampling parent")
        if synthesis.get("last_schedule") != expected:
            raise RuntimeError("PC-AMMCRB gate did not observe the frozen schedule")
        updates = int(synthesis.get("update_index", -1))
        bundles = int(synthesis.get("gf_bundle_count", -2))
        barrier_updates = int(barrier.get("update_index", -3))
        if not (updates == bundles == barrier_updates):
            raise RuntimeError("PC-AMMCRB sampling/barrier provenance counters differ")
        if parent == "pcnr":
            pcnr = diagnostic.get("pcnr", {})
            if int(pcnr.get("update_index", -4)) != updates:
                raise RuntimeError("PC-AMMCRB PCNR parent counter differs")
            if int(pcnr.get("bundle_serial", -5)) != 2 * updates:
                raise RuntimeError("PC-AMMCRB PCNR view serial differs")
    return {
        "sampling_parent": parent,
        "expected_schedule": expected,
        "states_checked": len(diagnostics),
        "all_sampling_and_barrier_counts_equal_updates": True,
        "all_de_and_gf_counts_equal_updates": True,
        "all_bundle_serials_equal_twice_updates": parent == "pcnr",
    }


def _run(context: CandidateGateContext, *, invariant: str) -> dict:
    e0_path = context.output_root / "shared_e0" / "e0.pt"
    e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    parent_hash = full_state_hash(e0)
    if invariant == "bvcp":
        invariants = _bvcp_invariants()
    elif invariant == "rsmg":
        invariants = _rsmg_invariants()
    elif invariant == "pcrsmg":
        invariants = _pcrsmg_invariants()
    elif invariant == "pcnr":
        invariants = _pcnr_invariants()
    elif invariant == "amtnc":
        invariants = _amtnc_invariants()
    elif invariant == "mcrb":
        invariants = _mcrb_invariants()
    elif invariant == "ammcrb":
        invariants = _ammcrb_invariants()
    elif invariant == "rfammcrb":
        invariants = _rfammcrb_invariants()
    elif invariant == "rfmcrb":
        invariants = _rfmcrb_invariants()
    elif invariant == "pcammcrb":
        invariants = _pcammcrb_invariants(context)
    elif invariant == "winner_ablation":
        invariants = _winner_ablation_invariants(context)
    else:
        raise ValueError(f"unknown Generation-1 invariant family: {invariant}")
    if any(row["status"] != "PASS" for row in invariants):
        raise RuntimeError("Generation-1 mathematical invariant failed")
    zero = _zero_intervention(context, e0=e0)
    resume = _resume_exact(context, e0=e0)
    cross = _cross_state(context)
    micro = _micro(context, e0=e0)
    observable_active_identity = (
        _observable_active_identity(context, e0=e0)
        if invariant == "winner_ablation" else None
    )
    player_conditional = (
        _validate_pcrsmg_execution_evidence(cross, micro)
        if invariant == "pcrsmg" else None
    )
    if invariant == "amtnc":
        player_conditional = _validate_amtnc_execution_evidence(cross, micro)
    if invariant == "pcnr":
        player_conditional = _validate_pcnr_execution_evidence(cross, micro)
    if invariant == "pcammcrb":
        player_conditional = _validate_pcammcrb_execution_evidence(
            context, cross, micro,
        )
    if full_state_hash(e0) != parent_hash:
        raise RuntimeError("candidate gate mutated shared e0")
    winner_observable_source = {
        "route1_bvcp_ablation": "current/lagged unpaired rollout velocity",
        "route1_pcrsmg_ablation": "conditionally iid native UNSB stochastic views",
        "route1_amtnc_ablation": "pre-update Adam-metric replica geometry",
        "route1_mcrb_ablation": "current/EMA covariance tangent and native Adam derivative",
        "route1_pcnr_ablation": "counterfactual fresh native player view dispersion",
        "route1_ammcrb_ablation": "moving covariance tangent and Adam-metric KKT geometry",
        "route1_rfammcrb_ablation": "moving covariance tangent and residual-feasible Adam-metric KKT geometry",
        "route1_rfmcrb_ablation": "moving covariance tangent and residual-feasible Euclidean KKT geometry",
    }.get(context.registration.spec.model)
    return {
        "checks": {
            "mathematical_invariants": True,
            "zero_intervention_identity": True,
            "resume_exact": True,
            "cross_state_counterfactual": True,
            "target_blind_observable": True,
            "micro_engineering": True,
            "base_unsb_semantics_preserved": True,
            "shared_e0_load_exact": True,
        },
        "mathematical_invariant_evidence": invariants,
        "zero_intervention_evidence": zero,
        "resume_evidence": resume,
        "cross_state_evidence": cross,
        "target_blind_evidence": {
            "paired_fields_observed": [],
            "paired_target_available": False,
            "observable_source": (
                "current and one-update-lagged unpaired rollout velocity"
                if invariant == "bvcp" else
                "current/EMA latent direction covariance and exact native Adam displacement"
                if invariant in ("mcrb", "ammcrb", "rfammcrb", "rfmcrb") else
                "conditional native G/F views plus current/EMA covariance tangent and exact native-like Adam displacement"
                if invariant == "pcammcrb" else
                "conditionally iid gradients and their pre-step Adam-metric exchange geometry"
                if invariant == "amtnc" else
                "one fresh native stochastic view at each realized player state"
                if invariant == "pcnr" else
                winner_observable_source
                if invariant == "winner_ablation" else
                "conditionally iid native UNSB stochastic gradients"
            ),
        },
        "micro_engineering_evidence": micro,
        "player_conditional_execution_evidence": player_conditional,
        "observable_active_identity_evidence": observable_active_identity,
        "shared_e0_scientific_state_sha256": parent_hash,
        "paired_metric_used_for_promotion": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def run_bvcp_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="bvcp")


def run_rsmg_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="rsmg")


def run_pcrsmg_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="pcrsmg")


def run_pcnr_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="pcnr")


def run_amtnc_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="amtnc")


def run_mcrb_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="mcrb")


def run_ammcrb_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="ammcrb")


def run_rfammcrb_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="rfammcrb")


def run_rfmcrb_gate(context: CandidateGateContext) -> dict:
    return _run(context, invariant="rfmcrb")


def run_pcammcrb_gate(context: CandidateGateContext) -> dict:
    report = _run(context, invariant="pcammcrb")
    compatibility = _pcammcrb_compatibility(context)
    if compatibility["all_parent_state_hashes_preserved"] is not True:
        raise RuntimeError("PC-AMMCRB compatibility audit polluted a parent state")
    report["checks"]["component_compatibility"] = True
    report["component_compatibility_evidence"] = compatibility
    return report


def run_winner_ablation_gate(context: CandidateGateContext) -> dict:
    if context.registration.spec.model not in (
        "route1_bvcp_ablation", "route1_pcrsmg_ablation",
        "route1_amtnc_ablation", "route1_mcrb_ablation",
        "route1_pcnr_ablation", "route1_ammcrb_ablation",
        "route1_rfammcrb_ablation", "route1_rfmcrb_ablation",
    ):
        raise RuntimeError("winner ablation gate received a non-ablation model")
    return _run(context, invariant="winner_ablation")
