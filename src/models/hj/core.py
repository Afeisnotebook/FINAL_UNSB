"""Minimal self-contained HJ-PatchNCE structure-projection loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .projection import (
    apply_absolute_evidence_gate,
    apply_factorial_structure_control,
    conservative_central_delta,
    positive_quantile_gate,
    project_conflicting_gradient,
)
from .structure import source_structure_direction


def finite_step_window_active(step: int, start: int, duration: int) -> bool:
    """Return whether ``step`` is inside a deterministic half-open HJ window.

    A non-positive duration preserves the historical open-ended behavior.
    """
    if step < start:
        return False
    return duration <= 0 or step < start + duration


@dataclass
class StructureProjectConfig:
    """Frozen best-branch defaults for continuous layer0-HJ."""

    direction: str = "joint"
    scales: str = "1,2,4"
    step: float = 0.01
    quantile: float = 0.75
    gate_quantile: float = 0.75
    strength: float = 0.5
    boundary_scale: float = 0.001
    min_risk: float = 0.05
    min_delta: float = 0.0
    probe_mode: str = "central_consensus"
    control: str = "true"
    amplitude: str = "constant"
    update_mode: str = "remove"
    eps: float = 1e-6
    nce_temperature: float = 0.07
    start_epoch: int = 5
    direction_alpha: float = 0.0
    random_seed: int = 2026


def _as_batched(feat, batch_size):
    if feat.ndim != 2:
        raise ValueError("expected [B*N, D] features")
    if feat.shape[0] % batch_size != 0:
        raise ValueError("feature count is not divisible by batch size")
    return feat.view(batch_size, feat.shape[0] // batch_size, feat.shape[1])


def correspondence_statistics(feat_q, feat_k, batch_size, temperature=0.07):
    """Detached PatchNCE correspondence reliability statistics."""
    q = _as_batched(feat_q.detach(), batch_size)
    k = _as_batched(feat_k.detach(), batch_size)
    similarity = torch.bmm(q, k.transpose(1, 2))
    num_patches = similarity.shape[1]
    diagonal = torch.eye(num_patches, device=similarity.device, dtype=torch.bool)[None]
    positive = similarity.diagonal(dim1=1, dim2=2)
    if num_patches > 1:
        alternatives = similarity.masked_fill(diagonal, -torch.inf)
        hard_negative = alternatives.amax(dim=2)
        reverse_hard_negative = alternatives.amax(dim=1)
    else:
        hard_negative = positive
        reverse_hard_negative = positive
    margin = positive - hard_negative
    reverse_margin = positive - reverse_hard_negative
    reciprocal_margin = torch.minimum(margin, reverse_margin)
    return {"reciprocal_margin": reciprocal_margin.flatten().detach()}


def correspondence_boundary_instability(reciprocal_margin, scale=0.05, eps=1e-6):
    """High uncertainty only near the exact/alternative match boundary."""
    probability = torch.sigmoid(reciprocal_margin.detach() / max(float(scale), float(eps)))
    return (4.0 * probability * (1.0 - probability)).detach()


def structure_project_nce_step(
    *,
    feat_q: torch.Tensor,
    feat_k: torch.Tensor,
    criterion,
    source: torch.Tensor,
    tgt_nce: torch.Tensor,
    probe_fn,
    batch_size: int,
    cfg: StructureProjectConfig,
    lambda_nce: float = 1.0,
    schedule_weight: float = 1.0,
):
    """One PatchNCE layer: forward unchanged, backward projects structural conflict.

    ``feat_q`` / ``feat_k`` are pooled ``[B*N, D]`` query/key features.
    ``probe_fn(tgt)`` returns pooled ``[B*N, D]`` query features for a perturbed
    target, so the function stays independent of the concrete encoder stack.
    """
    eps = float(cfg.eps)
    step = max(float(cfg.step), 1e-6)

    structure_gradient = source_structure_direction(
        tgt_nce, source.detach(), direction=cfg.direction, scales=cfg.scales
    )
    gradient_rms = structure_gradient.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
    normalized_gradient = structure_gradient / gradient_rms.clamp_min(eps)
    normalized_gradient = normalized_gradient.clamp(-5.0, 5.0)
    if cfg.direction_alpha > 0.0:
        g = torch.Generator(device=normalized_gradient.device)
        g.manual_seed(int(cfg.random_seed))
        d_rand = torch.randn(
            normalized_gradient.shape,
            generator=g,
            device=normalized_gradient.device,
            dtype=normalized_gradient.dtype,
        )
        rand_rms = d_rand.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        d_rand = d_rand / rand_rms.clamp_min(eps)
        blended = (1.0 - cfg.direction_alpha) * normalized_gradient + cfg.direction_alpha * d_rand
        blend_rms = blended.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        normalized_gradient = blended / blend_rms.clamp_min(eps)
        normalized_gradient = normalized_gradient.clamp(-5.0, 5.0)
    perturbed_tgt = (tgt_nce.detach() - step * normalized_gradient).clamp(-1.0, 1.0)
    opposite_tgt = None
    if cfg.probe_mode == "central_consensus":
        opposite_tgt = (tgt_nce.detach() + step * normalized_gradient).clamp(-1.0, 1.0)

    with torch.no_grad():
        perturbed_q = probe_fn(perturbed_tgt)
        opposite_q = probe_fn(opposite_tgt) if opposite_tgt is not None else None

    loss_raw = criterion(feat_q, feat_k) * lambda_nce
    with torch.no_grad():
        perturbed_loss = criterion(perturbed_q, feat_k) * lambda_nce
        if opposite_q is None:
            delta = (perturbed_loss - loss_raw.detach()) / step
            probe_agreement = delta.new_tensor(1.0)
        else:
            opposite_loss = criterion(opposite_q, feat_k) * lambda_nce
            delta, one_sided, central = conservative_central_delta(
                loss_raw.detach(), perturbed_loss, opposite_loss, step
            )
            probe_agreement = ((one_sided > 0.0) == (central > 0.0)).float().mean()

        positive_delta = delta.clamp_min(0.0)
        batched = positive_delta.view(batch_size, -1)
        scale = torch.quantile(batched, cfg.quantile, dim=1, keepdim=True)
        risk = (batched / scale.clamp_min(eps)).clamp_max(1.0)

        if cfg.boundary_scale > 0.0:
            stats = correspondence_statistics(
                feat_q, feat_k, batch_size, temperature=cfg.nce_temperature
            )
            boundary_risk = correspondence_boundary_instability(
                stats["reciprocal_margin"], scale=cfg.boundary_scale, eps=eps
            ).view(batch_size, -1)
            risk = torch.sqrt((risk * boundary_risk).clamp_min(0.0))
            gate = positive_quantile_gate(risk.flatten(), batch_size, cfg.gate_quantile)
            gate = gate * (risk >= cfg.min_risk).to(gate.dtype)
        else:
            gate = positive_quantile_gate(delta, batch_size, cfg.gate_quantile)

        gate = apply_absolute_evidence_gate(gate, scale, cfg.min_delta)
        applicability = (scale >= cfg.min_delta).to(risk.dtype).mean()

    if opposite_q is None:
        direction = ((perturbed_q - feat_q.detach()) / step).detach()
    else:
        direction = ((perturbed_q - opposite_q) / (2.0 * step)).detach()

    batched_direction = direction.view(batch_size, -1, direction.shape[-1])
    controlled_direction, effective_gate = apply_factorial_structure_control(
        batched_direction, gate, cfg.control
    )

    if cfg.amplitude == "risk":
        project_dose = (cfg.strength * schedule_weight * risk).reshape(-1, 1)
    else:
        project_dose = torch.as_tensor(
            cfg.strength * schedule_weight, dtype=direction.dtype, device=direction.device
        )

    direction = controlled_direction.reshape_as(direction)
    projected_q = project_conflicting_gradient(
        feat_q, direction, strength=project_dose, eps=eps, update_mode=cfg.update_mode
    )
    projected_loss = criterion(projected_q, feat_k) * lambda_nce

    diag = {
        "probe_agreement": float(probe_agreement.detach().item()),
        "risk_mean": float(risk.detach().mean().item()),
        "risk_positive": float((delta > 0.0).float().mean().item()),
        "gate_active": float(effective_gate.detach().mean().item()),
        "applicability": float(applicability.detach().item()),
        "project_dose": float(
            project_dose.detach().mean().item()
            if project_dose.ndim > 0
            else project_dose.item()
        ),
    }
    return projected_loss.mean(), diag
