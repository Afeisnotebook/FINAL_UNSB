"""Gradient projection and gate helpers for HJ-PatchNCE."""

from __future__ import annotations

import torch


class _ProjectConflictingGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feature, bridge_direction, strength, eps, update_mode):
        if feature.shape != bridge_direction.shape:
            raise ValueError("feature and bridge_direction must have identical shapes")
        ctx.save_for_backward(bridge_direction.detach(), strength.detach())
        ctx.eps = float(eps)
        ctx.update_mode = str(update_mode)
        return feature

    @staticmethod
    def backward(ctx, grad_output):
        bridge_direction, strength = ctx.saved_tensors
        norm_sq = bridge_direction.square().sum(dim=1, keepdim=True)
        alignment = (grad_output * bridge_direction).sum(dim=1, keepdim=True)
        conflicting = alignment.clamp_min(0.0)
        removed = conflicting / norm_sq.clamp_min(ctx.eps) * bridge_direction
        projected = grad_output - strength * removed
        active = (alignment > 0.0) & (norm_sq > ctx.eps) & (strength > 0.0)

        if ctx.update_mode in ("norm_preserve", "norm_redistribute"):
            original_norm = grad_output.square().sum(dim=1, keepdim=True).sqrt()
            projected_norm = projected.square().sum(dim=1, keepdim=True).sqrt()
            restored = projected * (original_norm / projected_norm.clamp_min(ctx.eps))
            projected = torch.where(active, restored, projected)
        if ctx.update_mode in ("safe_redistribute", "norm_redistribute"):
            safe = ~active
            safe_count = safe.to(grad_output.dtype).sum().clamp_min(1.0)
            net_removed = grad_output - projected
            compensation = net_removed.sum(dim=0, keepdim=True) / safe_count
            projected = projected + safe.to(grad_output.dtype) * compensation
        elif ctx.update_mode not in ("remove", "norm_preserve"):
            raise ValueError(f"unknown conflicting-gradient update mode: {ctx.update_mode}")

        return projected, None, None, None, None


def project_conflicting_gradient(
    feature,
    bridge_direction,
    strength=1.0,
    eps=1e-6,
    update_mode="remove",
):
    """Forward identity; backward removes the positive-aligned conflict component."""
    strength = torch.as_tensor(strength, dtype=feature.dtype, device=feature.device).detach()
    if strength.ndim == 0:
        strength = strength.reshape(1, 1)
    elif strength.ndim == 1 and strength.shape[0] == feature.shape[0]:
        strength = strength.unsqueeze(1)
    if strength.ndim != 2 or strength.shape not in ((1, 1), (feature.shape[0], 1)):
        raise ValueError("strength must be scalar, N, or N x 1")
    strength = strength.clamp(0.0, 1.0)
    update_mode = str(update_mode).lower()
    if update_mode not in ("remove", "norm_preserve", "safe_redistribute", "norm_redistribute"):
        raise ValueError(f"unknown conflicting-gradient update mode: {update_mode}")
    return _ProjectConflictingGradient.apply(feature, bridge_direction, strength, eps, update_mode)


def conservative_central_delta(center_loss, descent_loss, ascent_loss, step):
    """Keep only directional conflicts supported by one- and two-sided probes."""
    step = max(float(step), 1e-12)
    one_sided = (descent_loss - center_loss) / step
    central = (descent_loss - ascent_loss) / (2.0 * step)
    return torch.minimum(one_sided, central), one_sided, central


def positive_quantile_gate(values, batch_size, quantile):
    """Top-quantile positive gate, computed per image without touching RNG."""
    batched = values.detach().reshape(batch_size, -1)
    gates = []
    for row in batched:
        positive = row[row > 0.0]
        if positive.numel() == 0:
            gates.append(torch.zeros_like(row))
            continue
        threshold = torch.quantile(positive, float(quantile))
        gates.append(((row > 0.0) & (row >= threshold)).to(row.dtype))
    return torch.stack(gates, dim=0)


def apply_factorial_structure_control(direction, gate, control):
    """Factor risk-location and projection-direction alignment without RNG."""
    if direction.ndim != 3 or gate.ndim != 2:
        raise ValueError("direction must be BPC and gate must be BP")
    if direction.shape[:2] != gate.shape:
        raise ValueError("direction and gate batch/patch dimensions must match")
    control = str(control).lower()
    shift = max(1, direction.shape[1] // 2)
    rolled_direction = torch.roll(direction, shifts=shift, dims=1)
    rolled_gate = torch.roll(gate, shifts=shift, dims=1)

    if control == "true":
        effective_gate = gate
        controlled = direction * effective_gate.unsqueeze(-1)
    elif control == "roll":
        effective_gate = rolled_gate
        controlled = torch.roll(direction * gate.unsqueeze(-1), shifts=shift, dims=1)
    elif control == "gate_roll":
        effective_gate = rolled_gate
        controlled = direction * effective_gate.unsqueeze(-1)
    elif control == "direction_roll":
        effective_gate = gate
        controlled = rolled_direction * effective_gate.unsqueeze(-1)
    elif control == "uniform":
        effective_gate = torch.ones_like(gate)
        gated = direction * gate.unsqueeze(-1)
        controlled = gated.mean(dim=1, keepdim=True).expand_as(gated)
    else:
        raise ValueError(f"unknown structure projection control: {control}")
    return controlled, effective_gate


def apply_absolute_evidence_gate(gate, scale, minimum):
    """Disable an image/layer intervention below an absolute evidence threshold."""
    if gate.ndim != 2:
        raise ValueError("gate must be a BP tensor")
    scale = torch.as_tensor(scale, dtype=gate.dtype, device=gate.device).detach()
    if scale.ndim == 1:
        scale = scale.unsqueeze(1)
    if scale.shape != (gate.shape[0], 1):
        raise ValueError("scale must contain one value per image")
    minimum = max(float(minimum), 0.0)
    return gate * (scale >= minimum).to(gate.dtype)
