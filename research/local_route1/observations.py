"""Target-blind actual-update geometry for local route-1 causal audits.

The audit compares *committed optimizer displacements*, not raw loss gradients.
This matters for UNSB because the optimizer state, discriminator/critic updates,
and method co-state are part of the state whose long-horizon compatibility is
being investigated.
"""

from __future__ import annotations

import math
from typing import Mapping

import torch


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    first = first.detach().double().reshape(-1)
    second = second.detach().double().reshape(-1)
    left = float(torch.linalg.vector_norm(first).item())
    right = float(torch.linalg.vector_norm(second).item())
    denominator = left * right
    if denominator == 0.0:
        return 0.0
    value = float(torch.dot(first, second).item() / denominator)
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def tensor_geometry(reference: torch.Tensor, proposal: torch.Tensor) -> dict[str, float]:
    """Summarize a native displacement and a proposal displacement."""
    reference = reference.detach().double().reshape(-1)
    proposal = proposal.detach().double().reshape(-1)
    correction = proposal - reference
    return {
        "reference_norm": float(torch.linalg.vector_norm(reference).item()),
        "proposal_norm": float(torch.linalg.vector_norm(proposal).item()),
        "correction_norm": float(torch.linalg.vector_norm(correction).item()),
        "reference_proposal_cosine": _cosine(reference, proposal),
        "correction_reference_cosine": _cosine(correction, reference),
    }


def _floating_displacements(
    start: Mapping[str, torch.Tensor], end: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if tuple(start) != tuple(end):
        raise ValueError("state-dict identities differ")
    return {
        key: end[key].detach().cpu() - start[key].detach().cpu()
        for key in start
        if torch.is_floating_point(start[key])
    }


def state_dict_update_geometry(
    before: Mapping[str, torch.Tensor],
    reference_after: Mapping[str, torch.Tensor],
    proposal_after: Mapping[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Compare actual netG displacements globally and by top-level block."""
    reference = _floating_displacements(before, reference_after)
    proposal = _floating_displacements(before, proposal_after)
    if tuple(reference) != tuple(proposal) or not reference:
        raise ValueError("floating state-dict identities differ or are empty")
    global_geometry = tensor_geometry(
        torch.cat([reference[key].reshape(-1) for key in reference]),
        torch.cat([proposal[key].reshape(-1) for key in reference]),
    )
    blocks: dict[str, list[str]] = {}
    for key in reference:
        blocks.setdefault(key.split(".", 1)[0], []).append(key)
    block_geometry = {
        block: tensor_geometry(
            torch.cat([reference[key].reshape(-1) for key in keys]),
            torch.cat([proposal[key].reshape(-1) for key in keys]),
        )
        for block, keys in sorted(blocks.items())
    }
    return global_geometry, block_geometry


def state_dict_delta_cosine(
    first_start: Mapping[str, torch.Tensor],
    first_end: Mapping[str, torch.Tensor],
    second_start: Mapping[str, torch.Tensor],
    second_end: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Compare two displacements that may originate at adjacent states."""
    first = _floating_displacements(first_start, first_end)
    second = _floating_displacements(second_start, second_end)
    if tuple(first) != tuple(second) or not first:
        raise ValueError("floating state-dict identities differ or are empty")
    left = torch.cat([first[key].reshape(-1) for key in first]).double()
    right = torch.cat([second[key].reshape(-1) for key in first]).double()
    return {
        "first_norm": float(torch.linalg.vector_norm(left).item()),
        "second_norm": float(torch.linalg.vector_norm(right).item()),
        "cosine": _cosine(left, right),
    }


def component_directional_derivatives(
    *,
    before: Mapping[str, torch.Tensor],
    reference_after: Mapping[str, torch.Tensor],
    proposal_after: Mapping[str, torch.Tensor],
    native_component_gradients: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, float]]:
    """Evaluate native first-step component gradients along proposal correction.

    The correction is ``u_i - u_0`` in parameter-displacement coordinates.
    Values are descriptive target-blind first-order quantities; they are not
    paired-quality labels and are never exposed to a paired controller.
    """
    native = _floating_displacements(before, reference_after)
    proposal = _floating_displacements(before, proposal_after)
    correction = {key: proposal[key] - native[key] for key in native}
    result: dict[str, dict[str, float]] = {}
    for component, gradients in sorted(native_component_gradients.items()):
        dot = 0.0
        grad_sq = 0.0
        correction_sq = 0.0
        matched = 0
        for key, gradient in gradients.items():
            if key not in correction:
                continue
            grad = gradient.detach().cpu().double()
            corr = correction[key].detach().cpu().double()
            dot += float(torch.sum(grad * corr).item())
            grad_sq += float(torch.sum(grad * grad).item())
            correction_sq += float(torch.sum(corr * corr).item())
            matched += 1
        denominator = (grad_sq * correction_sq) ** 0.5
        result[component] = {
            "gradient_dot_correction": dot,
            "gradient_correction_cosine": 0.0 if denominator == 0.0 else max(-1.0, min(1.0, dot / denominator)),
            "gradient_norm": grad_sq ** 0.5,
            "correction_norm_on_parameters": correction_sq ** 0.5,
            "matched_parameter_tensors": float(matched),
            "interpretation": "negative dot predicts a first-order decrease of this native loss component",
        }
    return result
