"""Target-blind numerical audit for the AM-MCRB projection semantics."""

from __future__ import annotations

from typing import Any

import torch

from models.route1.ammcrb import project_actual_displacement_adam_metric
from models.route1.rfammcrb import (
    project_actual_displacement_residual_feasible_adam_metric,
)


SCALES = (1e-2, 1e-4, 1e-6, 1e-8)
TANGENT_SCALE = 1e-8


def deterministic_scale_counterexample() -> list[dict[str, Any]]:
    """Show whether projection cost remains proportional to a native update."""
    rows: list[dict[str, Any]] = []
    for scale in SCALES:
        native = [torch.tensor([scale], dtype=torch.float32)]
        tangent = [torch.tensor([TANGENT_SCALE], dtype=torch.float32)]
        inverse_metric = [torch.ones(1, dtype=torch.float32)]
        old, old_diag = project_actual_displacement_adam_metric(
            native, tangent, inverse_metric,
        )
        repaired, repaired_diag = (
            project_actual_displacement_residual_feasible_adam_metric(
                native, tangent, inverse_metric,
            )
        )
        rows.append({
            "native_displacement": scale,
            "defect_tangent": TANGENT_SCALE,
            "superseded_projected_displacement": float(old[0].item()),
            "superseded_correction_l2": old_diag.correction_l2,
            "superseded_correction_to_native_ratio": old_diag.correction_l2 / scale,
            "residual_feasible_projected_displacement": float(repaired[0].item()),
            "residual_feasible_correction_l2": repaired_diag.correction_l2,
            "residual_feasible_correction_to_native_ratio": (
                repaired_diag.correction_l2 / scale
            ),
            "residual_refinement_steps": repaired_diag.residual_refinement_steps,
        })
    return rows


def invariant_summary() -> dict[str, Any]:
    rows = deterministic_scale_counterexample()
    return {
        "superseded_max_correction_to_native_ratio": max(
            row["superseded_correction_to_native_ratio"] for row in rows
        ),
        "repaired_max_correction_to_native_ratio": max(
            row["residual_feasible_correction_to_native_ratio"] for row in rows
        ),
        "repaired_all_represented_feasible": all(
            row["residual_feasible_projected_displacement"] <= 0.0
            for row in rows
        ),
        "paired_target_read": False,
        "rows": rows,
    }
