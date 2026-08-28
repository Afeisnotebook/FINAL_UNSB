"""Clean-room HJ-PatchNCE core."""

from .projection import (
    apply_absolute_evidence_gate,
    apply_factorial_structure_control,
    conservative_central_delta,
    positive_quantile_gate,
    project_conflicting_gradient,
)
from .structure import source_structure_direction

__all__ = [
    "apply_absolute_evidence_gate",
    "apply_factorial_structure_control",
    "conservative_central_delta",
    "positive_quantile_gate",
    "project_conflicting_gradient",
    "source_structure_direction",
]
