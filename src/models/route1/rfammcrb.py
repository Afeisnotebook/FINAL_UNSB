"""Residual-feasible Adam-Metric Moving Covariance Rate Barrier.

This module is a new scientific identity rather than a silent edit of the
already-running AM-MCRB operator.  It implements the registered Adam-metric
half-space projection without the fixed absolute float32 margin that can
dominate a small native optimizer displacement.  The exact coefficient is
formed in float64; only when the displacement *as represented in its parameter
dtype* remains on the unsafe side is the coefficient advanced by a
dtype-relative ULP and the measured residual.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from models.route1.ammcrb import _adam_inverse_metric
from models.route1.mcrb import MCRBMixin


@dataclass(frozen=True)
class ResidualFeasibleAdamMetricProjection:
    unsafe: bool
    native_defect_directional_derivative: float
    projected_defect_directional_derivative: float
    correction_l2: float
    metric_correction_l2: float
    native_displacement_l2: float
    projection_coefficient: float
    residual_refinement_steps: int


def _represented_projection(
    native_displacement: list[torch.Tensor],
    defect_gradient: list[torch.Tensor | None],
    inverse_metric: list[torch.Tensor],
    coefficient: float,
) -> list[torch.Tensor]:
    return [
        displacement if tangent is None else displacement - (
            metric_inverse.to(displacement)
            * tangent.to(displacement)
            * coefficient
        )
        for displacement, tangent, metric_inverse in zip(
            native_displacement, defect_gradient, inverse_metric
        )
    ]


def _represented_directional_derivative(
    displacement: list[torch.Tensor],
    defect_gradient: list[torch.Tensor | None],
) -> float:
    first = displacement[0]
    value = torch.zeros((), dtype=torch.float64, device=first.device)
    for block, tangent in zip(displacement, defect_gradient):
        if tangent is not None:
            value.add_((block.detach().double() * tangent.detach().double()).sum())
    return float(value.item())


def _next_parameter_dtype_coefficient(
    coefficient: float, tensors: list[torch.Tensor],
) -> float:
    """Advance by at least one ULP in every floating parameter dtype.

    A Python ``nextafter`` is too small to change a coefficient after it is
    represented as float32.  This operation is relative to the coefficient's
    own scale; unlike the superseded absolute margin it cannot dominate merely
    because the native update is small.
    """
    values = [coefficient]
    for tensor in tensors:
        if not torch.is_floating_point(tensor):
            continue
        represented = torch.tensor(coefficient, dtype=tensor.dtype, device="cpu")
        advanced = torch.nextafter(
            represented,
            torch.tensor(float("inf"), dtype=tensor.dtype, device="cpu"),
        )
        values.append(float(advanced.double().item()))
    result = max(values)
    if not math.isfinite(result) or result <= coefficient:
        result = math.nextafter(coefficient, math.inf)
    if not math.isfinite(result) or result <= coefficient:
        raise RuntimeError("RF-AMMCRB could not advance its represented coefficient")
    return result


def project_actual_displacement_residual_feasible_adam_metric(
    native_displacement: list[torch.Tensor],
    defect_gradient: list[torch.Tensor | None],
    inverse_metric: list[torch.Tensor],
    *,
    eps: float = 1e-24,
    maximum_residual_refinements: int = 8,
) -> tuple[list[torch.Tensor], ResidualFeasibleAdamMetricProjection]:
    """Return the represented Adam-metric closest feasible displacement.

    In exact arithmetic this solves

    ``argmin_d 1/2 ||d-d0||_H^2`` subject to ``<a,d> <= 0``.

    Float64 is used for the two global scalar products.  No fixed numerical
    margin is added.  If dtype rounding leaves a positive residual, the exact
    residual correction ``residual / <a,H^-1 a>`` plus one relative parameter-
    dtype ULP is applied and rechecked.  Failure to represent a feasible point
    is fatal rather than silently accepting an unsafe or unbounded correction.
    """
    if not native_displacement or not (
        len(native_displacement) == len(defect_gradient) == len(inverse_metric)
    ):
        raise ValueError("RF-AMMCRB displacement, tangent and metric structures differ")
    if int(maximum_residual_refinements) < 1:
        raise ValueError("RF-AMMCRB requires at least one residual refinement")
    first = native_displacement[0]
    dot_tensor = torch.zeros((), dtype=torch.float64, device=first.device)
    denominator_tensor = torch.zeros_like(dot_tensor)
    displacement_tensor = torch.zeros_like(dot_tensor)
    for displacement, tangent, metric_inverse in zip(
        native_displacement, defect_gradient, inverse_metric
    ):
        if displacement.shape != metric_inverse.shape:
            raise ValueError("RF-AMMCRB inverse metric shape differs")
        if not bool(torch.isfinite(metric_inverse).all().item()) or not bool(
            (metric_inverse > 0).all().item()
        ):
            raise RuntimeError("RF-AMMCRB inverse metric must be finite and positive")
        displacement_tensor.add_(displacement.detach().double().square().sum())
        if tangent is None:
            continue
        if displacement.shape != tangent.shape:
            raise ValueError("RF-AMMCRB displacement and tangent shapes differ")
        dot_tensor.add_((displacement.detach().double() * tangent.detach().double()).sum())
        denominator_tensor.add_((
            tangent.detach().double().square() * metric_inverse.detach().double()
        ).sum())
    dot = float(dot_tensor.item())
    denominator = float(denominator_tensor.item())
    displacement_sq = float(displacement_tensor.item())
    if not math.isfinite(dot + denominator + displacement_sq):
        raise RuntimeError("RF-AMMCRB projection geometry is nonfinite")
    if dot <= 0.0 or denominator <= float(eps):
        return native_displacement, ResidualFeasibleAdamMetricProjection(
            unsafe=False,
            native_defect_directional_derivative=dot,
            projected_defect_directional_derivative=dot,
            correction_l2=0.0,
            metric_correction_l2=0.0,
            native_displacement_l2=math.sqrt(max(displacement_sq, 0.0)),
            projection_coefficient=0.0,
            residual_refinement_steps=0,
        )

    coefficient = dot / denominator
    if not math.isfinite(coefficient) or coefficient <= 0.0:
        raise RuntimeError("RF-AMMCRB analytical coefficient is invalid")
    projected = _represented_projection(
        native_displacement, defect_gradient, inverse_metric, coefficient,
    )
    projected_dot = _represented_directional_derivative(projected, defect_gradient)
    refinements = 0
    while projected_dot > 0.0 and refinements < int(maximum_residual_refinements):
        candidate = coefficient + projected_dot / denominator
        if not math.isfinite(candidate):
            raise RuntimeError("RF-AMMCRB residual correction is nonfinite")
        coefficient = _next_parameter_dtype_coefficient(
            max(coefficient, candidate), native_displacement,
        )
        projected = _represented_projection(
            native_displacement, defect_gradient, inverse_metric, coefficient,
        )
        projected_dot = _represented_directional_derivative(
            projected, defect_gradient,
        )
        refinements += 1
    if projected_dot > 0.0:
        raise RuntimeError(
            "RF-AMMCRB represented displacement remains infeasible after "
            f"{refinements} residual refinements: {projected_dot:.17g}"
        )

    correction_sq = torch.zeros_like(dot_tensor)
    for original, value in zip(native_displacement, projected):
        correction_sq.add_((
            value.detach().double() - original.detach().double()
        ).square().sum())
    return projected, ResidualFeasibleAdamMetricProjection(
        unsafe=True,
        native_defect_directional_derivative=dot,
        projected_defect_directional_derivative=projected_dot,
        correction_l2=math.sqrt(max(float(correction_sq.item()), 0.0)),
        metric_correction_l2=abs(coefficient) * math.sqrt(denominator),
        native_displacement_l2=math.sqrt(max(displacement_sq, 0.0)),
        projection_coefficient=coefficient,
        residual_refinement_steps=refinements,
    )


class RFAMMCRBMixin(MCRBMixin):
    """AM-MCRB with a scale-safe, represented-feasible projection."""

    def _mcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "rfammcrb_enable", True))

    def _generator_optimizer_step(self):
        if not self._mcrb_enabled():
            return super()._generator_optimizer_step()
        parameters = [parameter for parameter in self.netG.parameters() if parameter.requires_grad]
        defect, observable = self._mcrb_direction_covariance_defect()
        tangents = list(torch.autograd.grad(
            defect, parameters, allow_unused=True, retain_graph=False,
        ))
        before = [parameter.detach().clone() for parameter in parameters]
        # Bypass MCRB's Euclidean projector and realize native Adam exactly once.
        super(MCRBMixin, self)._generator_optimizer_step()
        native = [parameter.detach() - old for parameter, old in zip(parameters, before)]
        inverse_metric = _adam_inverse_metric(parameters, self.optimizer_G)
        projected, diag = project_actual_displacement_residual_feasible_adam_metric(
            native,
            tangents,
            inverse_metric,
            eps=float(getattr(self.opt, "rfammcrb_projection_epsilon", 1e-24)),
        )
        if diag.unsafe:
            with torch.no_grad():
                for parameter, old, displacement in zip(parameters, before, projected):
                    parameter.copy_(old + displacement.to(parameter))
            self._mcrb_intervention_count += 1
        self._mcrb_update_index += 1
        self._mcrb_eligible_count += int(diag.unsafe)
        self._mcrb_correction_l2_sum += float(diag.correction_l2)
        self._mcrb_last = {
            **observable,
            "defect": float(defect.detach().item()),
            "metric": "normalized_post_step_adam_inverse_root_second_moment",
            "projection_representation": "residual_feasible_without_absolute_margin",
            "native_defect_directional_derivative": diag.native_defect_directional_derivative,
            "projected_defect_directional_derivative": diag.projected_defect_directional_derivative,
            "native_displacement_l2": diag.native_displacement_l2,
            "correction_l2": diag.correction_l2,
            "metric_correction_l2": diag.metric_correction_l2,
            "projection_coefficient": diag.projection_coefficient,
            "residual_refinement_steps": diag.residual_refinement_steps,
            "intervened": bool(diag.unsafe),
        }
        self._update_mcrb_teacher()
