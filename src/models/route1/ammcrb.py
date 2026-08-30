"""Adam-Metric Moving Covariance Rate Barrier (AM-MCRB).

The original MCRB projects unsafe native generator displacements in ordinary
parameter-space Euclidean distance.  AM-MCRB keeps the same target-blind
moving covariance defect and exact identity condition, but computes the
closest feasible displacement in the diagonal metric induced by Adam's
post-native-step second moment.  Adam moments themselves remain untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from models.route1.mcrb import MCRBMixin


@dataclass(frozen=True)
class AdamMetricBarrierProjection:
    unsafe: bool
    native_defect_directional_derivative: float
    projected_defect_directional_derivative: float
    correction_l2: float
    metric_correction_l2: float
    native_displacement_l2: float


def project_actual_displacement_adam_metric(
    native_displacement: list[torch.Tensor],
    defect_gradient: list[torch.Tensor | None],
    inverse_metric: list[torch.Tensor],
    *,
    eps: float = 1e-24,
) -> tuple[list[torch.Tensor], AdamMetricBarrierProjection]:
    """Solve ``min 1/2 ||d-d0||_H^2`` subject to ``<a,d> <= 0``.

    ``inverse_metric`` is ``H^-1`` and must be strictly positive.  The closed
    form correction is ``-lambda H^-1 a`` with
    ``lambda=<a,d0>/<a,H^-1 a>``.  A common rescaling of ``H^-1`` cancels.
    """
    if not native_displacement or not (
        len(native_displacement) == len(defect_gradient) == len(inverse_metric)
    ):
        raise ValueError("AM-MCRB displacement, tangent and metric structures differ")
    first = native_displacement[0]
    dot_tensor = torch.zeros((), dtype=torch.float64, device=first.device)
    denominator_tensor = torch.zeros_like(dot_tensor)
    displacement_tensor = torch.zeros_like(dot_tensor)
    for displacement, tangent, metric_inverse in zip(
        native_displacement, defect_gradient, inverse_metric
    ):
        if displacement.shape != metric_inverse.shape:
            raise ValueError("AM-MCRB inverse metric shape differs")
        if not bool(torch.isfinite(metric_inverse).all().item()) or not bool(
            (metric_inverse > 0).all().item()
        ):
            raise RuntimeError("AM-MCRB inverse metric must be finite and positive")
        displacement_tensor.add_(displacement.detach().double().square().sum())
        if tangent is None:
            continue
        if displacement.shape != tangent.shape:
            raise ValueError("AM-MCRB displacement and tangent shapes differ")
        dot_tensor.add_((displacement.detach().double() * tangent.detach().double()).sum())
        denominator_tensor.add_((
            tangent.detach().double().square() * metric_inverse.detach().double()
        ).sum())
    dot = float(dot_tensor.item())
    denominator = float(denominator_tensor.item())
    displacement_sq = float(displacement_tensor.item())
    if not math.isfinite(dot + denominator + displacement_sq):
        raise RuntimeError("AM-MCRB projection geometry is nonfinite")
    if dot <= 0.0 or denominator <= float(eps):
        return native_displacement, AdamMetricBarrierProjection(
            unsafe=False,
            native_defect_directional_derivative=dot,
            projected_defect_directional_derivative=dot,
            correction_l2=0.0,
            metric_correction_l2=0.0,
            native_displacement_l2=math.sqrt(max(displacement_sq, 0.0)),
        )
    low_precision_eps = max(
        (
            float(torch.finfo(value.dtype).eps)
            for value in native_displacement
            if torch.is_floating_point(value) and value.dtype != torch.float64
        ),
        default=0.0,
    )
    numeric_margin = 8.0 * low_precision_eps * max(1.0, abs(dot))
    coefficient = (dot + numeric_margin) / denominator
    projected = [
        displacement if tangent is None else displacement - (
            metric_inverse.to(displacement) * tangent.to(displacement) * coefficient
        )
        for displacement, tangent, metric_inverse in zip(
            native_displacement, defect_gradient, inverse_metric
        )
    ]
    projected_dot_tensor = torch.zeros_like(dot_tensor)
    correction_sq = torch.zeros_like(dot_tensor)
    for original, value, tangent in zip(native_displacement, projected, defect_gradient):
        correction_sq.add_((value.detach().double() - original.detach().double()).square().sum())
        if tangent is not None:
            projected_dot_tensor.add_((value.detach().double() * tangent.detach().double()).sum())
    projected_dot = float(projected_dot_tensor.item())
    tolerance = 1e-10 * max(1.0, abs(dot))
    if projected_dot > tolerance:
        raise RuntimeError("AM-MCRB represented displacement violates its half-space")
    return projected, AdamMetricBarrierProjection(
        unsafe=True,
        native_defect_directional_derivative=dot,
        projected_defect_directional_derivative=projected_dot,
        correction_l2=math.sqrt(max(float(correction_sq.item()), 0.0)),
        metric_correction_l2=abs(coefficient) * math.sqrt(denominator),
        native_displacement_l2=math.sqrt(max(displacement_sq, 0.0)),
    )


def _adam_inverse_metric(parameters, optimizer) -> list[torch.Tensor]:
    groups = {}
    for group in optimizer.param_groups:
        epsilon = float(group.get("eps", 1e-8))
        for parameter in group["params"]:
            groups[id(parameter)] = epsilon
    values = []
    global_max = 0.0
    for parameter in parameters:
        if id(parameter) not in groups:
            raise RuntimeError("AM-MCRB generator parameter has no Adam group")
        state = optimizer.state.get(parameter, {})
        second_moment = state.get("exp_avg_sq")
        value = (
            torch.ones_like(parameter)
            if second_moment is None
            else second_moment.detach().sqrt().add(groups[id(parameter)]).reciprocal()
        )
        if not bool(torch.isfinite(value).all().item()) or not bool((value > 0).all().item()):
            raise RuntimeError("AM-MCRB Adam inverse metric is invalid")
        global_max = max(global_max, float(value.max().item()))
        values.append(value)
    if not math.isfinite(global_max) or global_max <= 0.0:
        raise RuntimeError("AM-MCRB Adam inverse metric has no positive scale")
    # A common scale cancels analytically and normalization avoids unnecessary
    # float32 range loss in the represented correction.
    return [value / global_max for value in values]


class AMMCRBMixin(MCRBMixin):
    def _mcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "ammcrb_enable", True))

    def _generator_optimizer_step(self):
        if not self._mcrb_enabled():
            return super()._generator_optimizer_step()
        parameters = [parameter for parameter in self.netG.parameters() if parameter.requires_grad]
        defect, observable = self._mcrb_direction_covariance_defect()
        tangents = list(torch.autograd.grad(
            defect, parameters, allow_unused=True, retain_graph=False,
        ))
        before = [parameter.detach().clone() for parameter in parameters]
        # Bypass MCRB's Euclidean projection and realize native Adam once.
        super(MCRBMixin, self)._generator_optimizer_step()
        native = [parameter.detach() - old for parameter, old in zip(parameters, before)]
        inverse_metric = _adam_inverse_metric(parameters, self.optimizer_G)
        projected, diag = project_actual_displacement_adam_metric(
            native,
            tangents,
            inverse_metric,
            eps=float(getattr(self.opt, "ammcrb_projection_epsilon", 1e-24)),
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
            "native_defect_directional_derivative": diag.native_defect_directional_derivative,
            "projected_defect_directional_derivative": diag.projected_defect_directional_derivative,
            "native_displacement_l2": diag.native_displacement_l2,
            "correction_l2": diag.correction_l2,
            "metric_correction_l2": diag.metric_correction_l2,
            "intervened": bool(diag.unsafe),
        }
        self._update_mcrb_teacher()
