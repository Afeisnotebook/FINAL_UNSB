"""Residual-feasible Euclidean Moving Covariance Rate Barrier."""

from __future__ import annotations

import torch

from models.route1.mcrb import MCRBMixin
from models.route1.rfammcrb import (
    ResidualFeasibleAdamMetricProjection,
    project_actual_displacement_residual_feasible_adam_metric,
)


def project_actual_displacement_residual_feasible(
    native_displacement: list[torch.Tensor],
    defect_gradient: list[torch.Tensor | None],
    *,
    eps: float = 1e-24,
) -> tuple[list[torch.Tensor], ResidualFeasibleAdamMetricProjection]:
    """Project in Euclidean geometry without a fixed absolute margin."""
    inverse_metric = [torch.ones_like(value) for value in native_displacement]
    return project_actual_displacement_residual_feasible_adam_metric(
        native_displacement,
        defect_gradient,
        inverse_metric,
        eps=eps,
    )


class RFMCRBMixin(MCRBMixin):
    """MCRB whose represented KKT projection is scale-safe and feasible."""

    def _mcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "rfmcrb_enable", True))

    def _generator_optimizer_step(self):
        if not self._mcrb_enabled():
            return super()._generator_optimizer_step()
        parameters = [parameter for parameter in self.netG.parameters() if parameter.requires_grad]
        defect, observable = self._mcrb_direction_covariance_defect()
        tangents = list(torch.autograd.grad(
            defect, parameters, allow_unused=True, retain_graph=False,
        ))
        before = [parameter.detach().clone() for parameter in parameters]
        super(MCRBMixin, self)._generator_optimizer_step()
        native = [parameter.detach() - old for parameter, old in zip(parameters, before)]
        projected, diag = project_actual_displacement_residual_feasible(
            native,
            tangents,
            eps=float(getattr(self.opt, "rfmcrb_projection_epsilon", 1e-24)),
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
            "metric": "euclidean",
            "projection_representation": "residual_feasible_without_absolute_margin",
            "native_defect_directional_derivative": diag.native_defect_directional_derivative,
            "projected_defect_directional_derivative": diag.projected_defect_directional_derivative,
            "native_displacement_l2": diag.native_displacement_l2,
            "correction_l2": diag.correction_l2,
            "projection_coefficient": diag.projection_coefficient,
            "residual_refinement_steps": diag.residual_refinement_steps,
            "intervened": bool(diag.unsafe),
        }
        self._update_mcrb_teacher()
