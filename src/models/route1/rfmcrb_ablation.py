"""Mechanism ablations for the residual-feasible Euclidean barrier."""

from __future__ import annotations

import copy
import math

import torch

from models.route1.mcrb import MCRBMixin
from models.route1.mcrb_ablation import norm_matched_negative_tangent
from models.route1.rfmcrb import project_actual_displacement_residual_feasible


class RFMCRBAblationMixin(MCRBMixin):
    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(
            self.opt, "rfmcrb_ablation_role", "proposal_only",
        ))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown RF-MCRB ablation role: {role}")
        return role

    def _mcrb_enabled(self) -> bool:
        return self._ablation_enabled()

    @staticmethod
    def _l2(values: list[torch.Tensor]) -> float:
        total = torch.zeros((), dtype=torch.float64, device=values[0].device)
        for value in values:
            total.add_(value.detach().double().square().sum())
        return math.sqrt(max(float(total.item()), 0.0))

    @staticmethod
    def _dot(
        left: list[torch.Tensor], right: list[torch.Tensor | None],
    ) -> float:
        total = torch.zeros((), dtype=torch.float64, device=left[0].device)
        for first, second in zip(left, right):
            if second is not None:
                total.add_((first.detach().double() * second.detach().double()).sum())
        return float(total.item())

    def _generator_optimizer_step(self):
        if not self._ablation_enabled():
            return super(MCRBMixin, self)._generator_optimizer_step()
        parameters = [
            parameter for parameter in self.netG.parameters()
            if parameter.requires_grad
        ]
        defect, observable = self._mcrb_direction_covariance_defect()
        tangents = list(torch.autograd.grad(
            defect, parameters, allow_unused=True, retain_graph=False,
        ))
        before = [parameter.detach().clone() for parameter in parameters]
        super(MCRBMixin, self)._generator_optimizer_step()
        native = [
            parameter.detach() - old
            for parameter, old in zip(parameters, before)
        ]
        epsilon = float(getattr(
            self.opt, "rfmcrb_projection_epsilon", 1e-24,
        ))
        safe, safe_diag = project_actual_displacement_residual_feasible(
            native, tangents, eps=epsilon,
        )
        proposal, proposal_diag = norm_matched_negative_tangent(
            native, tangents, eps=epsilon,
        )
        role = self._ablation_role()
        applied = role == "proposal_only" and bool(proposal_diag["applied"])
        committed = proposal if applied else native
        if applied:
            with torch.no_grad():
                for parameter, old, displacement in zip(
                    parameters, before, committed,
                ):
                    parameter.copy_(old + displacement.to(parameter))
        correction_l2 = self._l2([
            value - original for value, original in zip(committed, native)
        ])
        self._mcrb_update_index += 1
        self._mcrb_eligible_count += int(bool(proposal_diag["applied"]))
        self._mcrb_intervention_count += int(applied)
        self._mcrb_correction_l2_sum += correction_l2
        self._mcrb_last = {
            **observable,
            "defect": float(defect.detach().item()),
            "metric": "euclidean",
            "projection_representation": (
                "residual_feasible_without_absolute_margin"
            ),
            "native_defect_directional_derivative": self._dot(native, tangents),
            "projected_defect_directional_derivative": self._dot(
                committed, tangents,
            ),
            "full_safe_projection_directional_derivative": (
                safe_diag.projected_defect_directional_derivative
            ),
            "native_displacement_l2": self._l2(native),
            "correction_l2": correction_l2,
            "intervened": applied,
            "ablation_role": role,
            "full_safe_projection_differs_from_native": safe is not native,
            "full_safe_projection_residual_refinements": (
                safe_diag.residual_refinement_steps
            ),
        }
        self._update_mcrb_teacher()

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._ablation_enabled():
            return state
        if self._ablation_role() == "observable_only":
            observer = state.pop("mcrb")
            state["route1_observer"] = {
                "family": "rfmcrb",
                "role": "observable_only",
                **observer,
            }
        else:
            state["mcrb"]["ablation_family"] = "rfmcrb"
            state["mcrb"]["ablation_role"] = "proposal_only"
        return state

    def load_extra_training_state(self, state):
        value = copy.deepcopy(state or {})
        if self._ablation_enabled() and self._ablation_role() == "observable_only":
            observer = value.pop("route1_observer", None)
            if observer is not None:
                if (
                    observer.get("family") != "rfmcrb"
                    or observer.get("role") != "observable_only"
                ):
                    raise RuntimeError(
                        "RF-MCRB observable-only checkpoint role mismatch"
                    )
                value["mcrb"] = {
                    key: item for key, item in observer.items()
                    if key not in ("family", "role")
                }
        super().load_extra_training_state(value)
        if self._ablation_enabled() and self._ablation_role() == "proposal_only":
            saved = value.get("mcrb")
            if saved is not None and (
                saved.get("ablation_family") != "rfmcrb"
                or saved.get("ablation_role") != "proposal_only"
            ):
                raise RuntimeError(
                    "RF-MCRB proposal-only checkpoint role mismatch"
                )
