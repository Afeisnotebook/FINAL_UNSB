"""Source-bound proposal and observable ablations for AM-MCRB."""

from __future__ import annotations

import copy
import math

import torch

from models.route1.ammcrb import (
    _adam_inverse_metric,
    project_actual_displacement_adam_metric,
)
from models.route1.mcrb import MCRBMixin


def adam_metric_norm_matched_negative_normal(
    native: list[torch.Tensor],
    tangents: list[torch.Tensor | None],
    inverse_metric: list[torch.Tensor],
    *,
    eps: float,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Return ``-P a`` with the native displacement's Adam-metric norm."""
    if not native or not (len(native) == len(tangents) == len(inverse_metric)):
        raise ValueError("AM-MCRB proposal structures must be nonempty and equal")
    first = native[0]
    native_metric_sq = torch.zeros((), dtype=torch.float64, device=first.device)
    normal_metric_sq = torch.zeros_like(native_metric_sq)
    native_l2_sq = torch.zeros_like(native_metric_sq)
    for displacement, tangent, metric_inverse in zip(native, tangents, inverse_metric):
        if displacement.shape != metric_inverse.shape:
            raise ValueError("AM-MCRB proposal inverse metric shape differs")
        if not bool(torch.isfinite(metric_inverse).all().item()) or not bool(
            (metric_inverse > 0).all().item()
        ):
            raise RuntimeError("AM-MCRB proposal inverse metric must be positive")
        native_metric_sq.add_((
            displacement.detach().double().square()
            / metric_inverse.detach().double()
        ).sum())
        native_l2_sq.add_(displacement.detach().double().square().sum())
        if tangent is not None:
            if tangent.shape != displacement.shape:
                raise ValueError("AM-MCRB proposal tangent shape differs")
            normal_metric_sq.add_((
                tangent.detach().double().square()
                * metric_inverse.detach().double()
            ).sum())
    native_metric = float(native_metric_sq.item())
    normal_metric = float(normal_metric_sq.item())
    native_l2 = float(native_l2_sq.item())
    if not math.isfinite(native_metric + normal_metric + native_l2):
        raise RuntimeError("AM-MCRB proposal geometry is nonfinite")
    if native_metric <= 0.0 or normal_metric <= float(eps):
        return native, {
            "applied": False,
            "native_metric_norm": math.sqrt(max(native_metric, 0.0)),
            "normal_metric_norm": math.sqrt(max(normal_metric, 0.0)),
            "native_displacement_l2": math.sqrt(max(native_l2, 0.0)),
        }
    scale = math.sqrt(native_metric / normal_metric)
    proposal = [
        torch.zeros_like(displacement)
        if tangent is None else -scale * metric_inverse.to(displacement) * tangent.to(displacement)
        for displacement, tangent, metric_inverse in zip(native, tangents, inverse_metric)
    ]
    return proposal, {
        "applied": True,
        "native_metric_norm": math.sqrt(native_metric),
        "normal_metric_norm": math.sqrt(normal_metric),
        "native_displacement_l2": math.sqrt(native_l2),
    }


class AMMCRBAblationMixin(MCRBMixin):
    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(self.opt, "ammcrb_ablation_role", "proposal_only"))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown AM-MCRB ablation role: {role}")
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
    def _dot(left, right) -> float:
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
        # Bypass MCRB and realize the exact native Adam displacement once.
        super(MCRBMixin, self)._generator_optimizer_step()
        native = [parameter.detach() - old for parameter, old in zip(parameters, before)]
        inverse_metric = _adam_inverse_metric(parameters, self.optimizer_G)
        epsilon = float(getattr(self.opt, "ammcrb_projection_epsilon", 1e-24))
        safe, safe_diag = project_actual_displacement_adam_metric(
            native, tangents, inverse_metric, eps=epsilon,
        )
        proposal, proposal_diag = adam_metric_norm_matched_negative_normal(
            native, tangents, inverse_metric, eps=epsilon,
        )
        role = self._ablation_role()
        applied = role == "proposal_only" and bool(proposal_diag["applied"])
        committed = proposal if applied else native
        if applied:
            with torch.no_grad():
                for parameter, old, displacement in zip(parameters, before, committed):
                    parameter.copy_(old + displacement.to(parameter))
        correction_l2 = self._l2([
            value - original for value, original in zip(committed, native)
        ])
        native_dot = self._dot(native, tangents)
        committed_dot = self._dot(committed, tangents)
        self._mcrb_update_index += 1
        self._mcrb_eligible_count += int(bool(proposal_diag["applied"]))
        self._mcrb_intervention_count += int(applied)
        self._mcrb_correction_l2_sum += correction_l2
        self._mcrb_last = {
            **observable,
            "defect": float(defect.detach().item()),
            "metric": "normalized_post_step_adam_inverse_root_second_moment",
            "native_defect_directional_derivative": native_dot,
            "projected_defect_directional_derivative": committed_dot,
            "full_safe_projection_directional_derivative": (
                safe_diag.projected_defect_directional_derivative
            ),
            "native_displacement_l2": self._l2(native),
            "correction_l2": correction_l2,
            "intervened": applied,
            "ablation_role": role,
            "proposal_metric_norm": proposal_diag["native_metric_norm"],
            "full_safe_projection_differs_from_native": safe is not native,
        }
        self._update_mcrb_teacher()

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._ablation_enabled():
            return state
        if self._ablation_role() == "observable_only":
            observer = state.pop("mcrb")
            state["route1_observer"] = {
                "family": "ammcrb",
                "role": "observable_only",
                **observer,
            }
        else:
            state["mcrb"]["ablation_role"] = "proposal_only"
        return state

    def load_extra_training_state(self, state):
        value = copy.deepcopy(state or {})
        if self._ablation_enabled() and self._ablation_role() == "observable_only":
            observer = value.pop("route1_observer", None)
            if observer is not None:
                if observer.get("family") != "ammcrb" or observer.get("role") != "observable_only":
                    raise RuntimeError("AM-MCRB observable-only checkpoint role mismatch")
                value["mcrb"] = {
                    key: item for key, item in observer.items()
                    if key not in ("family", "role")
                }
        super().load_extra_training_state(value)
        if self._ablation_enabled() and self._ablation_role() == "proposal_only":
            saved = value.get("mcrb")
            if saved is not None and saved.get("ablation_role") != "proposal_only":
                raise RuntimeError("AM-MCRB proposal-only checkpoint role mismatch")
