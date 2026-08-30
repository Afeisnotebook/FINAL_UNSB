"""Source-bound long-horizon ablations for MCRB.

``proposal_only`` replaces the native generator displacement by a norm-matched
negative moving-covariance tangent whenever that tangent is nonzero.  It
isolates the moving proposal without the native-safe half-space projection.
``observable_only`` computes the same target-blind defect and its derivative
but commits the native Adam displacement unchanged.  Its EMA and diagnostics
are serialized only under ``route1_observer``.
"""

from __future__ import annotations

import copy
import math

import torch

from models.route1.mcrb import MCRBMixin


def norm_matched_negative_tangent(
    native: list[torch.Tensor], tangents: list[torch.Tensor | None], *, eps: float,
) -> tuple[list[torch.Tensor], dict[str, float | bool]]:
    """Return the pure covariance proposal with the native displacement norm."""
    if not native or len(native) != len(tangents):
        raise ValueError("MCRB proposal structures must be nonempty and equal")
    native_sq = sum(
        float((value.detach().double() * value.detach().double()).sum().item())
        for value in native
    )
    tangent_sq = sum(
        float((value.detach().double() * value.detach().double()).sum().item())
        for value in tangents if value is not None
    )
    if not math.isfinite(native_sq + tangent_sq):
        raise RuntimeError("MCRB proposal geometry is nonfinite")
    if tangent_sq <= float(eps) or native_sq <= 0.0:
        return native, {
            "applied": False,
            "native_displacement_l2": math.sqrt(max(native_sq, 0.0)),
            "tangent_l2": math.sqrt(max(tangent_sq, 0.0)),
        }
    scale = -math.sqrt(native_sq / tangent_sq)
    proposal = [
        torch.zeros_like(displacement)
        if tangent is None else tangent.detach() * scale
        for displacement, tangent in zip(native, tangents)
    ]
    return proposal, {
        "applied": True,
        "native_displacement_l2": math.sqrt(native_sq),
        "tangent_l2": math.sqrt(tangent_sq),
    }


class MCRBAblationMixin(MCRBMixin):
    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(self.opt, "mcrb_ablation_role", "proposal_only"))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown MCRB ablation role: {role}")
        return role

    def _mcrb_enabled(self) -> bool:
        return self._ablation_enabled()

    @staticmethod
    def _l2(values) -> float:
        total = torch.zeros((), dtype=torch.float64, device=values[0].device)
        for value in values:
            total.add_((value.detach().double() * value.detach().double()).sum())
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
            return super()._generator_optimizer_step()
        parameters = [
            parameter for parameter in self.netG.parameters()
            if parameter.requires_grad
        ]
        defect, observable = self._mcrb_direction_covariance_defect()
        tangents = list(torch.autograd.grad(
            defect, parameters, allow_unused=True, retain_graph=False,
        ))
        before = [parameter.detach().clone() for parameter in parameters]
        # Skip MCRBMixin: both ablations first realize the exact native Adam step.
        super(MCRBMixin, self)._generator_optimizer_step()
        native = [parameter.detach() - old for parameter, old in zip(parameters, before)]
        native_l2 = self._l2(native)
        native_dot = self._dot(native, tangents)
        proposal, proposal_diag = norm_matched_negative_tangent(
            native, tangents,
            eps=float(getattr(self.opt, "mcrb_projection_epsilon", 1e-24)),
        )
        applied = (
            bool(proposal_diag["applied"])
            and self._ablation_role() == "proposal_only"
        )
        correction_l2 = 0.0
        projected_dot = native_dot
        if applied:
            with torch.no_grad():
                for parameter, old, displacement in zip(parameters, before, proposal):
                    parameter.copy_(old + displacement.to(parameter))
            correction_l2 = self._l2([
                proposed - original for proposed, original in zip(proposal, native)
            ])
            projected_dot = self._dot(proposal, tangents)
            applied = True

        self._mcrb_update_index += 1
        self._mcrb_eligible_count += int(bool(proposal_diag["applied"]))
        self._mcrb_intervention_count += int(applied)
        self._mcrb_correction_l2_sum += correction_l2
        self._mcrb_last = {
            **observable,
            "defect": float(defect.detach().item()),
            "native_defect_directional_derivative": native_dot,
            "projected_defect_directional_derivative": projected_dot,
            "native_displacement_l2": native_l2,
            "correction_l2": correction_l2,
            "intervened": applied,
            "ablation_role": self._ablation_role(),
        }
        self._update_mcrb_teacher()

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._ablation_enabled():
            return state
        if self._ablation_role() == "observable_only":
            observer = state.pop("mcrb")
            state["route1_observer"] = {
                "family": "mcrb",
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
                if observer.get("family") != "mcrb" or observer.get("role") != "observable_only":
                    raise RuntimeError("MCRB observable-only checkpoint role mismatch")
                value["mcrb"] = {
                    key: item for key, item in observer.items()
                    if key not in ("family", "role")
                }
        super().load_extra_training_state(value)
        if self._ablation_enabled() and self._ablation_role() == "proposal_only":
            saved = value.get("mcrb")
            if saved is not None and saved.get("ablation_role") != "proposal_only":
                raise RuntimeError("MCRB proposal-only checkpoint role mismatch")
