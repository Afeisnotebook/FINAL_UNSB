"""Long-horizon proposal/observable ablations for BVCP."""

from __future__ import annotations

import copy

import torch

from models.route1.bvcp import BVCPMixin, minimum_velocity_chord_endpoint


class BVCPAblationMixin(BVCPMixin):
    """Reuse the exact lagged-state machinery while changing only its role."""

    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(self.opt, "bvcp_ablation_role", "proposal_only"))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown BVCP ablation role: {role}")
        return role

    def _bvcp_enabled(self) -> bool:
        return self._ablation_enabled()

    def _rollout_endpoint(self, rollout_net, x, time_idx, z, *, stream):
        if not self._ablation_enabled():
            # Skip BVCPMixin and dispatch the canonical SBModel hook.
            return super(BVCPMixin, self)._rollout_endpoint(
                rollout_net, x, time_idx, z, stream=stream,
            )
        current = rollout_net(x, time_idx, z)
        lagged_net = self._ensure_bvcp_lagged()
        with torch.no_grad():
            lagged = lagged_net(x, time_idx, z)
            _, diag = minimum_velocity_chord_endpoint(
                x, current, lagged,
                eps=float(getattr(self.opt, "bvcp_root_epsilon", 1e-12)),
            )
        self._bvcp_eligible_transition_count += int(diag.eligible)
        self._bvcp_endpoint_count += int(x.shape[0])
        self._bvcp_last = {
            "role": self._ablation_role(),
            "stream": str(stream),
            "time_index": int(time_idx.reshape(-1)[0].detach().item()),
            "current_rms": diag.current_rms,
            "lagged_rms": diag.lagged_rms,
            "velocity_growth_margin": diag.current_rms - diag.lagged_rms,
        }
        if self._ablation_role() == "proposal_only":
            self._bvcp_intervention_count += int(x.shape[0])
            self._bvcp_lambda_sum += float(x.shape[0])
            return lagged
        # Observable-only returns the exact current endpoint.  The lagged copy
        # and counters are observer state and never participate in an update.
        return current

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._ablation_enabled():
            return state
        if self._ablation_role() == "observable_only":
            observer = state.pop("bvcp")
            observer["family"] = "bvcp"
            observer["role"] = "observable_only"
            state["route1_observer"] = observer
        else:
            state["bvcp"]["role"] = "proposal_only"
        return state

    def load_extra_training_state(self, state):
        value = copy.deepcopy(state or {})
        if self._ablation_enabled() and self._ablation_role() == "observable_only":
            observer = value.pop("route1_observer", None)
            if observer is not None:
                if observer.get("family") != "bvcp" or observer.get("role") != "observable_only":
                    raise RuntimeError("BVCP observable-only checkpoint role mismatch")
                observer.pop("family", None)
                observer.pop("role", None)
                value["bvcp"] = observer
        super().load_extra_training_state(value)
        if self._ablation_enabled() and self._ablation_role() == "proposal_only":
            saved = (state or {}).get("bvcp")
            if saved is not None and saved.get("role") != "proposal_only":
                raise RuntimeError("BVCP proposal-only checkpoint role mismatch")

