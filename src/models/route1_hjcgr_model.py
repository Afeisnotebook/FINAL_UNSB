"""HJ-objective conditional G/F resampling for route-1 discovery.

The HJ structure projection defines the stochastic G/F objective.  Conditional
resampling then averages two iid gradients of that *same* objective at the
realized post-D/E state.  The controller bookkeeping performed while each loss
graph is built is reduced to one unbiased per-update transition, so replica
count cannot silently change the HJ physical-epoch state.
"""

from __future__ import annotations

import copy
import numbers

from models.hj.model import SBModelHJPatchNCE
from models.route1.pcrsmg_ablation import PCRSMGAblationMixin


_ROLES = ("full", "objective_only", "estimator_only", "observable_only")
_HJ_VIEW_STATE = (
    "_hj_step_in_epoch",
    "_hj_gate_sum",
    "_hj_risk_sum",
    "_hj_probe_sum",
    "_hj_risk_positive_sum",
    "_hj_sb_grad_norm",
    "_hj_active_optimizer_steps",
)
_ADDITIVE_INTEGER_STATE = {"_hj_step_in_epoch", "_hj_active_optimizer_steps"}
_ADDITIVE_FLOAT_STATE = {
    "_hj_gate_sum", "_hj_risk_sum", "_hj_probe_sum", "_hj_risk_positive_sum",
}


def reduce_hj_replica_transitions(baseline: dict, transitions: list[dict]) -> dict:
    """Reduce two loss-graph side effects to one unbiased HJ update record."""
    if len(transitions) != 2:
        raise RuntimeError("HJCGR requires exactly two HJ G/F state transitions")
    committed = copy.deepcopy(baseline)
    for name in _ADDITIVE_INTEGER_STATE:
        deltas = [int(row[name]) - int(baseline[name]) for row in transitions]
        if len(set(deltas)) != 1:
            raise RuntimeError(f"HJCGR replica integer transition differs: {name}")
        committed[name] = int(baseline[name]) + deltas[0]
    for name in _ADDITIVE_FLOAT_STATE:
        deltas = [float(row[name]) - float(baseline[name]) for row in transitions]
        committed[name] = float(baseline[name]) + sum(deltas) / len(deltas)
    norm_values = [row["_hj_sb_grad_norm"] for row in transitions]
    if all(isinstance(value, numbers.Real) for value in norm_values):
        committed["_hj_sb_grad_norm"] = sum(float(v) for v in norm_values) / 2.0
    else:
        committed["_hj_sb_grad_norm"] = copy.deepcopy(norm_values[0])
    return committed


class Route1HjcgrModel(PCRSMGAblationMixin, SBModelHJPatchNCE):
    """Frozen continuous HJ objective with conditional two-view G/F mean."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModelHJPatchNCE.modify_commandline_options(parser, is_train)
        parser.add_argument("--route1_hjcgr_enable", type=_str2bool, default=True)
        parser.add_argument("--hjcgr_role", choices=_ROLES, default="full")
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcrsmg_ablation_state()
        self._hjcgr_capture_replica_state = False
        self._hjcgr_replica_baseline = None
        self._hjcgr_replica_transitions = []

    def _hjcgr_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_hjcgr_enable", True))

    def _hjcgr_role(self) -> str:
        role = str(getattr(self.opt, "hjcgr_role", "full"))
        if role not in _ROLES:
            raise ValueError(f"unknown HJCGR role: {role}")
        return role

    def _hjcgr_hj_enabled(self) -> bool:
        return self._hjcgr_enabled() and self._hjcgr_role() in (
            "full", "objective_only", "observable_only",
        )

    def _ablation_enabled(self) -> bool:
        return self._hjcgr_enabled() and self._hjcgr_role() in (
            "full", "estimator_only", "observable_only",
        )

    def _ablation_role(self) -> str:
        return (
            "observable_only"
            if self._hjcgr_role() == "observable_only"
            else "proposal_only"
        )

    def _hj_active(self):
        return self._hjcgr_hj_enabled() and super()._hj_active()

    def _hjcgr_view_state(self) -> dict:
        return {name: copy.deepcopy(getattr(self, name)) for name in _HJ_VIEW_STATE}

    def _hjcgr_restore_view_state(self, state: dict) -> None:
        for name in _HJ_VIEW_STATE:
            setattr(self, name, copy.deepcopy(state[name]))

    def compute_G_loss(self):
        if not self._hjcgr_capture_replica_state:
            return super().compute_G_loss()
        baseline = self._hjcgr_replica_baseline
        if baseline is None or self._hjcgr_view_state() != baseline:
            raise RuntimeError("HJCGR replica loss did not start from its common HJ state")
        loss = super().compute_G_loss()
        self._hjcgr_replica_transitions.append(self._hjcgr_view_state())
        self._hjcgr_restore_view_state(baseline)
        return loss

    def _hjcgr_commit_replica_state(self) -> None:
        baseline = self._hjcgr_replica_baseline
        transitions = self._hjcgr_replica_transitions
        if baseline is None:
            raise RuntimeError("HJCGR replica baseline is missing")
        committed = reduce_hj_replica_transitions(baseline, transitions)
        self._hjcgr_restore_view_state(committed)

    def _proposal_only(self) -> None:
        self._hjcgr_replica_baseline = self._hjcgr_view_state()
        self._hjcgr_replica_transitions = []
        self._hjcgr_capture_replica_state = True
        try:
            super()._proposal_only()
            self._hjcgr_commit_replica_state()
        finally:
            self._hjcgr_capture_replica_state = False
            self._hjcgr_replica_baseline = None
            self._hjcgr_replica_transitions = []

    def get_extra_training_state(self):
        state = PCRSMGAblationMixin.get_extra_training_state(self)
        if not self._hjcgr_hj_enabled():
            state.pop("hj_controller", None)
        return state

    def load_extra_training_state(self, state):
        PCRSMGAblationMixin.load_extra_training_state(self, state)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
