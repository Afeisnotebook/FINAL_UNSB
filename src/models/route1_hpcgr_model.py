"""Physical-horizon conditional G/F resampling for route-1 discovery.

HPCGR composes two operators that act on different mathematical objects:

* HNEK defines the physical-horizon residual bridge game; and
* the PC-RSMG proposal-only estimator replaces only the joint G/F stochastic
  gradient by the mean of two conditionally iid views at the realized
  post-D/E state.

The composition is deliberately exposed with component roles.  They are used
by executable gates to prove that the estimator-only role is the existing
PC-RSMG proposal, the coordinate-only role is the frozen HNEK game, and the
observable-only role is pathwise HNEK after excluding diagnostics.
"""

from __future__ import annotations

from models.hnek.hnek_search import (
    HnekSearchConfig,
    install_hnek_search_model,
    set_hnek_search_active,
)
from models.route1.pcrsmg_ablation import PCRSMGAblationMixin
from models.sb_model import SBModel


_ROLES = ("full", "coordinate_only", "estimator_only", "observable_only")


class Route1HpcgrModel(PCRSMGAblationMixin, SBModel):
    """HNEK base field with target-blind conditional G/F replication."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--route1_hpcgr_enable", type=_str2bool, default=True)
        parser.add_argument("--hpcgr_role", choices=_ROLES, default="full")
        parser.add_argument("--hnek_gamma", type=float, default=0.25)
        parser.add_argument(
            "--hnek_coord", choices=["residual", "endpoint"], default="residual",
        )
        parser.add_argument(
            "--hnek_horizon_mode", choices=["physical", "index", "mix"],
            default="physical",
        )
        parser.add_argument(
            "--hnek_partial", choices=["all", "entropy_only", "endpoint_only"],
            default="all",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcrsmg_ablation_state()
        cfg = HnekSearchConfig(
            gamma=float(getattr(opt, "hnek_gamma", 0.25)),
            coord=str(getattr(opt, "hnek_coord", "residual")),
            horizon_mode=str(getattr(opt, "hnek_horizon_mode", "physical")),
            partial=str(getattr(opt, "hnek_partial", "all")),
        )
        install_hnek_search_model(self, cfg)
        if not self._hpcgr_hnek_enabled():
            set_hnek_search_active(self, False)

    def _hpcgr_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_hpcgr_enable", True))

    def _hpcgr_role(self) -> str:
        role = str(getattr(self.opt, "hpcgr_role", "full"))
        if role not in _ROLES:
            raise ValueError(f"unknown HPCGR role: {role}")
        return role

    def _hpcgr_hnek_enabled(self) -> bool:
        return self._hpcgr_enabled() and self._hpcgr_role() in (
            "full", "coordinate_only", "observable_only",
        )

    def _ablation_enabled(self) -> bool:
        return self._hpcgr_enabled() and self._hpcgr_role() in (
            "full", "estimator_only", "observable_only",
        )

    def _ablation_role(self) -> str:
        return (
            "observable_only"
            if self._hpcgr_role() == "observable_only"
            else "proposal_only"
        )

    def get_extra_training_state(self):
        # PCRSMGAblationMixin delegates to SBModel and adds exactly the same
        # proposal/observer state used by the already-audited parent.  Do not
        # add a hybrid-only key: component equivalence is then byte-auditable.
        state = PCRSMGAblationMixin.get_extra_training_state(self)
        if self._hpcgr_hnek_enabled():
            state["hnek_active"] = bool(getattr(self, "hnek_active", True))
        return state

    def load_extra_training_state(self, state):
        PCRSMGAblationMixin.load_extra_training_state(self, state)
        desired = self._hpcgr_hnek_enabled()
        if desired:
            desired = bool((state or {}).get("hnek_active", True))
        if desired != bool(getattr(self, "hnek_active", False)):
            set_hnek_search_active(self, desired)


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
