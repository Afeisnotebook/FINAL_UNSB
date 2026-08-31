"""HJ-objective player-conditional native G/F resampling.

This is the one-view gain-source control for HJCGR.  It keeps the frozen
continuous HJ objective, commits native one-view D/E, and draws exactly one
fresh HJ G/F view after those opponent updates.  Unlike HJCGR it does not
average replicas, so HJ-PCNR separates player-boundary resampling from the
two-view conditional-variance reduction.
"""

from __future__ import annotations

from models.hj.model import SBModelHJPatchNCE
from models.route1.pcnr import PCNRMixin


class Route1HjpcnrModel(PCNRMixin, SBModelHJPatchNCE):
    """Frozen continuous HJ objective with one fresh post-D/E G/F view."""

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModelHJPatchNCE.modify_commandline_options(parser, is_train)
        parser.add_argument("--route1_hjpcnr_enable", type=_str2bool, default=True)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcnr_state()

    def _hjpcnr_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_hjpcnr_enable", True))

    def _pcnr_enabled(self) -> bool:
        return self._hjpcnr_enabled()

    def _hj_active(self):
        return self._hjpcnr_enabled() and super()._hj_active()

    def get_extra_training_state(self):
        """Do not serialize dormant HJ bookkeeping in exact-plain mode.

        The shared HJ base advances read-only epoch/step diagnostics even when
        its objective is inactive.  Those values do not affect the native
        update, but including them in a disabled HJ-PCNR checkpoint would make
        the full-state identity gate differ from a plain checkpoint.  HJCGR
        already applies the same rule: a disabled composite operator must have
        no method-owned state at all.
        """
        state = super().get_extra_training_state()
        if not self._hjpcnr_enabled():
            state.pop("hj_controller", None)
        return state


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
