"""Model registry entry for Replicated Stochastic-Measure Gradient."""

from __future__ import annotations

from models.route1.rsmg import RSMGMixin
from models.sb_model import SBModel


class Route1RsmgModel(RSMGMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--rsmg_replicates", type=int, default=2, choices=[1, 2])
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_rsmg_state()

