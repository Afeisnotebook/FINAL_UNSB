"""Model registry entry for Player-Conditional RSMG."""

from __future__ import annotations

from models.route1.pcrsmg import PCRSMGMixin
from models.sb_model import SBModel


class Route1PcrsmgModel(PCRSMGMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--pcrsmg_replicates", type=int, default=2, choices=[1, 2])
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcrsmg_state()
