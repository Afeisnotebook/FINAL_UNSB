"""Model registry entry for Adam-Metric Tangential Noise Conservation."""

from __future__ import annotations

from models.route1.amtnc import AMTNCMixin
from models.sb_model import SBModel


class Route1AmtncModel(AMTNCMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--amtnc_replicates", type=int, default=2, choices=[1, 2])
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_amtnc_state()
