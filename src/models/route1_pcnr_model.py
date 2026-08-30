"""Model registry entry for Player-Conditional Native Resampling."""

from __future__ import annotations

from models.route1.pcnr import PCNRMixin
from models.sb_model import SBModel


class Route1PcnrModel(PCNRMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--pcnr_enable", type=_str2bool, default=True)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcnr_state()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
