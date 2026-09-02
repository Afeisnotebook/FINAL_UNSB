"""Model entry for Stratified-Time Conditional G/F Resampling (ST-CGR)."""

from __future__ import annotations

from models.route1.pcrsmg_ablation import PCRSMGAblationMixin
from models.route1.stratified_time import StratifiedTimeConditionalGFMixin
from models.sb_model import SBModel


class Route1StcgrModel(
    StratifiedTimeConditionalGFMixin,
    PCRSMGAblationMixin,
    SBModel,
):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--route1_stcgr_enable", type=_str2bool, default=True)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcrsmg_ablation_state()
        self._initialize_stcgr_state()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
