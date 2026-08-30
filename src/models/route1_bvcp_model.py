"""Model registry entry for Bridge-Velocity Chord Projection."""

from __future__ import annotations

from models.route1.bvcp import BVCPMixin
from models.sb_model import SBModel


class Route1BvcpModel(BVCPMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--bvcp_enable", type=_str2bool, default=True)
        parser.add_argument("--bvcp_root_epsilon", type=float, default=1e-12)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_bvcp_state()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

