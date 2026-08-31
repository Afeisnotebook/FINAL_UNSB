"""Model registry for RF-MCRB mechanism ablations."""

from __future__ import annotations

from models.route1.rfmcrb_ablation import RFMCRBAblationMixin
from models.sb_model import SBModel


class Route1RfmcrbAblationModel(RFMCRBAblationMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--route1_ablation_enable", type=_str2bool, default=True)
        parser.add_argument(
            "--rfmcrb_ablation_role",
            choices=["proposal_only", "observable_only"],
            default="proposal_only",
        )
        parser.add_argument("--mcrb_m", type=int, default=4)
        parser.add_argument("--mcrb_region_patch", type=int, default=32)
        parser.add_argument("--mcrb_u_floor", type=float, default=1e-30)
        parser.add_argument("--mcrb_teacher_half_life_updates", type=int, default=150)
        parser.add_argument("--rfmcrb_projection_epsilon", type=float, default=1e-24)
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_mcrb_state()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
