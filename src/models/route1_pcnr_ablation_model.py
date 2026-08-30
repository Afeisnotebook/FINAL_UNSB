"""Model registry entry for PCNR mechanism ablations."""

from __future__ import annotations

from models.route1.pcnr_ablation import PCNRAblationMixin
from models.sb_model import SBModel


class Route1PcnrAblationModel(PCNRAblationMixin, SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument("--route1_ablation_enable", type=_str2bool, default=True)
        parser.add_argument(
            "--pcnr_ablation_role",
            choices=["proposal_only", "observable_only"],
            default="proposal_only",
        )
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self._initialize_pcnr_ablation_state()


def _str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
