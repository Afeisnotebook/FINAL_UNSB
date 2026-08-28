"""Model registration for the HNEK bridge-native development candidate."""

from __future__ import annotations

from models.sb_model import SBModel
from models.hnek.hnek_search import (
    HnekSearchConfig,
    install_hnek_search_model,
    set_hnek_search_active,
)


class HnekSearchModel(SBModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = SBModel.modify_commandline_options(parser, is_train)
        parser.add_argument(
            "--hnek_gamma",
            type=float,
            default=0.25,
            help=(
                "remaining-horizon normalization exponent; 0.25 is the only "
                "e200 development survivor (still non-confirmatory)"
            ),
        )
        parser.add_argument("--hnek_coord", type=str, default="residual",
                            choices=["residual", "endpoint"])
        parser.add_argument("--hnek_horizon_mode", type=str, default="physical",
                            choices=["physical", "index", "mix"])
        parser.add_argument("--hnek_partial", type=str, default="all",
                            choices=["all", "entropy_only", "endpoint_only"])
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        cfg = HnekSearchConfig(
            gamma=float(getattr(opt, "hnek_gamma", 0.25)),
            coord=str(getattr(opt, "hnek_coord", "residual")),
            horizon_mode=str(getattr(opt, "hnek_horizon_mode", "physical")),
            partial=str(getattr(opt, "hnek_partial", "all")),
        )
        install_hnek_search_model(self, cfg)

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        state["hnek_active"] = bool(getattr(self, "hnek_active", True))
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        desired = bool((state or {}).get("hnek_active", True))
        if desired != bool(getattr(self, "hnek_active", True)):
            set_hnek_search_active(self, desired)
