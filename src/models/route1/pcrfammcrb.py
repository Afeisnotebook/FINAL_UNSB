"""Conditional sampling with a residual-feasible Adam covariance barrier.

This module is deliberately a new algorithm identity.  The earlier
``PCAMMCRBMixin`` composes conditional sampling with the superseded
fixed-absolute-margin AM-MCRB implementation.  Reusing that class unchanged
would silently reintroduce the numerical scale pathology found at e200.

The sampling transition remains PCNR or the PC-RSMG G/F proposal.  The only
change is the post-native generator operator: it dispatches the represented,
residual-checked RF-AMMCRB closest-feasible projection.  No strength, window,
paired metric, or checkpoint-dependent branch is introduced.
"""

from __future__ import annotations

from models.route1.pcammcrb import PCAMMCRBMixin, SAMPLING_PARENTS
from models.route1.rfammcrb import RFAMMCRBMixin


class PCRFAMMCRBMixin(PCAMMCRBMixin):
    """Compose a frozen conditional sampler with RF-AMMCRB."""

    def _pcammcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "pcrfammcrb_enable", True))

    def _pcammcrb_sampling_parent(self) -> str:
        parent = str(getattr(self.opt, "pcrfammcrb_sampling_parent", "pcnr"))
        if parent not in SAMPLING_PARENTS:
            raise ValueError(f"unknown PC-RF-AMMCRB sampling parent: {parent}")
        return parent

    def _generator_optimizer_step(self):
        # Calling the repaired mixin explicitly is intentional: inheriting it
        # after PCAMMCRB would leave the old AMMCRB method earlier in the MRO.
        return RFAMMCRBMixin._generator_optimizer_step(self)

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if self._pcammcrb_enabled():
            state["pcammcrb"]["barrier_operator"] = (
                "residual_feasible_adam_metric_without_absolute_margin"
            )
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._pcammcrb_enabled():
            return
        saved = (state or {}).get("pcammcrb")
        if saved is not None and saved.get("barrier_operator") != (
            "residual_feasible_adam_metric_without_absolute_margin"
        ):
            raise RuntimeError("PC-RF-AMMCRB checkpoint barrier identity changed")

