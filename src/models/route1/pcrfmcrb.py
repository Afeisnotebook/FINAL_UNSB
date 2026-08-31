"""Conditional sampling with a residual-feasible Euclidean barrier.

This is the Euclidean sibling of :mod:`models.route1.pcrfammcrb`.  It keeps
the same source-frozen PCNR or PC-RSMG proposal transition, but applies the
represented, residual-checked RF-MCRB closest feasible displacement instead
of the Adam-metric projection.  The two variants are distinct optimization
geometries, not strength settings of one operator.
"""

from __future__ import annotations

from models.route1.mcrb import MCRBMixin
from models.route1.pcammcrb import PCAMMCRBMixin, SAMPLING_PARENTS
from models.route1.rfmcrb import RFMCRBMixin


class PCRFMCRBMixin(PCAMMCRBMixin):
    """Compose a frozen conditional sampler with RF-MCRB."""

    def _pcammcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "pcrfmcrb_enable", True))

    def _pcammcrb_sampling_parent(self) -> str:
        parent = str(getattr(self.opt, "pcrfmcrb_sampling_parent", "pcnr"))
        if parent not in SAMPLING_PARENTS:
            raise ValueError(f"unknown PC-RF-MCRB sampling parent: {parent}")
        return parent

    def _generator_optimizer_step(self):
        if not self._pcammcrb_enabled():
            return super(MCRBMixin, self)._generator_optimizer_step()
        # Explicit dispatch avoids the superseded absolute-margin MCRB method
        # inherited earlier through PCAMMCRBMixin's MRO.
        return RFMCRBMixin._generator_optimizer_step(self)

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if self._pcammcrb_enabled():
            state["pcammcrb"]["barrier_operator"] = (
                "residual_feasible_euclidean_without_absolute_margin"
            )
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._pcammcrb_enabled():
            return
        saved = (state or {}).get("pcammcrb")
        if saved is not None and saved.get("barrier_operator") != (
            "residual_feasible_euclidean_without_absolute_margin"
        ):
            raise RuntimeError("PC-RF-MCRB checkpoint barrier identity changed")
