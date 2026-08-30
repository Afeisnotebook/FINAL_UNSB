"""Conditional sampling followed by an Adam-metric covariance barrier.

This is the only preregistered two-component route-1 synthesis.  Its sampling
parent is frozen before execution to either PCNR (one fresh G/F view after the
opponent commits) or the already-audited PC-RSMG proposal (two fresh G/F views
whose gradients are averaged).  The resulting native-like Adam displacement
is then passed through AM-MCRB's target-blind closest-feasible projection.

There is no strength, window, paired metric or checkpoint-dependent branch.
When disabled, the mixin dispatches the native ``SBModel`` transition without
creating method state.  For the two-view sampling parent, the covariance
tangent is the arithmetic mean over the same two G/F bridge views.  This keeps
the constraint exchange-symmetric and binds it to the stochastic measure that
actually produced the generator gradient.
"""

from __future__ import annotations

from models.route1.ammcrb import AMMCRBMixin
from models.route1.mcrb import MCRBMixin
from models.route1.pcnr import EXPECTED_PCNR_SCHEDULE, PCNRMixin
from models.route1.pcrsmg import PCRSMGMixin, _VIEW_NAMES


SAMPLING_PARENTS = ("pcnr", "pcrsmg_proposal")
EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE = (
    "NATIVE_DE_VIEW",
    "D_COMMIT",
    "E_COMMIT",
    "GF_BUNDLE",
    "GF_BARRIER_COMMIT",
)


class PCAMMCRBMixin(PCNRMixin, AMMCRBMixin):
    """Compose one frozen conditional sampler with the AM-MCRB operator."""

    def _pcammcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "pcammcrb_enable", True))

    def _pcammcrb_sampling_parent(self) -> str:
        parent = str(getattr(self.opt, "pcammcrb_sampling_parent", "pcnr"))
        if parent not in SAMPLING_PARENTS:
            raise ValueError(f"unknown PC-AMMCRB sampling parent: {parent}")
        return parent

    def _pcnr_enabled(self) -> bool:
        return self._pcammcrb_enabled() and self._pcammcrb_sampling_parent() == "pcnr"

    def _mcrb_enabled(self) -> bool:
        return self._pcammcrb_enabled()

    @staticmethod
    def _capture_synthesis_view(model) -> dict:
        return {name: getattr(model, name) for name in _VIEW_NAMES if hasattr(model, name)}

    @staticmethod
    def _restore_synthesis_view(model, view: dict) -> None:
        for name, value in view.items():
            setattr(model, name, value)

    @staticmethod
    def _mean_loss_records(records):
        return PCRSMGMixin._mean_loss_records(records)

    def _set_all_train(self) -> None:
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()

    def _mcrb_direction_covariance_defect(self):
        views = getattr(self, "_pcammcrb_constraint_views", None)
        if not views:
            defect, observable = MCRBMixin._mcrb_direction_covariance_defect(self)
            return defect, {**observable, "constraint_view_count": 1}

        restored = self._capture_synthesis_view(self)
        defects = []
        observables = []
        try:
            for view in views:
                self._restore_synthesis_view(self, view)
                defect, observable = MCRBMixin._mcrb_direction_covariance_defect(self)
                defects.append(defect)
                observables.append(observable)
        finally:
            self._restore_synthesis_view(self, restored)
        if len(defects) != 2:
            raise RuntimeError("PC-RSMG synthesis requires exactly two constraint views")
        return sum(defects) / 2.0, {
            "constraint_view_count": 2,
            "constraint_view_aggregation": "arithmetic_mean_common_latents",
            "time_indices": [int(row["time_index"]) for row in observables],
            "mean_t_norm": sum(float(row["t_norm"]) for row in observables) / 2.0,
            "mean_covariance_gap_rms": sum(
                float(row["covariance_gap_rms"]) for row in observables
            ) / 2.0,
        }

    def _pcrsmg_proposal_barrier_step(self) -> None:
        self.forward()
        self._set_all_train()
        self._pcammcrb_last_schedule = ["NATIVE_DE_VIEW"]

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        self._pcammcrb_last_schedule.append("D_COMMIT")

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        self._pcammcrb_last_schedule.append("E_COMMIT")

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        gf_views = []
        for _ in range(2):
            self.forward()
            gf_views.append(self._capture_synthesis_view(self))
        self._set_all_train()
        self._pcammcrb_gf_bundle_count += 1
        self._pcammcrb_last_schedule.append("GF_BUNDLE")

        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        records = []
        for view in gf_views:
            self._restore_synthesis_view(self, view)
            loss = self.compute_G_loss()
            record = {
                "loss_G": loss,
                "loss_G_GAN": self.loss_G_GAN,
                "loss_SB": self.loss_SB,
                "loss_NCE": self.loss_NCE,
            }
            if hasattr(self, "loss_NCE_Y"):
                record["loss_NCE_Y"] = self.loss_NCE_Y
            records.append(record)
            (loss / 2.0).backward()
        self._before_generator_optimizer_step()
        self._pcammcrb_constraint_views = gf_views
        try:
            self._generator_optimizer_step()
        finally:
            self._pcammcrb_constraint_views = None
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()
        self._pcammcrb_last_schedule.append("GF_BARRIER_COMMIT")

        if tuple(self._pcammcrb_last_schedule) != EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE:
            raise RuntimeError("PC-RSMG/AM-MCRB synthesis execution order changed")
        self._restore_synthesis_view(self, gf_views[-1])
        for name, value in self._mean_loss_records(records).items():
            setattr(self, name, value)
        self._pcammcrb_update_index += 1

    def optimize_parameters(self):
        if not self._pcammcrb_enabled():
            return super().optimize_parameters()
        if self._pcammcrb_sampling_parent() == "pcnr":
            result = PCNRMixin.optimize_parameters(self)
            self._pcammcrb_update_index += 1
            self._pcammcrb_gf_bundle_count += 1
            self._pcammcrb_last_schedule = list(self._pcnr_last_schedule)
            return result
        return self._pcrsmg_proposal_barrier_step()

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._pcammcrb_enabled():
            return state
        state["pcammcrb"] = {
            "enabled": True,
            "sampling_parent": self._pcammcrb_sampling_parent(),
            "update_index": int(self._pcammcrb_update_index),
            "gf_bundle_count": int(self._pcammcrb_gf_bundle_count),
            "last_schedule": list(self._pcammcrb_last_schedule),
            "two_view_constraint": "arithmetic_mean_common_latents",
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._pcammcrb_enabled():
            return
        saved = (state or {}).get("pcammcrb")
        if saved is None:
            self._pcammcrb_update_index = 0
            self._pcammcrb_gf_bundle_count = 0
            self._pcammcrb_last_schedule = []
            return
        if saved.get("enabled") is not True:
            raise RuntimeError("PC-AMMCRB checkpoint does not contain an active operator")
        if saved.get("sampling_parent") != self._pcammcrb_sampling_parent():
            raise RuntimeError("PC-AMMCRB sampling parent changed across resume")
        if saved.get("two_view_constraint") != "arithmetic_mean_common_latents":
            raise RuntimeError("PC-AMMCRB constraint aggregation changed across resume")
        self._pcammcrb_update_index = int(saved["update_index"])
        self._pcammcrb_gf_bundle_count = int(saved["gf_bundle_count"])
        self._pcammcrb_last_schedule = list(saved["last_schedule"])
        expected = (
            EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE
            if self._pcammcrb_sampling_parent() == "pcrsmg_proposal"
            else EXPECTED_PCNR_SCHEDULE
        )
        if self._pcammcrb_last_schedule and tuple(self._pcammcrb_last_schedule) != expected:
            raise RuntimeError("PC-AMMCRB checkpoint schedule mismatch")

    def _initialize_pcammcrb_state(self) -> None:
        self._initialize_pcnr_state()
        self._initialize_mcrb_state()
        self._pcammcrb_update_index = 0
        self._pcammcrb_gf_bundle_count = 0
        self._pcammcrb_last_schedule = []
        self._pcammcrb_constraint_views = None
