"""Player-Conditional Native Resampling (PCNR).

Native UNSB reuses one stochastic forward view across the sequential D, E and
joint G/F updates.  Once D has been updated by that view, reusing it for G/F
makes the generator estimator conditionally correlated with the realized
opponent state.  PCNR keeps the native one-view D/E transition, then draws one
fresh native view for G/F after both opponent commits.  It adds no loss,
average, projection, schedule window or paired signal.
"""

from __future__ import annotations

from models.route1.pcrsmg import _VIEW_NAMES


EXPECTED_PCNR_SCHEDULE = (
    "DE_VIEW",
    "D_COMMIT",
    "E_COMMIT",
    "GF_VIEW",
    "GF_COMMIT",
)


class PCNRMixin:
    def _pcnr_enabled(self) -> bool:
        return bool(getattr(self.opt, "pcnr_enable", True))

    def _capture_pcnr_view(self) -> dict:
        return {name: getattr(self, name) for name in _VIEW_NAMES if hasattr(self, name)}

    def _restore_pcnr_view(self, view: dict) -> None:
        for name, value in view.items():
            setattr(self, name, value)

    def _pcnr_forward(self, event: str) -> dict:
        self.forward()
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self._pcnr_bundle_serial += 1
        self._pcnr_last_schedule.append(event)
        if event == "DE_VIEW":
            self._pcnr_de_view_count += 1
        elif event == "GF_VIEW":
            self._pcnr_gf_view_count += 1
        else:
            raise ValueError(f"unknown PCNR view event: {event}")
        return self._capture_pcnr_view()

    def optimize_parameters(self):
        if not self._pcnr_enabled():
            return super().optimize_parameters()

        self._pcnr_last_schedule = []
        de_view = self._pcnr_forward("DE_VIEW")

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self._restore_pcnr_view(de_view)
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        self._pcnr_last_schedule.append("D_COMMIT")

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self._restore_pcnr_view(de_view)
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        self._pcnr_last_schedule.append("E_COMMIT")

        # The G/F draw occurs after the opponent state is realized.  It is a
        # single native view, so native stochastic variance is retained.
        del de_view
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        gf_view = self._pcnr_forward("GF_VIEW")
        self._restore_pcnr_view(gf_view)

        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self._before_generator_optimizer_step()
        self._generator_optimizer_step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()
        self._pcnr_last_schedule.append("GF_COMMIT")

        if tuple(self._pcnr_last_schedule) != EXPECTED_PCNR_SCHEDULE:
            raise RuntimeError("PCNR player-conditional execution order changed")
        self._restore_pcnr_view(gf_view)
        self._pcnr_update_index += 1

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._pcnr_enabled():
            return state
        state["pcnr"] = {
            "enabled": True,
            "update_index": int(self._pcnr_update_index),
            "bundle_serial": int(self._pcnr_bundle_serial),
            "de_view_count": int(self._pcnr_de_view_count),
            "gf_view_count": int(self._pcnr_gf_view_count),
            "last_schedule": list(self._pcnr_last_schedule),
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._pcnr_enabled():
            return
        saved = (state or {}).get("pcnr")
        if saved is None:
            self._initialize_pcnr_state()
            return
        if saved.get("enabled") is not True:
            raise RuntimeError("PCNR checkpoint does not contain an active operator")
        self._pcnr_update_index = int(saved["update_index"])
        self._pcnr_bundle_serial = int(saved["bundle_serial"])
        self._pcnr_de_view_count = int(saved["de_view_count"])
        self._pcnr_gf_view_count = int(saved["gf_view_count"])
        self._pcnr_last_schedule = list(saved["last_schedule"])
        if self._pcnr_last_schedule and tuple(
            self._pcnr_last_schedule
        ) != EXPECTED_PCNR_SCHEDULE:
            raise RuntimeError("PCNR checkpoint schedule mismatch")

    def _initialize_pcnr_state(self) -> None:
        self._pcnr_update_index = 0
        self._pcnr_bundle_serial = 0
        self._pcnr_de_view_count = 0
        self._pcnr_gf_view_count = 0
        self._pcnr_last_schedule = []
