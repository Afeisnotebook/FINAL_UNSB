"""Long-horizon mechanism ablations for PC-RSMG.

``proposal_only`` keeps native one-view D/E updates and applies a fresh
two-replica conditional estimator only to joint G/F.  ``observable_only``
draws a second diagnostic view but restores every RNG stream and commits the
first native view exactly.  Observer state is recoverable but is placed under
``route1_observer`` so the terminal verifier can exclude diagnostics, and only
diagnostics, when proving next-update dynamics identity with plain.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch

from models.route1.pcrsmg import _VIEW_NAMES, PCRSMGMixin


PROPOSAL_SCHEDULE = (
    "NATIVE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE", "GF_COMMIT",
)
OBSERVABLE_SCHEDULE = (
    "NATIVE_VIEW", "DIAGNOSTIC_VIEW_DISCARDED", "D_COMMIT", "E_COMMIT",
    "GF_COMMIT",
)


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [state.clone() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class PCRSMGAblationMixin:
    """Mixin installed directly ahead of ``SBModel`` in the MRO."""

    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(self.opt, "pcrsmg_ablation_role", "proposal_only"))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown PC-RSMG ablation role: {role}")
        return role

    @staticmethod
    def _capture_view(model) -> dict[str, Any]:
        return {
            name: getattr(model, name)
            for name in _VIEW_NAMES if hasattr(model, name)
        }

    @staticmethod
    def _restore_view(model, view: dict[str, Any]) -> None:
        for name, value in view.items():
            setattr(model, name, value)

    @staticmethod
    def _mean_records(records):
        return PCRSMGMixin._mean_loss_records(records)

    def _set_all_train(self) -> None:
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()

    def _prepare_second_gf_view(self, first_view: dict[str, Any]) -> None:
        """Candidate hook immediately before the second conditional G/F view."""
        del first_view

    def _finalize_gf_view_bundle(self, views: list[dict[str, Any]]) -> None:
        """Candidate hook after both conditional G/F views are materialized."""
        del views

    def _commit_native_current_view(self) -> None:
        self._set_all_train()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self._before_generator_optimizer_step()
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()

    def _proposal_only(self) -> None:
        self.forward()
        self._set_all_train()
        self._pcrsmg_ablation_last_schedule = ["NATIVE_VIEW"]

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        self._pcrsmg_ablation_last_schedule.append("D_COMMIT")

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        self.loss_E = self.compute_E_loss()
        self.loss_E.backward()
        self.optimizer_E.step()
        self._pcrsmg_ablation_last_schedule.append("E_COMMIT")

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        gf_views = []
        for replica in range(2):
            if replica == 1:
                self._prepare_second_gf_view(gf_views[0])
            self.forward()
            gf_views.append(self._capture_view(self))
        self._finalize_gf_view_bundle(gf_views)
        self._set_all_train()
        self._pcrsmg_ablation_gf_bundle_count += 1
        self._pcrsmg_ablation_last_schedule.append("GF_BUNDLE")

        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        records = []
        for view in gf_views:
            self._restore_view(self, view)
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
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()
        self._pcrsmg_ablation_last_schedule.append("GF_COMMIT")
        if tuple(self._pcrsmg_ablation_last_schedule) != PROPOSAL_SCHEDULE:
            raise RuntimeError("PC-RSMG proposal-only execution order changed")
        self._restore_view(self, gf_views[-1])
        for name, value in self._mean_records(records).items():
            setattr(self, name, value)
        self._pcrsmg_ablation_update_index += 1

    def _observable_only(self) -> None:
        # First view and all RNG calls up to this point are exactly native.
        self.forward()
        first = self._capture_view(self)
        post_native_view_rng = _capture_rng()
        with torch.no_grad():
            self.forward()
            second = self._capture_view(self)
            first_fake = first["fake_B"].detach().double()
            second_fake = second["fake_B"].detach().double()
            dispersion = float((first_fake - second_fake).square().mean().sqrt().item())
            second_time = int(second["time_idx"].reshape(-1)[0].detach().item())
        # The diagnostic draw is counterfactual: neither its tensors nor the RNG
        # advance is allowed to enter the native optimizer transition.
        _restore_rng(post_native_view_rng)
        self._restore_view(self, first)
        self._commit_native_current_view()
        self._pcrsmg_observer_update_index += 1
        self._pcrsmg_observer_last = {
            "native_time_index": int(first["time_idx"].reshape(-1)[0].detach().item()),
            "diagnostic_time_index": second_time,
            "fake_endpoint_rms_dispersion": dispersion,
            "diagnostic_rng_restored": True,
            "schedule": list(OBSERVABLE_SCHEDULE),
        }

    def optimize_parameters(self):
        if not self._ablation_enabled():
            return super().optimize_parameters()
        if self._ablation_role() == "proposal_only":
            return self._proposal_only()
        return self._observable_only()

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._ablation_enabled():
            return state
        if self._ablation_role() == "proposal_only":
            state["pcrsmg_proposal"] = {
                "role": "proposal_only",
                "update_index": int(self._pcrsmg_ablation_update_index),
                "gf_bundle_count": int(self._pcrsmg_ablation_gf_bundle_count),
                "last_schedule": list(self._pcrsmg_ablation_last_schedule),
            }
        else:
            state["route1_observer"] = {
                "family": "pcrsmg",
                "role": "observable_only",
                "update_index": int(self._pcrsmg_observer_update_index),
                "last": copy.deepcopy(self._pcrsmg_observer_last),
            }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._ablation_enabled():
            return
        if self._ablation_role() == "proposal_only":
            saved = (state or {}).get("pcrsmg_proposal")
            if saved is None:
                self._initialize_pcrsmg_ablation_state()
                return
            if saved.get("role") != "proposal_only":
                raise RuntimeError("PC-RSMG proposal-only checkpoint role mismatch")
            self._pcrsmg_ablation_update_index = int(saved["update_index"])
            self._pcrsmg_ablation_gf_bundle_count = int(saved["gf_bundle_count"])
            self._pcrsmg_ablation_last_schedule = list(saved["last_schedule"])
            if self._pcrsmg_ablation_last_schedule and tuple(
                self._pcrsmg_ablation_last_schedule
            ) != PROPOSAL_SCHEDULE:
                raise RuntimeError("PC-RSMG proposal-only checkpoint schedule mismatch")
        else:
            saved = (state or {}).get("route1_observer")
            if saved is None:
                self._initialize_pcrsmg_ablation_state()
                return
            if saved.get("family") != "pcrsmg" or saved.get("role") != "observable_only":
                raise RuntimeError("PC-RSMG observable-only checkpoint role mismatch")
            self._pcrsmg_observer_update_index = int(saved["update_index"])
            self._pcrsmg_observer_last = copy.deepcopy(saved.get("last", {}))

    def _initialize_pcrsmg_ablation_state(self) -> None:
        self._pcrsmg_ablation_update_index = 0
        self._pcrsmg_ablation_gf_bundle_count = 0
        self._pcrsmg_ablation_last_schedule = []
        self._pcrsmg_observer_update_index = 0
        self._pcrsmg_observer_last = {}
