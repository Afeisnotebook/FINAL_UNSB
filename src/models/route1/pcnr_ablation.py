"""Source-bound mechanism ablations for PCNR.

``proposal_only`` is intentionally the complete PCNR transition: PCNR contains
only the player-conditional resampling proposal and no separate projection or
loss.  ``observable_only`` draws the counterfactual views, restores every RNG
stream, and then executes the native UNSB transition.  Its diagnostics are
recoverable but cannot enter the next optimizer update.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import torch

from models.route1.pcnr import PCNRMixin


def _capture_rng() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": (
            [value.clone() for value in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available() else []
        ),
    }


def _restore_rng(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


class PCNRAblationMixin(PCNRMixin):
    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(self.opt, "pcnr_ablation_role", "proposal_only"))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown PCNR ablation role: {role}")
        return role

    def _pcnr_enabled(self) -> bool:
        return self._ablation_enabled() and self._ablation_role() == "proposal_only"

    def _observable_only(self) -> None:
        # Restore to the state before either diagnostic draw.  Calling the
        # native transition after restoration is therefore pathwise identical
        # to SBModel.optimize_parameters, including all latent/time/NCE RNG.
        before_rng = _capture_rng()
        with torch.no_grad():
            self.forward()
            first = self.fake_B.detach().double().clone()
            first_time = int(self.time_idx.reshape(-1)[0].detach().item())
            self.forward()
            second = self.fake_B.detach().double().clone()
            second_time = int(self.time_idx.reshape(-1)[0].detach().item())
            dispersion = float((first - second).square().mean().sqrt().item())
        _restore_rng(before_rng)
        # PCNRMixin is immediately before native SBModel in the MRO.
        super(PCNRMixin, self).optimize_parameters()
        self._pcnr_observer_update_index += 1
        self._pcnr_observer_last = {
            "role": "observable_only",
            "first_time_index": first_time,
            "counterfactual_time_index": second_time,
            "fake_endpoint_rms_dispersion": dispersion,
            "all_rng_restored_before_native_transition": True,
        }

    def optimize_parameters(self):
        if not self._ablation_enabled():
            return super(PCNRMixin, self).optimize_parameters()
        if self._ablation_role() == "proposal_only":
            return PCNRMixin.optimize_parameters(self)
        return self._observable_only()

    def get_extra_training_state(self):
        if not self._ablation_enabled():
            return super(PCNRMixin, self).get_extra_training_state()
        if self._ablation_role() == "proposal_only":
            state = PCNRMixin.get_extra_training_state(self)
            state["pcnr"]["ablation_role"] = "proposal_only"
            return state
        state = super(PCNRMixin, self).get_extra_training_state()
        state["route1_observer"] = {
            "family": "pcnr",
            "role": "observable_only",
            "update_index": int(self._pcnr_observer_update_index),
            "last": copy.deepcopy(self._pcnr_observer_last),
        }
        return state

    def load_extra_training_state(self, state):
        value = copy.deepcopy(state or {})
        if not self._ablation_enabled():
            return super(PCNRMixin, self).load_extra_training_state(value)
        if self._ablation_role() == "proposal_only":
            saved = value.get("pcnr")
            if saved is not None and saved.get("ablation_role") != "proposal_only":
                raise RuntimeError("PCNR proposal-only checkpoint role mismatch")
            return PCNRMixin.load_extra_training_state(self, value)
        super(PCNRMixin, self).load_extra_training_state(value)
        observer = value.get("route1_observer")
        if observer is None:
            self._pcnr_observer_update_index = 0
            self._pcnr_observer_last = {}
            return
        if observer.get("family") != "pcnr" or observer.get("role") != "observable_only":
            raise RuntimeError("PCNR observable-only checkpoint role mismatch")
        self._pcnr_observer_update_index = int(observer["update_index"])
        self._pcnr_observer_last = copy.deepcopy(observer.get("last", {}))

    def _initialize_pcnr_ablation_state(self) -> None:
        self._initialize_pcnr_state()
        self._pcnr_observer_update_index = 0
        self._pcnr_observer_last = {}
