"""Source-bound long-horizon ablations for AM-TNC.

``proposal_only`` preserves AM-TNC's fresh player-conditional two-view
schedule but commits the ordered first native gradient.  It therefore removes
the Adam-metric tangential operator while retaining the conditional sampling
protocol.  ``observable_only`` measures the pre-update two-view geometry,
restores RNG, buffers, modes and view tensors, and then dispatches the exact
native UNSB transition.
"""

from __future__ import annotations

import copy
from typing import Any

import torch

from models.route1.amtnc import (
    AMTNCMixin,
    _VIEW_NAMES,
    _adam_scales,
    _assign_gradients,
    _combine_optional_gradients,
    _loss_gradients,
    _network_parameters,
)
from models.route1.pcrsmg_ablation import _capture_rng, _restore_rng


class AMTNCAblationMixin(AMTNCMixin):
    def _ablation_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_ablation_enable", True))

    def _ablation_role(self) -> str:
        role = str(getattr(self.opt, "amtnc_ablation_role", "proposal_only"))
        if role not in ("proposal_only", "observable_only"):
            raise ValueError(f"unknown AM-TNC ablation role: {role}")
        return role

    def _amtnc_replicates(self) -> int:
        return 2 if self._ablation_enabled() and self._ablation_role() == "proposal_only" else 1

    def _commit_player(
        self, *, parameters: tuple[torch.nn.Parameter, ...],
        optimizers: tuple[torch.optim.Optimizer, ...],
        losses: tuple[torch.Tensor, torch.Tensor], player: str,
    ) -> None:
        """Commit replica one while recording the discarded AM-TNC geometry."""
        scales = _adam_scales(parameters, optimizers)
        first = _loss_gradients(losses[0], parameters)
        second = _loss_gradients(losses[1], parameters)
        _, diagnostics = _combine_optional_gradients(first, second, scales)
        for optimizer in optimizers:
            optimizer.zero_grad()
        _assign_gradients(parameters, first)
        if player == "GF":
            self._before_generator_optimizer_step()
            self._generator_optimizer_step()
            for optimizer in optimizers[1:]:
                optimizer.step()
        else:
            if len(optimizers) != 1:
                raise RuntimeError("AM-TNC proposal opponent has multiple optimizers")
            optimizers[0].step()
        self._amtnc_last_geometry[player] = {
            **diagnostics,
            "committed_first_replica": True,
        }
        self._amtnc_last_schedule.append(f"{player}_COMMIT")

    def _capture_view_presence(self) -> tuple[dict[str, Any], set[str]]:
        present = {
            name: getattr(self, name) for name in _VIEW_NAMES if hasattr(self, name)
        }
        return present, set(_VIEW_NAMES) - set(present)

    def _restore_view_presence(
        self, present: dict[str, Any], absent: set[str],
    ) -> None:
        for name, value in present.items():
            setattr(self, name, value)
        for name in absent:
            if hasattr(self, name):
                delattr(self, name)

    def _diagnostic_player_geometry(
        self, views: list[dict[str, Any]], *, player: str,
    ) -> dict[str, float]:
        if player == "D":
            parameters = _network_parameters(self.netD)
            optimizers = (self.optimizer_D,)
            loss_fn = self.compute_D_loss
        elif player == "E":
            parameters = _network_parameters(self.netE)
            optimizers = (self.optimizer_E,)
            loss_fn = self.compute_E_loss
        elif player == "GF":
            parameters = _network_parameters(self.netG, self.netF)
            optimizers = [self.optimizer_G]
            if self.opt.netF == "mlp_sample":
                optimizers.append(self.optimizer_F)
            optimizers = tuple(optimizers)
            loss_fn = self.compute_G_loss
        else:
            raise ValueError(f"unknown AM-TNC diagnostic player: {player}")
        gradients = []
        for view in views:
            self._restore_amtnc_view(view)
            gradients.append(_loss_gradients(loss_fn(), parameters))
        _, diagnostics = _combine_optional_gradients(
            gradients[0], gradients[1], _adam_scales(parameters, optimizers),
        )
        return diagnostics

    def _observe_without_transition(self) -> dict[str, Any]:
        rng = _capture_rng()
        present, absent = self._capture_view_presence()
        networks = (self.netG, self.netE, self.netD, self.netF)
        modes = [network.training for network in networks]
        parameters = [parameter for network in networks for parameter in network.parameters()]
        requires = [parameter.requires_grad for parameter in parameters]
        buffers = [
            (buffer, buffer.detach().clone())
            for network in networks for buffer in network.buffers()
        ]
        try:
            for network in networks:
                network.train()
            self.set_requires_grad(self.netD, True)
            self.set_requires_grad(self.netE, True)
            de_views = []
            for _ in range(2):
                self.forward()
                de_views.append(self._capture_amtnc_view())
            geometry = {
                "D": self._diagnostic_player_geometry(de_views, player="D"),
                "E": self._diagnostic_player_geometry(de_views, player="E"),
            }
            self.set_requires_grad(self.netD, False)
            self.set_requires_grad(self.netE, False)
            gf_views = []
            for _ in range(2):
                self.forward()
                gf_views.append(self._capture_amtnc_view())
            geometry["GF"] = self._diagnostic_player_geometry(gf_views, player="GF")
            return {
                "geometry": geometry,
                "diagnostic_rng_restored": True,
                "diagnostic_reference_state": "pre_native_update",
            }
        finally:
            with torch.no_grad():
                for buffer, value in buffers:
                    buffer.copy_(value)
            for parameter, required in zip(parameters, requires):
                parameter.requires_grad_(required)
            for network, mode in zip(networks, modes):
                network.train(mode)
            self._restore_view_presence(present, absent)
            _restore_rng(rng)

    def optimize_parameters(self):
        if not self._ablation_enabled() or self._ablation_role() == "proposal_only":
            return super().optimize_parameters()
        observed = self._observe_without_transition()
        # Skip AMTNCMixin and dispatch the original SBModel implementation.
        result = super(AMTNCMixin, self).optimize_parameters()
        self._amtnc_observer_update_index += 1
        self._amtnc_observer_last = observed
        return result

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if self._ablation_enabled() and self._ablation_role() == "observable_only":
            state["route1_observer"] = {
                "family": "amtnc",
                "role": "observable_only",
                "update_index": int(self._amtnc_observer_update_index),
                "last": copy.deepcopy(self._amtnc_observer_last),
            }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if self._ablation_enabled() and self._ablation_role() == "observable_only":
            saved = (state or {}).get("route1_observer")
            if saved is None:
                self._amtnc_observer_update_index = 0
                self._amtnc_observer_last = {}
            else:
                if saved.get("family") != "amtnc" or saved.get("role") != "observable_only":
                    raise RuntimeError("AM-TNC observable-only checkpoint role mismatch")
                self._amtnc_observer_update_index = int(saved["update_index"])
                self._amtnc_observer_last = copy.deepcopy(saved.get("last", {}))

    def _initialize_amtnc_ablation_state(self) -> None:
        self._initialize_amtnc_state()
        self._amtnc_observer_update_index = 0
        self._amtnc_observer_last = {}

