"""Replicated Stochastic-Measure Gradient (RSMG)."""

from __future__ import annotations

import copy

import torch


_VIEW_NAMES = (
    "times", "time_idx", "timestep", "real_A_noisy", "real_A_noisy2",
    "real", "realt", "fake", "fake_B", "fake_B2", "XtB", "idt_B",
    "flipped_for_equivariance",
)


def average_replica_gradients(gradients: list[tuple[torch.Tensor, ...]]):
    """Pure helper used by invariant tests and diagnostic code."""
    if not gradients:
        raise ValueError("at least one gradient replica is required")
    width = len(gradients[0])
    if any(len(row) != width for row in gradients):
        raise ValueError("gradient replica structures differ")
    return tuple(
        sum((row[index] for row in gradients), torch.zeros_like(gradients[0][index]))
        / float(len(gradients))
        for index in range(width)
    )


class RSMGMixin:
    def _rsmg_replicates(self) -> int:
        return int(getattr(self.opt, "rsmg_replicates", 2))

    def _capture_rsmg_view(self) -> dict:
        return {name: getattr(self, name) for name in _VIEW_NAMES if hasattr(self, name)}

    def _restore_rsmg_view(self, view: dict) -> None:
        for name, value in view.items():
            setattr(self, name, value)

    @staticmethod
    def _mean_loss_records(records: list[dict[str, torch.Tensor | float]]) -> dict:
        result = {}
        for name in sorted({key for record in records for key in record}):
            values = [record[name] for record in records if name in record]
            tensors = [
                value.detach() if torch.is_tensor(value) else torch.tensor(float(value))
                for value in values
            ]
            device = next((value.device for value in tensors if value.device.type != "cpu"), None)
            if device is not None:
                tensors = [value.to(device) for value in tensors]
            result[name] = sum(tensors) / float(len(tensors))
        return result

    def _rsmg_views(self, count: int) -> list[dict]:
        views = []
        for _ in range(count):
            self.forward()
            views.append(self._capture_rsmg_view())
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        return views

    def optimize_parameters(self):
        replicas = self._rsmg_replicates()
        if replicas == 1:
            return super().optimize_parameters()
        if replicas != 2:
            raise ValueError("the frozen RSMG derivation requires exactly two replicas")

        views = self._rsmg_views(replicas)

        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        d_records = []
        for view in views:
            self._restore_rsmg_view(view)
            loss = self.compute_D_loss()
            d_records.append({
                "loss_D": loss,
                "loss_D_real": self.loss_D_real,
                "loss_D_fake": self.loss_D_fake,
            })
            (loss / replicas).backward()
        self.optimizer_D.step()

        self.set_requires_grad(self.netE, True)
        self.optimizer_E.zero_grad()
        e_records = []
        for view in views:
            self._restore_rsmg_view(view)
            loss = self.compute_E_loss()
            e_records.append({"loss_E": loss})
            (loss / replicas).backward()
        self.optimizer_E.step()

        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.zero_grad()
        g_records = []
        for view in views:
            self._restore_rsmg_view(view)
            loss = self.compute_G_loss()
            record = {
                "loss_G": loss,
                "loss_G_GAN": self.loss_G_GAN,
                "loss_SB": self.loss_SB,
                "loss_NCE": self.loss_NCE,
            }
            if hasattr(self, "loss_NCE_Y"):
                record["loss_NCE_Y"] = self.loss_NCE_Y
            g_records.append(record)
            (loss / replicas).backward()
        self._before_generator_optimizer_step()
        self.optimizer_G.step()
        if self.opt.netF == "mlp_sample":
            self.optimizer_F.step()

        self._restore_rsmg_view(views[-1])
        for records in (d_records, e_records, g_records):
            for name, value in self._mean_loss_records(records).items():
                setattr(self, name, value)
        self._rsmg_update_index += 1

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if self._rsmg_replicates() == 1:
            return state
        state["rsmg"] = {
            "replicates": 2,
            "update_index": int(self._rsmg_update_index),
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if self._rsmg_replicates() == 1:
            return
        saved = (state or {}).get("rsmg")
        if saved is None:
            self._rsmg_update_index = 0
            return
        if int(saved.get("replicates", -1)) != 2:
            raise RuntimeError("RSMG checkpoint replica count mismatch")
        self._rsmg_update_index = int(saved["update_index"])

    def _initialize_rsmg_state(self) -> None:
        self._rsmg_update_index = 0

