"""Adam-Metric Tangential Noise Conservation (AM-TNC).

PC-RSMG averages two conditionally iid gradients and therefore deletes every
component of their disagreement.  The e200 target-blind audit found that the
deleted disagreement was overwhelmingly orthogonal to the consensus in the
pre-step Adam metric.  AM-TNC keeps that exchange-antisymmetric tangential
component and removes only radial step-length noise.
"""

from __future__ import annotations

from typing import Iterable

import torch


_VIEW_NAMES = (
    "times", "time_idx", "timestep", "real_A_noisy", "real_A_noisy2",
    "real", "realt", "fake", "fake_B", "fake_B2", "XtB", "idt_B",
    "flipped_for_equivariance",
)

EXPECTED_AMTNC_SCHEDULE = (
    "DE_BUNDLE",
    "D_COMMIT",
    "E_COMMIT",
    "GF_BUNDLE",
    "GF_COMMIT",
)


def adam_metric_tangential_gradient(
    first: tuple[torch.Tensor, ...],
    second: tuple[torch.Tensor, ...],
    scales: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], dict[str, float]]:
    """Cancel radial replica disagreement and retain its tangential part.

    ``scales`` is the frozen pre-step Adam diagonal ``1/(sqrt(v)+eps)``.
    If ``m=(g1+g2)/2`` and ``d=(g1-g2)/2``, this returns

        m + d - <A m,A d>/<A m,A m> m.

    Swapping the iid replicas negates the residual around ``m``.  Averaging
    over their exchange therefore returns ``m`` exactly in expectation.
    """
    if not first or len(first) != len(second) or len(first) != len(scales):
        raise ValueError("AM-TNC gradient structures must be nonempty and equal")
    if any(
        left.shape != right.shape or left.shape != scale.shape
        for left, right, scale in zip(first, second, scales)
    ):
        raise ValueError("AM-TNC gradient tensor shapes differ")

    if all(torch.equal(left, right) for left, right in zip(first, second)):
        consensus = sum(
            float(torch.sum(
                (scale * value) * (scale * value), dtype=torch.float64,
            ).item())
            for value, scale in zip(first, scales)
        )
        return first, {
            "consensus_update_energy": consensus,
            "disagreement_update_energy": 0.0,
            "radial_disagreement_energy": 0.0,
            "tangential_disagreement_energy": 0.0,
            "radial_fraction": 0.0,
            "projection_coefficient": 0.0,
        }

    means = tuple((left + right) * 0.5 for left, right in zip(first, second))
    differences = tuple((left - right) * 0.5 for left, right in zip(first, second))
    consensus_energy = torch.zeros((), dtype=torch.float64, device=first[0].device)
    disagreement_energy = torch.zeros_like(consensus_energy)
    cross = torch.zeros_like(consensus_energy)
    for mean, difference, scale in zip(means, differences, scales):
        adam_mean = scale * mean
        adam_difference = scale * difference
        consensus_energy = consensus_energy + torch.sum(
            adam_mean * adam_mean, dtype=torch.float64,
        )
        disagreement_energy = disagreement_energy + torch.sum(
            adam_difference * adam_difference, dtype=torch.float64,
        )
        cross = cross + torch.sum(
            adam_mean * adam_difference, dtype=torch.float64,
        )

    consensus_value = float(consensus_energy.item())
    disagreement_value = float(disagreement_energy.item())
    if not torch.isfinite(consensus_energy + disagreement_energy + cross).item():
        raise RuntimeError("AM-TNC replica geometry is nonfinite")
    if consensus_value == 0.0:
        # Here m is zero in the positive diagonal metric, hence g1=d.  Keeping
        # the ordered first draw is exchange-antisymmetric and unbiased.
        result = first
        coefficient = 0.0
        radial = 0.0
    else:
        coefficient = float((cross / consensus_energy).item())
        if not torch.isfinite(torch.tensor(coefficient)).item():
            raise RuntimeError("AM-TNC radial projection coefficient is nonfinite")
        result = tuple(
            mean + difference - coefficient * mean
            for mean, difference in zip(means, differences)
        )
        radial = min(
            max(float((cross * cross / consensus_energy).item()), 0.0),
            max(disagreement_value, 0.0),
        )
    tangential = max(disagreement_value - radial, 0.0)
    return result, {
        "consensus_update_energy": consensus_value,
        "disagreement_update_energy": disagreement_value,
        "radial_disagreement_energy": radial,
        "tangential_disagreement_energy": tangential,
        "radial_fraction": (
            0.0 if disagreement_value <= 0.0 else radial / disagreement_value
        ),
        "projection_coefficient": coefficient,
    }


def _network_parameters(*networks) -> tuple[torch.nn.Parameter, ...]:
    return tuple(
        parameter
        for network in networks
        for parameter in network.parameters()
        if parameter.requires_grad
    )


def _adam_scales(
    parameters: tuple[torch.nn.Parameter, ...],
    optimizers: Iterable[torch.optim.Optimizer],
) -> tuple[torch.Tensor, ...]:
    records: dict[int, tuple[torch.optim.Optimizer, float]] = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            epsilon = float(group.get("eps", 1e-8))
            for parameter in group["params"]:
                if id(parameter) in records:
                    raise RuntimeError("AM-TNC parameter appears in two optimizers")
                records[id(parameter)] = (optimizer, epsilon)
    values = []
    for parameter in parameters:
        if id(parameter) not in records:
            raise RuntimeError("AM-TNC trainable parameter has no optimizer")
        optimizer, epsilon = records[id(parameter)]
        second_moment = optimizer.state.get(parameter, {}).get("exp_avg_sq")
        scale = (
            torch.ones_like(parameter)
            if second_moment is None
            else second_moment.detach().sqrt().add(epsilon).reciprocal()
        )
        if not bool(torch.isfinite(scale).all().item()):
            raise RuntimeError("AM-TNC pre-step Adam metric is nonfinite")
        values.append(scale)
    return tuple(values)


def _loss_gradients(
    loss: torch.Tensor, parameters: tuple[torch.nn.Parameter, ...],
) -> tuple[torch.Tensor | None, ...]:
    return tuple(torch.autograd.grad(
        loss, parameters, allow_unused=True, retain_graph=False,
    ))


def _combine_optional_gradients(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
    scales: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor | None, ...], dict[str, float]]:
    if len(first) != len(second) or len(first) != len(scales):
        raise RuntimeError("AM-TNC optional gradient structures differ")
    active = [
        index for index, (left, right) in enumerate(zip(first, second))
        if left is not None or right is not None
    ]
    if not active:
        raise RuntimeError("AM-TNC player loss has no trainable gradient")
    left_active = tuple(
        torch.zeros_like(scales[index]) if first[index] is None else first[index]
        for index in active
    )
    right_active = tuple(
        torch.zeros_like(scales[index]) if second[index] is None else second[index]
        for index in active
    )
    combined, diagnostics = adam_metric_tangential_gradient(
        left_active, right_active, tuple(scales[index] for index in active),
    )
    by_index = dict(zip(active, combined))
    return tuple(by_index.get(index) for index in range(len(first))), diagnostics


def _assign_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    gradients: tuple[torch.Tensor | None, ...],
) -> None:
    if len(parameters) != len(gradients):
        raise RuntimeError("AM-TNC parameter/gradient structures differ")
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.detach()


class AMTNCMixin:
    def _amtnc_replicates(self) -> int:
        return int(getattr(self.opt, "amtnc_replicates", 2))

    def _capture_amtnc_view(self) -> dict:
        return {name: getattr(self, name) for name in _VIEW_NAMES if hasattr(self, name)}

    def _restore_amtnc_view(self, view: dict) -> None:
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
            device = next(
                (value.device for value in tensors if value.device.type != "cpu"), None,
            )
            if device is not None:
                tensors = [value.to(device) for value in tensors]
            result[name] = sum(tensors) / float(len(tensors))
        return result

    def _amtnc_views(self, count: int, *, player_bundle: str) -> list[dict]:
        if player_bundle not in ("DE", "GF"):
            raise ValueError(f"unknown AM-TNC player bundle: {player_bundle}")
        views = []
        for _ in range(count):
            self.forward()
            views.append(self._capture_amtnc_view())
        self.netG.train()
        self.netE.train()
        self.netD.train()
        self.netF.train()
        self._amtnc_bundle_serial += 1
        if player_bundle == "DE":
            self._amtnc_de_bundle_count += 1
        else:
            self._amtnc_gf_bundle_count += 1
        self._amtnc_last_schedule.append(f"{player_bundle}_BUNDLE")
        return views

    def _commit_player(
        self, *, parameters: tuple[torch.nn.Parameter, ...],
        optimizers: tuple[torch.optim.Optimizer, ...],
        losses: tuple[torch.Tensor, torch.Tensor], player: str,
    ) -> None:
        scales = _adam_scales(parameters, optimizers)
        first = _loss_gradients(losses[0], parameters)
        second = _loss_gradients(losses[1], parameters)
        gradients, diagnostics = _combine_optional_gradients(
            first, second, scales,
        )
        for optimizer in optimizers:
            optimizer.zero_grad()
        _assign_gradients(parameters, gradients)
        if player == "GF":
            self._before_generator_optimizer_step()
            self._generator_optimizer_step()
            for optimizer in optimizers[1:]:
                optimizer.step()
        else:
            if len(optimizers) != 1:
                raise RuntimeError("AM-TNC opponent player has multiple optimizers")
            optimizers[0].step()
        self._amtnc_last_geometry[player] = diagnostics
        self._amtnc_last_schedule.append(f"{player}_COMMIT")

    def optimize_parameters(self):
        replicas = self._amtnc_replicates()
        if replicas == 1:
            return super().optimize_parameters()
        if replicas != 2:
            raise ValueError("the frozen AM-TNC derivation requires exactly two replicas")

        self._amtnc_last_schedule = []
        self._amtnc_last_geometry = {}
        de_views = self._amtnc_views(replicas, player_bundle="DE")

        self.set_requires_grad(self.netD, True)
        d_parameters = _network_parameters(self.netD)
        d_losses = []
        d_records = []
        for view in de_views:
            self._restore_amtnc_view(view)
            loss = self.compute_D_loss()
            d_losses.append(loss)
            d_records.append({
                "loss_D": loss,
                "loss_D_real": self.loss_D_real,
                "loss_D_fake": self.loss_D_fake,
            })
        self._commit_player(
            parameters=d_parameters, optimizers=(self.optimizer_D,),
            losses=tuple(d_losses), player="D",
        )

        self.set_requires_grad(self.netE, True)
        e_parameters = _network_parameters(self.netE)
        e_losses = []
        e_records = []
        for view in de_views:
            self._restore_amtnc_view(view)
            loss = self.compute_E_loss()
            e_losses.append(loss)
            e_records.append({"loss_E": loss})
        self._commit_player(
            parameters=e_parameters, optimizers=(self.optimizer_E,),
            losses=tuple(e_losses), player="E",
        )

        del de_views, d_losses, e_losses
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netE, False)
        gf_views = self._amtnc_views(replicas, player_bundle="GF")
        gf_parameters = _network_parameters(self.netG, self.netF)
        gf_optimizers = [self.optimizer_G]
        if self.opt.netF == "mlp_sample":
            gf_optimizers.append(self.optimizer_F)
        g_losses = []
        g_records = []
        for view in gf_views:
            self._restore_amtnc_view(view)
            loss = self.compute_G_loss()
            g_losses.append(loss)
            record = {
                "loss_G": loss,
                "loss_G_GAN": self.loss_G_GAN,
                "loss_SB": self.loss_SB,
                "loss_NCE": self.loss_NCE,
            }
            if hasattr(self, "loss_NCE_Y"):
                record["loss_NCE_Y"] = self.loss_NCE_Y
            g_records.append(record)
        self._commit_player(
            parameters=gf_parameters, optimizers=tuple(gf_optimizers),
            losses=tuple(g_losses), player="GF",
        )

        if tuple(self._amtnc_last_schedule) != EXPECTED_AMTNC_SCHEDULE:
            raise RuntimeError("AM-TNC player-conditional execution order changed")
        self._restore_amtnc_view(gf_views[-1])
        for records in (d_records, e_records, g_records):
            for name, value in self._mean_loss_records(records).items():
                setattr(self, name, value)
        self._amtnc_update_index += 1

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if self._amtnc_replicates() == 1:
            return state
        state["amtnc"] = {
            "replicates": 2,
            "update_index": int(self._amtnc_update_index),
            "bundle_serial": int(self._amtnc_bundle_serial),
            "de_bundle_count": int(self._amtnc_de_bundle_count),
            "gf_bundle_count": int(self._amtnc_gf_bundle_count),
            "last_schedule": list(self._amtnc_last_schedule),
            "last_geometry": dict(self._amtnc_last_geometry),
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if self._amtnc_replicates() == 1:
            return
        saved = (state or {}).get("amtnc")
        if saved is None:
            self._initialize_amtnc_state()
            return
        if int(saved.get("replicates", -1)) != 2:
            raise RuntimeError("AM-TNC checkpoint replica count mismatch")
        self._amtnc_update_index = int(saved["update_index"])
        self._amtnc_bundle_serial = int(saved["bundle_serial"])
        self._amtnc_de_bundle_count = int(saved["de_bundle_count"])
        self._amtnc_gf_bundle_count = int(saved["gf_bundle_count"])
        self._amtnc_last_schedule = list(saved["last_schedule"])
        self._amtnc_last_geometry = dict(saved.get("last_geometry", {}))
        if self._amtnc_last_schedule and (
            tuple(self._amtnc_last_schedule) != EXPECTED_AMTNC_SCHEDULE
        ):
            raise RuntimeError("AM-TNC checkpoint schedule mismatch")

    def _initialize_amtnc_state(self) -> None:
        self._amtnc_update_index = 0
        self._amtnc_bundle_serial = 0
        self._amtnc_de_bundle_count = 0
        self._amtnc_gf_bundle_count = 0
        self._amtnc_last_schedule = []
        self._amtnc_last_geometry = {}
