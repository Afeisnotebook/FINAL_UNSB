"""Configurable HNEK-style bridge-native search installer.

This module contains the single frozen production variant. It does not add
learnable parameters and does not change state-dict keys.
"""

from __future__ import annotations

import types
from dataclasses import dataclass

import torch

from .hnek_kernel import horizon_from_condition


@dataclass(frozen=True)
class HnekSearchConfig:
    # gamma=0.25 is the only historical e200 development survivor.
    gamma: float = 0.25
    coord: str = "residual"          # residual: (x_t,r), endpoint: (x_t,y)
    horizon_mode: str = "physical"   # physical, index, mix
    partial: str = "all"             # all, entropy_only, endpoint_only

    def __post_init__(self):
        if self.gamma <= 0:
            raise ValueError("gamma must be positive")
        if self.coord not in {"residual", "endpoint"}:
            raise ValueError(f"unknown coord: {self.coord}")
        if self.horizon_mode not in {"physical", "index", "mix"}:
            raise ValueError(f"unknown horizon_mode: {self.horizon_mode}")
        if self.partial not in {"all", "entropy_only", "endpoint_only"}:
            raise ValueError(f"unknown partial mode: {self.partial}")


def _inner(net):
    return net.module if hasattr(net, "module") else net


def _broadcast(value: float, like: torch.Tensor) -> torch.Tensor:
    return torch.full_like(like, value)


def physical_horizon(model, like: torch.Tensor) -> torch.Tensor:
    return horizon_from_condition(
        model.time_idx,
        num_timesteps=int(model.opt.num_timesteps),
        like=like,
    )


def entropy_weight(model, like: torch.Tensor, mode: str) -> torch.Tensor:
    """Return a broadcastable entropy weight of the same shape as ``like``."""
    if mode == "physical":
        return physical_horizon(model, like)
    if mode == "index":
        t_idx = int(model.time_idx.reshape(-1)[0])
        value = (int(model.opt.num_timesteps) - t_idx) / int(model.opt.num_timesteps)
        return _broadcast(value, like)
    if mode == "mix":
        h = physical_horizon(model, like)
        t_idx = int(model.time_idx.reshape(-1)[0])
        value = (int(model.opt.num_timesteps) - t_idx) / int(model.opt.num_timesteps)
        return 0.5 * h + 0.5 * _broadcast(value, like)
    raise ValueError(f"unknown horizon_mode: {mode}")


def normalized_residual_gamma(
    x_t: torch.Tensor,
    endpoint: torch.Tensor,
    horizon: torch.Tensor,
    *,
    gamma: float,
    eps: float = 1e-8,
) -> torch.Tensor:
    if x_t.shape != endpoint.shape:
        raise ValueError("x_t and endpoint must have identical shapes")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    h = horizon
    while h.ndim < x_t.ndim:
        h = h.unsqueeze(-1)
    h = h.to(device=x_t.device, dtype=x_t.dtype)
    return (endpoint - x_t) / h.clamp_min(eps).pow(gamma)


def endpoint_from_residual_gamma(
    x_t: torch.Tensor,
    residual: torch.Tensor,
    horizon: torch.Tensor,
    *,
    gamma: float,
) -> torch.Tensor:
    if x_t.shape != residual.shape:
        raise ValueError("x_t and residual must have identical shapes")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    h = horizon
    while h.ndim < x_t.ndim:
        h = h.unsqueeze(-1)
    h = h.to(device=x_t.device, dtype=x_t.dtype)
    endpoint = x_t + h.pow(gamma) * residual
    proposal = x_t + residual
    endpoint = torch.where(h == 1, proposal, endpoint)
    endpoint = torch.where(h == 0, x_t, endpoint)
    return endpoint


def install_hnek_search_generator(net_g, *, num_timesteps: int, cfg: HnekSearchConfig):
    if cfg.partial == "entropy_only":
        return None
    generator = _inner(net_g)
    if hasattr(generator, "_hnek_search_original_forward"):
        raise RuntimeError("HNEK search generator adapter is already installed")
    before_keys = tuple(generator.state_dict().keys())
    before_count = sum(parameter.numel() for parameter in generator.parameters())
    original_forward = generator.forward
    generator._hnek_search_original_forward = original_forward
    generator._hnek_search_num_timesteps = int(num_timesteps)
    generator._hnek_search_gamma = float(cfg.gamma)

    generator.forward = types.MethodType(_make_hnek_search_forward(), generator)
    after_keys = tuple(generator.state_dict().keys())
    after_count = sum(parameter.numel() for parameter in generator.parameters())
    if before_keys != after_keys or before_count != after_count:
        raise RuntimeError("adapter changed generator state identity")
    return {
        "num_timesteps": int(num_timesteps),
        "parameter_count": before_count,
        "state_keys": before_keys,
    }


def _make_hnek_search_forward():
    def hnek_search_forward(self, x, time_cond, z, layers=None, encode_only=False):
        requested_layers = [] if layers is None else layers
        result = self._hnek_search_original_forward(
            x, time_cond, z, requested_layers, encode_only
        )
        if len(requested_layers) > 0:
            return result
        horizon = horizon_from_condition(
            time_cond,
            num_timesteps=self._hnek_search_num_timesteps,
            like=x,
        )
        residual = result - x
        return endpoint_from_residual_gamma(
            x, residual, horizon, gamma=self._hnek_search_gamma
        )

    return hnek_search_forward


def _residual_pair(model, *, detach: bool, cfg: HnekSearchConfig):
    h1 = physical_horizon(model, model.fake_B)
    h2 = physical_horizon(model, model.fake_B2)
    y1 = model.fake_B.detach() if detach else model.fake_B
    y2 = model.fake_B2.detach() if detach else model.fake_B2
    r1 = normalized_residual_gamma(
        model.real_A_noisy, y1, h1, gamma=cfg.gamma
    )
    r2 = normalized_residual_gamma(
        model.real_A_noisy2, y2, h2, gamma=cfg.gamma
    )
    return r1, r2


def _critic_inputs(model, *, detach: bool, cfg: HnekSearchConfig):
    if cfg.coord == "endpoint":
        y1 = model.fake_B.detach() if detach else model.fake_B
        y2 = model.fake_B2.detach() if detach else model.fake_B2
        return y1, y2
    return _residual_pair(model, detach=detach, cfg=cfg)


def hnek_search_compute_E_loss(self):
    cfg = self._hnek_search_cfg
    e1, e2 = _critic_inputs(self, detach=True, cfg=cfg)
    x_e1 = torch.cat([self.real_A_noisy, e1], dim=1)
    x_e2 = torch.cat([self.real_A_noisy2, e2], dim=1)
    temp = torch.logsumexp(
        self.netE(x_e1, self.time_idx, x_e2).reshape(-1), dim=0
    ).mean()
    self.loss_E = -self.netE(x_e1, self.time_idx, x_e1).mean() + temp + temp ** 2
    return self.loss_E


def hnek_search_compute_G_loss(self):
    cfg = self._hnek_search_cfg
    fake = self.fake_B
    tau = float(self.opt.tau)
    # Frozen coupled-runner invariant: the plain SBModel lane performs one
    # unused CPU RNG draw here.  Keep the identical draw so both lanes consume
    # byte-identical RNG bundles; the value is intentionally unused.
    std = torch.rand(size=[1]).item() * self.opt.std

    if self.opt.lambda_GAN > 0.0:
        pred_fake = self.netD(fake, self.time_idx)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
    else:
        self.loss_G_GAN = 0.0

    self.loss_SB = 0.0
    if self.opt.lambda_SB > 0.0:
        e1, e2 = _critic_inputs(self, detach=False, cfg=cfg)
        x_e1 = torch.cat([self.real_A_noisy, e1], dim=1)
        x_e2 = torch.cat([self.real_A_noisy2, e2], dim=1)
        et = self.netE(x_e1, self.time_idx, x_e1).mean() - torch.logsumexp(
            self.netE(x_e1, self.time_idx, x_e2).reshape(-1), dim=0
        )
        weight = entropy_weight(self, fake, cfg.horizon_mode).reshape(-1)[0]
        self.loss_SB = -weight * tau * et
        self.loss_SB = self.loss_SB + tau * torch.mean(
            (self.real_A_noisy - fake) ** 2
        )

    if self.opt.lambda_NCE > 0.0:
        self.loss_NCE = self.calculate_NCE_loss(self.real_A, fake)
    else:
        self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

    if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
        self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
        loss_nce_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
    else:
        loss_nce_both = self.loss_NCE

    self.loss_G = (
        self.loss_G_GAN
        + self.opt.lambda_SB * self.loss_SB
        + self.opt.lambda_NCE * loss_nce_both
    )
    return self.loss_G


def install_hnek_search_model(model, cfg: HnekSearchConfig):
    record = install_hnek_search_generator(
        model.netG, num_timesteps=int(model.opt.num_timesteps), cfg=cfg
    )
    if hasattr(model, "_hnek_search_original_compute_E_loss"):
        raise RuntimeError("HNEK search model adapter is already installed")
    model._hnek_search_cfg = cfg
    if cfg.partial in ("all", "entropy_only"):
        model._hnek_search_original_compute_E_loss = model.compute_E_loss
        model._hnek_search_original_compute_G_loss = model.compute_G_loss
        model.compute_E_loss = types.MethodType(hnek_search_compute_E_loss, model)
        model.compute_G_loss = types.MethodType(hnek_search_compute_G_loss, model)
    model.hnek_active = True
    model._hnek_search_install_record = record
    return record


def set_hnek_search_active(model, active: bool) -> None:
    """Toggle an installed HNEK adapter without changing learned state.

    The switch is deliberately parameter-free and idempotent. It is suitable
    for a target-blind handoff only when ``hnek_active`` and the controller
    state are saved in the same full-state checkpoint.
    """
    if not hasattr(model, "_hnek_search_cfg"):
        raise RuntimeError("HNEK search model adapter is not installed")

    cfg = model._hnek_search_cfg
    generator = _inner(model.netG)
    active = bool(active)

    if cfg.partial != "entropy_only":
        if not hasattr(generator, "_hnek_search_original_forward"):
            raise RuntimeError("HNEK generator adapter is not installed")
        generator.forward = (
            types.MethodType(_make_hnek_search_forward(), generator)
            if active
            else generator._hnek_search_original_forward
        )

    if cfg.partial in ("all", "entropy_only"):
        if not hasattr(model, "_hnek_search_original_compute_E_loss"):
            raise RuntimeError("HNEK objective adapter is not installed")
        if active:
            model.compute_E_loss = types.MethodType(hnek_search_compute_E_loss, model)
            model.compute_G_loss = types.MethodType(hnek_search_compute_G_loss, model)
        else:
            model.compute_E_loss = model._hnek_search_original_compute_E_loss
            model.compute_G_loss = model._hnek_search_original_compute_G_loss

    model.hnek_active = active


def hnek_search_installation_status(model) -> dict:
    generator = _inner(model.netG)
    cfg = getattr(model, "_hnek_search_cfg", None)
    return {
        "installed": cfg is not None,
        "generator_wrapped": hasattr(generator, "_hnek_search_original_forward"),
        "compute_wrapped": hasattr(model, "_hnek_search_original_compute_G_loss"),
        "gamma": None if cfg is None else cfg.gamma,
        "coord": None if cfg is None else cfg.coord,
        "horizon_mode": None if cfg is None else cfg.horizon_mode,
        "partial": None if cfg is None else cfg.partial,
        "active": bool(getattr(model, "hnek_active", False)),
    }
