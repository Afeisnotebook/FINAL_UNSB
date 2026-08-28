"""Pure HNEK kernel mathematics.

HNEK keeps the official time-dead UNSB backbone and changes only the
coordinate chart used for its restricted endpoint law.  For remaining bridge
horizon ``h = 1 - t`` it predicts a normalized residual ``r`` and constructs

    y = x_t + sqrt(h) * r.

The exponent is deliberately fixed to 1/2.  It is not a hyperparameter.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
import torch


TensorLike = Union[torch.Tensor, float]


def bridge_schedule(num_timesteps: int = 5, *, device=None) -> torch.Tensor:
    """Return the *actual* physical schedule used by official ``SBModel``.

    The returned tensor contains all T+1 grid points.  Generator calls use
    indices 0..T-1; the final value 1 is the pinned endpoint.
    """
    if num_timesteps < 2:
        raise ValueError("num_timesteps must be at least 2")
    incs = np.array(
        [0.0] + [1.0 / (i + 1) for i in range(num_timesteps - 1)],
        dtype=np.float64,
    )
    times = np.cumsum(incs)
    times = times / times[-1]
    times = 0.5 + 0.5 * times
    times = np.concatenate([np.zeros(1, dtype=np.float64), times])
    result = torch.tensor(times, dtype=torch.float32)
    return result.to(device) if device is not None else result


def physical_time_from_condition(
    time_cond: torch.Tensor,
    *,
    num_timesteps: int = 5,
) -> torch.Tensor:
    """Map an integer SBModel index, or validated physical float, to ``t``."""
    if time_cond.ndim == 0:
        time_cond = time_cond.reshape(1)
    if time_cond.dtype.is_floating_point:
        time = time_cond.float().reshape(-1)
        if not bool(torch.isfinite(time).all()):
            raise ValueError("physical time must be finite")
        if bool((time < 0).any()) or bool((time > 1).any()):
            raise ValueError("physical time must lie in [0, 1]")
        return time

    index = time_cond.long().reshape(-1)
    if bool((index < 0).any()) or bool((index >= num_timesteps).any()):
        raise ValueError("time index is outside generator-call range [0, T-1]")
    schedule = bridge_schedule(num_timesteps, device=time_cond.device)
    return schedule[index]


def horizon_from_condition(
    time_cond: torch.Tensor,
    *,
    num_timesteps: int = 5,
    like: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return broadcastable remaining horizon ``h = 1-t``."""
    time = physical_time_from_condition(time_cond, num_timesteps=num_timesteps)
    horizon = (1.0 - time).clamp(min=0.0, max=1.0)
    if like is None:
        return horizon
    while horizon.ndim < like.ndim:
        horizon = horizon.unsqueeze(-1)
    return horizon.to(device=like.device, dtype=like.dtype)


def endpoint_from_residual(
    x_t: torch.Tensor,
    residual: torch.Tensor,
    horizon: TensorLike,
) -> torch.Tensor:
    """Construct the endpoint from a normalized residual.

    ``h=1`` recovers ``x_t + residual`` and ``h=0`` is exactly the identity.
    """
    if x_t.shape != residual.shape:
        raise ValueError("x_t and residual must have identical shapes")
    h = torch.as_tensor(horizon, device=x_t.device, dtype=x_t.dtype)
    if not bool(torch.isfinite(h).all()):
        raise ValueError("horizon must be finite")
    if bool((h < 0).any()) or bool((h > 1).any()):
        raise ValueError("horizon must lie in [0, 1]")
    while h.ndim < x_t.ndim:
        h = h.unsqueeze(-1)
    endpoint = x_t + torch.sqrt(h) * residual
    # These explicit branches are semantically important for coupled audits:
    # h=1 must be bitwise the legacy proposal and h=0 bitwise the input.
    proposal = x_t + residual
    endpoint = torch.where(h == 1, proposal, endpoint)
    endpoint = torch.where(h == 0, x_t, endpoint)
    return endpoint


def normalized_residual(
    x_t: torch.Tensor,
    endpoint: torch.Tensor,
    horizon: TensorLike,
    *,
    eps: float = 0.0,
) -> torch.Tensor:
    """Invert :func:`endpoint_from_residual` for strictly positive horizons."""
    if x_t.shape != endpoint.shape:
        raise ValueError("x_t and endpoint must have identical shapes")
    if eps < 0:
        raise ValueError("eps must be nonnegative")
    h = torch.as_tensor(horizon, device=x_t.device, dtype=x_t.dtype)
    if bool((h <= 0).any()) and eps == 0:
        raise ValueError("cannot recover residual at zero horizon")
    while h.ndim < x_t.ndim:
        h = h.unsqueeze(-1)
    return (endpoint - x_t) / torch.sqrt(h.clamp_min(eps))


def transformed_restricted_objective(
    residual_sq_mean: torch.Tensor,
    entropy_residual: torch.Tensor,
    horizon: torch.Tensor,
    tau: float,
) -> torch.Tensor:
    """Parameter-dependent part of the restricted SB objective in r-space."""
    if tau <= 0 or not math.isfinite(tau):
        raise ValueError("tau must be positive and finite")
    return horizon * (residual_sq_mean - 2.0 * tau * entropy_residual)
