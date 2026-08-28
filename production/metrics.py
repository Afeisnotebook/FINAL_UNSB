"""Deterministic common-random-number rollout and image metrics."""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F


def canonical_seed(*fields: object) -> int:
    payload = "|".join(str(field) for field in fields).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _cpu_randn(shape: tuple[int, ...], *fields: object) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(canonical_seed(*fields))
    return torch.randn(shape, generator=generator, dtype=torch.float32)


def build_rollout_bundle(
    *, protocol_hash: str, domain: str, stem: str, replicate: int,
    latent_dim: int = 256, height: int = 128, width: int = 128,
    num_timesteps: int = 5,
) -> dict:
    """Create lane-blind CRN tensors for a matched evaluation cell."""
    z, noise = [], []
    for step in range(num_timesteps):
        base = ("FINAL_UNSB_CRN_V1", protocol_hash, domain, stem, replicate, step)
        z.append(_cpu_randn((1, latent_dim), *base, "latent"))
        noise.append(_cpu_randn((1, 3, height, width), *base, "bridge_noise"))
    return {"z": z, "noise": noise}


def bundle_hash(bundle: dict) -> str:
    digest = hashlib.sha256()
    for value in bundle["z"] + bundle["noise"]:
        digest.update(value.contiguous().numpy().tobytes())
    return digest.hexdigest()


def bridge_times(num_timesteps: int = 5) -> np.ndarray:
    increments = np.asarray(
        [0.0] + [1.0 / (index + 1) for index in range(num_timesteps - 1)],
        dtype=np.float64,
    )
    times = np.cumsum(increments)
    times = times / times[-1]
    times = 0.5 * times[-1] + 0.5 * times
    return np.concatenate([np.zeros(1), times])


@torch.no_grad()
def rollout_endpoint(net_g, source: torch.Tensor, bundle: dict, *, tau: float = 0.01):
    times = bridge_times(len(bundle["z"]))
    state = source
    endpoint = None
    for step in range(len(bundle["z"])):
        if step > 0:
            delta = float(times[step] - times[step - 1])
            denominator = float(times[-1] - times[step - 1])
            alpha = delta / denominator
            variance = delta * (1.0 - alpha)
            state = (
                (1.0 - alpha) * state
                + alpha * endpoint.detach()
                + math.sqrt(variance * tau) * bundle["noise"][step].to(source.device)
            )
        time_index = torch.full(
            (source.shape[0],), step, dtype=torch.long, device=source.device
        )
        endpoint = net_g(state, time_index, bundle["z"][step].to(source.device))
    return endpoint


def to_unit(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor.clamp(-1.0, 1.0) + 1.0) / 2.0)


def psnr_unit(prediction: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(F.mse_loss(prediction, target).detach().cpu())
    return 10.0 * math.log10(1.0 / max(mse, 1e-12))


def ssim_unit(prediction: torch.Tensor, target: torch.Tensor, window: int = 7) -> float:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window // 2
    mu_p = F.avg_pool2d(prediction, window, stride=1, padding=pad)
    mu_t = F.avg_pool2d(target, window, stride=1, padding=pad)
    var_p = F.avg_pool2d(prediction.square(), window, 1, pad) - mu_p.square()
    var_t = F.avg_pool2d(target.square(), window, 1, pad) - mu_t.square()
    covariance = F.avg_pool2d(prediction * target, window, 1, pad) - mu_p * mu_t
    score = ((2 * mu_p * mu_t + c1) * (2 * covariance + c2)) / (
        (mu_p.square() + mu_t.square() + c1) * (var_p + var_t + c2)
    )
    return float(score.mean().detach().cpu())
