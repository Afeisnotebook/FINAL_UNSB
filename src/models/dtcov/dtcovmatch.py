"""Minimal, self-contained DT-CovMatch algorithm.

The original implementation scattered the same logic across
``uncertainty_rollout.py`` and ``sb_model.py``, mixing several unused schemes,
diagnostics, side-car heads, and half-wired knobs.  This module keeps only the
best-branch mechanism:

1. Sample ``m`` stochastic endpoint proposals from the current generator and
   from a frozen first-use teacher.
2. Convert each proposal to a bridge direction ``(Y-X_t)/(1-t)`` and compute a
   regional, signal-normalized MC disagreement ``U_reg_norm``.
3. Compare ``log U`` between current and teacher in *domain x time* z-score
   space, using an EMA of the teacher distribution per ``(domain, time)``.
4. Aggregate per-domain groups with equal weight and add a small additive
   regularization to the usual UNSB generator loss.  Evaluation is always
   plain: the regularizer is a no-op unless explicitly enabled during training.
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EPS = 1e-6


def _deterministic_adaptive_avg_pool2d(
    x: torch.Tensor, output_size: Tuple[int, int]
) -> torch.Tensor:
    """Drop-in replacement for ``F.adaptive_avg_pool2d`` with deterministic backward.

    PyTorch's CUDA adaptive-average-pooling backward has no deterministic
    implementation.  When the input is evenly divisible by the requested output
    (the normal DT-CovMatch setting, 128 -> 4 with a 32-pixel region patch), the
    operation is exactly a non-overlapping ``avg_pool2d`` and we use that path.
    The remaining fallback computes the same adaptive bin boundaries with
    explicit slicing + mean, which is deterministic on both CPU and CUDA.
    """
    if x.dim() != 4:
        raise ValueError("expected 4D input [N,C,H,W]")
    oh, ow = int(output_size[0]), int(output_size[1])
    h, w = int(x.shape[-2]), int(x.shape[-1])
    if oh <= 0 or ow <= 0 or oh > h or ow > w:
        raise ValueError("invalid adaptive pool output size")

    if h % oh == 0 and w % ow == 0:
        kh = h // oh
        kw = w // ow
        return F.avg_pool2d(x, kernel_size=(kh, kw), stride=(kh, kw))

    def bounds(size: int, out: int, i: int) -> Tuple[int, int]:
        start = (i * size) // out
        end = ((i + 1) * size + out - 1) // out
        return start, end

    cols = [bounds(h, oh, i) for i in range(oh)]
    rows = [bounds(w, ow, j) for j in range(ow)]
    chunks = []
    for r0, r1 in cols:
        for c0, c1 in rows:
            chunks.append(x[..., r0:r1, c0:c1].mean(dim=(-2, -1), keepdim=True))
    return torch.cat(chunks, dim=-1).view(x.shape[0], x.shape[1], oh, ow)


EndpointFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass
class DirectionStats:
    """Outputs of ``compute_direction_statistics``.

    ``U_reg_norm`` is the quantity actually regularized.  ``U_pix``/``U_reg``
    are kept only for tests and diagnostics; they are not needed by the loss.
    """

    t_norm: torch.Tensor
    mean_endpoint: torch.Tensor
    v_bar: torch.Tensor
    U_pix: torch.Tensor
    U_reg: torch.Tensor
    U_reg_norm: torch.Tensor


def scheduled_lambda(
    base: float,
    epoch: int,
    schedule: str = "fixed",
    *,
    ramp_start: int = 1,
    ramp_end: int = 0,
    decay_start: int = 0,
    decay_end: int = 0,
    min_value: float = 0.0,
) -> float:
    """Epoch-level scalar for the additive regularizer.

    This is a direct, uncluttered port of the original
    ``scheduled_ua_train_reg_lambda``.  It is deliberately independent of the
    torch model so it can be unit-tested without building a network.
    """
    schedule = str(schedule).lower()
    if schedule == "fixed":
        return float(base)

    if schedule in ("ramp_hold_cosine_decay", "ramp_hold_linear_decay"):
        if ramp_end <= 0 or epoch > ramp_end:
            ramp_factor = 1.0
        elif epoch < ramp_start:
            ramp_factor = 0.0
        else:
            span = max(1, ramp_end - ramp_start + 1)
            ramp_factor = float(epoch - ramp_start + 1) / float(span)

        if decay_end <= decay_start:
            decay_factor = 0.0 if epoch >= decay_end else 1.0
        elif epoch <= decay_start:
            decay_factor = 1.0
        elif epoch >= decay_end:
            decay_factor = 0.0
        else:
            progress = float(epoch - decay_start) / float(decay_end - decay_start)
            if schedule == "ramp_hold_cosine_decay":
                decay_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                decay_factor = 1.0 - progress
        factor = ramp_factor * decay_factor
    else:
        if decay_end <= decay_start:
            factor = 0.0 if epoch >= decay_end else 1.0
        elif epoch <= decay_start:
            factor = 1.0
        elif epoch >= decay_end:
            factor = 0.0
        else:
            progress = float(epoch - decay_start) / float(decay_end - decay_start)
            if schedule == "cosine_decay":
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            elif schedule == "linear_decay":
                factor = 1.0 - progress
            else:
                raise ValueError(f"unknown dtcov schedule: {schedule}")

    factor = max(0.0, min(1.0, factor))
    return min_value + (base - min_value) * factor


def domain_key_from_path(path: object) -> str:
    """Stable lowercase domain key from final6-style image paths.

    ``image_paths`` is the only place the original implementation carried domain
    identity.  This function is intentionally small and deterministic.
    """
    text = str(path).replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    if "__" in name:
        return name.split("__", 1)[0].strip().lower() or "unknown"
    stem = name.rsplit(".", 1)[0]
    if "__" in stem:
        return stem.split("__", 1)[0].strip().lower() or "unknown"
    for part in reversed([item for item in text.split("/") if item]):
        if part not in {"trainA", "trainB", "testA", "testB"} and "__" not in part:
            lowered = part.strip().lower()
            if lowered and lowered not in {"input", "target", "datasets", "data"}:
                return lowered
    return "unknown"


def time_norm_from_times(times: torch.Tensor, t_idx, fallback_num_timesteps: int) -> float:
    """Normalize a rollout index using UNSB's actual non-linear time grid."""
    try:
        idx = int(float(t_idx.detach().item() if torch.is_tensor(t_idx) else t_idx))
        flat = times.detach().flatten()
        if flat.numel() == 0:
            return 0.0
        idx = max(0, min(idx, int(flat.numel()) - 1))
        terminal = float(flat[-1].item())
        if abs(terminal) <= EPS:
            return 0.0
        return max(0.0, min(1.0, float(flat[idx].item()) / terminal))
    except Exception:
        if fallback_num_timesteps <= 1:
            return 0.0
        return max(0.0, min(1.0, float(t_idx) / float(fallback_num_timesteps - 1)))


def compute_direction_statistics(
    *,
    X_t: torch.Tensor,
    endpoint_samples: torch.Tensor,
    t_norm: float,
    region_patch: int,
    detach_uncertainty: bool = False,
    eps: float = EPS,
    signal_normalize: bool = True,
) -> DirectionStats:
    """MC bridge-direction disagreement, normalized by directional signal."""
    if endpoint_samples.dim() != 5:
        raise ValueError("endpoint_samples must have shape [M,B,C,H,W]")
    if endpoint_samples.size(0) < 2:
        raise ValueError("DT-CovMatch requires at least two endpoint samples")

    M, B, C, H, W = endpoint_samples.shape
    t_norm = max(0.0, min(1.0, float(t_norm)))
    denom_t = max(1.0 - t_norm, eps)

    directions = (endpoint_samples - X_t.unsqueeze(0)) / denom_t
    v_bar = directions.mean(dim=0)
    centered = directions - v_bar.unsqueeze(0)
    var_dir = (centered * centered).sum(dim=0) / float(M - 1)

    U_pix = var_dir.mean(dim=1, keepdim=True)
    signal = (v_bar * v_bar).mean(dim=1, keepdim=True)

    patch = max(int(region_patch), 1)
    h_reg = max(1, math.ceil(H / patch))
    w_reg = max(1, math.ceil(W / patch))
    U_reg = _deterministic_adaptive_avg_pool2d(U_pix, (h_reg, w_reg))
    if signal_normalize:
        signal_reg = _deterministic_adaptive_avg_pool2d(signal, (h_reg, w_reg))
        U_reg_norm = U_reg / (signal_reg + eps)
    else:
        U_reg_norm = U_reg

    if detach_uncertainty:
        U_pix = U_pix.detach()
        U_reg = U_reg.detach()
        U_reg_norm = U_reg_norm.detach()

    return DirectionStats(
        t_norm=torch.tensor(t_norm, dtype=X_t.dtype, device=X_t.device),
        mean_endpoint=endpoint_samples.mean(dim=0),
        v_bar=v_bar,
        U_pix=U_pix,
        U_reg=U_reg,
        U_reg_norm=U_reg_norm,
    )


@dataclass
class DomainTimeStats:
    """EMA of teacher ``log U`` statistics keyed by ``(domain, time)``.

    This class replaces four free functions (``_dtcov_stats_store``,
    ``_dtcov_batch_teacher_stats``, ``_dtcov_mu_sigma_for_loss`` and
    ``_dtcov_update_stats``) that mutated the model with a bare dict.
    """

    eps: float = 1e-4
    momentum: float = 0.98
    norm_mode: str = "domain_time"
    store: Dict[Tuple[str, int], Dict[str, float]] = field(default_factory=dict)

    def _key(self, domain: str, time_id: int) -> Tuple[str, int]:
        if self.norm_mode == "global":
            return ("__global__", 0)
        return (str(domain), int(time_id))

    def batch_stats(
        self,
        log_teacher: torch.Tensor,
        domain_keys: Sequence[str],
        time_id: int,
    ) -> Dict[Tuple[str, int], Tuple[float, float]]:
        grouped: Dict[Tuple[str, int], List[torch.Tensor]] = {}
        for idx, domain in enumerate(domain_keys):
            key = self._key(domain, time_id)
            grouped.setdefault(key, []).append(log_teacher[idx].detach().reshape(-1))

        min_var = self.eps * self.eps
        result: Dict[Tuple[str, int], Tuple[float, float]] = {}
        for key, values in grouped.items():
            flat = torch.cat(values, dim=0)
            mean = float(flat.mean().item())
            var = float(flat.var(unbiased=False).item()) if flat.numel() > 1 else min_var
            result[key] = (mean, max(var, min_var))
        return result

    def mu_sigma(
        self,
        log_teacher: torch.Tensor,
        domain_keys: Sequence[str],
        time_id: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        B = int(log_teacher.size(0))
        mu_values: List[float] = []
        sigma_values: List[float] = []
        known = 0
        for idx, domain in enumerate(domain_keys):
            key = self._key(domain, time_id)
            stats = self.store.get(key)
            if stats is not None and float(stats.get("count", 0.0)) > 0:
                mean = float(stats.get("mean", 0.0))
                var = max(float(stats.get("var", self.eps * self.eps)), self.eps * self.eps)
                known += 1
            else:
                mean = 0.0
                var = 1.0
            mu_values.append(mean)
            sigma_values.append(math.sqrt(max(var, self.eps * self.eps)))

        mu = torch.tensor(
            mu_values, dtype=log_teacher.dtype, device=log_teacher.device
        ).reshape(B, 1, 1, 1)
        sigma = torch.tensor(
            sigma_values, dtype=log_teacher.dtype, device=log_teacher.device
        ).reshape(B, 1, 1, 1)
        return mu, sigma.clamp_min(self.eps), float(known) / float(max(B, 1))

    def update(self, batch_stats: Dict[Tuple[str, int], Tuple[float, float]]) -> None:
        momentum = max(0.0, min(float(self.momentum), 0.9999))
        min_var = self.eps * self.eps
        for key, (batch_mean, batch_var) in batch_stats.items():
            batch_var = max(float(batch_var), min_var)
            old = self.store.get(key)
            if old is None or float(old.get("count", 0.0)) <= 0:
                self.store[key] = {"count": 1.0, "mean": float(batch_mean), "var": batch_var}
                continue
            self.store[key] = {
                "count": float(old.get("count", 0.0)) + 1.0,
                "mean": momentum * float(old.get("mean", 0.0)) + (1.0 - momentum) * batch_mean,
                "var": max(
                    momentum * float(old.get("var", min_var)) + (1.0 - momentum) * batch_var,
                    min_var,
                ),
            }


@dataclass
class DTCovMatchConfig:
    """Best-branch hyper-parameters as explicit, named fields."""

    m: int = 4
    region_patch: int = 32
    u_floor: float = 1e-8
    norm_eps: float = 1e-4
    norm_momentum: float = 0.98
    norm_clip: float = 3.0
    domain_balance: str = "grouped_domain"
    lambda_value: float = 0.0
    warmup_iters: int = 300
    time_mode: str = "actual"
    latent_dim: int = 256
    freeze_teacher: bool = True
    norm_mode: str = "domain_time"
    signal_normalize: bool = True


def domain_index_groups(domain_keys: Sequence[str], device: torch.device) -> List[Tuple[str, torch.Tensor]]:
    groups: List[Tuple[str, torch.Tensor]] = []
    for domain in sorted(set(str(item) for item in domain_keys)):
        idxs = [idx for idx, key in enumerate(domain_keys) if str(key) == domain]
        if idxs:
            groups.append((domain, torch.tensor(idxs, dtype=torch.long, device=device)))
    return groups


def _select_time_idx(time_idx: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(time_idx):
        return time_idx
    if time_idx.dim() > 0 and int(time_idx.size(0)) > int(index.detach().max().item()):
        return time_idx.index_select(0, index.to(device=time_idx.device))
    return time_idx


@contextmanager
def _preserve_rng_state() -> Iterator[None]:
    """Prevent auxiliary MC sampling from shifting the main training RNG."""
    cpu_state = torch.random.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@contextmanager
def _eval_mode(module: torch.nn.Module) -> Iterator[None]:
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        if was_training:
            module.train()


def _sample_endpoints(
    net: EndpointFn,
    X_t: torch.Tensor,
    time_idx: torch.Tensor,
    m: int,
    latent_dim: int,
) -> torch.Tensor:
    samples = []
    for _ in range(int(m)):
        z = torch.randn(size=[X_t.size(0), latent_dim], device=X_t.device, dtype=X_t.dtype)
        samples.append(net(X_t, time_idx, z))
    return torch.stack(samples, dim=0)


class DTCovMatch:
    """Regularizer that owns all mutable state and performs one train step."""

    def __init__(
        self,
        *,
        netG: EndpointFn,
        config: DTCovMatchConfig,
    ) -> None:
        self.netG = netG
        self.config = config
        self.stats = DomainTimeStats(
            eps=config.norm_eps,
            momentum=config.norm_momentum,
            norm_mode=config.norm_mode,
        )
        self.teacher: Optional[torch.nn.Module] = None
        self.iter = 0

    @property
    def enabled(self) -> bool:
        return (
            self.config.lambda_value > 0.0
            and self.iter >= max(0, self.config.warmup_iters)
        )

    def ensure_teacher(self) -> torch.nn.Module:
        if self.teacher is None:
            source = self.netG.module if isinstance(self.netG, torch.nn.DataParallel) else self.netG
            import copy

            teacher = copy.deepcopy(source)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad_(False)
            self.teacher = teacher
        return self.teacher

    def inject_teacher(self, state_dict: dict) -> torch.nn.Module:
        """Install an immutable teacher from an explicit canonical netG state.

        This must be used for the DT fork: lazily cloning the already-diverged
        live generator would silently change the method being tested.
        """
        import copy

        source = self.netG.module if isinstance(self.netG, torch.nn.DataParallel) else self.netG
        teacher = copy.deepcopy(source)
        teacher.load_state_dict(state_dict, strict=True)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher = teacher
        self._teacher_netG_sha256 = _state_dict_sha256(teacher.state_dict())
        return teacher

    def _current_and_teacher_stats(
        self,
        X_g: torch.Tensor,
        time_g: torch.Tensor,
        t_norm: float,
    ) -> Tuple[DirectionStats, DirectionStats]:
        m = max(2, int(self.config.m))
        with _preserve_rng_state(), _eval_mode(self.netG):
            current = compute_direction_statistics(
                X_t=X_g,
                endpoint_samples=_sample_endpoints(
                    self.netG, X_g, time_g, m, self.config.latent_dim
                ),
                t_norm=t_norm,
                region_patch=self.config.region_patch,
                detach_uncertainty=False,
                signal_normalize=self.config.signal_normalize,
            )

        if self.config.freeze_teacher:
            teacher = self.ensure_teacher()
        else:
            teacher = (
                self.netG.module
                if isinstance(self.netG, torch.nn.DataParallel)
                else self.netG
            )
        with torch.no_grad(), _preserve_rng_state(), _eval_mode(teacher):
            teacher_stats = compute_direction_statistics(
                X_t=X_g,
                endpoint_samples=_sample_endpoints(
                    teacher, X_g, time_g, m, self.config.latent_dim
                ),
                t_norm=t_norm,
                region_patch=self.config.region_patch,
                detach_uncertainty=True,
                signal_normalize=self.config.signal_normalize,
            )
        return current, teacher_stats

    def forward(
        self,
        *,
        X_t: torch.Tensor,
        time_idx: torch.Tensor,
        time_id: int,
        t_norm: float,
        domain_keys: Sequence[str],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        if not self.enabled:
            return torch.zeros((), dtype=X_t.dtype, device=X_t.device), {}

        groups = domain_index_groups(domain_keys, X_t.device)
        if not groups:
            return torch.zeros((), dtype=X_t.dtype, device=X_t.device), {}

        floor = max(float(self.config.u_floor), EPS)
        clip = float(self.config.norm_clip)
        group_losses: List[torch.Tensor] = []
        raw_per_image: List[torch.Tensor] = []
        current_u: List[torch.Tensor] = []
        teacher_u: List[torch.Tensor] = []
        batch_stats_all: Dict[Tuple[str, int], Tuple[float, float]] = {}
        known_fracs: List[float] = []

        for domain, index in groups:
            X_g = X_t.index_select(0, index)
            time_g = _select_time_idx(time_idx, index)
            current, teacher_stats = self._current_and_teacher_stats(X_g, time_g, t_norm)

            log_current = torch.log(current.U_reg_norm.clamp_min(floor))
            log_teacher = torch.log(teacher_stats.U_reg_norm.detach().clamp_min(floor))
            raw_per_image.append(
                F.smooth_l1_loss(log_current, log_teacher, reduction="none").mean(dim=(1, 2, 3))
            )

            group_keys = [str(domain)] * int(index.numel())
            batch_stats_all.update(self.stats.batch_stats(log_teacher, group_keys, time_id))
            mu, sigma, known_frac = self.stats.mu_sigma(log_teacher, group_keys, time_id)

            z_current = (log_current - mu) / sigma
            z_teacher = (log_teacher.detach() - mu) / sigma
            if clip > 0.0:
                z_current = z_current.clamp(-clip, clip)
                z_teacher = z_teacher.clamp(-clip, clip)

            per_image = F.smooth_l1_loss(z_current, z_teacher.detach(), reduction="none").mean(
                dim=(1, 2, 3)
            )
            group_losses.append(per_image.mean())
            current_u.append(current.U_reg_norm.detach())
            teacher_u.append(teacher_stats.U_reg_norm.detach())
            known_fracs.append(known_frac)

        self.stats.update(batch_stats_all)
        norm_loss = torch.stack(group_losses).mean() if group_losses else torch.zeros_like(X_t)
        raw_loss = torch.cat(raw_per_image).mean() if raw_per_image else torch.zeros_like(X_t)
        loss = norm_loss
        self.iter += 1

        diag = {
            "u_match_loss": float(loss.detach().item()),
            "u_match_raw_loss": float(raw_loss.detach().item()),
            "u_match_norm_loss": float(norm_loss.detach().item()),
            "group_count": float(len(groups)),
            "known_frac": float(sum(known_fracs) / max(len(known_fracs), 1)),
            "t_norm": float(t_norm),
        }
        return loss, diag


def _state_dict_sha256(state_dict: dict) -> str:
    """Return a stable SHA-256 over tensor names, dtypes, shapes and bytes."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\x00")
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(b"\x00")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\x00")
    return digest.hexdigest()

