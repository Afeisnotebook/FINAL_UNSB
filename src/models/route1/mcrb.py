"""Moving Covariance Rate Barrier (MCRB).

MCRB does not match a frozen teacher and never changes an endpoint forward.
It uses a one-data-epoch EMA generator only to define a moving, target-blind
direction-covariance defect.  After native Adam has produced its exact
generator displacement, the closest Euclidean displacement whose first-order
effect does not increase that defect is committed.  Safe native displacements
are left byte-for-byte unchanged.
"""

from __future__ import annotations

import copy
import math
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from models.dtcov.dtcovmatch import compute_direction_statistics


def _inner(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def log_direction_covariance(
    covariance: torch.Tensor, *, configured_floor: float,
) -> torch.Tensor:
    """Log a nonnegative covariance with a numerical-only zero guard."""
    if not torch.is_floating_point(covariance):
        raise TypeError("MCRB covariance must be floating point")
    floor = max(float(configured_floor), float(torch.finfo(covariance.dtype).tiny))
    return torch.log(covariance.clamp_min(floor))


@contextmanager
def _preserve_rng_state():
    cpu = torch.random.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        torch.random.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


@contextmanager
def _eval_mode(module):
    training = bool(module.training)
    module.eval()
    try:
        yield
    finally:
        module.train(training)


@dataclass(frozen=True)
class RateBarrierProjection:
    unsafe: bool
    native_defect_directional_derivative: float
    projected_defect_directional_derivative: float
    correction_l2: float
    native_displacement_l2: float


def project_actual_displacement(
    native_displacement: list[torch.Tensor],
    defect_gradient: list[torch.Tensor | None],
    *,
    eps: float = 1e-24,
) -> tuple[list[torch.Tensor], RateBarrierProjection]:
    """Project an actual optimizer displacement onto ``<a,d> <= 0``.

    The projection is the unique minimum-Euclidean-change solution.  It is
    exact identity when the native displacement is safe or the defect tangent
    is zero.  Computation of the two scalar products is promoted to float64.
    """
    if len(native_displacement) != len(defect_gradient):
        raise ValueError("MCRB displacement and tangent lengths differ")
    first = next((value for value in native_displacement if value is not None), None)
    if first is None:
        raise ValueError("MCRB requires at least one displacement tensor")
    dot_tensor = torch.zeros((), dtype=torch.float64, device=first.device)
    norm_tensor = torch.zeros_like(dot_tensor)
    displacement_tensor = torch.zeros_like(dot_tensor)
    for displacement, tangent in zip(native_displacement, defect_gradient):
        displacement_tensor.add_(displacement.detach().double().square().sum())
        if tangent is None:
            continue
        if displacement.shape != tangent.shape:
            raise ValueError("MCRB displacement and tangent shapes differ")
        dot_tensor.add_((displacement.detach().double() * tangent.detach().double()).sum())
        norm_tensor.add_(tangent.detach().double().square().sum())
    dot = float(dot_tensor.item())
    norm_sq = float(norm_tensor.item())
    displacement_sq = float(displacement_tensor.item())
    if not math.isfinite(dot) or not math.isfinite(norm_sq):
        raise RuntimeError("MCRB projection geometry is nonfinite")
    if dot <= 0.0 or norm_sq <= float(eps):
        return native_displacement, RateBarrierProjection(
            unsafe=False,
            native_defect_directional_derivative=dot,
            projected_defect_directional_derivative=dot,
            correction_l2=0.0,
            native_displacement_l2=math.sqrt(max(displacement_sq, 0.0)),
        )
    # Float32 multiplication/subtraction can round the exact boundary back to
    # the unsafe side.  Add a dtype-derived (not user-tuned) margin so the
    # displacement that is actually represented remains feasible.  Float64
    # keeps the exact analytical projection used by the invariant tests.
    low_precision_eps = max(
        (
            float(torch.finfo(value.dtype).eps)
            for value in native_displacement
            if torch.is_floating_point(value) and value.dtype != torch.float64
        ),
        default=0.0,
    )
    numeric_margin = 8.0 * low_precision_eps * max(1.0, abs(dot))
    coefficient = (dot + numeric_margin) / norm_sq
    projected = [
        displacement if tangent is None
        else displacement - tangent.to(displacement) * coefficient
        for displacement, tangent in zip(native_displacement, defect_gradient)
    ]
    projected_dot_tensor = torch.zeros_like(dot_tensor)
    for value, tangent in zip(projected, defect_gradient):
        if tangent is not None:
            projected_dot_tensor.add_(
                (value.detach().double() * tangent.detach().double()).sum()
            )
    projected_dot = float(projected_dot_tensor.item())
    correction_sq = coefficient * coefficient * norm_sq
    tolerance = 1e-10 * max(1.0, abs(dot))
    if projected_dot > tolerance:
        raise RuntimeError(
            "MCRB projected displacement violates its halfspace: "
            f"native_dot={dot:.17g}, tangent_norm_sq={norm_sq:.17g}, "
            f"projected_dot={projected_dot:.17g}, tolerance={tolerance:.17g}"
        )
    return projected, RateBarrierProjection(
        unsafe=True,
        native_defect_directional_derivative=dot,
        projected_defect_directional_derivative=projected_dot,
        correction_l2=math.sqrt(max(correction_sq, 0.0)),
        native_displacement_l2=math.sqrt(max(displacement_sq, 0.0)),
    )


class MCRBMixin:
    """Target-blind moving directional-covariance rate barrier."""

    def _mcrb_enabled(self) -> bool:
        return bool(getattr(self.opt, "mcrb_enable", True))

    def _ensure_mcrb_teacher(self):
        if not self._mcrb_enabled():
            return None
        if getattr(self, "_mcrb_teacher", None) is None:
            self._mcrb_teacher = copy.deepcopy(_inner(self.netG)).to(self.device)
            self._mcrb_teacher.eval()
            for parameter in self._mcrb_teacher.parameters():
                parameter.requires_grad_(False)
        return self._mcrb_teacher

    def _sync_mcrb_teacher(self) -> None:
        teacher = self._ensure_mcrb_teacher()
        if teacher is None:
            return
        teacher.load_state_dict(_inner(self.netG).state_dict(), strict=True)
        teacher.eval()

    def _mcrb_decay(self) -> float:
        half_life = int(getattr(self.opt, "mcrb_teacher_half_life_updates", 150))
        if half_life <= 0:
            raise RuntimeError("MCRB teacher half-life must be positive")
        return math.exp(-math.log(2.0) / float(half_life))

    @torch.no_grad()
    def _update_mcrb_teacher(self) -> None:
        teacher = self._ensure_mcrb_teacher()
        source = _inner(self.netG)
        decay = self._mcrb_decay()
        source_state = source.state_dict()
        teacher_state = teacher.state_dict()
        for key, target in teacher_state.items():
            value = source_state[key].to(device=target.device, dtype=target.dtype)
            if torch.is_floating_point(target):
                target.mul_(decay).add_(value, alpha=1.0 - decay)
            else:
                target.copy_(value)
        teacher.eval()

    def _mcrb_direction_covariance_defect(self):
        teacher = self._ensure_mcrb_teacher()
        current = _inner(self.netG)
        x = self.real_A_noisy.detach()
        time_idx = self.time_idx
        time_value = int(time_idx.reshape(-1)[0].detach().item())
        times = self.times.detach().reshape(-1)
        terminal = float(times[-1].item()) if int(times.numel()) else 1.0
        t_norm = 0.0 if abs(terminal) <= 1e-12 else float(times[time_value].item()) / terminal
        m = max(2, int(getattr(self.opt, "mcrb_m", 4)))
        latent_dim = 4 * int(self.opt.ngf)
        patch = int(getattr(self.opt, "mcrb_region_patch", 32))
        # The deterministic clean generator starts with latent-direction
        # covariance around 1e-14.  Reusing DT's much larger loss floor would
        # clamp every region to the same value and silently turn MCRB into
        # plain.  Here the floor is only a log(0) guard; the half-space
        # projection is invariant to the overall tangent scale.
        configured_floor = float(getattr(self.opt, "mcrb_u_floor", 1e-30))
        with _preserve_rng_state():
            latents = [
                torch.randn(x.size(0), latent_dim, device=x.device, dtype=x.dtype)
                for _ in range(m)
            ]
            with _eval_mode(current):
                current_endpoints = torch.stack(
                    [current(x, time_idx, latent) for latent in latents], dim=0
                )
                current_stats = compute_direction_statistics(
                    X_t=x,
                    endpoint_samples=current_endpoints,
                    t_norm=t_norm,
                    region_patch=patch,
                    detach_uncertainty=False,
                    signal_normalize=True,
                )
            with torch.no_grad(), _eval_mode(teacher):
                teacher_endpoints = torch.stack(
                    [teacher(x, time_idx, latent) for latent in latents], dim=0
                )
                teacher_stats = compute_direction_statistics(
                    X_t=x,
                    endpoint_samples=teacher_endpoints,
                    t_norm=t_norm,
                    region_patch=patch,
                    detach_uncertainty=True,
                    signal_normalize=True,
                )
        current_log = log_direction_covariance(
            current_stats.U_reg_norm, configured_floor=configured_floor,
        )
        teacher_log = log_direction_covariance(
            teacher_stats.U_reg_norm, configured_floor=configured_floor,
        ).detach()
        difference = current_log - teacher_log
        defect = 0.5 * difference.square().mean()
        return defect, {
            "time_index": time_value,
            "t_norm": t_norm,
            "covariance_gap_rms": float(difference.detach().double().square().mean().sqrt().item()),
        }

    def _generator_optimizer_step(self):
        if not self._mcrb_enabled():
            return super()._generator_optimizer_step()
        parameters = [parameter for parameter in self.netG.parameters() if parameter.requires_grad]
        defect, observable = self._mcrb_direction_covariance_defect()
        tangents = list(torch.autograd.grad(
            defect, parameters, allow_unused=True, retain_graph=False,
        ))
        before = [parameter.detach().clone() for parameter in parameters]
        super()._generator_optimizer_step()
        native = [parameter.detach() - old for parameter, old in zip(parameters, before)]
        projected, diag = project_actual_displacement(
            native,
            tangents,
            eps=float(getattr(self.opt, "mcrb_projection_epsilon", 1e-24)),
        )
        if diag.unsafe:
            with torch.no_grad():
                for parameter, old, displacement in zip(parameters, before, projected):
                    parameter.copy_(old + displacement.to(parameter))
            self._mcrb_intervention_count += 1
        self._mcrb_update_index += 1
        self._mcrb_eligible_count += int(diag.unsafe)
        self._mcrb_correction_l2_sum += float(diag.correction_l2)
        self._mcrb_last = {
            **observable,
            "defect": float(defect.detach().item()),
            "native_defect_directional_derivative": diag.native_defect_directional_derivative,
            "projected_defect_directional_derivative": diag.projected_defect_directional_derivative,
            "native_displacement_l2": diag.native_displacement_l2,
            "correction_l2": diag.correction_l2,
            "intervened": bool(diag.unsafe),
        }
        self._update_mcrb_teacher()

    def set_search_step(self, step, total_steps=None):
        super().set_search_step(step, total_steps)
        if self._mcrb_enabled() and int(step) == 0 and not self._mcrb_loaded_state:
            self._sync_mcrb_teacher()
            self._mcrb_update_index = 0
            self._mcrb_eligible_count = 0
            self._mcrb_intervention_count = 0
            self._mcrb_correction_l2_sum = 0.0
            self._mcrb_last = {}

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._mcrb_enabled():
            return state
        teacher = self._ensure_mcrb_teacher()
        state["mcrb"] = {
            "enabled": True,
            "update_index": int(self._mcrb_update_index),
            "eligible_count": int(self._mcrb_eligible_count),
            "intervention_count": int(self._mcrb_intervention_count),
            "correction_l2_sum": float(self._mcrb_correction_l2_sum),
            "teacher_half_life_updates": int(getattr(self.opt, "mcrb_teacher_half_life_updates", 150)),
            "last": copy.deepcopy(self._mcrb_last),
            "teacher_netG": {
                key: value.detach().cpu() for key, value in teacher.state_dict().items()
            },
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._mcrb_enabled():
            return
        saved = (state or {}).get("mcrb")
        if saved is None:
            self._mcrb_loaded_state = False
            return
        if saved.get("enabled") is not True:
            raise RuntimeError("MCRB checkpoint does not contain an active operator")
        if int(saved.get("teacher_half_life_updates", -1)) != int(
            getattr(self.opt, "mcrb_teacher_half_life_updates", 150)
        ):
            raise RuntimeError("MCRB teacher half-life changed across resume")
        teacher_state = saved.get("teacher_netG")
        if not isinstance(teacher_state, dict) or not teacher_state:
            raise RuntimeError("MCRB checkpoint is missing its moving teacher")
        self._ensure_mcrb_teacher().load_state_dict(teacher_state, strict=True)
        self._mcrb_update_index = int(saved["update_index"])
        self._mcrb_eligible_count = int(saved["eligible_count"])
        self._mcrb_intervention_count = int(saved["intervention_count"])
        self._mcrb_correction_l2_sum = float(saved["correction_l2_sum"])
        self._mcrb_last = copy.deepcopy(saved.get("last", {}))
        self._mcrb_loaded_state = True

    def _initialize_mcrb_state(self):
        self._mcrb_teacher = None
        self._mcrb_loaded_state = False
        self._mcrb_update_index = 0
        self._mcrb_eligible_count = 0
        self._mcrb_intervention_count = 0
        self._mcrb_correction_l2_sum = 0.0
        self._mcrb_last = {}
