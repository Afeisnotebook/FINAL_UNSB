"""Bridge-Velocity Chord Projection (BVCP).

BVCP changes only the no-gradient rollout used to construct the endogenous
training bridge state.  The differentiable endpoint, native losses and
inference path remain the canonical UNSB implementation.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass

import torch


def _inner(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


@dataclass(frozen=True)
class ChordProjectionDiagnostics:
    eligible: int
    intervened: int
    mean_lambda: float
    current_rms: float
    lagged_rms: float
    projected_rms: float


def minimum_velocity_chord_endpoint(
    x: torch.Tensor,
    current: torch.Tensor,
    lagged: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, ChordProjectionDiagnostics]:
    """Return the closest point on ``current -> lagged`` no faster than lagged.

    The operation is per image.  If the current endpoint residual is already
    no larger than the lagged residual it is returned exactly.  Otherwise the
    feasible chord is nonempty because lambda=1 is the lagged endpoint.
    """
    if x.shape != current.shape or x.shape != lagged.shape:
        raise ValueError("BVCP tensors must have identical shapes")
    if x.ndim < 2:
        raise ValueError("BVCP expects a batch dimension")

    batch = int(x.shape[0])
    a = (current - x).reshape(batch, -1)
    b = (lagged - current).reshape(batch, -1)
    current_sq = a.double().square().mean(dim=1)
    lagged_sq = (a + b).double().square().mean(dim=1)
    output = current.clone()
    lambdas: list[float] = []
    eligible = intervened = 0

    for index in range(batch):
        c0 = float(current_sq[index].item())
        c1 = float(lagged_sq[index].item())
        if c0 <= c1 + eps:
            lambdas.append(0.0)
            continue
        eligible += 1
        av = a[index].double()
        bv = b[index].double()
        qa = float(bv.square().mean().item())
        qb = float((2.0 * av * bv).mean().item())
        qc = c0 - c1
        lam = 1.0
        if qa > eps:
            discriminant = max(qb * qb - 4.0 * qa * qc, 0.0)
            roots = sorted((
                (-qb - math.sqrt(discriminant)) / (2.0 * qa),
                (-qb + math.sqrt(discriminant)) / (2.0 * qa),
                1.0,
            ))
            for candidate in roots:
                if candidate < -eps or candidate > 1.0 + eps:
                    continue
                value = min(1.0, max(0.0, float(candidate)))
                trial = av + value * bv
                if float(trial.square().mean().item()) <= c1 + 32.0 * eps:
                    lam = value
                    break
        if lam > eps:
            intervened += 1
            output[index] = current[index] + lam * (lagged[index] - current[index])
        lambdas.append(lam)

    projected_sq = (output - x).detach().double().reshape(batch, -1).square().mean(dim=1)
    return output, ChordProjectionDiagnostics(
        eligible=eligible,
        intervened=intervened,
        mean_lambda=float(sum(lambdas) / max(len(lambdas), 1)),
        current_rms=float(current_sq.mean().clamp_min(0).sqrt().item()),
        lagged_rms=float(lagged_sq.mean().clamp_min(0).sqrt().item()),
        projected_rms=float(projected_sq.mean().clamp_min(0).sqrt().item()),
    )


class BVCPMixin:
    """Mixin installed ahead of ``SBModel`` in the model MRO."""

    def _bvcp_enabled(self) -> bool:
        return bool(getattr(self.opt, "bvcp_enable", True))

    def _ensure_bvcp_lagged(self):
        if not self._bvcp_enabled():
            return None
        if getattr(self, "_bvcp_lagged_netG", None) is None:
            self._bvcp_lagged_netG = copy.deepcopy(_inner(self.netG)).to(self.device)
            self._bvcp_lagged_netG.eval()
            for parameter in self._bvcp_lagged_netG.parameters():
                parameter.requires_grad_(False)
        return self._bvcp_lagged_netG

    def _sync_bvcp_lagged(self) -> None:
        if not self._bvcp_enabled():
            return
        lagged = self._ensure_bvcp_lagged()
        lagged.load_state_dict(_inner(self.netG).state_dict(), strict=True)
        lagged.eval()

    def _rollout_endpoint(self, rollout_net, x, time_idx, z, *, stream):
        if not self._bvcp_enabled():
            return super()._rollout_endpoint(
                rollout_net, x, time_idx, z, stream=stream
            )
        current = rollout_net(x, time_idx, z)
        lagged_net = self._ensure_bvcp_lagged()
        with torch.no_grad():
            lagged = lagged_net(x, time_idx, z)
            projected, diag = minimum_velocity_chord_endpoint(
                x, current, lagged,
                eps=float(getattr(self.opt, "bvcp_root_epsilon", 1e-12)),
            )
        self._bvcp_eligible_transition_count += int(diag.eligible)
        self._bvcp_intervention_count += int(diag.intervened)
        self._bvcp_lambda_sum += float(diag.mean_lambda) * int(x.shape[0])
        self._bvcp_endpoint_count += int(x.shape[0])
        self._bvcp_last = {
            "stream": str(stream),
            "time_index": int(time_idx.reshape(-1)[0].detach().item()),
            "current_rms": diag.current_rms,
            "lagged_rms": diag.lagged_rms,
            "projected_rms": diag.projected_rms,
            "mean_lambda": diag.mean_lambda,
            "eligible": diag.eligible,
            "intervened": diag.intervened,
        }
        return projected

    def _before_generator_optimizer_step(self):
        if self._bvcp_enabled():
            # The current parameters are theta_k here.  Capturing before the
            # optimizer step makes the next forward compare theta_{k+1} with
            # exactly theta_k.
            self._sync_bvcp_lagged()
            self._bvcp_update_index += 1
        return super()._before_generator_optimizer_step()

    def set_search_step(self, step, total_steps=None):
        super().set_search_step(step, total_steps)
        if self._bvcp_enabled() and int(step) == 0 and not self._bvcp_loaded_state:
            self._sync_bvcp_lagged()
            self._bvcp_update_index = 0
            self._bvcp_eligible_transition_count = 0
            self._bvcp_intervention_count = 0
            self._bvcp_lambda_sum = 0.0
            self._bvcp_endpoint_count = 0
            self._bvcp_last = {}

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._bvcp_enabled():
            return state
        lagged = self._ensure_bvcp_lagged()
        state["bvcp"] = {
            "enabled": True,
            "update_index": int(self._bvcp_update_index),
            "eligible_transition_count": int(self._bvcp_eligible_transition_count),
            "intervention_count": int(self._bvcp_intervention_count),
            "lambda_sum": float(self._bvcp_lambda_sum),
            "endpoint_count": int(self._bvcp_endpoint_count),
            "last": copy.deepcopy(self._bvcp_last),
            "lagged_netG": {
                key: value.detach().cpu()
                for key, value in lagged.state_dict().items()
            },
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._bvcp_enabled():
            return
        saved = (state or {}).get("bvcp")
        if saved is None:
            # Shared e0 is a plain checkpoint.  set_search_step(0) will sync
            # the lagged copy to the loaded canonical generator.
            self._bvcp_loaded_state = False
            return
        if saved.get("enabled") is not True:
            raise RuntimeError("BVCP checkpoint does not contain an active operator")
        lagged_state = saved.get("lagged_netG")
        if not isinstance(lagged_state, dict) or not lagged_state:
            raise RuntimeError("BVCP checkpoint is missing the lagged generator")
        self._ensure_bvcp_lagged().load_state_dict(lagged_state, strict=True)
        self._bvcp_update_index = int(saved["update_index"])
        self._bvcp_eligible_transition_count = int(saved["eligible_transition_count"])
        self._bvcp_intervention_count = int(saved["intervention_count"])
        self._bvcp_lambda_sum = float(saved["lambda_sum"])
        self._bvcp_endpoint_count = int(saved["endpoint_count"])
        self._bvcp_last = copy.deepcopy(saved.get("last", {}))
        self._bvcp_loaded_state = True

    def _initialize_bvcp_state(self) -> None:
        self._bvcp_lagged_netG = None
        self._bvcp_loaded_state = False
        self._bvcp_update_index = 0
        self._bvcp_eligible_transition_count = 0
        self._bvcp_intervention_count = 0
        self._bvcp_lambda_sum = 0.0
        self._bvcp_endpoint_count = 0
        self._bvcp_last = {}

