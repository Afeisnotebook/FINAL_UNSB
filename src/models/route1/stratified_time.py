"""Without-replacement bridge-time coupling for conditional G/F replicas.

The native UNSB bridge-time law is uniform on ``{0, ..., T-1}``.  This mixin
leaves the first post-D/E G/F view untouched and draws the second time
uniformly from the remaining ``T-1`` indices.  Both ordered-pair marginals are
therefore exactly uniform, while duplicate-time pairs are removed.

All latent, rollout-noise and PatchNCE randomness remains replica-local.  The
operator changes only the coupling between the two time indices and is used
with the arithmetic pre-Adam mean already implemented by Proposal-only.
"""

from __future__ import annotations

import copy
from typing import Any

import torch


def map_excluding_index(raw: torch.Tensor, excluded: int, size: int) -> torch.Tensor:
    """Map ``0..size-2`` bijectively onto ``0..size-1`` except ``excluded``."""
    if int(size) < 2:
        raise ValueError("without-replacement time sampling needs at least two times")
    if int(excluded) < 0 or int(excluded) >= int(size):
        raise ValueError(f"excluded time {excluded} is outside 0..{size - 1}")
    if raw.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("raw exclusion samples must be integer tensors")
    if bool((raw < 0).any()) or bool((raw >= int(size) - 1).any()):
        raise ValueError("raw exclusion sample is outside its support")
    return raw + (raw >= int(excluded)).to(dtype=raw.dtype)


def ordered_time_pairs(size: int) -> tuple[tuple[int, int], ...]:
    """Return the exact ordered support used by the stratified estimator."""
    if int(size) < 2:
        raise ValueError("ordered time pairs need at least two times")
    return tuple(
        (first, second)
        for first in range(int(size))
        for second in range(int(size))
        if second != first
    )


def between_time_covariance_coefficient(size: int) -> float:
    """Coefficient of the finite-population mean covariance for two draws."""
    if int(size) < 2:
        raise ValueError("covariance coefficient needs at least two times")
    return float(int(size) - 2) / float(2 * (int(size) - 1))


class StratifiedTimeConditionalGFMixin:
    """Couple the two Proposal G/F times without replacement."""

    def _stcgr_enabled(self) -> bool:
        return bool(getattr(self.opt, "route1_stcgr_enable", True))

    def _ablation_enabled(self) -> bool:
        return self._stcgr_enabled()

    @staticmethod
    def _view_time(view: dict[str, Any]) -> int:
        value = view.get("time_idx")
        if not torch.is_tensor(value) or value.numel() != 1:
            raise RuntimeError("stratified G/F view lacks one scalar time_idx")
        return int(value.detach().reshape(-1)[0].item())

    def _prepare_second_gf_view(self, first_view: dict[str, Any]) -> None:
        super()._prepare_second_gf_view(first_view)
        if not self._stcgr_enabled():
            return
        if self._stcgr_excluded_time is not None:
            raise RuntimeError("a stratified second-time request is already pending")
        self._stcgr_excluded_time = self._view_time(first_view)

    def _sample_training_time_idx(self, T):
        excluded = self._stcgr_excluded_time
        if excluded is None or not self._stcgr_enabled():
            return super()._sample_training_time_idx(T)
        raw = torch.randint(int(T) - 1, size=[1])
        mapped = map_excluding_index(raw, int(excluded), int(T))
        self._stcgr_excluded_time = None
        self._stcgr_pending_second_time = int(mapped.reshape(-1)[0].item())
        return (
            mapped.to(self.device)
            * torch.ones(size=[1], device=self.device, dtype=torch.long)
        ).long()

    def _finalize_gf_view_bundle(self, views: list[dict[str, Any]]) -> None:
        super()._finalize_gf_view_bundle(views)
        if not self._stcgr_enabled():
            return
        if len(views) != 2:
            raise RuntimeError("stratified conditional G/F requires two views")
        if self._stcgr_excluded_time is not None:
            raise RuntimeError("stratified time request was not consumed")
        first, second = (self._view_time(view) for view in views)
        if first == second:
            raise RuntimeError("without-replacement G/F times unexpectedly coincide")
        if self._stcgr_pending_second_time != second:
            raise RuntimeError("recorded stratified second time differs from the view")
        size = int(self.opt.num_timesteps)
        if not self._stcgr_pair_counts:
            self._stcgr_pair_counts = [[0 for _ in range(size)] for _ in range(size)]
            self._stcgr_first_counts = [0 for _ in range(size)]
            self._stcgr_second_counts = [0 for _ in range(size)]
        self._stcgr_pair_counts[first][second] += 1
        self._stcgr_first_counts[first] += 1
        self._stcgr_second_counts[second] += 1
        self._stcgr_last_pair = [first, second]
        self._stcgr_bundle_count += 1
        self._stcgr_pending_second_time = None

    def get_extra_training_state(self):
        state = super().get_extra_training_state()
        if not self._stcgr_enabled():
            return state
        if self._stcgr_excluded_time is not None or self._stcgr_pending_second_time is not None:
            raise RuntimeError("cannot checkpoint inside a stratified G/F bundle")
        state["stcgr"] = {
            "schema": "final-unsb-stcgr-state-v1",
            "num_timesteps": int(self.opt.num_timesteps),
            "bundle_count": int(self._stcgr_bundle_count),
            "last_pair": copy.deepcopy(self._stcgr_last_pair),
            "first_counts": list(self._stcgr_first_counts),
            "second_counts": list(self._stcgr_second_counts),
            "pair_counts": copy.deepcopy(self._stcgr_pair_counts),
            "time_marginal": "uniform",
            "pair_coupling": "ordered_without_replacement",
        }
        return state

    def load_extra_training_state(self, state):
        super().load_extra_training_state(state)
        if not self._stcgr_enabled():
            return
        saved = (state or {}).get("stcgr")
        if saved is None:
            self._initialize_stcgr_state()
            return
        if saved.get("schema") != "final-unsb-stcgr-state-v1":
            raise RuntimeError("stratified-time checkpoint schema mismatch")
        size = int(self.opt.num_timesteps)
        if int(saved.get("num_timesteps", -1)) != size:
            raise RuntimeError("stratified-time checkpoint changed num_timesteps")
        pair_counts = copy.deepcopy(saved.get("pair_counts", []))
        if len(pair_counts) != size or any(len(row) != size for row in pair_counts):
            raise RuntimeError("stratified-time pair-count shape mismatch")
        if any(int(pair_counts[index][index]) != 0 for index in range(size)):
            raise RuntimeError("stratified-time checkpoint contains diagonal pairs")
        bundle_count = int(saved.get("bundle_count", -1))
        if sum(sum(int(value) for value in row) for row in pair_counts) != bundle_count:
            raise RuntimeError("stratified-time pair counts do not match bundle count")
        first_counts = [int(value) for value in saved.get("first_counts", [])]
        second_counts = [int(value) for value in saved.get("second_counts", [])]
        if len(first_counts) != size or len(second_counts) != size:
            raise RuntimeError("stratified-time marginal-count shape mismatch")
        if sum(first_counts) != bundle_count or sum(second_counts) != bundle_count:
            raise RuntimeError("stratified-time marginal counts do not match bundle count")
        self._stcgr_bundle_count = bundle_count
        self._stcgr_last_pair = copy.deepcopy(saved.get("last_pair"))
        self._stcgr_first_counts = first_counts
        self._stcgr_second_counts = second_counts
        self._stcgr_pair_counts = pair_counts
        self._stcgr_excluded_time = None
        self._stcgr_pending_second_time = None

    def _initialize_stcgr_state(self) -> None:
        size = int(self.opt.num_timesteps)
        self._stcgr_bundle_count = 0
        self._stcgr_last_pair = None
        self._stcgr_first_counts = [0 for _ in range(size)]
        self._stcgr_second_counts = [0 for _ in range(size)]
        self._stcgr_pair_counts = [[0 for _ in range(size)] for _ in range(size)]
        self._stcgr_excluded_time = None
        self._stcgr_pending_second_time = None
