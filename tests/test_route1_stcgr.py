from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from research.local_route1.candidates import CARD_REQUIRED_FIELDS, CARD_SCHEMA
from research.local_route1.protocol import ROOT, file_sha256
from models.route1.stratified_time import (
    StratifiedTimeConditionalGFMixin,
    between_time_covariance_coefficient,
    map_excluding_index,
    ordered_time_pairs,
)


def test_derivation_card_is_complete_target_blind_and_evidence_bound() -> None:
    path = (
        ROOT / "research" / "local_route1" / "derivation_cards"
        / "G4-01-STRATIFIED-TIME-CONDITIONAL-GF.json"
    )
    card = json.loads(path.read_text(encoding="utf-8"))
    assert card["schema"] == CARD_SCHEMA
    assert all(card.get(field) not in (None, "") for field in CARD_REQUIRED_FIELDS)
    assert card["paired_target_available_to_training"] is False
    assert card["objective_change"] is False
    assert card["endpoint_law_change"] is False
    assert card["historical_evidence_index_sha256"] == file_sha256(
        ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl"
    )
    assert card["mechanism_object_map_sha256"] == file_sha256(
        ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json"
    )
    assert card["reuse_boundary_sha256"] == file_sha256(
        ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json"
    )


def test_exclusion_map_enumerates_every_off_diagonal_pair_once() -> None:
    size = 5
    observed = []
    for first in range(size):
        raw = torch.arange(size - 1, dtype=torch.long)
        second = map_excluding_index(raw, first, size)
        observed.extend((first, int(value)) for value in second)
    assert tuple(observed) == ordered_time_pairs(size)
    assert all(first != second for first, second in observed)
    assert [sum(first == index for first, _ in observed) for index in range(size)] == [4] * 5
    assert [sum(second == index for _, second in observed) for index in range(size)] == [4] * 5


def test_without_replacement_mean_is_unbiased_and_psd_better_than_iid() -> None:
    values = np.asarray([
        [1.0, -2.0],
        [3.0, 0.5],
        [-1.0, 4.0],
        [2.0, 2.0],
        [6.0, -3.0],
    ])
    pairs = ordered_time_pairs(len(values))
    estimates = np.stack([(values[i] + values[j]) / 2.0 for i, j in pairs])
    uniform_mean = values.mean(axis=0)
    assert np.allclose(estimates.mean(axis=0), uniform_mean)
    centered = values - uniform_mean
    between = centered.T @ centered / len(values)
    wor_covariance = np.cov(estimates, rowvar=False, bias=True)
    expected = between_time_covariance_coefficient(len(values)) * between
    assert np.allclose(wor_covariance, expected)
    iid_covariance = 0.5 * between
    eigvals = np.linalg.eigvalsh(iid_covariance - wor_covariance)
    assert np.all(eigvals >= -1e-12)
    assert np.any(eigvals > 1e-8)


def test_exclusion_map_fails_closed() -> None:
    with pytest.raises(ValueError, match="at least two"):
        map_excluding_index(torch.tensor([0]), 0, 1)
    with pytest.raises(ValueError, match="outside"):
        map_excluding_index(torch.tensor([0]), 5, 5)
    with pytest.raises(TypeError, match="integer"):
        map_excluding_index(torch.tensor([0.0]), 0, 5)
    with pytest.raises(ValueError, match="support"):
        map_excluding_index(torch.tensor([4]), 0, 5)


class _BaseSampler:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.opt = SimpleNamespace(route1_stcgr_enable=True, num_timesteps=5)

    def _prepare_second_gf_view(self, first_view) -> None:
        del first_view

    def _finalize_gf_view_bundle(self, views) -> None:
        del views

    def _sample_training_time_idx(self, size):
        return torch.randint(int(size), size=[1]).long()

    def get_extra_training_state(self):
        return {"base": True}

    def load_extra_training_state(self, state):
        assert state["base"] is True


class _Sampler(StratifiedTimeConditionalGFMixin, _BaseSampler):
    def __init__(self) -> None:
        super().__init__()
        self._initialize_stcgr_state()


def test_stratified_sampler_records_and_restores_complete_state() -> None:
    torch.manual_seed(2026)
    sampler = _Sampler()
    first = {"time_idx": sampler._sample_training_time_idx(5)}
    sampler._prepare_second_gf_view(first)
    second = {"time_idx": sampler._sample_training_time_idx(5)}
    sampler._finalize_gf_view_bundle([first, second])
    assert int(first["time_idx"]) != int(second["time_idx"])
    state = sampler.get_extra_training_state()
    assert state["stcgr"]["bundle_count"] == 1
    assert sum(state["stcgr"]["first_counts"]) == 1
    assert sum(state["stcgr"]["second_counts"]) == 1

    restored = _Sampler()
    restored.load_extra_training_state(state)
    assert restored.get_extra_training_state() == state


def test_disabled_sampler_dispatches_base_time_law_without_method_state() -> None:
    sampler = _Sampler()
    sampler.opt.route1_stcgr_enable = False
    torch.manual_seed(7)
    expected = torch.randint(5, size=[1]).long()
    torch.manual_seed(7)
    actual = sampler._sample_training_time_idx(5)
    assert torch.equal(actual, expected)
    assert "stcgr" not in sampler.get_extra_training_state()
