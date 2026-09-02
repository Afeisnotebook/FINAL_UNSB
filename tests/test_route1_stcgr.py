from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from operations.local_route1_freeze_stcgr import (
    CANDIDATE_ID,
    EVIDENCE,
    PARENT_IDS,
    SPEC,
)
from research.local_route1.candidates import (
    CARD_REQUIRED_FIELDS,
    CARD_SCHEMA,
    register_target_blind_successor,
)
from research.local_route1.generation1_gates import (
    _stcgr_invariants,
    _validate_stcgr_execution_evidence,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json
from models.route1.stratified_time import (
    StratifiedTimeConditionalGFMixin,
    between_time_covariance_coefficient,
    map_excluding_index,
    ordered_time_pairs,
)
from research.local_route1.stratified_time_audit import (
    _TimeMoment,
    covariance_trace_prediction,
    summarize_time_moments,
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


def test_fixed_state_receipt_and_frozen_source_spec_are_small25_only() -> None:
    receipt = json.loads((ROOT / EVIDENCE).read_text(encoding="utf-8"))
    assert receipt["candidate_id"] == CANDIDATE_ID
    assert receipt["small25_e200_authorized"] is True
    assert receipt["full_data_training_authorized"] is False
    assert receipt["paired_metric_control"] is False
    assert set(receipt["source_lanes"].values()) == set(PARENT_IDS)
    assert SPEC["model"] == "route1_stcgr"
    assert SPEC["method"] == {"route1_stcgr_enable": True}
    assert all((ROOT / relative).is_file() for relative in SPEC["sources"])


def test_target_blind_successor_registration_is_evidence_bound_and_idempotent(tmp_path) -> None:
    ledger = {
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [
            {"candidate_id": value, "status": "FROZEN_FOR_GATES"}
            for value in PARENT_IDS
        ],
    }
    write_json(tmp_path / "derive" / "HYPOTHESIS_LEDGER.json", ledger)
    first = register_target_blind_successor(
        tmp_path, CANDIDATE_ID, parent_candidate_ids=PARENT_IDS,
    )
    second = register_target_blind_successor(
        tmp_path, CANDIDATE_ID, parent_candidate_ids=PARENT_IDS,
    )
    assert first["status"] == "DERIVATION_REQUIRED"
    assert second["record"] == first["record"]
    assert first["record"]["generation"] == 4
    assert first["record"]["paired_controller_access"] is False


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


def test_fixed_state_covariance_prediction_has_registered_identity() -> None:
    result = covariance_trace_prediction(
        within_trace=8.0, between_trace=4.0, time_strata=5,
    )
    assert result["iid_pair_mean_covariance_trace"] == pytest.approx(6.0)
    assert result["without_replacement_pair_mean_covariance_trace"] == pytest.approx(5.5)
    assert result["without_replacement_to_iid_trace_ratio"] == pytest.approx(11 / 12)
    identity = covariance_trace_prediction(
        within_trace=8.0, between_trace=0.0, time_strata=5,
    )
    assert identity["without_replacement_to_iid_trace_ratio"] == 1.0


def test_stcgr_executable_invariants_and_pair_provenance() -> None:
    assert all(row["status"] == "PASS" for row in _stcgr_invariants())

    def diagnostic(updates: int) -> dict:
        pairs = [[0] * 5 for _ in range(5)]
        for index in range(updates):
            first = index % 5
            second = (first + 1) % 5
            pairs[first][second] += 1
        first_counts = [sum(row) for row in pairs]
        second_counts = [sum(pairs[row][column] for row in range(5)) for column in range(5)]
        return {
            "pcrsmg_proposal": {
                "update_index": updates,
                "gf_bundle_count": updates,
                "last_schedule": [
                    "NATIVE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE",
                    "GF_COMMIT",
                ],
            },
            "stcgr": {
                "num_timesteps": 5,
                "bundle_count": updates,
                "last_pair": [4, 0],
                "first_counts": first_counts,
                "second_counts": second_counts,
                "pair_counts": pairs,
            },
        }

    cross = {"rows": [{"candidate": {"method_diagnostics": diagnostic(8)}}]}
    result = _validate_stcgr_execution_evidence(
        cross, {"method_diagnostics": diagnostic(400)},
    )
    assert result["all_stcgr_pair_counts_equal_updates"] is True
    assert result["pair_coupling"] == "ordered_without_replacement"


def test_time_moment_summary_removes_finite_replicate_mean_noise() -> None:
    # All strata share the same true mean and have symmetric finite-replicate
    # noise.  The bias-corrected between-time term must self-null.
    moments = []
    for time_index in range(5):
        moment = _TimeMoment()
        for sign in (-1.0, 1.0):
            value = torch.tensor([3.0 + sign * (time_index + 1)], dtype=torch.float32)
            moment.add((value,))
        moments.append(moment)
    summary = summarize_time_moments(moments)
    assert summary["between_time_mean_covariance_trace"] == 0.0
    assert summary["without_replacement_to_iid_trace_ratio"] == 1.0


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
