from __future__ import annotations

import json
import math

import torch

from models.route1.amtnc import adam_metric_tangential_gradient
from operations.local_route1_freeze_amtnc_revision import (
    CANDIDATE_ID,
    PARENT_ID,
    SPEC,
)
from research.local_route1.candidates import CARD_REQUIRED_FIELDS, CARD_SCHEMA
from research.local_route1.protocol import ROOT, file_sha256


def test_radial_disagreement_is_cancelled_and_tangent_is_retained():
    scales = (torch.ones(2),)
    radial, radial_diagnostics = adam_metric_tangential_gradient(
        (torch.tensor([3.0, 3.0]),),
        (torch.tensor([1.0, 1.0]),),
        scales,
    )
    assert torch.equal(radial[0], torch.tensor([2.0, 2.0]))
    assert radial_diagnostics["radial_fraction"] == 1.0

    tangent_first = (torch.tensor([3.0, 1.0]),)
    tangent, tangent_diagnostics = adam_metric_tangential_gradient(
        tangent_first,
        (torch.tensor([1.0, 3.0]),),
        scales,
    )
    assert torch.equal(tangent[0], tangent_first[0])
    assert tangent_diagnostics["radial_fraction"] == 0.0


def test_replica_exchange_average_is_native_consensus_in_weighted_metric():
    first = (torch.tensor([4.0, -1.0]), torch.tensor([2.0]))
    second = (torch.tensor([0.0, 3.0]), torch.tensor([-2.0]))
    scales = (torch.tensor([0.5, 2.0]), torch.tensor([3.0]))
    forward, _ = adam_metric_tangential_gradient(first, second, scales)
    reverse, _ = adam_metric_tangential_gradient(second, first, scales)
    for left, right, first_block, second_block in zip(
        forward, reverse, first, second,
    ):
        assert torch.allclose(
            (left + right) * 0.5,
            (first_block + second_block) * 0.5,
            atol=1e-7,
            rtol=0.0,
        )


def test_identical_replicas_are_returned_by_exact_identity():
    first = (torch.tensor([1.0, 2.0]),)
    result, diagnostics = adam_metric_tangential_gradient(
        first, (first[0].clone(),), (torch.ones(2),),
    )
    assert result is first
    assert torch.equal(result[0], first[0])
    assert diagnostics["disagreement_update_energy"] == 0.0


def test_zero_consensus_keeps_exchange_antisymmetric_native_draw():
    first = (torch.tensor([1.0, -2.0]),)
    second = (-first[0],)
    forward, _ = adam_metric_tangential_gradient(first, second, (torch.ones(2),))
    reverse, _ = adam_metric_tangential_gradient(second, first, (torch.ones(2),))
    assert torch.equal(forward[0], first[0])
    assert torch.equal(reverse[0], second[0])
    assert torch.equal((forward[0] + reverse[0]) * 0.5, torch.zeros(2))


def _legacy_finite_amtnc(first, second, scales):
    means = tuple((left + right) * 0.5 for left, right in zip(first, second))
    differences = tuple((left - right) * 0.5 for left, right in zip(first, second))
    consensus = torch.zeros((), dtype=torch.float64)
    disagreement = torch.zeros_like(consensus)
    cross = torch.zeros_like(consensus)
    for mean, difference, scale in zip(means, differences, scales):
        adam_mean = scale * mean
        adam_difference = scale * difference
        consensus += torch.sum(adam_mean * adam_mean, dtype=torch.float64)
        disagreement += torch.sum(
            adam_difference * adam_difference, dtype=torch.float64,
        )
        cross += torch.sum(adam_mean * adam_difference, dtype=torch.float64)
    coefficient = float((cross / consensus).item())
    return tuple(
        mean + difference - coefficient * mean
        for mean, difference in zip(means, differences)
    )


def test_finite_path_remains_bitwise_equal_to_frozen_operator():
    generator = torch.Generator().manual_seed(2026)
    first = (torch.randn(31, generator=generator), torch.randn(7, generator=generator))
    second = (torch.randn(31, generator=generator), torch.randn(7, generator=generator))
    scales = (
        torch.rand(31, generator=generator) + 0.25,
        torch.rand(7, generator=generator) + 0.25,
    )
    expected = _legacy_finite_amtnc(first, second, scales)
    observed, diagnostics = adam_metric_tangential_gradient(first, second, scales)
    assert all(torch.equal(left, right) for left, right in zip(expected, observed))
    assert "metric_product_precision" not in diagnostics


def test_float32_metric_product_overflow_uses_equivalent_float64_reduction():
    first = (torch.tensor([1.0e10, -2.0e10], dtype=torch.float32),)
    second = (torch.tensor([5.0e16, -3.0e16], dtype=torch.float32),)
    scales = (torch.full((2,), 1.0e8, dtype=torch.float32),)
    forward, diagnostics = adam_metric_tangential_gradient(first, second, scales)
    reverse, reverse_diagnostics = adam_metric_tangential_gradient(
        second, first, scales,
    )
    assert torch.isfinite(forward[0]).all()
    assert torch.isfinite(reverse[0]).all()
    assert diagnostics["metric_product_precision"] == "float64_overflow_fallback"
    assert reverse_diagnostics["metric_product_precision"] == (
        "float64_overflow_fallback"
    )
    assert torch.allclose(
        (forward[0] + reverse[0]) * 0.5,
        (first[0] + second[0]) * 0.5,
        rtol=1e-6,
        atol=0.0,
    )
    assert all(math.isfinite(float(value)) for key, value in diagnostics.items()
               if key != "metric_product_precision")


def test_identical_replica_overflow_retains_exact_identity():
    first = (torch.tensor([5.0e16], dtype=torch.float32),)
    result, diagnostics = adam_metric_tangential_gradient(
        first, (first[0].clone(),),
        (torch.tensor([1.0e8], dtype=torch.float32),),
    )
    assert result is first
    assert torch.equal(result[0], first[0])
    assert diagnostics["metric_product_precision"] == "float64_overflow_fallback"
    assert math.isfinite(diagnostics["consensus_update_energy"])


def test_revision_card_request_and_source_spec_are_frozen():
    request_path = (
        ROOT / "research" / "local_route1" / "revision_requests"
        / f"{CANDIDATE_ID}.json"
    )
    card_path = (
        ROOT / "research" / "local_route1" / "derivation_cards"
        / f"{CANDIDATE_ID}.json"
    )
    request = json.loads(request_path.read_text(encoding="utf-8"))
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert request["parent_candidate_id"] == PARENT_ID
    assert request["revision_candidate_id"] == CANDIDATE_ID
    assert request["fixed_window_or_handoff"] is False
    assert request["hyperparameter_grid_search"] is False
    assert request["paired_target_available_to_revision"] is False
    assert card["schema"] == CARD_SCHEMA
    assert card["candidate_id"] == CANDIDATE_ID
    assert card["parent_candidate_id"] == PARENT_ID
    assert card["revision_request_sha256"] == file_sha256(request_path)
    assert all(field in card and card[field] not in (None, "") for field in CARD_REQUIRED_FIELDS)
    assert card["construction_authority"] == "independent_unbiased_reparameterization"
    assert card["objective_change"] is False
    assert card["estimator_change"] is True
    assert card["endpoint_law_change"] is False
    assert SPEC["model"] == "route1_amtnc"
    assert SPEC["gate_callable"] == "run_amtnc_gate"
    for relative in SPEC["sources"]:
        assert (ROOT / relative).is_file(), relative
