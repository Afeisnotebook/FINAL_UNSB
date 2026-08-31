from __future__ import annotations

from types import SimpleNamespace

import torch

from models.route1.pcammcrb import (
    EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE,
    PCAMMCRBMixin,
    SAMPLING_PARENTS,
)
from models.route1_pcammcrb_model import Route1PcammcrbModel
from operations.local_route1_freeze_generation3_synthesis import (
    AMMCRB_ID,
    PCNR_ID,
    REPLACEMENT_CANDIDATE_ID,
    materialize,
    select_sampling_parent,
)
from research.local_route1.frontier_adjudication import SCHEMA as FRONTIER_SCHEMA
from research.local_route1.generation1_gates import (
    _disabled_spec,
    _displacement_correction_cosine,
    _pcammcrb_component_specs,
    _pcammcrb_invariants,
)
from research.local_route1.protocol import ProbeSpec


def _context(parent: str = "pcnr"):
    spec = ProbeSpec(
        id="G3-01-PLAYER-CONDITIONAL-ADAM-METRIC-COVARIANCE-BARRIER",
        contract_id="G3-01-PLAYER-CONDITIONAL-ADAM-METRIC-COVARIANCE-BARRIER",
        model="route1_pcammcrb",
        role="route1_candidate",
        method={
            "pcammcrb_enable": True,
            "pcammcrb_sampling_parent": parent,
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "ammcrb_projection_epsilon": 1e-24,
        },
    )
    return SimpleNamespace(registration=SimpleNamespace(spec=spec))


def test_synthesis_model_mro_places_sampler_before_barrier_before_unsb():
    names = [value.__name__ for value in Route1PcammcrbModel.__mro__]
    assert names.index("PCAMMCRBMixin") < names.index("PCNRMixin")
    assert names.index("PCNRMixin") < names.index("AMMCRBMixin")
    assert names.index("AMMCRBMixin") < names.index("SBModel")
    assert issubclass(Route1PcammcrbModel, PCAMMCRBMixin)


def test_synthesis_parent_and_two_view_schedule_are_closed():
    assert SAMPLING_PARENTS == ("pcnr", "pcrsmg_proposal")
    assert EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE == (
        "NATIVE_DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE",
        "GF_BARRIER_COMMIT",
    )
    for parent in SAMPLING_PARENTS:
        rows = _pcammcrb_invariants(_context(parent))
        assert rows
        assert all(row["status"] == "PASS" for row in rows)


def test_synthesis_zero_intervention_spec_disables_both_components():
    disabled = _disabled_spec(_context("pcnr"))
    assert disabled.model == "route1_pcammcrb"
    assert disabled.method["pcammcrb_enable"] is False
    assert disabled.method["pcammcrb_sampling_parent"] == "pcnr"


def test_component_specs_are_source_bound_and_never_mix_sampling_families():
    sampling, barrier = _pcammcrb_component_specs(_context("pcnr"))
    assert sampling.model == "route1_pcnr"
    assert sampling.method == {"pcnr_enable": True}
    assert barrier.model == "route1_ammcrb"
    assert barrier.method["ammcrb_enable"] is True

    sampling, barrier = _pcammcrb_component_specs(_context("pcrsmg_proposal"))
    assert sampling.model == "route1_pcrsmg_ablation"
    assert sampling.method["pcrsmg_ablation_role"] == "proposal_only"
    assert barrier.model == "route1_ammcrb"


def test_component_correction_cosine_and_self_null_rule():
    plain = [torch.tensor([1.0, 1.0], dtype=torch.float64)]
    sampling = [torch.tensor([2.0, 1.0], dtype=torch.float64)]
    barrier = [torch.tensor([1.5, 1.5], dtype=torch.float64)]
    result = _displacement_correction_cosine(sampling, barrier, plain)
    assert result["self_null_compatible"] is False
    assert abs(result["cosine"] - 2 ** -0.5) < 1e-12

    self_null = _displacement_correction_cosine(plain, barrier, plain)
    assert self_null["self_null_compatible"] is True
    assert self_null["cosine"] == 1.0


def _terminal(strict):
    return {
        "schema": FRONTIER_SCHEMA,
        "selection_seeds": [2026],
        "strict_gate_pass_candidate_ids": list(strict),
        "intermediate_metrics_used_for_routing": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_synthesis_route_prefers_strict_pcnr_and_requires_strict_ammcrb():
    both = select_sampling_parent(_terminal([PCNR_ID, AMMCRB_ID]))
    assert both["eligible"] is True
    assert both["sampling_parent"] == "pcnr"

    fallback = select_sampling_parent(_terminal([AMMCRB_ID]))
    assert fallback["eligible"] is True
    assert fallback["sampling_parent"] == "pcrsmg_proposal"

    blocked = select_sampling_parent(_terminal([PCNR_ID]))
    assert blocked["eligible"] is False
    assert blocked["sampling_parent"] is None


def test_old_synthesis_materializer_is_fail_closed_after_margin_incident(tmp_path):
    adjudication = tmp_path / "frontier.json"
    adjudication.write_text(
        __import__("json").dumps(_terminal([PCNR_ID, AMMCRB_ID])),
        encoding="utf-8",
    )
    result = materialize(tmp_path, adjudication)
    assert result["status"] == "SYNTHESIS_SUPERSEDED_NUMERICAL_SEMANTIC_INCIDENT"
    assert result["candidate_id"] is None
    assert result["replacement_candidate_id"] == REPLACEMENT_CANDIDATE_ID
    assert result["old_operator_long_run_authorized"] is False
    assert not (tmp_path / "derive" / "cards").exists()
