from __future__ import annotations

from research.local_route1.frontier_adjudication import SCHEMA as FRONTIER_SCHEMA
from research.local_route1.frontier_second_wave import (
    AMMCRB_ID,
    PCNR_ID,
    select_second_wave_parent_ablation,
)


def _adjudication(strict, recommended=None):
    return {
        "schema": FRONTIER_SCHEMA,
        "strict_gate_pass_candidate_ids": list(strict),
        "recommended_4090_replay_candidate_id": recommended,
    }


def _advancement(strict, near=()):
    return {
        "schema": "final-unsb-route1-frontier-advancement-classification-v1",
        "strict_candidate_ids": list(strict),
        "near_boundary_pending_target_blind_audit_ids": list(near),
    }


def test_both_strict_routes_unselected_independent_parent_ablation():
    route = select_second_wave_parent_ablation(
        _adjudication([AMMCRB_ID, PCNR_ID], recommended=AMMCRB_ID),
        _advancement([AMMCRB_ID, PCNR_ID]),
    )
    assert route["eligible"] is True
    assert route["parent_candidate_id"] == PCNR_ID
    assert route["ablation_role"] == "observable_only"


def test_sole_ammcrb_strict_routes_fixed_proposal_ablation():
    route = select_second_wave_parent_ablation(
        _adjudication([AMMCRB_ID], recommended=AMMCRB_ID),
        _advancement([AMMCRB_ID]),
    )
    assert route["eligible"] is True
    assert route["parent_candidate_id"] == AMMCRB_ID
    assert route["ablation_role"] == "proposal_only"


def test_near_boundary_reserves_slot_for_target_blind_revision_audit():
    route = select_second_wave_parent_ablation(
        _adjudication([PCNR_ID], recommended=PCNR_ID),
        _advancement([PCNR_ID], near=[AMMCRB_ID]),
    )
    assert route["eligible"] is False
    assert "TARGET_BLIND" in route["reason"]
    assert route["near_boundary_candidate_ids"] == [AMMCRB_ID]


def test_no_strict_parent_does_not_fill_gpu_with_an_ablation():
    route = select_second_wave_parent_ablation(
        _adjudication([], recommended=None), _advancement([]),
    )
    assert route["eligible"] is False
    assert route["parent_candidate_id"] is None

