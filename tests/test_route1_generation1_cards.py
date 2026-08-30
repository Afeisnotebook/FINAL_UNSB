from __future__ import annotations

import json
from pathlib import Path

from research.local_route1.candidates import CARD_REQUIRED_FIELDS, CARD_SCHEMA
from research.local_route1.protocol import ROOT, file_sha256
from operations.local_route1_freeze_generation1 import SPECS


CARD_ROOT = ROOT / "research" / "local_route1" / "derivation_cards"
MATRIX_SHA256 = "dc54569ac474706cbe001c061f94836f90b7a1baf0ba9be944e5ebbf4f87e0d3"
ATLAS_SHA256 = "965faf9a4eaf7279aed4caddc4379b5da3892d3024652210238e90d3fad3d2e1"


def _cards() -> list[dict]:
    paths = sorted(CARD_ROOT.glob("G1-*.json"))
    assert [path.name for path in paths] == [
        "G1-01-ROLLOUT-DISTRIBUTION-SPEED.json",
        "G1-02-SAMPLING-VARIANCE.json",
    ]
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def test_generation1_cards_are_complete_and_evidence_bound():
    expected_sources = {
        "historical_evidence_index_sha256": ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl",
        "mechanism_object_map_sha256": ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json",
        "reuse_boundary_sha256": ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json",
    }
    for card in _cards():
        assert card["schema"] == CARD_SCHEMA
        assert all(field in card and card[field] not in (None, "") for field in CARD_REQUIRED_FIELDS)
        assert card["prior_equivalence_audit"]["equivalent_rerun"] is False
        assert card["paired_target_available_to_training"] is False
        assert card["causal_matrix_sha256"] == MATRIX_SHA256
        assert card["reversal_atlas_sha256"] == ATLAS_SHA256
        assert set(card["ablation_definitions"]) == {
            "proposal_only", "observable_only", "projected_or_full",
        }
        for field, path in expected_sources.items():
            assert card[field] == file_sha256(path)


def test_generation1_authority_matches_the_frozen_evidence_boundary():
    cards = {card["candidate_id"]: card for card in _cards()}
    rollout = cards["G1-01-ROLLOUT-DISTRIBUTION-SPEED"]
    assert rollout["construction_authority"] == "eligible_method_specific_signal"
    assert rollout["target_blind_driver_probe"] == "dt"
    assert rollout["target_blind_driver_signal"] == "rollout_velocity_growth_margin"
    assert rollout["endpoint_law_change"] is False

    variance = cards["G1-02-SAMPLING-VARIANCE"]
    assert variance["construction_authority"] == "independent_unbiased_reparameterization"
    assert variance["objective_change"] is False
    assert variance["estimator_change"] is True
    assert variance["endpoint_law_change"] is False
    assert "Var((g1+g2)/2)=Var(g)/2" in variance["unbiased_proof"]


def test_generation1_materializer_registers_all_frozen_sources():
    assert set(SPECS) == {
        "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
        "G1-02-SAMPLING-VARIANCE",
    }
    for candidate_id, spec in SPECS.items():
        assert spec["model"] in {"route1_bvcp", "route1_rsmg"}
        assert spec["gate_callable"].startswith("run_")
        for relative in spec["sources"]:
            assert (ROOT / relative).is_file(), (candidate_id, relative)
