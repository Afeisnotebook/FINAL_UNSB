from __future__ import annotations

import json

import pytest
import torch

from models.route1.ammcrb import project_actual_displacement_adam_metric
from models.route1.pcnr import EXPECTED_PCNR_SCHEDULE
from models.route1.pcrsmg import coupled_game_conditional_bias_example
from operations.local_route1_freeze_frontier_expansion import SPECS
from research.local_route1.candidates import CARD_REQUIRED_FIELDS, CARD_SCHEMA
from research.local_route1.generation1_gates import (
    _ammcrb_invariants,
    _pcnr_invariants,
)
from research.local_route1.protocol import ROOT, file_sha256


CARD_IDS = (
    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING",
    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER",
)


def test_frontier_cards_are_complete_and_evidence_bound():
    sources = {
        "historical_evidence_index_sha256": ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl",
        "mechanism_object_map_sha256": ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json",
        "reuse_boundary_sha256": ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json",
    }
    for candidate_id in CARD_IDS:
        path = ROOT / "research" / "local_route1" / "derivation_cards" / f"{candidate_id}.json"
        card = json.loads(path.read_text(encoding="utf-8"))
        assert card["schema"] == CARD_SCHEMA
        assert card["candidate_id"] == candidate_id
        assert all(card.get(field) not in (None, "") for field in CARD_REQUIRED_FIELDS)
        assert card["prior_equivalence_audit"]["equivalent_rerun"] is False
        assert card["paired_target_available_to_training"] is False
        for field, source in sources.items():
            assert card[field] == file_sha256(source)


def test_frontier_materializer_freezes_exactly_two_distinct_models():
    assert set(SPECS) == set(CARD_IDS)
    assert SPECS[CARD_IDS[0]]["model"] == "route1_pcnr"
    assert SPECS[CARD_IDS[1]]["model"] == "route1_ammcrb"
    assert SPECS[CARD_IDS[0]]["gate_callable"] == "run_pcnr_gate"
    assert SPECS[CARD_IDS[1]]["gate_callable"] == "run_ammcrb_gate"
    for candidate_id, spec in SPECS.items():
        for relative in spec["sources"]:
            assert (ROOT / relative).is_file(), (candidate_id, relative)


def test_pcnr_keeps_single_view_variance_and_refreshes_after_opponents():
    evidence = coupled_game_conditional_bias_example()
    assert evidence["fresh_conditional_bias_max"] == 0.0
    assert EXPECTED_PCNR_SCHEDULE == (
        "DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_VIEW", "GF_COMMIT",
    )
    assert all(row["status"] == "PASS" for row in _pcnr_invariants())


def test_ammcrb_safe_displacement_is_exact_identity():
    native = [torch.tensor([-2.0, 3.0], dtype=torch.float64)]
    tangent = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    inverse_metric = [torch.tensor([4.0, 1.0], dtype=torch.float64)]
    projected, diagnostics = project_actual_displacement_adam_metric(
        native, tangent, inverse_metric,
    )
    assert projected is native
    assert torch.equal(projected[0], native[0])
    assert diagnostics.unsafe is False


def test_ammcrb_projection_is_feasible_and_metric_scale_invariant():
    native = [torch.tensor([2.0, 3.0], dtype=torch.float64)]
    tangent = [torch.tensor([1.0, 1.0], dtype=torch.float64)]
    inverse_metric = [torch.tensor([4.0, 1.0], dtype=torch.float64)]
    projected, diagnostics = project_actual_displacement_adam_metric(
        native, tangent, inverse_metric,
    )
    scaled, scaled_diagnostics = project_actual_displacement_adam_metric(
        native, tangent, [inverse_metric[0] * 17.0],
    )
    assert diagnostics.unsafe is True
    assert float((projected[0] * tangent[0]).sum().item()) <= 0.0
    assert abs(float((projected[0] * tangent[0]).sum().item())) < 1e-10
    assert torch.allclose(projected[0], scaled[0], atol=1e-12, rtol=0.0)
    assert diagnostics.native_defect_directional_derivative == (
        scaled_diagnostics.native_defect_directional_derivative
    )


def test_ammcrb_rejects_nonpositive_metric():
    with pytest.raises(RuntimeError, match="finite and positive"):
        project_actual_displacement_adam_metric(
            [torch.tensor([1.0])],
            [torch.tensor([1.0])],
            [torch.tensor([0.0])],
        )


def test_frontier_executable_invariants_report_pass():
    for report in (_pcnr_invariants(), _ammcrb_invariants()):
        assert report
        assert all(row["status"] == "PASS" for row in report)
