from __future__ import annotations

import json

import torch

from models.route1.rfammcrb import (
    project_actual_displacement_residual_feasible_adam_metric,
)
from operations.local_route1_freeze_rfammcrb_replacement import (
    CANDIDATE_ID,
    INCIDENT,
    PARENT_ID,
    SPEC,
)
from research.local_route1.candidates import (
    CARD_REQUIRED_FIELDS,
    CARD_SCHEMA,
    register_engineering_replacement,
)
from research.local_route1.generation1_gates import _rfammcrb_invariants
from research.local_route1.projection_semantic_audit import invariant_summary
from research.local_route1.protocol import ROOT, file_sha256


def test_rfammcrb_safe_displacement_is_object_identity():
    native = [torch.tensor([-2.0, 3.0], dtype=torch.float32)]
    projected, diagnostics = (
        project_actual_displacement_residual_feasible_adam_metric(
            native,
            [torch.tensor([1.0, 0.0])],
            [torch.tensor([4.0, 1.0])],
        )
    )
    assert projected is native
    assert projected[0] is native[0]
    assert diagnostics.unsafe is False
    assert diagnostics.correction_l2 == 0.0


def test_rfammcrb_tiny_tangent_has_no_absolute_margin_blowup():
    summary = invariant_summary()
    assert summary["superseded_max_correction_to_native_ratio"] > 1e9
    assert summary["repaired_max_correction_to_native_ratio"] <= 1.000001
    assert summary["repaired_all_represented_feasible"] is True
    assert summary["paired_target_read"] is False


def test_rfammcrb_metric_common_scale_invariance():
    native = [torch.tensor([2.0, 3.0], dtype=torch.float32)]
    tangent = [torch.tensor([1.0, 1.0], dtype=torch.float32)]
    inverse = [torch.tensor([4.0, 1.0], dtype=torch.float32)]
    first, _ = project_actual_displacement_residual_feasible_adam_metric(
        native, tangent, inverse,
    )
    scaled, _ = project_actual_displacement_residual_feasible_adam_metric(
        native, tangent, [inverse[0] * 17.0],
    )
    assert torch.allclose(first[0], scaled[0], atol=2e-6, rtol=0.0)
    assert float((first[0].double() * tangent[0].double()).sum().item()) <= 0.0
    assert float((scaled[0].double() * tangent[0].double()).sum().item()) <= 0.0


def test_rfammcrb_random_represented_projections_are_feasible():
    generator = torch.Generator().manual_seed(2026)
    for index in range(20):
        native = [torch.randn(2048, generator=generator) * 10 ** (-(index % 7))]
        tangent = [torch.randn(2048, generator=generator) * 10 ** (-(index % 5))]
        if float((native[0].double() * tangent[0].double()).sum().item()) <= 0.0:
            tangent = [-tangent[0]]
        inverse = [torch.rand(2048, generator=generator) * 0.999 + 0.001]
        projected, diagnostics = (
            project_actual_displacement_residual_feasible_adam_metric(
                native, tangent, inverse,
            )
        )
        represented = float(
            (projected[0].double() * tangent[0].double()).sum().item()
        )
        assert represented <= 0.0
        assert diagnostics.residual_refinement_steps <= 8


def test_rfammcrb_incident_card_and_sources_are_immutable():
    incident_path = ROOT / INCIDENT
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert incident["candidate_id"] == PARENT_ID
    assert incident["classification"] == "implementation_failure"
    assert incident["scientific_conclusion_allowed"] is False
    assert incident["parent_mechanism_falsified"] is False
    assert incident["paired_metric_used_for_discovery_or_repair"] is False
    assert incident["invalid_identity"]["projection_source_sha256"] == file_sha256(
        ROOT / incident["invalid_identity"]["projection_source"]
    )

    card_path = (
        ROOT / "research" / "local_route1" / "derivation_cards"
        / f"{CANDIDATE_ID}.json"
    )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["schema"] == CARD_SCHEMA
    assert card["candidate_id"] == CANDIDATE_ID
    assert all(card.get(field) not in (None, "") for field in CARD_REQUIRED_FIELDS)
    assert card["engineering_replacement_for"] == PARENT_ID
    assert card["engineering_incident_path"] == INCIDENT
    assert card["engineering_incident_sha256"] == file_sha256(incident_path)
    assert card["paired_target_available_to_training"] is False
    for relative in SPEC["sources"]:
        assert (ROOT / relative).is_file()


def test_rfammcrb_engineering_replacement_keeps_parent_generation(tmp_path):
    ledger_path = tmp_path / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [{
            "candidate_id": PARENT_ID,
            "generation": 3,
            "parent_candidate_id": "G1-03-STATE-FEEDBACK-MISSING",
            "parent_evidence": {"failure_type": "state_feedback_missing"},
            "construction_route": "user_authorized_near_miss_frontier_expansion",
            "status": "FROZEN_FOR_GATES",
            "revision_count": 0,
            "algorithm_fingerprint": (
                "d6317303f3aa124b677d7637c97d5d08f24eb11ff526a83a2fe7d2f3b30851ea"
            ),
        }],
    }), encoding="utf-8")
    result = register_engineering_replacement(
        tmp_path, PARENT_ID, CANDIDATE_ID, incident_relative=INCIDENT,
    )
    assert result["status"] == "DERIVATION_REQUIRED"
    record = result["record"]
    assert record["generation"] == 3
    assert record["engineering_replacement"]["consumes_causal_revision"] is False
    assert record["engineering_replacement"]["incident_path"] == INCIDENT


def test_rfammcrb_executable_invariants_pass():
    reports = _rfammcrb_invariants()
    assert reports
    assert all(row["status"] == "PASS" for row in reports)
