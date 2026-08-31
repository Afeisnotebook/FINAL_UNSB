from __future__ import annotations

import json

import torch

from models.route1.mcrb import project_actual_displacement as superseded_projection
from models.route1.rfmcrb import project_actual_displacement_residual_feasible
from operations.local_route1_freeze_rfmcrb_replacement import (
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
from research.local_route1.generation1_gates import _rfmcrb_invariants
from research.local_route1.protocol import ROOT, file_sha256


def test_rfmcrb_safe_displacement_is_exact_identity():
    native = [torch.tensor([-2.0, 3.0], dtype=torch.float32)]
    projected, diagnostics = project_actual_displacement_residual_feasible(
        native, [torch.tensor([1.0, 0.0], dtype=torch.float32)],
    )
    assert projected is native
    assert projected[0] is native[0]
    assert diagnostics.unsafe is False


def test_rfmcrb_removes_small_tangent_absolute_margin_blowup():
    for scale in (1e-2, 1e-4, 1e-6, 1e-8):
        native = [torch.tensor([scale], dtype=torch.float32)]
        tangent = [torch.tensor([1e-8], dtype=torch.float32)]
        old, old_diag = superseded_projection(native, tangent)
        repaired, repaired_diag = project_actual_displacement_residual_feasible(
            native, tangent,
        )
        assert old_diag.correction_l2 / scale > 1e3
        assert float(old[0].item()) < -90.0
        assert repaired_diag.correction_l2 / scale <= 1.000001
        assert float(repaired[0].item()) <= 0.0


def test_rfmcrb_incident_card_and_sources_are_bound():
    incident_path = ROOT / INCIDENT
    incident = json.loads(incident_path.read_text(encoding="utf-8"))
    assert incident["candidate_id"] == PARENT_ID
    assert incident["classification"] == "implementation_failure"
    assert incident["scientific_conclusion_allowed"] is False
    assert incident["invalid_identity"]["projection_source_sha256"] == file_sha256(
        ROOT / incident["invalid_identity"]["projection_source"]
    )
    card_path = (
        ROOT / "research" / "local_route1" / "derivation_cards"
        / f"{CANDIDATE_ID}.json"
    )
    card = json.loads(card_path.read_text(encoding="utf-8"))
    assert card["schema"] == CARD_SCHEMA
    assert all(card.get(field) not in (None, "") for field in CARD_REQUIRED_FIELDS)
    assert card["engineering_replacement_for"] == PARENT_ID
    assert card["engineering_incident_path"] == INCIDENT
    assert card["engineering_incident_sha256"] == file_sha256(incident_path)
    assert card["paired_target_available_to_training"] is False
    for relative in SPEC["sources"]:
        assert (ROOT / relative).is_file()


def test_rfmcrb_engineering_replacement_is_generation_preserving(tmp_path):
    ledger_path = tmp_path / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(json.dumps({
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [{
            "candidate_id": PARENT_ID,
            "generation": 1,
            "parent_candidate_id": None,
            "parent_evidence": {"failure_type": "state_feedback_missing"},
            "construction_route": "state_conditional_self_null_intervention",
            "status": "FROZEN_FOR_GATES",
            "revision_count": 0,
            "algorithm_fingerprint": (
                "99aea7dd8caf48524dcd8a7216c403afee24c9365204d7df3bb3c3657de24df7"
            ),
        }],
    }), encoding="utf-8")
    result = register_engineering_replacement(
        tmp_path, PARENT_ID, CANDIDATE_ID, incident_relative=INCIDENT,
    )
    assert result["record"]["generation"] == 1
    assert result["record"]["engineering_replacement"][
        "consumes_causal_revision"
    ] is False


def test_rfmcrb_executable_invariants_pass():
    reports = _rfmcrb_invariants()
    assert reports
    assert all(row["status"] == "PASS" for row in reports)
