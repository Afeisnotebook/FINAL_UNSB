from __future__ import annotations

from types import SimpleNamespace

import json
import pytest

from models.route1.pcrfammcrb import PCRFAMMCRBMixin
from models.route1.rfammcrb import RFAMMCRBMixin
from models.route1_pcrfammcrb_model import Route1PcrfammcrbModel
from operations import local_route1_freeze_residual_synthesis as freeze
from research.local_route1.candidates import _validate_card
from research.local_route1.generation1_gates import (
    _disabled_spec,
    _pcammcrb_component_specs,
    _pcammcrb_invariants,
)
from research.local_route1.protocol import ProbeSpec
from research.local_route1.protocol import file_sha256


def _context(parent: str = "pcnr"):
    spec = ProbeSpec(
        id=freeze.CANDIDATE_ID,
        contract_id=freeze.CANDIDATE_ID,
        model="route1_pcrfammcrb",
        role="route1_candidate",
        method={
            "pcrfammcrb_enable": True,
            "pcrfammcrb_sampling_parent": parent,
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "rfammcrb_projection_epsilon": 1e-24,
        },
    )
    return SimpleNamespace(registration=SimpleNamespace(spec=spec))


def test_residual_synthesis_has_distinct_model_identity_and_dispatch(monkeypatch):
    names = [value.__name__ for value in Route1PcrfammcrbModel.__mro__]
    assert names.index("PCRFAMMCRBMixin") < names.index("PCAMMCRBMixin")
    assert issubclass(Route1PcrfammcrbModel, PCRFAMMCRBMixin)
    sentinel = object()
    monkeypatch.setattr(
        RFAMMCRBMixin,
        "_generator_optimizer_step",
        lambda self: sentinel,
    )
    instance = object.__new__(PCRFAMMCRBMixin)
    assert PCRFAMMCRBMixin._generator_optimizer_step(instance) is sentinel


def test_residual_synthesis_gate_uses_only_repaired_barrier():
    disabled = _disabled_spec(_context())
    assert disabled.model == "route1_pcrfammcrb"
    assert disabled.method["pcrfammcrb_enable"] is False

    sampling, barrier = _pcammcrb_component_specs(_context("pcnr"))
    assert sampling.model == "route1_pcnr"
    assert barrier.model == "route1_rfammcrb"
    assert barrier.method["rfammcrb_enable"] is True
    assert "ammcrb_enable" not in barrier.method

    rows = _pcammcrb_invariants(_context("pcrsmg_proposal"))
    assert rows and all(row["status"] == "PASS" for row in rows)
    identity = next(
        row for row in rows
        if row["name"] == "composite_barrier_identity_is_source_frozen"
    )
    assert identity["observed"] == (
        "residual_feasible_adam_metric_without_absolute_margin"
    )


def _receipt(candidate_id: str, authority_suffix: str = "same"):
    return {
        "candidate_id": candidate_id,
        "base_e0_scientific_state_sha256": "e0-" + authority_suffix,
        "base_protocol_fingerprint": "protocol-" + authority_suffix,
        "manifest_sha256": "manifest-" + authority_suffix,
        "plain_e200_verification_sha256": "plain-" + authority_suffix,
        "trajectory_path": "unused-by-monkeypatch",
        "trajectory_sha256": "unused-by-monkeypatch",
    }


def test_parent_route_requires_two_strict_same_host_receipts(tmp_path, monkeypatch):
    sampling_path = tmp_path / "sampling.json"
    barrier_path = tmp_path / "barrier.json"
    sampling_path.write_text("{}\n", encoding="utf-8")
    barrier_path.write_text("{}\n", encoding="utf-8")
    receipts = {
        sampling_path.resolve(): _receipt(freeze.PCNR_ID),
        barrier_path.resolve(): _receipt(freeze.RFAMMCRB_ID),
    }
    monkeypatch.setattr(
        freeze, "_validate_receipt", lambda path: receipts[path.resolve()]
    )
    monkeypatch.setattr(freeze, "_trajectory_for_receipt", lambda receipt: {})
    monkeypatch.setattr(
        freeze,
        "classify_complete_trajectory",
        lambda receipt, trajectory: {
            "classification": freeze.STRICT,
            "checks": {"complete": True},
        },
    )
    route = freeze.adjudicate_parent_route(sampling_path, barrier_path)
    assert route["eligible"] is True
    assert route["sampling_parent"] == "pcnr"
    assert route["barrier_parent_candidate_id"] == freeze.RFAMMCRB_ID

    receipts[barrier_path.resolve()] = _receipt(
        freeze.RFAMMCRB_ID, authority_suffix="other"
    )
    with pytest.raises(RuntimeError, match="same-host authority"):
        freeze.adjudicate_parent_route(sampling_path, barrier_path)


def test_eligible_route_materializes_new_source_bound_identity(tmp_path, monkeypatch):
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / "derive").mkdir()
    (tmp_path / "derive" / "HYPOTHESIS_LEDGER.json").write_text(
        json.dumps({
            "schema": "final-unsb-route1-hypothesis-ledger-v1",
            "records": [],
        }),
        encoding="utf-8",
    )
    route = {
        "eligible": True,
        "reason": "TWO_INDEPENDENT_STRICT_SAME_HOST_PARENTS",
        "sampling_parent": "pcnr",
        "sampling_parent_candidate_id": freeze.PCNR_ID,
        "barrier_parent_candidate_id": freeze.RFAMMCRB_ID,
        "sampling_classification": {
            "classification": freeze.STRICT, "checks": {"complete": True},
        },
        "barrier_classification": {
            "classification": freeze.STRICT, "checks": {"complete": True},
        },
        "same_host_authority": {
            field: field + "-value" for field in freeze.AUTHORITY_FIELDS
        },
        "sampling_receipt_path": "sampling.json",
        "sampling_receipt_sha256": "1" * 64,
        "barrier_receipt_path": "barrier.json",
        "barrier_receipt_sha256": "2" * 64,
    }
    monkeypatch.setattr(
        freeze, "adjudicate_parent_route", lambda *args, **kwargs: route
    )
    monkeypatch.setattr(
        freeze,
        "freeze_candidate_derivation",
        lambda output, candidate: SimpleNamespace(
            to_dict=lambda: {
                "candidate_id": candidate,
                "algorithm_fingerprint": "frozen",
            }
        ),
    )
    result = freeze.materialize(
        tmp_path,
        sampling_receipt_path=tmp_path / "sampling.json",
        barrier_receipt_path=tmp_path / "barrier.json",
    )
    assert result["status"] == "SYNTHESIS_FROZEN_FOR_COMPATIBILITY_GATE"
    implementation = json.loads((
        tmp_path / "derive" / "implementations" / f"{freeze.CANDIDATE_ID}.json"
    ).read_text(encoding="utf-8"))
    assert implementation["model"] == "route1_pcrfammcrb"
    assert implementation["method"]["rfammcrb_projection_epsilon"] == 1e-24
    assert "ammcrb_projection_epsilon" not in implementation["method"]
    card = json.loads((
        tmp_path / "derive" / "cards" / f"{freeze.CANDIDATE_ID}.json"
    ).read_text(encoding="utf-8"))
    assert card["candidate_id"] == freeze.CANDIDATE_ID
    assert card["paired_target_available_to_training"] is False
    assert "fixed absolute" in card["prior_equivalence_audit"]["material_difference"]


def test_materialized_card_satisfies_target_blind_registration_contract(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    matrix_path = audit / "LONG_CAUSAL_MATRIX.json"
    atlas_path = audit / "LONG_REVERSAL_ATLAS.jsonl"
    matrix = {
        "status": "COMPLETE_CAUSAL_AUDIT",
        "ranked_failure_mechanisms": [{
            "failure_type": "state_feedback_missing",
            "candidate_generation_eligible": True,
            "supporting_probes": ["dt"],
            "eligible_method_specific_driver_signals_by_probe": {
                "dt": ["dt_covariance_mismatch_descent_margin"],
            },
        }],
        "target_blind_signal_screen": {
            "eligible_method_specific_driver_signals": {
                "dt": ["dt_covariance_mismatch_descent_margin"],
            },
        },
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    atlas_path.write_text("{}\n", encoding="utf-8")
    route = {
        "sampling_parent": "pcnr",
        "sampling_parent_candidate_id": freeze.PCNR_ID,
        "sampling_receipt_sha256": "1" * 64,
        "barrier_receipt_sha256": "2" * 64,
        "same_host_authority": {
            field: field + "-value" for field in freeze.AUTHORITY_FIELDS
        },
    }
    card = freeze._card(tmp_path, route)
    _validate_card(
        candidate_id=freeze.CANDIDATE_ID,
        card_path=tmp_path / "derive" / "cards" / f"{freeze.CANDIDATE_ID}.json",
        card=card,
        matrix=matrix,
        matrix_sha256=file_sha256(matrix_path),
        atlas_sha256=file_sha256(atlas_path),
    )
