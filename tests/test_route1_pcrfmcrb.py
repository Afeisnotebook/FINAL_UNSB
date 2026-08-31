from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from models.route1.pcrfmcrb import PCRFMCRBMixin
from models.route1.rfmcrb import RFMCRBMixin
from models.route1_pcrfmcrb_model import Route1PcrfmcrbModel
from operations import local_route1_freeze_residual_euclidean_synthesis as freeze
from research.local_route1.candidates import _validate_card
from research.local_route1.generation1_gates import (
    _disabled_spec,
    _pcammcrb_component_specs,
    _pcammcrb_invariants,
)
from research.local_route1.protocol import ProbeSpec, file_sha256


def _context(parent: str = "pcnr"):
    spec = ProbeSpec(
        id=freeze.CANDIDATE_ID,
        contract_id=freeze.CANDIDATE_ID,
        model="route1_pcrfmcrb",
        role="route1_candidate",
        method={
            "pcrfmcrb_enable": True,
            "pcrfmcrb_sampling_parent": parent,
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "rfmcrb_projection_epsilon": 1e-24,
        },
    )
    return SimpleNamespace(registration=SimpleNamespace(spec=spec))


def test_euclidean_synthesis_has_distinct_model_identity_and_dispatch(monkeypatch):
    names = [value.__name__ for value in Route1PcrfmcrbModel.__mro__]
    assert names.index("PCRFMCRBMixin") < names.index("PCAMMCRBMixin")
    assert issubclass(Route1PcrfmcrbModel, PCRFMCRBMixin)
    sentinel = object()
    monkeypatch.setattr(
        RFMCRBMixin,
        "_generator_optimizer_step",
        lambda self: sentinel,
    )
    instance = object.__new__(PCRFMCRBMixin)
    instance.opt = SimpleNamespace(pcrfmcrb_enable=True)
    assert PCRFMCRBMixin._generator_optimizer_step(instance) is sentinel


def test_euclidean_synthesis_gate_uses_only_repaired_euclidean_barrier():
    disabled = _disabled_spec(_context())
    assert disabled.model == "route1_pcrfmcrb"
    assert disabled.method["pcrfmcrb_enable"] is False

    sampling, barrier = _pcammcrb_component_specs(_context("pcnr"))
    assert sampling.model == "route1_pcnr"
    assert barrier.model == "route1_rfmcrb"
    assert barrier.method["rfmcrb_enable"] is True
    assert "rfammcrb_enable" not in barrier.method
    assert "ammcrb_enable" not in barrier.method

    rows = _pcammcrb_invariants(_context("pcrsmg_proposal"))
    assert rows and all(row["status"] == "PASS" for row in rows)
    identity = next(
        row for row in rows
        if row["name"] == "composite_barrier_identity_is_source_frozen"
    )
    assert identity["observed"] == (
        "residual_feasible_euclidean_without_absolute_margin"
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


def test_euclidean_route_requires_strict_same_host_rfmcrb(tmp_path, monkeypatch):
    sampling_path = tmp_path / "sampling.json"
    barrier_path = tmp_path / "barrier.json"
    sampling_path.write_text("{}\n", encoding="utf-8")
    barrier_path.write_text("{}\n", encoding="utf-8")
    receipts = {
        sampling_path.resolve(): _receipt(freeze.PCNR_ID),
        barrier_path.resolve(): _receipt(freeze.RFMCRB_ID),
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
    assert route["barrier_parent_candidate_id"] == freeze.RFMCRB_ID

    receipts[barrier_path.resolve()] = _receipt(
        freeze.RFMCRB_ID, authority_suffix="other"
    )
    with pytest.raises(RuntimeError, match="same-host authority"):
        freeze.adjudicate_parent_route(sampling_path, barrier_path)


def test_euclidean_card_satisfies_target_blind_registration_contract(tmp_path):
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
        card_path=(
            tmp_path / "derive" / "cards" / f"{freeze.CANDIDATE_ID}.json"
        ),
        card=card,
        matrix=matrix,
        matrix_sha256=file_sha256(matrix_path),
        atlas_sha256=file_sha256(atlas_path),
    )

