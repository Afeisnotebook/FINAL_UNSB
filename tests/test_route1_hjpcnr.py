from __future__ import annotations

from types import SimpleNamespace

from models.route1_hjpcnr_model import Route1HjpcnrModel
from research.local_route1.generation1_gates import (
    _disabled_spec,
    _hjpcnr_invariants,
    _validate_hjpcnr_execution_evidence,
)
from research.local_route1.protocol import ProbeSpec, load_protocol, probe_spec


def _context():
    spec = ProbeSpec(
        id="ABL-HJPCNR",
        contract_id="ABL-HJPCNR",
        model="route1_hjpcnr",
        role="gain_source_control",
        method={
            **probe_spec("hj", load_protocol()).method,
            "route1_hjpcnr_enable": True,
        },
    )
    return SimpleNamespace(registration=SimpleNamespace(spec=spec))


def _bare(enabled: bool):
    model = object.__new__(Route1HjpcnrModel)
    model.opt = SimpleNamespace(
        route1_hjpcnr_enable=enabled,
        hj_enable=True,
        hj_start_epoch=5,
        hj_search_start_step=-1,
        hj_search_duration_steps=0,
        lambda_NCE=1.0,
    )
    model.isTrain = True
    model.hj_epoch = 20
    return model


def test_hjpcnr_enable_controls_both_hj_objective_and_pcnr_transition():
    active = _bare(True)
    assert active._pcnr_enabled() is True
    assert active._hj_active() is True

    disabled = _bare(False)
    assert disabled._pcnr_enabled() is False
    assert disabled._hj_active() is False


def test_hjpcnr_disabled_spec_is_exact_plain_mode():
    disabled = _disabled_spec(_context())
    assert disabled.method["route1_hjpcnr_enable"] is False


def test_hjpcnr_invariants_keep_single_view_variance_boundary():
    rows = _hjpcnr_invariants()
    assert rows
    assert all(row["status"] == "PASS" for row in rows)
    names = {row["name"] for row in rows}
    assert "fresh_single_gf_view_preserves_hj_conditional_mean" in names
    assert "single_view_control_does_not_claim_variance_halving" in names


def test_hjpcnr_execution_evidence_binds_hj_steps_and_one_view_schedule():
    diagnostic = {
        "pcnr": {
            "update_index": 8,
            "de_view_count": 8,
            "gf_view_count": 8,
            "bundle_serial": 16,
            "last_schedule": [
                "DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_VIEW", "GF_COMMIT",
            ],
        },
        "hj_controller": {"_hj_active_optimizer_steps": 8},
    }
    cross = {"rows": [{"candidate": {"method_diagnostics": diagnostic}}]}
    micro = {"method_diagnostics": {
        "pcnr": {
            **diagnostic["pcnr"],
            "update_index": 400,
            "de_view_count": 400,
            "gf_view_count": 400,
            "bundle_serial": 800,
        },
        "hj_controller": {"_hj_active_optimizer_steps": 0},
    }}
    result = _validate_hjpcnr_execution_evidence(cross, micro)
    assert result["conditional_expected_field"] == "HJ"
    assert result["replica_averaging_used"] is False
    assert result["cross_state_hj_active_steps_equal_optimizer_updates"] is True
