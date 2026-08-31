from __future__ import annotations

from types import SimpleNamespace

from models.route1_hpcgr_model import Route1HpcgrModel
from research.local_route1.generation1_gates import (
    _disabled_spec,
    _hpcgr_component_specs,
    _hpcgr_invariants,
    _validate_hpcgr_execution_evidence,
)
from research.local_route1.protocol import ProbeSpec


def _context(role: str = "full"):
    spec = ProbeSpec(
        id="G3-HPCGR",
        contract_id="G3-HPCGR",
        model="route1_hpcgr",
        role="route1_candidate",
        method={
            "route1_hpcgr_enable": True,
            "hpcgr_role": role,
            "hnek_gamma": 0.25,
            "hnek_coord": "residual",
            "hnek_horizon_mode": "physical",
            "hnek_partial": "all",
        },
    )
    return SimpleNamespace(registration=SimpleNamespace(spec=spec))


def _bare_model(*, role: str, enabled: bool = True):
    model = object.__new__(Route1HpcgrModel)
    model.opt = SimpleNamespace(
        route1_hpcgr_enable=enabled,
        hpcgr_role=role,
    )
    return model


def test_hpcgr_role_routing_is_explicit_and_self_nulls():
    full = _bare_model(role="full")
    assert full._hpcgr_hnek_enabled() is True
    assert full._ablation_enabled() is True
    assert full._ablation_role() == "proposal_only"

    coordinate = _bare_model(role="coordinate_only")
    assert coordinate._hpcgr_hnek_enabled() is True
    assert coordinate._ablation_enabled() is False

    estimator = _bare_model(role="estimator_only")
    assert estimator._hpcgr_hnek_enabled() is False
    assert estimator._ablation_enabled() is True
    assert estimator._ablation_role() == "proposal_only"

    observer = _bare_model(role="observable_only")
    assert observer._hpcgr_hnek_enabled() is True
    assert observer._ablation_enabled() is True
    assert observer._ablation_role() == "observable_only"

    disabled = _bare_model(role="full", enabled=False)
    assert disabled._hpcgr_hnek_enabled() is False
    assert disabled._ablation_enabled() is False


def test_hpcgr_component_specs_freeze_both_parent_operators():
    coordinate, hnek, estimator, proposal, observable = _hpcgr_component_specs(
        _context()
    )
    assert coordinate.method["hpcgr_role"] == "coordinate_only"
    assert coordinate.method["hnek_gamma"] == 0.25
    assert hnek.model == "hnek_search"
    assert hnek.method == {
        "hnek_gamma": 0.25,
        "hnek_coord": "residual",
        "hnek_horizon_mode": "physical",
        "hnek_partial": "all",
    }
    assert estimator.method["hpcgr_role"] == "estimator_only"
    assert proposal.model == "route1_pcrsmg_ablation"
    assert proposal.method["pcrsmg_ablation_role"] == "proposal_only"
    assert observable.method["hpcgr_role"] == "observable_only"


def test_hpcgr_disabled_spec_is_exact_zero_intervention_role():
    disabled = _disabled_spec(_context())
    assert disabled.model == "route1_hpcgr"
    assert disabled.method["route1_hpcgr_enable"] is False


def test_hpcgr_math_invariants_all_pass():
    rows = _hpcgr_invariants()
    assert rows
    assert all(row["status"] == "PASS" for row in rows)
    names = {row["name"] for row in rows}
    assert "physical_horizon_coordinate_has_exact_boundary_identities" in names
    assert "conditional_gf_resampling_preserves_hnek_expected_field" in names


def test_hpcgr_execution_validator_requires_hnek_and_exact_schedule():
    method = {
        "pcrsmg_proposal": {
            "update_index": 8,
            "gf_bundle_count": 8,
            "last_schedule": [
                "NATIVE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE",
                "GF_COMMIT",
            ],
        },
        "hnek_active": True,
    }
    cross = {"rows": [{"candidate": {"method_diagnostics": method}}]}
    micro = {"method_diagnostics": {
        **method,
        "pcrsmg_proposal": {
            **method["pcrsmg_proposal"],
            "update_index": 400,
            "gf_bundle_count": 400,
        },
    }}
    result = _validate_hpcgr_execution_evidence(cross, micro)
    assert result["all_gf_bundle_counts_equal_updates"] is True
    assert result["hnek_active_all_states"] is True
    assert result["conditional_expected_field"] == "HNEK"

