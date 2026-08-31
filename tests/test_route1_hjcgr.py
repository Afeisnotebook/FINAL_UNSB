from __future__ import annotations

from types import SimpleNamespace

from models.route1_hjcgr_model import (
    Route1HjcgrModel,
    reduce_hj_replica_transitions,
)
from research.local_route1.generation1_gates import (
    _disabled_spec,
    _hjcgr_component_specs,
    _hjcgr_invariants,
    _validate_hjcgr_execution_evidence,
)
from research.local_route1.protocol import ProbeSpec, load_protocol, probe_spec


def _method(role: str = "full") -> dict:
    return {
        **probe_spec("hj", load_protocol()).method,
        "route1_hjcgr_enable": True,
        "hjcgr_role": role,
    }


def _context(role: str = "full"):
    spec = ProbeSpec(
        id="G3-HJCGR",
        contract_id="G3-HJCGR",
        model="route1_hjcgr",
        role="route1_candidate",
        method=_method(role),
    )
    return SimpleNamespace(registration=SimpleNamespace(spec=spec))


def _bare_model(*, role: str, enabled: bool = True):
    model = object.__new__(Route1HjcgrModel)
    model.opt = SimpleNamespace(route1_hjcgr_enable=enabled, hjcgr_role=role)
    return model


def test_hjcgr_role_routing_separates_objective_from_estimator():
    full = _bare_model(role="full")
    assert full._hjcgr_hj_enabled() is True
    assert full._ablation_enabled() is True

    objective = _bare_model(role="objective_only")
    assert objective._hjcgr_hj_enabled() is True
    assert objective._ablation_enabled() is False

    estimator = _bare_model(role="estimator_only")
    assert estimator._hjcgr_hj_enabled() is False
    assert estimator._ablation_enabled() is True
    assert estimator._ablation_role() == "proposal_only"

    observer = _bare_model(role="observable_only")
    assert observer._hjcgr_hj_enabled() is True
    assert observer._ablation_role() == "observable_only"

    disabled = _bare_model(role="full", enabled=False)
    assert disabled._hjcgr_hj_enabled() is False
    assert disabled._ablation_enabled() is False


def test_hjcgr_replica_state_reduction_advances_counts_once_and_means_diagnostics():
    baseline = {
        "_hj_step_in_epoch": 4,
        "_hj_gate_sum": 1.0,
        "_hj_risk_sum": 2.0,
        "_hj_probe_sum": 3.0,
        "_hj_risk_positive_sum": 4.0,
        "_hj_sb_grad_norm": 0.5,
        "_hj_active_optimizer_steps": 9,
    }
    first = {
        **baseline,
        "_hj_step_in_epoch": 5,
        "_hj_gate_sum": 1.2,
        "_hj_risk_sum": 2.4,
        "_hj_probe_sum": 3.6,
        "_hj_risk_positive_sum": 4.8,
        "_hj_sb_grad_norm": 1.0,
        "_hj_active_optimizer_steps": 10,
    }
    second = {
        **first,
        "_hj_gate_sum": 1.4,
        "_hj_risk_sum": 2.8,
        "_hj_probe_sum": 4.0,
        "_hj_risk_positive_sum": 5.2,
        "_hj_sb_grad_norm": 2.0,
    }
    reduced = reduce_hj_replica_transitions(baseline, [first, second])
    assert reduced["_hj_step_in_epoch"] == 5
    assert reduced["_hj_active_optimizer_steps"] == 10
    assert abs(reduced["_hj_gate_sum"] - 1.3) < 1e-12
    assert abs(reduced["_hj_risk_sum"] - 2.6) < 1e-12
    assert reduced["_hj_sb_grad_norm"] == 1.5


def test_hjcgr_component_specs_are_exact_frozen_parents():
    objective, hj, estimator, proposal, observer = _hjcgr_component_specs(_context())
    assert objective.method["hjcgr_role"] == "objective_only"
    assert hj.model == "hj"
    assert hj.method == probe_spec("hj", load_protocol()).method
    assert estimator.method["hjcgr_role"] == "estimator_only"
    assert proposal.model == "route1_pcrsmg_ablation"
    assert proposal.method["pcrsmg_ablation_role"] == "proposal_only"
    assert observer.method["hjcgr_role"] == "observable_only"


def test_hjcgr_disabled_spec_is_plain_identity_role():
    disabled = _disabled_spec(_context())
    assert disabled.method["route1_hjcgr_enable"] is False


def test_hjcgr_invariants_all_pass():
    rows = _hjcgr_invariants()
    assert rows
    assert all(row["status"] == "PASS" for row in rows)


def test_hjcgr_execution_validator_checks_once_per_update_hj_state():
    method = {
        "pcrsmg_proposal": {
            "update_index": 8,
            "gf_bundle_count": 8,
            "last_schedule": [
                "NATIVE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE",
                "GF_COMMIT",
            ],
        },
        "hj_controller": {
            "_hj_step_in_epoch": 8,
            "_hj_active_optimizer_steps": 8,
        },
    }
    cross = {"rows": [{"candidate": {"method_diagnostics": method}}]}
    micro = {"method_diagnostics": {
        **method,
        "pcrsmg_proposal": {
            **method["pcrsmg_proposal"],
            "update_index": 400,
            "gf_bundle_count": 400,
        },
        "hj_controller": {
            "_hj_step_in_epoch": 100,
            "_hj_active_optimizer_steps": 0,
        },
    }}
    result = _validate_hjcgr_execution_evidence(cross, micro)
    assert result["conditional_expected_field"] == "HJ"
    assert result["cross_state_hj_active_steps_equal_optimizer_updates"] is True

