from __future__ import annotations

from types import SimpleNamespace

import torch

from research.local_route1.protocol import ROOT
from research.local_route1.protocol import ProbeSpec
from research.local_route1.winner_ablations import WINNER_FAMILIES, _role_semantics
from research.local_route1.generation1_gates import _disabled_spec
from models.route1.amtnc_ablation import AMTNCAblationMixin
from models.route1.mcrb_ablation import norm_matched_negative_tangent


def test_generation2_and_third_mechanism_have_executable_ablation_families():
    expected = {
        "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS": "amtnc",
        "G1-03-STATE-FEEDBACK-MISSING": "mcrb",
    }
    for candidate_id, family in expected.items():
        row = WINNER_FAMILIES[candidate_id]
        assert row["family"] == family
        assert (ROOT / "src" / "models" / "route1" / f"{family}_ablation.py").is_file()
        assert (ROOT / "src" / "models" / f"route1_{family}_ablation_model.py").is_file()
        for role in ("proposal_only", "observable_only"):
            semantics = _role_semantics(family, role)
            assert semantics["method"][f"{family}_ablation_role"] == role
            assert semantics["identity"]
            assert semantics["falsifier"]


def test_amtnc_ablation_replica_dispatch_is_role_and_identity_bound():
    value = object.__new__(AMTNCAblationMixin)
    value.opt = SimpleNamespace(
        route1_ablation_enable=True,
        amtnc_ablation_role="proposal_only",
    )
    assert value._amtnc_replicates() == 2
    value.opt.amtnc_ablation_role = "observable_only"
    assert value._amtnc_replicates() == 1
    value.opt.route1_ablation_enable = False
    assert value._amtnc_replicates() == 1


def test_extended_ablation_zero_intervention_specs_disable_the_operator():
    for model, role_key in (
        ("route1_amtnc_ablation", "amtnc_ablation_role"),
        ("route1_mcrb_ablation", "mcrb_ablation_role"),
    ):
        spec = ProbeSpec(
            id=model, contract_id=model, model=model, role="candidate",
            method={"route1_ablation_enable": True, role_key: "proposal_only"},
        )
        context = SimpleNamespace(registration=SimpleNamespace(spec=spec))
        disabled = _disabled_spec(context)
        assert disabled.method["route1_ablation_enable"] is False
        assert disabled.method[role_key] == "proposal_only"


def test_mcrb_proposal_is_norm_matched_descent_and_zero_tangent_identity():
    native = [torch.tensor([3.0, 4.0])]
    tangent = [torch.tensor([0.0, 2.0])]
    proposal, diag = norm_matched_negative_tangent(native, tangent, eps=1e-24)
    assert diag["applied"] is True
    assert torch.isclose(proposal[0].norm(), native[0].norm())
    assert float((proposal[0] * tangent[0]).sum()) < 0.0

    identity, zero = norm_matched_negative_tangent(
        native, [torch.zeros_like(native[0])], eps=1e-24,
    )
    assert zero["applied"] is False
    assert identity[0] is native[0]
