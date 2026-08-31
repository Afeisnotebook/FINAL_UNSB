"""Freeze the one-view HJ conditional-resampling gain-source control.

HJ-PCNR is not a new hyperparameter point.  It is the unique factorial control
between the continuous HJ parent and HJCGR: native one-view D/E are followed by
one fresh HJ G/F view, while HJCGR averages two such fresh views.  A complete
e200 HJ-PCNR trajectory therefore separates player-boundary resampling from
the two-view conditional-variance reduction on the HJ parent field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.local_route1.candidates import (
    CARD_SCHEMA,
    IMPLEMENTATION_SCHEMA,
    freeze_candidate_derivation,
)
from research.local_route1.hj_multi_algorithm_frontier import HJCGR_ID
from research.local_route1.multi_algorithm_frontier import (
    _historical_bindings,
    _read_json,
    _write_exact,
)
from research.local_route1.protocol import (
    ROOT,
    file_sha256,
    load_protocol,
    probe_spec,
)
from research.local_route1.runtime import write_json


HJPCNR_ID = "ABL-G3-02-HJCGR-SINGLE-VIEW"


def _parent_evidence(output_root: Path) -> dict[str, Any]:
    receipt_path = (
        output_root / "operations" / "terminal_receipts" / f"{HJCGR_ID}.json"
    )
    anchor_path = output_root / "evidence" / "ANCHOR_TRAJECTORIES.json"
    receipt = _read_json(receipt_path)
    anchor = _read_json(anchor_path)
    if not (
        receipt.get("status") == "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT"
        and receipt.get("candidate_id") == HJCGR_ID
        and receipt.get("training_git_commit")
        == receipt.get("verification_git_commit")
        and float(receipt.get("ranking_fields", {}).get(
            "late_three_mean_macro_psnr_delta", -1e9,
        )) > 0.0
        and float(receipt.get("ranking_fields", {}).get(
            "e200_macro_psnr_delta", -1e9,
        )) > 0.0
        and receipt.get("paired_metrics_used_for_training_or_control") is False
        and receipt.get("confirmation20_opened") is False
    ):
        raise RuntimeError("HJ-PCNR requires the complete strict HJCGR parent receipt")
    hj = next(
        (
            row for row in anchor.get("summaries", [])
            if isinstance(row, dict) and row.get("probe_id") == "hj"
        ),
        None,
    )
    if not isinstance(hj, dict) or hj.get("complete_e200") is not True:
        raise RuntimeError("HJ-PCNR requires the complete HJ parent trajectory")
    e200 = next(
        (row for row in hj.get("trajectory", []) if int(row.get("epoch", -1)) == 200),
        None,
    )
    if not isinstance(e200, dict):
        raise RuntimeError("HJ-PCNR HJ parent lacks e200")
    return {
        "parent_candidate_id": HJCGR_ID,
        "parent_terminal_receipt_sha256": file_sha256(receipt_path),
        "parent_trajectory_sha256": receipt["trajectory_sha256"],
        "parent_algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "hj_parent": {
            "late_three_mean_macro_psnr_delta": float(
                hj["late_three_mean_macro_psnr_delta"]
            ),
            "e200_macro_psnr_delta": float(e200["macro_psnr_delta"]),
        },
        "factorial_question": (
            "Does HJCGR retain benefit because it draws after D/E, or because the "
            "two fresh HJ G/F gradients are averaged before Adam?"
        ),
        "paired_parent_result_used_only_after_complete_e200": True,
    }


def build_card(output_root: Path, evidence: dict[str, Any]) -> dict[str, Any]:
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    frozen_hj = dict(probe_spec("hj", load_protocol()).method)
    return {
        "schema": CARD_SCHEMA,
        "candidate_id": HJPCNR_ID,
        "name": "HJ Player-Conditional Native G/F Resampling",
        "parent_candidate_id": HJCGR_ID,
        "parent_terminal_receipt_sha256": evidence[
            "parent_terminal_receipt_sha256"
        ],
        "ablation_role": "proposal_only",
        "gain_source_control_role": "single_fresh_hj_gf_view",
        "parent_evidence": evidence,
        "lineage_evidence": [
            "HJ:continuous Layer-0 structure-projected PatchNCE parent",
            f"{HJCGR_ID}:complete source-bound e200 strict parent",
            "PCNR:one fresh post-D/E native G/F view defines the resampling control",
        ],
        "prior_equivalence_audit": {
            "compared_implementations": ["hj", HJCGR_ID, HJPCNR_ID],
            "material_difference": (
                "HJ-PCNR uses one fresh post-D/E HJ G/F view. HJCGR uses two "
                "conditionally iid fresh HJ G/F views and averages their gradients."
            ),
            "equivalent_rerun": False,
        },
        "unsb_object": (
            "the player-boundary stochastic coupling and conditional covariance of "
            "the continuous HJ joint G/F estimator"
        ),
        "formula": (
            "Commit native one-view D/E. Given the realized post-D/E state S and "
            "continuous HJ controller q, draw one fresh xi and commit G/F with "
            "g_HJ(S,q;xi). No replica mean, projection strength, window or paired "
            "controller is introduced."
        ),
        "identity_or_unbiased_condition": (
            "Disabled dispatches plain exactly. Conditional on S, q and the official "
            "unpaired batch, E[g_HJ(S,q;xi_fresh)]=E[g_HJ|S,q]; unlike HJCGR, the "
            "single-view conditional covariance is retained."
        ),
        "objective_change": True,
        "estimator_change": True,
        "coordinate_change": False,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": (
            "The operator reads only the official unpaired batch, current model/"
            "optimizer/HJ state and native stochastic draws. Discovery pairs, paired "
            "metrics and confirmation20 are not addressable."
        ),
        "expected_applicable_state": (
            "The same continuous HJ states used by HJCGR; this control is executed "
            "only to attribute the already completed HJCGR gain."
        ),
        "falsifying_experiment": (
            "If HJ-PCNR matches HJCGR at e150/e175/e200, the two-view mean is not "
            "needed for HJCGR's observed long-horizon gain. If it resembles HJ or "
            "fails while HJCGR passes, conditional variance reduction is implicated."
        ),
        "compute_cost": "one native D/E view plus one fresh HJ G/F view",
        "memory_cost": "one serial HJ G/F graph; no persistent model copy",
        "recovery_state_cost": "HJ controller, PCNR view counters and ordinary full state",
        "algorithm_hyperparameters": {
            **frozen_hj,
            "route1_hjpcnr_enable": True,
        },
        "algorithm_state_variables": ["hj_controller", "pcnr"],
        "ablation_definitions": {
            "proposal_only": HJPCNR_ID,
            "observable_only": "HJ objective-only / HJCGR observable-only exact HJ",
            "projected_or_full": HJCGR_ID,
        },
        **_historical_bindings(),
        "construction_authority": "independent_unbiased_reparameterization",
        "unbiased_proof": (
            "Condition on the realized post-D/E state, HJ controller and fixed "
            "official unpaired batch. The fresh view is drawn from the unchanged HJ "
            "stochastic measure, so its gradient has the HJ conditional mean. A "
            "single view makes no covariance-halving claim."
        ),
        "paired_target_available_to_training": False,
        "causal_matrix_sha256": file_sha256(matrix_path),
        "reversal_atlas_sha256": file_sha256(atlas_path),
        "contract_id": HJPCNR_ID,
    }


def build_implementation(card_path: Path) -> dict[str, Any]:
    method = dict(_read_json(card_path)["algorithm_hyperparameters"])
    sources = [
        "src/models/route1_hjpcnr_model.py",
        "src/models/route1/pcnr.py",
        "src/models/hj/model.py",
        "src/models/hj/core.py",
        "src/models/hj/projection.py",
        "src/models/hj/structure.py",
        "research/local_route1/generation1_gates.py",
    ]
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": HJPCNR_ID,
        "status": "FROZEN_FOR_GATES",
        "model": "route1_hjpcnr",
        "method": method,
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "zero_intervention": {"route1_hjpcnr_enable": False},
        "gate_hook": {
            "module": "research.local_route1.generation1_gates",
            "callable": "run_hjpcnr_gate",
        },
        "source_files": [
            {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for relative in sources
        ],
        "derivation_card_sha256": file_sha256(card_path),
    }


def materialize_hjpcnr_gain_source(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    evidence = _parent_evidence(output_root)
    card_path = output_root / "derive" / "cards" / f"{HJPCNR_ID}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{HJPCNR_ID}.json"
    )
    _write_exact(card_path, build_card(output_root, evidence))
    _write_exact(implementation_path, build_implementation(card_path))

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    expected = {
        "candidate_id": HJPCNR_ID,
        "generation": 3,
        "parent_candidate_id": HJCGR_ID,
        "parent_evidence": evidence,
        "construction_route": "completed_parent_gain_source_factorial_ablation",
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    matches = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == HJPCNR_ID
    ]
    if not matches:
        ledger["records"].append(expected)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("HJ-PCNR hypothesis identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES" and matches[0] != expected:
        raise RuntimeError("HJ-PCNR hypothesis ledger slot changed")
    registration = freeze_candidate_derivation(output_root, HJPCNR_ID)
    return {
        "schema": "final-unsb-route1-hjpcnr-gain-source-freeze-v1",
        "status": "HJPCNR_GAIN_SOURCE_CONTROL_FROZEN_FOR_GATE",
        "candidate_id": HJPCNR_ID,
        "registration": registration.to_dict(),
        "paired_parent_result_used_only_after_complete_e200": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        materialize_hjpcnr_gain_source(args.output_root), ensure_ascii=False,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
