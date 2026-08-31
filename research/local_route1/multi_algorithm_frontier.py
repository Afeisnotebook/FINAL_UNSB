"""Evidence-qualified multi-algorithm frontier for route-1 discovery.

This module deliberately separates *action priority* from *scientific
exclusivity*.  It can freeze a two-object Generation-3 composition only after
both parent mechanisms have completed e200 evidence, while preserving the
standalone parents and mathematically distinct siblings in the research
frontier.
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
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


HPCGR_ID = "G3-01-PHYSICAL-HORIZON-CONDITIONAL-GF-RESAMPLING"
PROPOSAL_ID = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
AMTNC_ID = "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS"
FRONTIER_SCHEMA = "final-unsb-route1-multi-algorithm-math-frontier-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_exact(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != payload:
            raise RuntimeError(f"frozen multi-algorithm artifact changed: {path}")
        return
    write_json(path, payload)


def _hnek_evidence(anchor: dict[str, Any], matrix: dict[str, Any]) -> dict[str, Any]:
    if anchor.get("schema") != "local-route1-anchor-summary-v1":
        raise RuntimeError("HPCGR requires the canonical long-anchor summary")
    summary = next(
        (row for row in anchor.get("summaries", []) if row.get("probe_id") == "hnek"),
        None,
    )
    causal = next(
        (row for row in matrix.get("probe_summaries", []) if row.get("probe") == "hnek"),
        None,
    )
    if not isinstance(summary, dict) or not isinstance(causal, dict):
        raise RuntimeError("HPCGR requires HNEK trajectory and causal evidence")
    e200 = next(
        (row for row in summary.get("trajectory", []) if int(row.get("epoch", -1)) == 200),
        None,
    )
    sustainable = int(causal.get("case_counts", {}).get("sustainable_on_both_states", 0))
    if not isinstance(e200, dict) or not (
        summary.get("complete_e200") is True
        and float(summary.get("late_three_mean_macro_psnr_delta", -1e9)) > 0.0
        and int(summary.get("late_points_with_four_of_six_positive_domains", 0)) >= 2
        and float(e200.get("macro_psnr_delta", -1e9)) > 0.0
        and e200.get("guardrails_pass") is True
        and sustainable >= 2
    ):
        raise RuntimeError("HNEK does not satisfy the preregistered composition parent gate")
    return {
        "probe_id": "hnek",
        "complete_e200": True,
        "late_three_mean_macro_psnr_delta": float(
            summary["late_three_mean_macro_psnr_delta"]
        ),
        "e200_macro_psnr_delta": float(e200["macro_psnr_delta"]),
        "e200_positive_domains": int(e200["positive_domains"]),
        "late_points_with_four_of_six_positive_domains": int(
            summary["late_points_with_four_of_six_positive_domains"]
        ),
        "sustainable_on_both_states": sustainable,
    }


def _proposal_evidence(output_root: Path) -> dict[str, Any]:
    receipt_path = (
        output_root / "operations" / "terminal_receipts" / f"{PROPOSAL_ID}.json"
    )
    card_path = output_root / "derive" / "cards" / f"{PROPOSAL_ID}.json"
    if not receipt_path.is_file() or not card_path.is_file():
        raise RuntimeError("HPCGR requires the source-bound proposal-only receipt/card")
    receipt = _read_json(receipt_path)
    trajectory_path = Path(str(receipt.get("trajectory_path", ""))).resolve()
    try:
        trajectory_path.relative_to(output_root.resolve())
    except ValueError as error:
        raise RuntimeError("proposal-only trajectory escapes the run root") from error
    if not trajectory_path.is_file():
        raise RuntimeError("proposal-only terminal trajectory is missing")
    trajectory = _read_json(trajectory_path)
    if not (
        receipt.get("candidate_id") == PROPOSAL_ID
        and receipt.get("status") == "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT"
        and receipt.get("trajectory_sha256") == file_sha256(trajectory_path)
        and receipt.get("derivation_card_sha256") == file_sha256(card_path)
        and receipt.get("paired_metrics_used_for_training_or_control") is False
        and receipt.get("confirmation20_opened") is False
        and trajectory.get("candidate_id") == PROPOSAL_ID
        and float(trajectory.get("late_three_mean_macro_psnr_delta", -1e9)) > 0.0
        and float(trajectory.get("e200_macro_psnr_delta", -1e9)) > 0.0
        and int(trajectory.get("late_points_with_four_of_six_positive_domains", 0)) >= 2
    ):
        raise RuntimeError("proposal-only does not satisfy the composition parent gate")
    return {
        "candidate_id": PROPOSAL_ID,
        "algorithm_fingerprint": receipt.get("algorithm_fingerprint"),
        "terminal_receipt_sha256": file_sha256(receipt_path),
        "derivation_card_sha256": file_sha256(card_path),
        "trajectory_sha256": file_sha256(trajectory_path),
        "late_three_mean_macro_psnr_delta": float(
            trajectory["late_three_mean_macro_psnr_delta"]
        ),
        "e200_macro_psnr_delta": float(trajectory["e200_macro_psnr_delta"]),
        "late_points_with_four_of_six_positive_domains": int(
            trajectory["late_points_with_four_of_six_positive_domains"]
        ),
    }


def select_hpcgr_parent_evidence(
    output_root: Path, *, anchor_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    if not matrix_path.is_file():
        raise RuntimeError("HPCGR requires a complete causal matrix")
    matrix = _read_json(matrix_path)
    if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
        raise RuntimeError("HPCGR causal matrix is incomplete")
    variance = next(
        (
            row for row in matrix.get("ranked_failure_mechanisms", [])
            if row.get("failure_type") == "sampling_variance"
        ),
        None,
    )
    if not isinstance(variance, dict) or (
        variance.get("candidate_generation_eligible") is not True
        or int(variance.get("cross_probe_support", 0)) < 3
    ):
        raise RuntimeError("sampling variance is not evidence-qualified across probes")
    anchor_path = Path(anchor_path or (
        output_root / "evidence" / "ANCHOR_TRAJECTORIES.json"
    )).resolve()
    anchor = _read_json(anchor_path)
    return {
        "hnek": _hnek_evidence(anchor, matrix),
        "pcrsmg_proposal_only": _proposal_evidence(output_root),
        "sampling_variance": {
            "evidence_rank": int(variance.get("evidence_rank", -1)),
            "cross_probe_support": int(variance["cross_probe_support"]),
            "supporting_probes": list(variance.get("supporting_probes", [])),
        },
        "anchor_summary_sha256": file_sha256(anchor_path),
        "causal_matrix_sha256": file_sha256(matrix_path),
    }


def _historical_bindings() -> dict[str, str]:
    return {
        "historical_evidence_index_sha256": file_sha256(
            ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl"
        ),
        "mechanism_object_map_sha256": file_sha256(
            ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json"
        ),
        "reuse_boundary_sha256": file_sha256(
            ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json"
        ),
    }


def build_hpcgr_card(
    output_root: Path, evidence: dict[str, Any],
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    if not atlas_path.is_file():
        raise RuntimeError("HPCGR requires the completed reversal atlas")
    cross_runtime = (
        ROOT / "evidence" / "remote_route1_offload"
        / "CROSS_RUNTIME_LONG_HORIZON_DIVERGENCE_20260830.json"
    )
    lineage = [
        "LONG_CAUSAL_MATRIX:HNEK:sustainable_on_both_states",
        "ANCHOR_TRAJECTORIES:HNEK:complete_e200_positive_parent",
        f"TERMINAL_RECEIPT:{PROPOSAL_ID}:complete_e200_positive_parent",
        "LONG_CAUSAL_MATRIX:sampling_variance:cross_probe_support=3",
        "GENERATION3:nested_coordinate_and_conditional_estimator_composition",
    ]
    if cross_runtime.is_file():
        lineage.append(
            "CROSS_RUNTIME_LONG_HORIZON_DIVERGENCE:HNEK:terminal_instability_not_mechanism_death:"
            + file_sha256(cross_runtime)
        )
    return {
        "schema": CARD_SCHEMA,
        "candidate_id": HPCGR_ID,
        "name": "Physical-Horizon Conditional G/F Resampling",
        "parent_evidence": evidence,
        "lineage_evidence": lineage,
        "prior_equivalence_audit": {
            "compared_implementations": [
                "hnek_search:gamma=.25/residual/physical/all",
                PROPOSAL_ID,
                HPCGR_ID,
            ],
            "material_difference": (
                "HNEK defines the physical-horizon residual bridge game while "
                "fresh two-view averaging is applied only to the joint G/F "
                "stochastic field at the realized post-D/E HNEK state. Neither "
                "standalone parent contains both operators."
            ),
            "equivalent_rerun": False,
        },
        "unsb_object": (
            "the physical-horizon residual bridge coordinate/objective and the "
            "player-conditional joint G/F stochastic gradient estimator"
        ),
        "formula": (
            "Let F_H(S;xi) be the frozen HNEK sequential game field. Commit D "
            "and E from one HNEK view. At the resulting state S_DE draw iid "
            "xi1,xi2 and commit G/F with g*=(g_H,GF(S_DE;xi1)+"
            "g_H,GF(S_DE;xi2))/2."
        ),
        "identity_or_unbiased_condition": (
            "route1_hpcgr_enable=false is exact plain; coordinate_only is exact "
            "HNEK; estimator_only is exact PC-RSMG proposal-only. Conditional "
            "on S_DE, E[g*|S_DE]=E[g_H,GF|S_DE]."
        ),
        "objective_change": True,
        "estimator_change": True,
        "coordinate_change": True,
        "endpoint_law_change": True,
        "target_inaccessibility_proof": (
            "The operator reads only current unpaired A/B samples, native bridge "
            "stochastic variables, HNEK physical horizon, and optimizer state. "
            "Discovery70, confirmation20 and paired metrics are not addressable."
        ),
        "expected_applicable_state": (
            "Long-horizon HNEK states where the physical coordinate remains useful "
            "but terminal G/F stochastic dispersion can erase its margin."
        ),
        "falsifying_experiment": (
            "From common e0, a complete small25 e200 trajectory whose late-three "
            "or e200 matched delta is non-positive, or whose guardrails are worse "
            "than both standalone parents, rejects the nested construction."
        ),
        "compute_cost": (
            "one HNEK view for D/E plus two serial fresh HNEK G/F views per update"
        ),
        "memory_cost": "two serial G/F graphs; no persistent teacher network",
        "recovery_state_cost": (
            "HNEK active bit, proposal update index, G/F bundle counter, schedule, "
            "ordinary optimizers/schedulers/samplers/RNG"
        ),
        "algorithm_hyperparameters": {
            "route1_hpcgr_enable": True,
            "hpcgr_role": "full",
            "hnek_gamma": 0.25,
            "hnek_coord": "residual",
            "hnek_horizon_mode": "physical",
            "hnek_partial": "all",
        },
        "algorithm_state_variables": [
            "hnek_active",
            "pcrsmg_proposal.update_index",
            "pcrsmg_proposal.gf_bundle_count",
            "pcrsmg_proposal.last_schedule",
        ],
        "ablation_definitions": {
            "proposal_only": (
                f"{PROPOSAL_ID} / route1_hpcgr:hpcgr_role=estimator_only"
            ),
            "observable_only": "route1_hpcgr:hpcgr_role=observable_only",
            "projected_or_full": HPCGR_ID,
            "coordinate_only": "hnek_search / route1_hpcgr:hpcgr_role=coordinate_only",
        },
        **_historical_bindings(),
        "construction_authority": "independent_unbiased_reparameterization",
        "unbiased_proof": (
            "Condition on the sigma-field after the single-view HNEK D and E "
            "commits. xi1 and xi2 are iid draws from the unchanged native HNEK "
            "measure, hence linearity gives E[(g1+g2)/2|S_DE]=E[g_H|S_DE] "
            "and conditional covariance Cov((g1+g2)/2|S_DE)=Cov(g_H|S_DE)/2."
        ),
        "paired_target_available_to_training": False,
        "causal_matrix_sha256": evidence["causal_matrix_sha256"],
        "reversal_atlas_sha256": file_sha256(atlas_path),
        "contract_id": HPCGR_ID,
    }


def build_hpcgr_implementation(card_path: Path) -> dict[str, Any]:
    sources = [
        "src/models/route1_hpcgr_model.py",
        "src/models/hnek/hnek_search.py",
        "src/models/hnek/hnek_kernel.py",
        "src/models/route1/pcrsmg_ablation.py",
        "src/models/route1/pcrsmg.py",
    ]
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": HPCGR_ID,
        "status": "FROZEN_FOR_GATES",
        "model": "route1_hpcgr",
        "method": {
            "route1_hpcgr_enable": True,
            "hpcgr_role": "full",
            "hnek_gamma": 0.25,
            "hnek_coord": "residual",
            "hnek_horizon_mode": "physical",
            "hnek_partial": "all",
        },
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "zero_intervention": {"route1_hpcgr_enable": False},
        "gate_hook": {
            "module": "research.local_route1.generation1_gates",
            "callable": "run_hpcgr_gate",
        },
        "source_files": [
            {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for relative in sources
        ],
        "derivation_card_sha256": file_sha256(card_path),
    }


def materialize_multi_algorithm_frontier(
    output_root: Path, *, anchor_path: Path | None = None,
) -> dict[str, Any]:
    """Freeze HPCGR without collapsing the research program to one winner."""
    output_root = Path(output_root).resolve()
    evidence = select_hpcgr_parent_evidence(output_root, anchor_path=anchor_path)
    card_path = output_root / "derive" / "cards" / f"{HPCGR_ID}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{HPCGR_ID}.json"
    )
    card = build_hpcgr_card(output_root, evidence)
    _write_exact(card_path, card)
    _write_exact(implementation_path, build_hpcgr_implementation(card_path))

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    expected = {
        "candidate_id": HPCGR_ID,
        "generation": 3,
        "parent_candidate_id": PROPOSAL_ID,
        "parent_evidence": evidence,
        "construction_route": "evidence_qualified_nested_coordinate_estimator",
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    matches = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == HPCGR_ID
    ]
    if not matches:
        ledger["records"].append(expected)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("HPCGR hypothesis identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES" and matches[0] != expected:
        raise RuntimeError("HPCGR hypothesis ledger slot changed")

    registration = freeze_candidate_derivation(output_root, HPCGR_ID)
    frontier = {
        "schema": FRONTIER_SCHEMA,
        "status": "MULTI_ALGORITHM_FRONTIER_FROZEN_FOR_HPCGR_GATE",
        "north_star": "discover sustained e200 algorithms, not select an early sole winner",
        "action_priority_candidate_id": HPCGR_ID,
        "action_priority_is_not_scientific_exclusivity": True,
        "maximum_concurrent_long_candidates": 3,
        "frontier": [
            {
                "id": "hnek_search:gamma=.25/residual/physical/all",
                "kind": "coordinate_objective_parent",
                "status": "POSITIVE_E200_PARENT_RUNTIME_FRAGILE",
                "evidence": evidence["hnek"],
            },
            {
                "id": PROPOSAL_ID,
                "kind": "conditional_unbiased_estimator_parent",
                "status": "POSITIVE_E200_SOURCE_BOUND_PARENT",
                "evidence": evidence["pcrsmg_proposal_only"],
            },
            {
                "id": HPCGR_ID,
                "kind": "nested_coordinate_estimator_composition",
                "status": "FROZEN_FOR_TARGET_BLIND_GATE",
                "algorithm_fingerprint": registration.algorithm_fingerprint,
            },
            {
                "id": AMTNC_ID,
                "kind": "exchange_unbiased_tangential_estimator",
                "status": "INDEPENDENT_E200_REPLAY_IN_FLIGHT",
            },
            {
                "id": "HJ-CONDITIONAL-GF-RESAMPLING",
                "kind": "possible_nce_objective_estimator_composition",
                "status": "MATHEMATICAL_INTERFACE_AND_NON_EQUIVALENCE_AUDIT_REQUIRED",
                "long_run_authorized": False,
            },
        ],
        "composition_policy": {
            "maximum_components": 2,
            "standalone_positive_required": True,
            "different_unsb_objects_required": True,
            "executable_component_identity_required": True,
            "paired_metrics_may_not_control_training": True,
        },
        "registration": registration.to_dict(),
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(
        output_root / "operations" / "MULTI_ALGORITHM_MATH_FRONTIER.json",
        frontier,
    )
    return frontier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-path", type=Path)
    args = parser.parse_args()
    result = materialize_multi_algorithm_frontier(
        args.output_root, anchor_path=args.anchor_path,
    )
    print(json.dumps({
        "status": result["status"],
        "action_priority_candidate_id": result["action_priority_candidate_id"],
        "algorithm_fingerprint": result["registration"]["algorithm_fingerprint"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
