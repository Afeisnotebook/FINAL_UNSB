"""Freeze the HJ-objective member of the related conditional-estimator family."""

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
from research.local_route1.multi_algorithm_frontier import (
    AMTNC_ID,
    FRONTIER_SCHEMA,
    HPCGR_ID,
    PROPOSAL_ID,
    _historical_bindings,
    _proposal_evidence,
    _read_json,
    _write_exact,
)
from research.local_route1.protocol import ROOT, file_sha256, object_sha256
from research.local_route1.runtime import write_json


HJCGR_ID = "G3-02-HJ-CONDITIONAL-GF-RESAMPLING"
PARENT_AUTHORITY_SCHEMA = "final-unsb-route1-hjcgr-parent-authority-v1"
PARENT_AUTHORITY_STATUS = "PORTABLE_HJCGR_CONSTRUCTION_AUTHORITY"
PARENT_AUTHORITY_HOST = "remote4090-authoritative-route1"


def _sha256_string(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_parent_evidence(value: dict[str, Any]) -> dict[str, Any]:
    """Validate evidence embedded in portable construction authority."""
    if set(value) != {
        "hj", "hj_sampling_variance", "pcrsmg_proposal_only",
        "anchor_summary_sha256", "causal_matrix_sha256",
    }:
        raise RuntimeError("HJCGR portable parent evidence shape changed")
    hj = value.get("hj")
    variance = value.get("hj_sampling_variance")
    proposal = value.get("pcrsmg_proposal_only")
    if not all(isinstance(row, dict) for row in (hj, variance, proposal)):
        raise RuntimeError("HJCGR portable parent evidence is malformed")
    if not (
        hj.get("probe_id") == "hj"
        and hj.get("complete_e200") is True
        and float(hj.get("late_three_mean_macro_psnr_delta", -1e9)) > 0.0
        and float(hj.get("e200_macro_psnr_delta", -1e9)) > 0.0
        and int(hj.get("late_points_with_four_of_six_positive_domains", 0)) >= 2
        and isinstance(hj.get("state_operator_case_counts"), dict)
        and int(hj.get("next_batch_consensus_negative_rows", 0)) > 0
    ):
        raise RuntimeError("portable HJ evidence no longer satisfies its source gate")
    if not (
        int(variance.get("independent_batch_variance_dominated_rows", 0)) > 0
        and int(variance.get("latent_time_bridge_variance_dominated_rows", 0)) > 0
        and 0.0 <= float(variance.get("independent_batch_variance_fraction", -1.0)) <= 1.0
        and 0.0 <= float(variance.get("latent_time_bridge_variance_fraction", -1.0)) <= 1.0
    ):
        raise RuntimeError("portable HJ variance evidence is inadmissible")
    if not (
        proposal.get("candidate_id") == PROPOSAL_ID
        and isinstance(proposal.get("algorithm_fingerprint"), str)
        and bool(proposal.get("algorithm_fingerprint"))
        and float(proposal.get("late_three_mean_macro_psnr_delta", -1e9)) > 0.0
        and float(proposal.get("e200_macro_psnr_delta", -1e9)) > 0.0
        and int(proposal.get("late_points_with_four_of_six_positive_domains", 0)) >= 2
    ):
        raise RuntimeError("portable proposal-only evidence no longer satisfies its source gate")
    for key in ("anchor_summary_sha256", "causal_matrix_sha256"):
        if not _sha256_string(value.get(key)):
            raise RuntimeError(f"portable HJCGR evidence lacks {key}")
    for key in (
        "terminal_receipt_sha256", "derivation_card_sha256", "trajectory_sha256",
    ):
        if not _sha256_string(proposal.get(key)):
            raise RuntimeError(f"portable proposal-only evidence lacks {key}")
    return value


def validate_hjcgr_parent_authority(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema") != PARENT_AUTHORITY_SCHEMA
        or value.get("status") != PARENT_AUTHORITY_STATUS
    ):
        raise RuntimeError("HJCGR parent authority schema/status mismatch")
    fixed = {
        "candidate_id": HJCGR_ID,
        "source_host_label": PARENT_AUTHORITY_HOST,
        "construction_authority_only": True,
        "destination_result_may_not_requalify_algorithm": True,
        "cross_host_metrics_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise RuntimeError(f"HJCGR parent authority changed: {key}")
    evidence = value.get("parent_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("HJCGR parent authority lacks embedded evidence")
    _validate_parent_evidence(evidence)
    if value.get("parent_evidence_sha256") != object_sha256(evidence):
        raise RuntimeError("HJCGR embedded parent evidence changed")
    artifacts = value.get("source_artifact_sha256")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "anchor_summary", "causal_matrix", "reversal_atlas",
        "proposal_terminal_receipt", "proposal_derivation_card",
        "proposal_trajectory",
    }:
        raise RuntimeError("HJCGR source artifact binding changed")
    proposal = evidence["pcrsmg_proposal_only"]
    expected_hashes = {
        "anchor_summary": evidence["anchor_summary_sha256"],
        "causal_matrix": evidence["causal_matrix_sha256"],
        "proposal_terminal_receipt": proposal["terminal_receipt_sha256"],
        "proposal_derivation_card": proposal["derivation_card_sha256"],
        "proposal_trajectory": proposal["trajectory_sha256"],
    }
    for key, expected in expected_hashes.items():
        if artifacts.get(key) != expected:
            raise RuntimeError(f"HJCGR source artifact/evidence mismatch: {key}")
    if not _sha256_string(artifacts.get("reversal_atlas")):
        raise RuntimeError("HJCGR parent authority lacks the source reversal atlas")
    return value


def export_hjcgr_parent_authority(
    output_root: Path,
    *,
    anchor_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Export authoritative-host construction evidence for exact replay."""
    output_root = Path(output_root).resolve()
    evidence = select_hjcgr_parent_evidence(output_root, anchor_path=anchor_path)
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    if not atlas_path.is_file():
        raise RuntimeError("HJCGR parent authority requires the reversal atlas")
    result = {
        "schema": PARENT_AUTHORITY_SCHEMA,
        "status": PARENT_AUTHORITY_STATUS,
        "candidate_id": HJCGR_ID,
        "source_host_label": PARENT_AUTHORITY_HOST,
        "construction_authority_only": True,
        "destination_result_may_not_requalify_algorithm": True,
        "parent_evidence": evidence,
        "parent_evidence_sha256": object_sha256(evidence),
        "source_artifact_sha256": {
            "anchor_summary": evidence["anchor_summary_sha256"],
            "causal_matrix": evidence["causal_matrix_sha256"],
            "reversal_atlas": file_sha256(atlas_path),
            "proposal_terminal_receipt": evidence["pcrsmg_proposal_only"][
                "terminal_receipt_sha256"
            ],
            "proposal_derivation_card": evidence["pcrsmg_proposal_only"][
                "derivation_card_sha256"
            ],
            "proposal_trajectory": evidence["pcrsmg_proposal_only"][
                "trajectory_sha256"
            ],
        },
        "cross_host_metrics_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    validate_hjcgr_parent_authority(result)
    output_path = Path(output_path or (
        output_root / "operations" / "HJCGR_PARENT_CONSTRUCTION_AUTHORITY.json"
    )).resolve()
    _write_exact(output_path, result)
    return result


def select_hjcgr_parent_evidence(
    output_root: Path,
    *,
    anchor_path: Path | None = None,
    proposal_parent_authority_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    anchor_path = Path(anchor_path or (
        output_root / "evidence" / "ANCHOR_TRAJECTORIES.json"
    )).resolve()
    if proposal_parent_authority_path is not None:
        authority_path = Path(proposal_parent_authority_path).resolve()
        authority = validate_hjcgr_parent_authority(_read_json(authority_path))
        atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
        local_artifacts = {
            "anchor_summary": anchor_path,
            "causal_matrix": matrix_path,
            "reversal_atlas": atlas_path,
        }
        for key, path in local_artifacts.items():
            if not path.is_file():
                raise RuntimeError(f"portable HJCGR replay lacks local {key}")
            if file_sha256(path) != authority["source_artifact_sha256"][key]:
                raise RuntimeError(f"portable HJCGR replay changed source {key}")
        return dict(authority["parent_evidence"])
    matrix = _read_json(matrix_path)
    if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
        raise RuntimeError("HJCGR requires a complete causal matrix")
    anchor = _read_json(anchor_path)
    if anchor.get("schema") != "local-route1-anchor-summary-v1":
        raise RuntimeError("HJCGR requires the canonical anchor summary")
    hj = next(
        (row for row in anchor.get("summaries", []) if row.get("probe_id") == "hj"),
        None,
    )
    causal = next(
        (row for row in matrix.get("probe_summaries", []) if row.get("probe") == "hj"),
        None,
    )
    variance = next(
        (
            row for row in matrix.get("sampling_variance_summaries", [])
            if row.get("probe") == "hj"
        ),
        None,
    )
    if not all(isinstance(row, dict) for row in (hj, causal, variance)):
        raise RuntimeError("HJCGR requires HJ trajectory, causal and variance evidence")
    e200 = next(
        (row for row in hj.get("trajectory", []) if int(row.get("epoch", -1)) == 200),
        None,
    )
    axes = variance.get("axes", {})
    independent = axes.get("independent_unpaired_batch", {})
    latent = axes.get("latent_time_bridge_rng", {})
    if not isinstance(e200, dict) or not (
        hj.get("complete_e200") is True
        and float(hj.get("late_three_mean_macro_psnr_delta", -1e9)) > 0.0
        and int(hj.get("late_points_with_four_of_six_positive_domains", 0)) >= 2
        and float(e200.get("macro_psnr_delta", -1e9)) > 0.0
        and e200.get("guardrails_pass") is True
        and int(independent.get("variance_dominated_rows", 0)) == int(
            independent.get("rows", -1)
        )
        and int(latent.get("variance_dominated_rows", 0)) == int(latent.get("rows", -1))
        and int(independent.get("rows", 0)) > 0
    ):
        raise RuntimeError("HJ does not satisfy the HJCGR composition parent gate")
    return {
        "hj": {
            "probe_id": "hj",
            "complete_e200": True,
            "late_three_mean_macro_psnr_delta": float(
                hj["late_three_mean_macro_psnr_delta"]
            ),
            "e200_macro_psnr_delta": float(e200["macro_psnr_delta"]),
            "e200_positive_domains": int(e200["positive_domains"]),
            "late_points_with_four_of_six_positive_domains": int(
                hj["late_points_with_four_of_six_positive_domains"]
            ),
            "state_operator_case_counts": dict(causal.get("case_counts", {})),
            "next_batch_consensus_negative_rows": int(
                causal.get("next_batch_consensus_negative_rows", 0)
            ),
        },
        "hj_sampling_variance": {
            "independent_batch_variance_fraction": float(
                independent["mean_variance_fraction"]
            ),
            "independent_batch_variance_dominated_rows": int(
                independent["variance_dominated_rows"]
            ),
            "latent_time_bridge_variance_fraction": float(
                latent["mean_variance_fraction"]
            ),
            "latent_time_bridge_variance_dominated_rows": int(
                latent["variance_dominated_rows"]
            ),
        },
        "pcrsmg_proposal_only": _proposal_evidence(output_root),
        "anchor_summary_sha256": file_sha256(anchor_path),
        "causal_matrix_sha256": file_sha256(matrix_path),
    }


def build_hjcgr_card(
    output_root: Path, evidence: dict[str, Any],
) -> dict[str, Any]:
    atlas_path = Path(output_root).resolve() / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    return {
        "schema": CARD_SCHEMA,
        "candidate_id": HJCGR_ID,
        "name": "HJ-Objective Conditional G/F Resampling",
        "parent_evidence": evidence,
        "lineage_evidence": [
            "ANCHOR_TRAJECTORIES:HJ:complete_e200_positive_parent",
            "LONG_CAUSAL_MATRIX:HJ:sampling_variance_dominated=22/22+22/22",
            "LONG_CAUSAL_MATRIX:HJ:next_batch_consensus_negative=22/22:no_controller",
            f"TERMINAL_RECEIPT:{PROPOSAL_ID}:complete_e200_positive_parent",
            "GENERATION3:nested_objective_and_conditional_estimator_composition",
        ],
        "prior_equivalence_audit": {
            "compared_implementations": [
                "hj:layer0/true/joint/central-consensus/continuous",
                PROPOSAL_ID,
                HJCGR_ID,
            ],
            "material_difference": (
                "The two fresh post-D/E G/F views each evaluate the frozen HJ "
                "structure-projected PatchNCE objective, while replica-local HJ "
                "bookkeeping is reduced to one physical optimizer transition."
            ),
            "equivalent_rerun": False,
        },
        "unsb_object": (
            "the Layer-0 finite-difference structure-projected PatchNCE objective "
            "and its player-conditional joint G/F stochastic gradient estimator"
        ),
        "formula": (
            "Commit native one-view D/E. Given the realized state and frozen HJ "
            "controller q, draw iid xi1,xi2 and commit G/F with "
            "g*=(grad L_HJ(S,q;xi1)+grad L_HJ(S,q;xi2))/2. Reduce the two "
            "diagnostic state deltas by their mean but advance integer physical "
            "optimizer counters exactly once."
        ),
        "identity_or_unbiased_condition": (
            "Disabled is exact plain; objective_only is exact continuous HJ; "
            "estimator_only is exact proposal-only. Conditional on post-D/E S,q, "
            "E[g*|S,q]=E[grad L_HJ|S,q]."
        ),
        "objective_change": True,
        "estimator_change": True,
        "coordinate_change": False,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": (
            "Only unpaired training samples, HJ feature probes, native stochastic "
            "variables, current physical epoch and optimizer state are read. No "
            "paired metric or confirmation image is addressable."
        ),
        "expected_applicable_state": (
            "Continuous HJ states whose structure correction is useful but whose "
            "joint G/F estimator is dominated by batch/latent/time/bridge variance."
        ),
        "falsifying_experiment": (
            "A common-e0 e200 run with non-positive late-three or e200 delta, or "
            "worse guardrails than both parent streams, rejects this composition."
        ),
        "compute_cost": (
            "one native D/E view plus two serial HJ structure-projected G/F views"
        ),
        "memory_cost": "two serial HJ G/F graphs; no persistent model copy",
        "recovery_state_cost": (
            "HJ controller, proposal update/bundle/schedule state, ordinary full state"
        ),
        "algorithm_hyperparameters": {
            "route1_hjcgr_enable": True,
            "hjcgr_role": "full",
            "hj_enable": True,
            "hj_layers": "0",
            "hj_direction": "joint",
            "hj_scales": "1,2,4",
            "hj_step": 0.01,
            "hj_quantile": 0.75,
            "hj_gate_quantile": 0.75,
            "hj_strength": 0.5,
            "hj_boundary_scale": 0.001,
            "hj_min_risk": 0.05,
            "hj_min_delta": 0.0,
            "hj_probe_mode": "central_consensus",
            "hj_control": "true",
            "hj_amplitude": "constant",
            "hj_update_mode": "remove",
            "hj_start_epoch": 5,
            "hj_search_start_step": -1,
            "hj_search_duration_steps": 0,
        },
        "algorithm_state_variables": [
            "hj_controller",
            "pcrsmg_proposal.update_index",
            "pcrsmg_proposal.gf_bundle_count",
            "pcrsmg_proposal.last_schedule",
        ],
        "ablation_definitions": {
            "proposal_only": f"{PROPOSAL_ID} / hjcgr_role=estimator_only",
            "observable_only": "route1_hjcgr:hjcgr_role=observable_only",
            "projected_or_full": HJCGR_ID,
            "objective_only": "hj / route1_hjcgr:hjcgr_role=objective_only",
        },
        **_historical_bindings(),
        "construction_authority": "independent_unbiased_reparameterization",
        "unbiased_proof": (
            "Condition on the post-D/E state and HJ controller. The two views are "
            "iid under the unchanged HJ stochastic measure, so linearity gives "
            "the HJ mean field and halves conditional covariance. Replica-local "
            "diagnostic deltas are likewise mean-reduced; integer epoch/update "
            "counters advance once and cannot scale with replica count."
        ),
        "paired_target_available_to_training": False,
        "causal_matrix_sha256": evidence["causal_matrix_sha256"],
        "reversal_atlas_sha256": file_sha256(atlas_path),
        "contract_id": HJCGR_ID,
    }


def build_hjcgr_implementation(card_path: Path) -> dict[str, Any]:
    method = dict(_read_json(card_path)["algorithm_hyperparameters"])
    sources = [
        "src/models/route1_hjcgr_model.py",
        "src/models/hj/model.py",
        "src/models/hj/core.py",
        "src/models/hj/projection.py",
        "src/models/hj/structure.py",
        "src/models/route1/pcrsmg_ablation.py",
        "src/models/route1/pcrsmg.py",
        "research/local_route1/generation1_gates.py",
    ]
    return {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": HJCGR_ID,
        "status": "FROZEN_FOR_GATES",
        "model": "route1_hjcgr",
        "method": method,
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "zero_intervention": {"route1_hjcgr_enable": False},
        "gate_hook": {
            "module": "research.local_route1.generation1_gates",
            "callable": "run_hjcgr_gate",
        },
        "source_files": [
            {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for relative in sources
        ],
        "derivation_card_sha256": file_sha256(card_path),
    }


def materialize_hjcgr_frontier(
    output_root: Path,
    *,
    anchor_path: Path | None = None,
    proposal_parent_authority_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    evidence = select_hjcgr_parent_evidence(
        output_root,
        anchor_path=anchor_path,
        proposal_parent_authority_path=proposal_parent_authority_path,
    )
    card_path = output_root / "derive" / "cards" / f"{HJCGR_ID}.json"
    implementation_path = output_root / "derive" / "implementations" / f"{HJCGR_ID}.json"
    _write_exact(card_path, build_hjcgr_card(output_root, evidence))
    _write_exact(implementation_path, build_hjcgr_implementation(card_path))

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    expected = {
        "candidate_id": HJCGR_ID,
        "generation": 3,
        "parent_candidate_id": PROPOSAL_ID,
        "parent_evidence": evidence,
        "construction_route": "evidence_qualified_nested_hj_objective_estimator",
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    matches = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == HJCGR_ID
    ]
    if not matches:
        ledger["records"].append(expected)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("HJCGR hypothesis identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES" and matches[0] != expected:
        raise RuntimeError("HJCGR hypothesis ledger slot changed")
    registration = freeze_candidate_derivation(output_root, HJCGR_ID)

    frontier_path = output_root / "operations" / "MULTI_ALGORITHM_MATH_FRONTIER.json"
    if frontier_path.is_file():
        frontier = _read_json(frontier_path)
        if frontier.get("schema") != FRONTIER_SCHEMA:
            raise RuntimeError("multi-algorithm frontier schema mismatch")
        rows = [
            row for row in frontier.get("frontier", [])
            if row.get("id") not in ("HJ-CONDITIONAL-GF-RESAMPLING", HJCGR_ID)
        ]
    else:
        frontier = {
            "schema": FRONTIER_SCHEMA,
            "status": "MULTI_ALGORITHM_FRONTIER_ACTIVE",
            "north_star": "discover sustained e200 algorithms, not select an early sole winner",
            "action_priority_is_not_scientific_exclusivity": True,
            "maximum_concurrent_long_candidates": 3,
        }
        rows = []
    rows.append({
        "id": HJCGR_ID,
        "kind": "nested_hj_objective_conditional_estimator_composition",
        "status": "FROZEN_FOR_TARGET_BLIND_GATE",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "evidence": evidence,
    })
    frontier.update({
        "status": "RELATED_MULTI_ALGORITHM_FRONTIER_ACTIVE",
        "action_priority_candidate_id": HPCGR_ID,
        "action_priority_is_not_scientific_exclusivity": True,
        "frontier": rows,
        "related_family": {
            "shared_operator": "post-D/E conditionally iid two-view G/F mean",
            "members": [PROPOSAL_ID, HPCGR_ID, HJCGR_ID],
            "distinct_base_objects": [
                "native UNSB field", "physical-horizon HNEK bridge game",
                "HJ structure-projected PatchNCE objective",
            ],
        },
        "independent_family_member": AMTNC_ID,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    write_json(frontier_path, frontier)
    return {
        "schema": "final-unsb-route1-hjcgr-frontier-freeze-v1",
        "status": "HJCGR_FROZEN_FOR_TARGET_BLIND_GATE",
        "candidate_id": HJCGR_ID,
        "registration": registration.to_dict(),
        "frontier_path": str(frontier_path),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-path", type=Path)
    parser.add_argument("--proposal-parent-authority", type=Path)
    args = parser.parse_args()
    result = materialize_hjcgr_frontier(
        args.output_root,
        anchor_path=args.anchor_path,
        proposal_parent_authority_path=args.proposal_parent_authority,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
