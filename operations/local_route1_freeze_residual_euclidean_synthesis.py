"""Freeze conditional sampling plus RF-MCRB after two strict same-host parents."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import _validate_receipt
from operations.local_route1_freeze_residual_synthesis import (
    AUTHORITY_FIELDS,
    PCNR_ID,
    PCRSMG_PROPOSAL_ID,
    SAMPLING_IDS,
    _read_json,
    _trajectory_for_receipt,
    _write_exact,
)
from research.local_route1.candidates import (
    CARD_SCHEMA,
    IMPLEMENTATION_SCHEMA,
    freeze_candidate_derivation,
)
from research.local_route1.frontier_advancement import (
    STRICT,
    classify_complete_trajectory,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


CANDIDATE_ID = (
    "G3-03-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-EUCLIDEAN-BARRIER"
)
RFMCRB_ID = "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER"
DECISION = (
    "decisions/DEC-20260831-RESIDUAL-FEASIBLE-EUCLIDEAN-"
    "CONDITIONAL-SYNTHESIS.md"
)
SOURCES = (
    "src/models/sb_model.py",
    "src/models/route1/pcnr.py",
    "src/models/route1/pcrsmg.py",
    "src/models/route1/mcrb.py",
    "src/models/route1/rfammcrb.py",
    "src/models/route1/rfmcrb.py",
    "src/models/route1/pcammcrb.py",
    "src/models/route1/pcrfmcrb.py",
    "src/models/route1_pcrfmcrb_model.py",
    "research/local_route1/generation1_gates.py",
)


def adjudicate_parent_route(
    sampling_receipt_path: Path, barrier_receipt_path: Path,
) -> dict[str, Any]:
    sampling_path = Path(sampling_receipt_path).resolve()
    barrier_path = Path(barrier_receipt_path).resolve()
    sampling = _validate_receipt(sampling_path)
    barrier = _validate_receipt(barrier_path)
    sampling_id = str(sampling.get("candidate_id", ""))
    if sampling_id not in SAMPLING_IDS:
        raise RuntimeError("Euclidean synthesis received an unknown sampling parent")
    if barrier.get("candidate_id") != RFMCRB_ID:
        raise RuntimeError("Euclidean synthesis requires RF-MCRB as barrier parent")
    authority = {}
    for field in AUTHORITY_FIELDS:
        values = {str(sampling.get(field, "")), str(barrier.get(field, ""))}
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError(
                f"Euclidean synthesis parents differ on same-host authority: {field}"
            )
        authority[field] = next(iter(values))
    sampling_class = classify_complete_trajectory(
        sampling, _trajectory_for_receipt(sampling),
    )
    barrier_class = classify_complete_trajectory(
        barrier, _trajectory_for_receipt(barrier),
    )
    eligible = (
        sampling_class["classification"] == STRICT
        and barrier_class["classification"] == STRICT
    )
    return {
        "eligible": eligible,
        "reason": (
            "TWO_INDEPENDENT_STRICT_SAME_HOST_PARENTS"
            if eligible else "ONE_OR_MORE_PARENTS_NOT_STRICT"
        ),
        "sampling_parent": (
            "pcnr" if sampling_id == PCNR_ID else "pcrsmg_proposal"
        ),
        "sampling_parent_candidate_id": sampling_id,
        "barrier_parent_candidate_id": RFMCRB_ID,
        "sampling_classification": sampling_class,
        "barrier_classification": barrier_class,
        "same_host_authority": authority,
        "sampling_receipt_path": str(sampling_path),
        "sampling_receipt_sha256": file_sha256(sampling_path),
        "barrier_receipt_path": str(barrier_path),
        "barrier_receipt_sha256": file_sha256(barrier_path),
    }


def _card(output_root: Path, route: dict[str, Any]) -> dict[str, Any]:
    rf_card = _read_json(
        ROOT / "research/local_route1/derivation_cards/"
        "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER.json"
    )
    sampling_id = str(route["sampling_parent_candidate_id"])
    sampling = str(route["sampling_parent"])
    sampling_formula = (
        "one fresh native G/F view after the realized D/E commits"
        if sampling == "pcnr" else
        "the arithmetic mean of two fresh native G/F gradients after native D/E commits"
    )
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    return {
        "schema": CARD_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "name": (
            "Conditional Sampling with Residual-Feasible Euclidean "
            "Covariance Barrier"
        ),
        "parent_candidate_id": RFMCRB_ID,
        "parent_evidence": {
            "failure_type": "state_feedback_missing",
            "secondary_failure_type": "sampling_variance_and_player_conditioning",
            "strict_sampling_parent_id": sampling_id,
            "strict_barrier_parent_id": RFMCRB_ID,
            "sampling_receipt_sha256": route["sampling_receipt_sha256"],
            "barrier_receipt_sha256": route["barrier_receipt_sha256"],
            "same_host_authority": route["same_host_authority"],
            "paired_metric_used_for_formula": False,
        },
        "lineage_evidence": [
            *rf_card["lineage_evidence"],
            f"strict conditional sampling parent:{sampling_id}",
            f"strict residual-feasible barrier parent:{RFMCRB_ID}",
            "DEC-20260831:Euclidean and Adam closest-feasible geometries remain distinct",
        ],
        "prior_equivalence_audit": {
            "compared_implementations": [
                sampling_id,
                RFMCRB_ID,
                "G3-02-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-ADAM-BARRIER",
                "G3-01-CONDITIONAL-SAMPLING-ADAM-BARRIER",
                "native UNSB",
            ],
            "material_difference": (
                "The conditional sampler first realizes its native-like Adam "
                "displacement. The composite then applies RF-MCRB's represented-"
                "residual Euclidean closest feasible operator on the same G/F "
                "measure. Unlike G3-02 it uses P=I rather than the Adam inverse "
                "metric; unlike G3-01 it has no fixed absolute margin."
            ),
            "equivalent_rerun": False,
        },
        "unsb_object": (
            "the coupled post-opponent stochastic G/F field and its actually "
            "represented native Adam displacement under a moving covariance-rate "
            "half-space in Euclidean parameter geometry"
        ),
        "formula": (
            f"After D/E commit form g_R from {sampling_formula} and realize "
            "d_R=AdamStep(g_R). On the same G/F measure compute a=grad C for the "
            "current/EMA moving covariance defect. If <a,d_R><=0 commit d_R byte-"
            "for-byte. Otherwise compute lambda=<a,d_R>/<a,a> in float64, "
            "represent d=d_R-lambda a in parameter dtype, and advance lambda only "
            "by measured residual/<a,a> plus a relative parameter-dtype ULP until "
            "<a,d><=0 (maximum eight fail-closed numerical refinements). Update "
            "the EMA once after the final G commit."
        ),
        "identity_or_unbiased_condition": (
            "The sampling parent retains its registered conditional native mean. "
            "The barrier returns the original displacement tensors when safe or "
            "tangent-zero; its active displacement is the residual-checked scale-"
            "free Euclidean KKT closest point. Disabling pcrfmcrb is byte-identical "
            "to plain."
        ),
        "objective_change": False,
        "estimator_change": True,
        "coordinate_change": False,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": (
            "Only official unpaired batches, native stochastic views, current/EMA "
            "netG, the native optimizer displacement and target-blind covariance "
            "geometry are addressable. No paired target, quality metric, plain "
            "output, domain controller, confirmation sample, epoch window or "
            "checkpoint-selection input exists in the model or gate."
        ),
        "expected_applicable_state": {
            "condition": (
                "conditional player sampling is beneficial and the represented "
                "native displacement would increase the moving covariance defect, "
                "with Euclidean rather than Adam-metric proximity being appropriate"
            ),
            "self_null_state": "the sampled displacement is rate-safe or tangent-zero",
            "causal_scope": (
                "composition of two independently strict target-blind mechanisms "
                "in Euclidean closest-feasible geometry"
            ),
        },
        "falsifying_experiment": (
            "Do not run unless both complete same-host parent trajectories are "
            "strict and every e20/e100/e200 1/8/32-step component correction cosine "
            "is at least -0.2. Kill on disabled identity, resume, parent isolation, "
            "represented feasibility, schedule or provenance failure. Close only "
            "this composite if matched e200 fails."
        ),
        "compute_cost": (
            "the selected conditional sampling transition plus one m=4 current/EMA "
            "covariance tangent and at most eight numerical-only residual reconstructions"
        ),
        "memory_cost": (
            "one EMA generator, serial sampling graphs, covariance endpoint graphs "
            "and one Euclidean tangent copy"
        ),
        "recovery_state_cost": (
            "frozen sampling-parent identity, sampling provenance, EMA generator, "
            "barrier counters, last residual-feasible Euclidean geometry, samplers "
            "and all RNG streams"
        ),
        "algorithm_hyperparameters": {
            "sampling_parent": sampling,
            "endpoint_samples": 4,
            "region_patch": 32,
            "u_floor": 1e-30,
            "teacher_half_life_updates": 150,
            "metric": "euclidean",
            "projection_epsilon": 1e-24,
            "absolute_numeric_margin": 0.0,
            "component_cosine_floor": -0.2,
            "fixed_window": False,
            "paired_threshold": False,
            "strength": "none_exact_constraint",
        },
        "algorithm_state_variables": [
            "sampling_parent", "sampling_update_index", "sampling_bundle_counters",
            "last_player_schedule", "ema_netG_state_dict", "barrier_update_index",
            "barrier_intervention_count", "last_residual_feasible_euclidean_geometry",
        ],
        "ablation_definitions": {
            "proposal_only": (
                f"Commit the strict sampling parent {sampling_id} without the barrier."
            ),
            "observable_only": (
                "Run the sampling transition and residual-feasible Euclidean barrier "
                "diagnostics while committing the sampled displacement unchanged."
            ),
            "projected_or_full": (
                "Commit the sampled displacement after the residual-checked RF-MCRB "
                "Euclidean closest feasible projection."
            ),
        },
        "historical_evidence_index_sha256": rf_card[
            "historical_evidence_index_sha256"
        ],
        "mechanism_object_map_sha256": rf_card["mechanism_object_map_sha256"],
        "reuse_boundary_sha256": rf_card["reuse_boundary_sha256"],
        "construction_authority": "eligible_method_specific_signal",
        "target_blind_driver_signal": "dt_covariance_mismatch_descent_margin",
        "target_blind_driver_probe": "dt",
        "paired_target_available_to_training": False,
        "causal_matrix_sha256": file_sha256(matrix_path),
        "reversal_atlas_sha256": file_sha256(atlas_path),
    }


def materialize(
    output_root: Path,
    *,
    sampling_receipt_path: Path,
    barrier_receipt_path: Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    route = adjudicate_parent_route(sampling_receipt_path, barrier_receipt_path)
    result_path = (
        output_root / "operations" / "RESIDUAL_EUCLIDEAN_SYNTHESIS_FREEZE.json"
    )
    common = {
        "schema": "final-unsb-route1-residual-euclidean-synthesis-freeze-v1",
        "decision": DECISION,
        "decision_sha256": file_sha256(ROOT / DECISION),
        "candidate_id": CANDIDATE_ID if route["eligible"] else None,
        "route": route,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metric_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if route["eligible"] is not True:
        result = {**common, "status": "SYNTHESIS_INAPPLICABLE"}
        write_json(result_path, result)
        return result

    card = _card(output_root, route)
    card_path = output_root / "derive" / "cards" / f"{CANDIDATE_ID}.json"
    _write_exact(card_path, card)
    implementation = {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": "FROZEN_FOR_GATES",
        "derivation_card_sha256": file_sha256(card_path),
        "model": "route1_pcrfmcrb",
        "method": {
            "pcrfmcrb_enable": True,
            "pcrfmcrb_sampling_parent": route["sampling_parent"],
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "rfmcrb_projection_epsilon": 1e-24,
        },
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "gate_hook": {
            "module": "research.local_route1.generation1_gates",
            "callable": "run_pcrfmcrb_gate",
        },
        "source_files": [
            {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for relative in SOURCES
        ],
    }
    implementation_path = (
        output_root / "derive" / "implementations" / f"{CANDIDATE_ID}.json"
    )
    _write_exact(implementation_path, implementation)

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    records = ledger.setdefault("records", [])
    matches = [row for row in records if row.get("candidate_id") == CANDIDATE_ID]
    expected = {
        "candidate_id": CANDIDATE_ID,
        "generation": 3,
        "parent_candidate_id": RFMCRB_ID,
        "parent_evidence": card["parent_evidence"],
        "construction_route": (
            "strict_same_host_residual_feasible_euclidean_two_component_synthesis"
        ),
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if not matches:
        records.append(expected)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("Euclidean synthesis ledger identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES":
        if matches[0] != expected:
            raise RuntimeError("Euclidean synthesis ledger slot changed before freeze")

    registration = freeze_candidate_derivation(output_root, CANDIDATE_ID)
    result = {
        **common,
        "status": "SYNTHESIS_FROZEN_FOR_COMPATIBILITY_GATE",
        "registration": registration.to_dict(),
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sampling-receipt", type=Path, required=True)
    parser.add_argument("--barrier-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(
        args.output,
        sampling_receipt_path=args.sampling_receipt,
        barrier_receipt_path=args.barrier_receipt,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
