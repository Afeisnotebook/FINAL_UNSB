"""Freeze the conditional-sampling/RF-AMMCRB synthesis after strict parents.

The operation accepts only two complete, same-host e200 receipts.  Paired
quality may decide whether the already-derived composition receives compute;
it cannot enter the formula, gate, or training transition.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import _validate_receipt
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


CANDIDATE_ID = "G3-02-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-ADAM-BARRIER"
RFAMMCRB_ID = "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER"
PCNR_ID = "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING"
PCRSMG_PROPOSAL_ID = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
SAMPLING_IDS = (PCNR_ID, PCRSMG_PROPOSAL_ID)
DECISION = "decisions/DEC-20260831-RESIDUAL-FEASIBLE-CONDITIONAL-SYNTHESIS.md"
SOURCES = (
    "src/models/sb_model.py",
    "src/models/route1/pcnr.py",
    "src/models/route1/pcrsmg.py",
    "src/models/route1/mcrb.py",
    "src/models/route1/ammcrb.py",
    "src/models/route1/rfammcrb.py",
    "src/models/route1/pcammcrb.py",
    "src/models/route1/pcrfammcrb.py",
    "src/models/route1_pcrfammcrb_model.py",
    "research/local_route1/generation1_gates.py",
)
AUTHORITY_FIELDS = (
    "base_e0_scientific_state_sha256",
    "base_protocol_fingerprint",
    "manifest_sha256",
    "plain_e200_verification_sha256",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_exact(path: Path, value: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != value:
            raise RuntimeError(f"non-identical residual synthesis artifact exists: {path}")
        return
    write_json(path, value)


def _trajectory_for_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(receipt["trajectory_path"])).resolve()
    if not path.is_file() or file_sha256(path) != receipt.get("trajectory_sha256"):
        raise RuntimeError("residual synthesis parent trajectory changed")
    return _read_json(path)


def adjudicate_parent_route(
    sampling_receipt_path: Path, barrier_receipt_path: Path,
) -> dict[str, Any]:
    sampling_path = Path(sampling_receipt_path).resolve()
    barrier_path = Path(barrier_receipt_path).resolve()
    sampling = _validate_receipt(sampling_path)
    barrier = _validate_receipt(barrier_path)
    sampling_id = str(sampling.get("candidate_id", ""))
    if sampling_id not in SAMPLING_IDS:
        raise RuntimeError("residual synthesis received an unknown sampling parent")
    if barrier.get("candidate_id") != RFAMMCRB_ID:
        raise RuntimeError("residual synthesis requires RF-AMMCRB as barrier parent")
    authority = {}
    for field in AUTHORITY_FIELDS:
        values = {str(sampling.get(field, "")), str(barrier.get(field, ""))}
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError(
                f"residual synthesis parents differ on same-host authority: {field}"
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
        "barrier_parent_candidate_id": RFAMMCRB_ID,
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
        "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER.json"
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
        "name": "Conditional Sampling with Residual-Feasible Adam Covariance Barrier",
        "parent_candidate_id": RFAMMCRB_ID,
        "parent_evidence": {
            "failure_type": "state_feedback_missing",
            "secondary_failure_type": "sampling_variance_and_player_conditioning",
            "strict_sampling_parent_id": sampling_id,
            "strict_barrier_parent_id": RFAMMCRB_ID,
            "sampling_receipt_sha256": route["sampling_receipt_sha256"],
            "barrier_receipt_sha256": route["barrier_receipt_sha256"],
            "same_host_authority": route["same_host_authority"],
            "paired_metric_used_for_formula": False,
        },
        "lineage_evidence": [
            *rf_card["lineage_evidence"],
            f"strict conditional sampling parent:{sampling_id}",
            f"strict residual-feasible barrier parent:{RFAMMCRB_ID}",
            "DEC-20260831:G3-01 superseded because its fixed absolute margin is invalid",
        ],
        "prior_equivalence_audit": {
            "compared_implementations": [
                sampling_id,
                RFAMMCRB_ID,
                "G3-01-CONDITIONAL-SAMPLING-ADAM-BARRIER",
                "native UNSB",
            ],
            "material_difference": (
                "The sampling parent first realizes its conditionally native Adam displacement. "
                "The composite then applies RF-AMMCRB's represented-residual closest feasible "
                "operator on the same stochastic G/F measure. It contains no fixed absolute "
                "projection margin and is neither parent alone nor the invalid G3-01 operator."
            ),
            "equivalent_rerun": False,
        },
        "unsb_object": (
            "the coupled post-opponent stochastic G/F field and its actually represented Adam "
            "generator displacement under a moving direction-covariance-rate half-space"
        ),
        "formula": (
            f"After D/E commit form g_R from {sampling_formula} and realize d_R=AdamStep(g_R). "
            "On the same G/F measure compute a=grad C for the current/EMA moving covariance "
            "defect and P=H^-1 from the post-step Adam second moment. If <a,d_R><=0 commit "
            "d_R byte-for-byte. Otherwise compute lambda=<a,d_R>/<a,P a> in float64, represent "
            "d=d_R-lambda P a in parameter dtype, and advance lambda only by measured residual/"
            "<a,P a> plus a relative parameter-dtype ULP until <a,d><=0 (maximum eight fail-"
            "closed numerical refinements). Update the EMA once after the final G commit."
        ),
        "identity_or_unbiased_condition": (
            "The sampling parent retains its registered conditional native mean. The barrier "
            "returns the original displacement tensors when safe or tangent-zero; its active "
            "represented displacement is the residual-checked scale-free KKT closest point. "
            "Disabling pcrfammcrb is byte-identical to plain."
        ),
        "objective_change": False,
        "estimator_change": True,
        "coordinate_change": True,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": (
            "Only official unpaired batches, native stochastic views, current/EMA netG, Adam "
            "moments and target-blind covariance geometry are addressable. No discovery target, "
            "paired metric, plain output, domain controller, confirmation sample, epoch window "
            "or checkpoint-selection input exists in the model or gate."
        ),
        "expected_applicable_state": {
            "condition": (
                "conditional player sampling is beneficial and a realized Adam displacement "
                "would otherwise increase the moving covariance-rate defect"
            ),
            "self_null_state": "the sampled displacement is rate-safe or the tangent vanishes",
            "causal_scope": "composition of two independently strict target-blind mechanisms",
        },
        "falsifying_experiment": (
            "Do not run unless both complete same-host parent trajectories are strict and every "
            "e20/e100/e200 1/8/32-step component correction cosine is at least -0.2. Kill on "
            "disabled identity, resume, parent isolation, represented feasibility, schedule or "
            "provenance failure. Close only this composite if matched e200 fails."
        ),
        "compute_cost": (
            "the selected conditional sampling transition plus one m=4 current/EMA covariance "
            "tangent and at most eight numerical-only residual reconstructions"
        ),
        "memory_cost": (
            "one EMA generator, serial sampling graphs, covariance endpoint graphs, one tangent "
            "copy and the normalized diagonal Adam inverse metric"
        ),
        "recovery_state_cost": (
            "frozen sampling-parent identity, sampling provenance, EMA generator, barrier "
            "counters, last residual-feasible KKT geometry, samplers and all RNG streams"
        ),
        "algorithm_hyperparameters": {
            "sampling_parent": sampling,
            "endpoint_samples": 4,
            "region_patch": 32,
            "u_floor": 1e-30,
            "teacher_half_life_updates": 150,
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
            "barrier_intervention_count", "last_residual_feasible_barrier_geometry",
        ],
        "ablation_definitions": {
            "proposal_only": f"Commit the strict sampling parent {sampling_id} without the barrier.",
            "observable_only": (
                "Run the sampling transition and residual-feasible barrier diagnostics while "
                "committing the sampled displacement unchanged."
            ),
            "projected_or_full": (
                "Commit the sampled displacement after the residual-checked RF-AMMCRB closest "
                "feasible projection."
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
    result_path = output_root / "operations" / "RESIDUAL_SYNTHESIS_FREEZE.json"
    common = {
        "schema": "final-unsb-route1-residual-synthesis-freeze-v1",
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
        "model": "route1_pcrfammcrb",
        "method": {
            "pcrfammcrb_enable": True,
            "pcrfammcrb_sampling_parent": route["sampling_parent"],
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "rfammcrb_projection_epsilon": 1e-24,
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
            "callable": "run_pcrfammcrb_gate",
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
        "parent_candidate_id": RFAMMCRB_ID,
        "parent_evidence": card["parent_evidence"],
        "construction_route": "strict_same_host_residual_feasible_two_component_synthesis",
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
        raise RuntimeError("residual synthesis ledger identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES":
        if matches[0] != expected:
            raise RuntimeError("residual synthesis ledger slot changed before freeze")

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
