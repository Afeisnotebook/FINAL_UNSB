"""Conditionally freeze the sole preregistered two-component route-1 synthesis.

The operation may run only after both 5090 frontier terminal receipts have
been adjudicated.  It chooses no hyperparameter: AM-MCRB must be a strict
parent, and the sampling parent is PCNR when strict, otherwise the already
strict PC-RSMG proposal-only operator.  An ineligible terminal outcome writes
an auditable ``SYNTHESIS_INAPPLICABLE`` record and creates no candidate.
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
from research.local_route1.frontier_adjudication import FRONTIER_IDS, SCHEMA as FRONTIER_SCHEMA
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


CANDIDATE_ID = "G3-01-CONDITIONAL-SAMPLING-ADAM-BARRIER"
PCNR_ID, AMMCRB_ID = FRONTIER_IDS
PCRSMG_PROPOSAL_ID = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
DECISION = "decisions/DEC-20260831-ROUTE1-CONDITIONAL-GENERATION3-SYNTHESIS.md"
FALLBACK_EVIDENCE = (
    "evidence/remote_route1_offload/"
    "PCRSMG_WINNER_ABLATION_E200_AND_RECOVERY_20260831.json"
)
SOURCES = (
    "src/models/sb_model.py",
    "src/models/dtcov/dtcovmatch.py",
    "src/models/route1/__init__.py",
    "src/models/route1/pcrsmg.py",
    "src/models/route1/pcnr.py",
    "src/models/route1/mcrb.py",
    "src/models/route1/ammcrb.py",
    "src/models/route1/pcammcrb.py",
    "src/models/route1_pcammcrb_model.py",
    "research/local_route1/generation1_gates.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_exact_json(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != payload:
            raise RuntimeError(f"non-identical frozen Generation-3 artifact exists: {path}")
        return
    write_json(path, payload)


def select_sampling_parent(adjudication: dict[str, Any]) -> dict[str, Any]:
    if adjudication.get("schema") != FRONTIER_SCHEMA:
        raise RuntimeError("Generation-3 requires the canonical frontier adjudication")
    if adjudication.get("selection_seeds") != [2026]:
        raise RuntimeError("Generation-3 parent adjudication is not seed2026-only")
    for key in (
        "intermediate_metrics_used_for_routing", "cross_host_deltas_merged",
        "paired_metrics_used_for_training_or_control", "paired_controller_access",
        "confirmation20_opened",
    ):
        if adjudication.get(key) is not False:
            raise RuntimeError(f"Generation-3 parent adjudication requires {key}=false")
    strict = set(adjudication.get("strict_gate_pass_candidate_ids", []))
    if AMMCRB_ID not in strict:
        return {
            "eligible": False,
            "reason": "AM_MCRB_PARENT_DID_NOT_PASS_STRICT_SUSTAINED_GATE",
            "strict_frontier_parent_ids": sorted(strict),
            "sampling_parent": None,
        }
    if PCNR_ID in strict:
        return {
            "eligible": True,
            "reason": "BOTH_5090_PARENTS_STRICT",
            "strict_frontier_parent_ids": sorted(strict),
            "sampling_parent": "pcnr",
            "sampling_parent_candidate_id": PCNR_ID,
        }
    fallback = _read_json(ROOT / FALLBACK_EVIDENCE)
    evidence = fallback.get("pcrsmg_source_bound_e200_ablation", {})
    if evidence.get("proposal_only_strict_gate_pass") is not True or (
        evidence.get("proposal_only", {}).get("candidate_id") != PCRSMG_PROPOSAL_ID
    ):
        raise RuntimeError("registered PC-RSMG proposal fallback is not strict-pass evidence")
    return {
        "eligible": True,
        "reason": "AM_MCRB_STRICT_WITH_REGISTERED_PCRSMG_PROPOSAL_FALLBACK",
        "strict_frontier_parent_ids": sorted(strict),
        "sampling_parent": "pcrsmg_proposal",
        "sampling_parent_candidate_id": PCRSMG_PROPOSAL_ID,
    }


def _card(
    *, output_root: Path, adjudication_path: Path, route: dict[str, Any],
) -> dict[str, Any]:
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    ammcrb = _read_json(
        ROOT / "research/local_route1/derivation_cards/"
        "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER.json"
    )
    sampling_parent = str(route["sampling_parent"])
    sampling_name = str(route["sampling_parent_candidate_id"])
    sampling_formula = (
        "one fresh native G/F view after the realized D/E commits"
        if sampling_parent == "pcnr" else
        "the arithmetic mean of two fresh native G/F gradients after native D/E commits"
    )
    return {
        "schema": CARD_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "name": "Conditional Sampling with Adam-Metric Moving Covariance Barrier",
        "parent_candidate_id": AMMCRB_ID,
        "parent_evidence": {
            **ammcrb["parent_evidence"],
            "strict_constraint_parent_id": AMMCRB_ID,
            "strict_sampling_parent_id": sampling_name,
            "frontier_adjudication_sha256": file_sha256(adjudication_path),
            "sampling_parent_fallback_evidence_sha256": (
                None if sampling_parent == "pcnr" else file_sha256(ROOT / FALLBACK_EVIDENCE)
            ),
        },
        "lineage_evidence": [
            *ammcrb["lineage_evidence"],
            f"strict sampling parent:{sampling_name}",
            f"strict constraint parent:{AMMCRB_ID}",
            "DEC-20260831 conditional Generation-3 synthesis fixed before terminal metrics",
        ],
        "prior_equivalence_audit": {
            "compared_implementations": [
                PCNR_ID, PCRSMG_PROPOSAL_ID, AMMCRB_ID,
                "native UNSB", "PC-RSMG full", "MCRB Euclidean barrier",
            ],
            "material_difference": (
                "The selected conditionally native G/F estimator first realizes Adam moments; "
                "the resulting native-like displacement is then mapped to the unique AM-MCRB "
                "closest feasible point. For the two-view parent, the barrier tangent is the "
                "exchange-symmetric mean over those same views with common latent probes."
            ),
            "equivalent_rerun": False,
        },
        "unsb_object": (
            "the joint coupling of the post-opponent stochastic G/F field and the exact "
            "Adam-metric generator displacement under a moving covariance-rate half-space"
        ),
        "formula": (
            f"After D/E commit, form g_R from {sampling_formula} and realize d_R=AdamStep(g_R). "
            "Let a be the current-versus-EMA log direction-covariance tangent evaluated on "
            "the same G/F stochastic measure, P=H^-1 the normalized post-step Adam inverse "
            "root-second-moment metric. Commit d_R exactly when <a,d_R><=0; otherwise commit "
            "d*=d_R-(<a,d_R>/<a,P a>)P a. Update the EMA once from the committed G."
        ),
        "identity_or_unbiased_condition": (
            "The sampling parent retains its registered conditional native mean. The barrier "
            "is exact identity when the native-like displacement is rate-safe or its tangent "
            "vanishes; disabling pcammcrb is byte-identical to plain. The active composite is "
            "not claimed unbiased because the target-blind safety constraint intentionally "
            "changes an unsafe finite-step displacement."
        ),
        "objective_change": False,
        "estimator_change": True,
        "coordinate_change": True,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": (
            "Only official unpaired batches, native stochastic views, current/EMA netG, Adam "
            "moments and target-blind covariance geometry are addressable. The gate and model "
            "contain no discovery target, PSNR, SSIM, LPIPS, plain output, domain controller, "
            "confirmation20, epoch window or checkpoint-selection input."
        ),
        "expected_applicable_state": {
            "condition": (
                "conditional player resampling is beneficial while some realized Adam "
                "displacements still increase the moving direction-covariance defect"
            ),
            "self_null_state": "the sampled displacement is covariance-rate-safe or tangent zero",
            "causal_scope": "tests whether two independently strict mechanisms compose safely",
        },
        "falsifying_experiment": (
            "Do not run if either required parent is not strict or any target-blind e20/e100/e200 "
            "1/8/32-step component-correction cosine is below -0.2. Kill as implementation invalid "
            "on disabled identity, resume, parent-hash, schedule or barrier-feasibility failure. "
            "Close the composite operator if its complete host-matched seed2026 e200 gate fails."
        ),
        "compute_cost": (
            "PCNR mode uses two serial native views plus one m=4 current/EMA covariance tangent; "
            "PC-RSMG proposal mode uses native D/E, two G/F views and an exchange-symmetric "
            "m=4 tangent on both views"
        ),
        "memory_cost": (
            "one EMA generator, serial sampling graphs, covariance endpoint graphs, tangent copy "
            "and normalized diagonal Adam inverse metric"
        ),
        "recovery_state_cost": (
            "sampling provenance, fixed sampling-parent identity, EMA generator, barrier counters, "
            "last KKT geometry and ordinary model/optimizer/sampler/RNG state"
        ),
        "algorithm_hyperparameters": {
            "sampling_parent": sampling_parent,
            "endpoint_samples": 4,
            "region_patch": 32,
            "u_floor": 1e-30,
            "teacher_half_life_updates": 150,
            "projection_epsilon": 1e-24,
            "component_cosine_floor": -0.2,
            "fixed_window": False,
            "paired_threshold": False,
            "strength": "none_exact_constraint",
        },
        "algorithm_state_variables": [
            "sampling_parent", "sampling_update_index", "sampling_bundle_counters",
            "last_player_schedule", "ema_netG_state_dict", "barrier_update_index",
            "barrier_intervention_count", "last_adam_metric_barrier_geometry",
        ],
        "ablation_definitions": {
            "proposal_only": f"Commit the frozen sampling parent {sampling_name} without the barrier.",
            "observable_only": (
                "Run the selected sampling transition and compute the moving Adam-metric barrier "
                "counterfactually while committing its sampled displacement unchanged."
            ),
            "projected_or_full": "Commit the selected sampling displacement after the exact AM-MCRB closest-feasible projection.",
        },
        "historical_evidence_index_sha256": ammcrb["historical_evidence_index_sha256"],
        "mechanism_object_map_sha256": ammcrb["mechanism_object_map_sha256"],
        "reuse_boundary_sha256": ammcrb["reuse_boundary_sha256"],
        "construction_authority": ammcrb["construction_authority"],
        "target_blind_driver_signal": ammcrb["target_blind_driver_signal"],
        "target_blind_driver_probe": ammcrb["target_blind_driver_probe"],
        "paired_target_available_to_training": False,
        "causal_matrix_sha256": file_sha256(matrix_path),
        "reversal_atlas_sha256": file_sha256(atlas_path),
    }


def materialize(output_root: Path, adjudication_path: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    adjudication_path = Path(adjudication_path).resolve()
    adjudication = _read_json(adjudication_path)
    route = select_sampling_parent(adjudication)
    result_path = output_root / "operations" / "GENERATION3_SYNTHESIS_FREEZE.json"
    common = {
        "schema": "final-unsb-route1-generation3-synthesis-freeze-v1",
        "decision": DECISION,
        "decision_sha256": file_sha256(ROOT / DECISION),
        "frontier_adjudication_path": str(adjudication_path),
        "frontier_adjudication_sha256": file_sha256(adjudication_path),
        "route": route,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_used_for_formula": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if route["eligible"] is not True:
        result = {**common, "status": "SYNTHESIS_INAPPLICABLE", "candidate_id": None}
        write_json(result_path, result)
        return result

    card = _card(
        output_root=output_root, adjudication_path=adjudication_path, route=route,
    )
    card_path = output_root / "derive" / "cards" / f"{CANDIDATE_ID}.json"
    _write_exact_json(card_path, card)
    implementation = {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": "FROZEN_FOR_GATES",
        "derivation_card_sha256": file_sha256(card_path),
        "model": "route1_pcammcrb",
        "method": {
            "pcammcrb_enable": True,
            "pcammcrb_sampling_parent": route["sampling_parent"],
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "ammcrb_projection_epsilon": 1e-24,
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
            "callable": "run_pcammcrb_gate",
        },
        "source_files": [
            {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for relative in SOURCES
        ],
    }
    implementation_path = (
        output_root / "derive" / "implementations" / f"{CANDIDATE_ID}.json"
    )
    _write_exact_json(implementation_path, implementation)

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    records = ledger.setdefault("records", [])
    matches = [row for row in records if row.get("candidate_id") == CANDIDATE_ID]
    expected = {
        "candidate_id": CANDIDATE_ID,
        "generation": 3,
        "parent_candidate_id": AMMCRB_ID,
        "parent_evidence": card["parent_evidence"],
        "construction_route": "preregistered_two_component_constrained_synthesis",
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
        raise RuntimeError("Generation-3 ledger identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES":
        if matches[0] != expected:
            raise RuntimeError("Generation-3 ledger slot changed before freeze")

    registration = freeze_candidate_derivation(output_root, CANDIDATE_ID)
    result = {
        **common,
        "status": "SYNTHESIS_FROZEN_FOR_COMPATIBILITY_GATE",
        "candidate_id": CANDIDATE_ID,
        "registration": registration.to_dict(),
    }
    write_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier-adjudication", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        materialize(args.output, args.frontier_adjudication),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
