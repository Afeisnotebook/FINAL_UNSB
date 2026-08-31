"""Freeze source-bound long-horizon ablations after the e200 winner is known."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .candidates import freeze_candidate_derivation
from .final_selection import (
    ALLOWED_STATUSES as TERMINAL_SELECTION_STATUSES,
    resolve_e200_selection_path,
    validate_e200_selection,
)
from .protocol import ROOT, file_sha256
from .runtime import write_json


SCHEMA = "final-unsb-route1-winner-ablation-freeze-v1"
POSITIVE_CROSS_STATUS = "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"
WINNER_FAMILIES = {
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED": {
        "family": "bvcp",
        "model": "route1_bvcp_ablation",
        "ids": {
            "proposal_only": "ABL-G1-01-BVCP-PROPOSAL-ONLY",
            "observable_only": "ABL-G1-01-BVCP-OBSERVABLE-ONLY",
        },
    },
    "G1-02B-PLAYER-CONDITIONAL-RSMG": {
        "family": "pcrsmg",
        "model": "route1_pcrsmg_ablation",
        "ids": {
            "proposal_only": "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
            "observable_only": "ABL-G1-02B-PCRSMG-OBSERVABLE-ONLY",
        },
    },
    "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS": {
        "family": "amtnc",
        "model": "route1_amtnc_ablation",
        "ids": {
            "proposal_only": "ABL-G2-01-AMTNC-PROPOSAL-ONLY",
            "observable_only": "ABL-G2-01-AMTNC-OBSERVABLE-ONLY",
        },
    },
    "G1-03-STATE-FEEDBACK-MISSING": {
        "family": "mcrb",
        "model": "route1_mcrb_ablation",
        "ids": {
            "proposal_only": "ABL-G1-03-MCRB-PROPOSAL-ONLY",
            "observable_only": "ABL-G1-03-MCRB-OBSERVABLE-ONLY",
        },
    },
    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING": {
        "family": "pcnr",
        "model": "route1_pcnr_ablation",
        "ids": {
            "proposal_only": "ABL-F1-01-PCNR-PROPOSAL-ONLY",
            "observable_only": "ABL-F1-01-PCNR-OBSERVABLE-ONLY",
        },
    },
    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER": {
        "family": "ammcrb",
        "model": "route1_ammcrb_ablation",
        "ids": {
            "proposal_only": "ABL-F1-02-AMMCRB-PROPOSAL-ONLY",
            "observable_only": "ABL-F1-02-AMMCRB-OBSERVABLE-ONLY",
        },
    },
    "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER": {
        "family": "rfammcrb",
        "model": "route1_rfammcrb_ablation",
        "ids": {
            "proposal_only": "ABL-F2-01-RFAMMCRB-PROPOSAL-ONLY",
            "observable_only": "ABL-F2-01-RFAMMCRB-OBSERVABLE-ONLY",
        },
    },
    "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER": {
        "family": "rfmcrb",
        "model": "route1_rfmcrb_ablation",
        "ids": {
            "proposal_only": "ABL-F2-02-RFMCRB-PROPOSAL-ONLY",
            "observable_only": "ABL-F2-02-RFMCRB-OBSERVABLE-ONLY",
        },
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _source_rows(paths: list[str]) -> list[dict[str, str]]:
    return [{"path": path, "sha256": file_sha256(ROOT / path)} for path in paths]


def _role_semantics(family: str, role: str) -> dict[str, Any]:
    if family == "bvcp" and role == "proposal_only":
        return {
            "name": "BVCP Proposal-Only Lagged Rollout Teacher",
            "formula": "Maintain theta_{k-1} exactly as BVCP, but for every no-gradient bridge rollout endpoint return G_{theta_{k-1}}(X,t,z) wholesale. The differentiable endpoint, native losses and inference remain current G_theta.",
            "identity": "When route1_ablation_enable=false the exact native UNSB path is dispatched. The active proposal is identity only when current and lagged rollout endpoints coincide.",
            "objective_change": True, "estimator_change": False,
            "compute": "one additional no-gradient lagged generator endpoint per rollout endpoint",
            "memory": "one full lagged generator copy",
            "recovery": "lagged generator and intervention counters",
            "state": ["lagged_netG_state_dict", "update_index", "endpoint_count"],
            "method": {"route1_ablation_enable": True, "bvcp_ablation_role": role, "bvcp_root_epsilon": 1e-12},
            "expected": "Tests whether the lagged feasible direction alone, without the BVCP minimum chord projection, explains sustained behavior.",
            "falsifier": "A complete e200 trajectory below the full operator closes the wholesale lagged-teacher explanation for the frozen winner.",
        }
    if family == "bvcp":
        return {
            "name": "BVCP Observable-Only Velocity Monitor",
            "formula": "Compute current and one-update-lagged rollout endpoints with common X,t,z and log their physical velocity margin, then return the current endpoint byte-for-byte. Observer state is recoverable but cannot enter outputs, gradients, RNG or samplers.",
            "identity": "After excluding the explicitly named route1_observer diagnostics, the complete next-update dynamics state must equal plain exactly at every executable gate and at e200.",
            "objective_change": False, "estimator_change": False,
            "compute": "one additional no-gradient lagged generator endpoint per rollout endpoint for diagnostics only",
            "memory": "one observer-only lagged generator copy",
            "recovery": "route1_observer lagged generator and counters; excluded only from dynamics identity comparison",
            "state": ["route1_observer.lagged_netG", "route1_observer.update_index"],
            "method": {"route1_ablation_enable": True, "bvcp_ablation_role": role, "bvcp_root_epsilon": 1e-12},
            "expected": "Negative control proving that target-blind observation and its extra compute do not change native UNSB dynamics.",
            "falsifier": "Any next-update dynamics or e200 evaluation mismatch with plain is an implementation failure, not a scientific result.",
        }
    if family == "pcrsmg" and role == "proposal_only":
        return {
            "name": "PC-RSMG Proposal-Only G/F Replication",
            "formula": "Commit native one-view D and E updates from the official first view. At the realized updated opponent state draw two fresh iid native-measure views and commit joint G/F once using their mean gradient.",
            "identity": "Disabled mode dispatches native UNSB exactly. Conditional on the realized D/E-updated state, the two-view G/F estimator has the native mean and half the single-view conditional variance.",
            "objective_change": False, "estimator_change": True,
            "compute": "one native view for D/E plus two fresh stochastic views for G/F per optimizer update",
            "memory": "two serial G/F replica graphs",
            "recovery": "update index, G/F bundle counter and schedule plus ordinary RNG",
            "state": ["pcrsmg_proposal.update_index", "pcrsmg_proposal.gf_bundle_count"],
            "method": {"route1_ablation_enable": True, "pcrsmg_ablation_role": role},
            "expected": "Isolates whether reducing only the generator player's conditional variance accounts for the full four-view result.",
            "falsifier": "A complete e200 trajectory below full PC-RSMG rejects G/F-only replication as the sufficient mechanism.",
            "unbiased": "D/E use their native estimator. Conditional on their realized update, linearity of expectation gives E[(g_GF(xi1)+g_GF(xi2))/2]=E[g_GF], with iid conditional variance halved.",
        }
    if family == "pcrsmg":
        return {
        "name": "PC-RSMG Observable-Only Replicate Monitor",
        "formula": "Draw the native first view, snapshot all Python/NumPy/CPU/CUDA RNG states, draw a second no-gradient diagnostic view, record endpoint dispersion, restore every RNG state and the first-view tensors, then execute the exact native D/E/G/F transition from the first view.",
        "identity": "After excluding only recoverable route1_observer diagnostics, networks, optimizers, schedulers, RNG, both samplers, step and native method state must equal plain exactly.",
        "objective_change": False, "estimator_change": False,
        "compute": "one discarded no-gradient diagnostic forward in addition to the native update",
        "memory": "one transient diagnostic view",
        "recovery": "route1_observer counters only; all RNG is restored before the native commit",
        "state": ["route1_observer.update_index", "route1_observer.last"],
        "method": {"route1_ablation_enable": True, "pcrsmg_ablation_role": role},
        "expected": "Negative control proving that observation, extra forward compute and logging alone do not change native UNSB dynamics.",
        "falsifier": "Any next-update dynamics or e200 evaluation mismatch with plain is an implementation failure, not a scientific result.",
        "unbiased": "The committed transition is pathwise identical to native UNSB because the diagnostic graph is discarded and all stochastic states are restored before the first-view losses are committed.",
        }
    if family == "amtnc" and role == "proposal_only":
        return {
            "name": "AM-TNC Proposal-Only Conditional First Replica",
            "formula": "Retain AM-TNC's fresh two-view DE and post-opponent GF bundles, compute the registered Adam-metric geometry, but commit the ordered first native gradient g1 for D, E and joint G/F. No radial component is removed.",
            "identity": "Disabled mode dispatches native UNSB exactly. Under replica exchange the committed residual around the two-view mean changes sign, and the ordered first replica retains the native conditional mean without AM-TNC's projection.",
            "objective_change": False, "estimator_change": True,
            "compute": "the same two fresh gradients per D/E/GF player as full AM-TNC; the second is diagnostic only",
            "memory": "two serial player graphs, matching full AM-TNC",
            "recovery": "player-conditional bundle counters, schedule and discarded geometry",
            "state": ["amtnc.update_index", "amtnc.bundle_serial", "amtnc.last_geometry"],
            "method": {"route1_ablation_enable": True, "amtnc_ablation_role": role},
            "expected": "Separates the fresh player-conditional sampling protocol from the Adam-metric radial cancellation operator.",
            "falsifier": "If full AM-TNC outranks this complete e200 trajectory, the tangential operator contributes beyond conditional resampling; if this wins, the projection is unnecessary or harmful.",
            "unbiased": "Conditional on each realized player state, g1 is an iid native-measure draw. Thus E[g1|S]=E[g_UNSB|S]; the unused second draw and recorded geometry cannot change the committed estimator.",
        }
    if family == "amtnc":
        return {
            "name": "AM-TNC Observable-Only Adam Geometry Monitor",
            "formula": "Compute pre-native-update two-view Adam-metric consensus/radial/tangential geometry, then restore every RNG stream, network buffer, train/eval flag and bridge-view tensor before executing the ordinary one-view UNSB update.",
            "identity": "After removing only route1_observer diagnostics, the complete next-update dynamics state and e200 evaluation must equal plain exactly.",
            "objective_change": False, "estimator_change": False,
            "compute": "one discarded two-view player-geometry audit plus the native update",
            "memory": "two transient diagnostic player graphs",
            "recovery": "route1_observer geometry counters only",
            "state": ["route1_observer.update_index", "route1_observer.last.geometry"],
            "method": {"route1_ablation_enable": True, "amtnc_ablation_role": role},
            "expected": "Negative control proving that observing AM-TNC geometry and paying its diagnostic compute cannot itself alter UNSB dynamics.",
            "falsifier": "Any next-update or e200 dynamics mismatch with plain is an implementation failure.",
            "unbiased": "The diagnostic transition is discarded and all mutable stochastic/buffer state is restored before the pathwise native update.",
        }
    if family in ("mcrb", "rfmcrb") and role == "proposal_only":
        repaired = family == "rfmcrb"
        role_key = "rfmcrb_ablation_role" if repaired else "mcrb_ablation_role"
        epsilon_key = (
            "rfmcrb_projection_epsilon" if repaired else "mcrb_projection_epsilon"
        )
        return {
            "name": (
                "RF-MCRB Proposal-Only Moving Covariance Tangent"
                if repaired else "MCRB Proposal-Only Moving Covariance Tangent"
            ),
            "formula": "After realizing native Adam moments and displacement Delta, replace the generator displacement by -||Delta|| grad C/||grad C|| whenever the moving current/EMA covariance-gap tangent is nonzero. The native-safe half-space projection is removed.",
            "identity": "Disabled mode is exact native UNSB; active mode is exact identity only when the covariance tangent or native displacement norm is zero.",
            "objective_change": True, "estimator_change": False,
            "compute": "one current/EMA covariance tangent plus the realized native Adam step, matching full MCRB order",
            "memory": "one EMA generator and covariance-tangent graph",
            "recovery": "EMA generator, tangent diagnostics and proposal counters",
            "state": ["mcrb.teacher_netG", "mcrb.update_index", "mcrb.last"],
            "method": {"route1_ablation_enable": True, role_key: role, "mcrb_m": 4, "mcrb_region_patch": 32, "mcrb_u_floor": 1e-30, "mcrb_teacher_half_life_updates": 150, epsilon_key: 1e-24},
            "expected": "Separates the moving covariance proposal from MCRB's native-safe half-space constraint.",
            "falsifier": (
                "A complete e200 trajectory below its source-bound full barrier shows that the native-safe projection, rather than the raw covariance tangent, is essential."
                if repaired else
                "A complete e200 trajectory below full MCRB shows that the native-safe projection, rather than the raw covariance tangent, is essential."
            ),
        }
    if family in ("mcrb", "rfmcrb"):
        repaired = family == "rfmcrb"
        role_key = "rfmcrb_ablation_role" if repaired else "mcrb_ablation_role"
        epsilon_key = (
            "rfmcrb_projection_epsilon" if repaired else "mcrb_projection_epsilon"
        )
        return {
            "name": (
                "RF-MCRB Observable-Only Moving Covariance Monitor"
                if repaired else "MCRB Observable-Only Moving Covariance Monitor"
            ),
            "formula": (
                "Compute the current/EMA covariance gap, its tangent, the derivative along the realized native Adam displacement and the hypothetical residual-feasible closest point, but commit that native displacement unchanged. The EMA and diagnostics live only under route1_observer."
                if repaired else
                "Compute the current/EMA covariance gap, its tangent and the derivative along the realized native Adam displacement, but commit that native displacement unchanged. The EMA and diagnostics live only under route1_observer."
            ),
            "identity": "After removing only route1_observer diagnostics, complete next-update dynamics and e200 evaluation must equal plain exactly.",
            "objective_change": False, "estimator_change": False,
            "compute": "one current/EMA covariance tangent in addition to the native generator update",
            "memory": "one observer-only EMA generator and one tangent graph",
            "recovery": "route1_observer EMA, counters and last derivative",
            "state": ["route1_observer.teacher_netG", "route1_observer.update_index"],
            "method": {"route1_ablation_enable": True, role_key: role, "mcrb_m": 4, "mcrb_region_patch": 32, "mcrb_u_floor": 1e-30, "mcrb_teacher_half_life_updates": 150, epsilon_key: 1e-24},
            "expected": "Negative control proving that the moving covariance observation and extra compute do not explain any gain.",
            "falsifier": "Any next-update or e200 dynamics mismatch with plain is an implementation failure.",
            "unbiased": "The committed parameter and optimizer transition is pathwise native; the observer EMA is excluded from and cannot enter subsequent native updates.",
        }
    if family == "pcnr" and role == "proposal_only":
        return {
            "name": "PCNR Proposal-Only Player-Conditional Resampling",
            "formula": "Commit one native D/E stochastic view, then draw one fresh native view after both opponent commits and use it for the single joint G/F update. This is the complete PCNR proposal because PCNR has no projection, auxiliary loss or controller.",
            "identity": "Disabled mode dispatches native UNSB exactly. Conditional on the realized D/E-updated state, the fresh one-view G/F estimator has the native mean and native single-view variance.",
            "objective_change": False, "estimator_change": True,
            "compute": "two serial native stochastic views per optimizer update, identical to full PCNR",
            "memory": "one stochastic view graph at a time",
            "recovery": "PCNR view counters, event schedule and ordinary RNG",
            "state": ["pcnr.update_index", "pcnr.bundle_serial", "pcnr.last_schedule"],
            "method": {"route1_ablation_enable": True, "pcnr_ablation_role": role},
            "expected": "Confirms that the source-frozen full result is attributable to player-conditional resampling itself rather than an unregistered side component.",
            "falsifier": "Any non-identical complete trajectory relative to the source-frozen full PCNR under the same host/e0 is an implementation-identity failure.",
            "unbiased": "D/E retain the native estimator. Given the realized D/E state, the fresh G/F view is distributed by the native measure, so its conditional expected vector field is unchanged.",
        }
    if family == "pcnr":
        return {
            "name": "PCNR Observable-Only Fresh-View Monitor",
            "formula": "Draw two counterfactual native endpoint views, record their target-blind dispersion, restore Python/NumPy/CPU/CUDA RNG to the pre-diagnostic state, and then execute the ordinary shared-view UNSB transition.",
            "identity": "After excluding route1_observer diagnostics, networks, optimizer states, schedulers, RNG, both samplers and the e200 evaluation must equal plain exactly.",
            "objective_change": False, "estimator_change": False,
            "compute": "two discarded no-gradient endpoint views plus the native update",
            "memory": "two transient detached endpoints",
            "recovery": "route1_observer counters and last dispersion only",
            "state": ["route1_observer.update_index", "route1_observer.last"],
            "method": {"route1_ablation_enable": True, "pcnr_ablation_role": role},
            "expected": "Negative control proving that observing fresh-view dispersion and paying its compute cannot change native UNSB dynamics.",
            "falsifier": "Any next-update or e200 dynamics mismatch with plain is an implementation failure.",
            "unbiased": "The diagnostic views are discarded and every RNG stream is restored before the pathwise native transition.",
        }
    if family in ("ammcrb", "rfammcrb") and role == "proposal_only":
        repaired = family == "rfammcrb"
        role_key = (
            "rfammcrb_ablation_role" if repaired else "ammcrb_ablation_role"
        )
        epsilon_key = (
            "rfammcrb_projection_epsilon"
            if repaired else "ammcrb_projection_epsilon"
        )
        return {
            "name": (
                "RF-AMMCRB Proposal-Only Adam-Metric Normal"
                if repaired else "AM-MCRB Proposal-Only Adam-Metric Normal"
            ),
            "formula": "Realize native Adam moments and displacement d0, compute the moving covariance tangent a and inverse trust metric P, then replace d0 by -sPa where s makes the proposal's H-metric norm equal to the native H-metric norm. The native-safe closest-point term is removed.",
            "identity": "Disabled mode is exact native UNSB; active identity holds only when the tangent or native displacement has zero Adam-metric norm.",
            "objective_change": True, "estimator_change": False,
            "compute": "the full AM-MCRB tangent and Adam metric plus one metric-norm scaling",
            "memory": "one EMA generator, tangent graph and diagonal inverse metric",
            "recovery": "EMA generator, proposal counters and Adam-metric geometry",
            "state": ["mcrb.teacher_netG", "mcrb.update_index", "mcrb.last"],
            "method": {"route1_ablation_enable": True, role_key: role, "mcrb_m": 4, "mcrb_region_patch": 32, "mcrb_u_floor": 1e-30, "mcrb_teacher_half_life_updates": 150, epsilon_key: 1e-24},
            "expected": "Separates the Adam-metric moving-covariance proposal from the native-safe closest feasible displacement.",
            "falsifier": (
                "A complete e200 trajectory below its source-bound full barrier shows that the safe closest-point term, not merely the metric normal, is necessary."
                if repaired else
                "A complete e200 trajectory below full AM-MCRB shows that the safe closest-point term, not merely the metric normal, is necessary."
            ),
        }
    if family in ("ammcrb", "rfammcrb"):
        repaired = family == "rfammcrb"
        role_key = (
            "rfammcrb_ablation_role" if repaired else "ammcrb_ablation_role"
        )
        epsilon_key = (
            "rfammcrb_projection_epsilon"
            if repaired else "ammcrb_projection_epsilon"
        )
        return {
            "name": (
                "RF-AMMCRB Observable-Only Adam Barrier Monitor"
                if repaired else "AM-MCRB Observable-Only Adam Barrier Monitor"
            ),
            "formula": (
                "Compute the moving covariance tangent, realized native Adam displacement, diagonal metric and hypothetical residual-feasible KKT correction, but commit the native displacement unchanged. The EMA and geometry remain observer-only."
                if repaired else
                "Compute the moving covariance tangent, realized native Adam displacement, diagonal metric and hypothetical KKT correction, but commit the native displacement unchanged. The EMA and geometry remain observer-only."
            ),
            "identity": "After excluding route1_observer diagnostics, complete next-update dynamics and e200 evaluation must equal plain exactly.",
            "objective_change": False, "estimator_change": False,
            "compute": "the full AM-MCRB target-blind geometry in addition to the native update",
            "memory": "one observer-only EMA generator, tangent graph and inverse metric",
            "recovery": "route1_observer EMA, counters and last geometry",
            "state": ["route1_observer.teacher_netG", "route1_observer.update_index"],
            "method": {"route1_ablation_enable": True, role_key: role, "mcrb_m": 4, "mcrb_region_patch": 32, "mcrb_u_floor": 1e-30, "mcrb_teacher_half_life_updates": 150, epsilon_key: 1e-24},
            "expected": "Negative control proving that the moving covariance/Adam observation and extra compute do not explain any gain.",
            "falsifier": "Any next-update or e200 dynamics mismatch with plain is an implementation failure.",
            "unbiased": "The committed network and optimizer transition is pathwise native; observer state cannot enter subsequent native updates.",
        }
    raise ValueError(f"unknown winner ablation family: {family}")


def _card(
    *, parent: dict[str, Any], parent_id: str, parent_receipt_sha256: str,
    candidate_id: str, family: str, role: str, sibling_ids: dict[str, str],
) -> dict[str, Any]:
    semantics = _role_semantics(family, role)
    card = copy.deepcopy(parent)
    for key in (
        "engineering_replacement_for", "engineering_incident_sha256",
        "finite_step_coupling_change", "unbiased_proof",
    ):
        card.pop(key, None)
    card.update({
        "candidate_id": candidate_id,
        "contract_id": candidate_id,
        "name": semantics["name"],
        "parent_candidate_id": parent_id,
        "parent_terminal_receipt_sha256": parent_receipt_sha256,
        "ablation_role": role,
        "lineage_evidence": list(parent["lineage_evidence"]) + [
            f"WINNER_ABLATION:{parent_id}:{role}:source-bound-e200-parent",
        ],
        "prior_equivalence_audit": {
            "compared_implementations": [parent_id, sibling_ids["proposal_only"], sibling_ids["observable_only"]],
            "material_difference": semantics["formula"],
            "equivalent_rerun": False,
        },
        "formula": semantics["formula"],
        "identity_or_unbiased_condition": semantics["identity"],
        "objective_change": semantics["objective_change"],
        "estimator_change": semantics["estimator_change"],
        "coordinate_change": False,
        "endpoint_law_change": False,
        "expected_applicable_state": semantics["expected"],
        "falsifying_experiment": semantics["falsifier"],
        "compute_cost": semantics["compute"],
        "memory_cost": semantics["memory"],
        "recovery_state_cost": semantics["recovery"],
        "algorithm_hyperparameters": semantics["method"],
        "algorithm_state_variables": semantics["state"],
        "ablation_definitions": {
            "proposal_only": sibling_ids["proposal_only"],
            "observable_only": sibling_ids["observable_only"],
            "projected_or_full": parent_id,
        },
        "target_inaccessibility_proof": (
            "This source-bound ablation reads only the same unpaired official batch, native stochastic variables and target-blind internal state as its parent. Discovery70, confirmation20 and paired metrics are not addressable."
        ),
        "paired_target_available_to_training": False,
    })
    if "unbiased" in semantics:
        card["unbiased_proof"] = semantics["unbiased"]
    return card


def _implementation(candidate_id: str, family: str, role: str, card_path: Path) -> dict:
    model = f"route1_{family}_ablation"
    shared = [
        "src/models/sb_model.py", "src/models/route1/__init__.py",
        f"src/models/route1/{family}.py",
        f"src/models/route1/{family}_ablation.py",
        f"src/models/route1_{family}_ablation_model.py",
        "research/local_route1/generation1_gates.py",
    ]
    method = dict(_role_semantics(family, role)["method"])
    if family == "bvcp":
        method["bvcp_root_epsilon"] = 1e-12
    return {
        "schema": "final-unsb-route1-candidate-implementation-v1",
        "candidate_id": candidate_id,
        "status": "FROZEN_FOR_GATES",
        "derivation_card_sha256": file_sha256(card_path),
        "model": model,
        "method": method,
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
            "observer_state_excluded_only_from_dynamics_identity": role == "observable_only",
        },
        "zero_intervention": {"route1_ablation_enable": False},
        "gate_hook": {
            "module": "research.local_route1.generation1_gates",
            "callable": "run_winner_ablation_gate",
        },
        "source_files": _source_rows(shared),
    }


def materialize_parent_ablation_definitions(
    output_root: Path, *, parent_id: str, authority_path: Path,
    authority_algorithm_fingerprint: str,
    freeze_filename: str,
    authority_kind: str,
) -> dict[str, Any]:
    """Freeze one complete parent's two mechanism ablations.

    This primitive deliberately accepts a source-bound terminal parent rather
    than requiring that the parent already be the unique final winner.  It is
    what lets the repaired frontier retain more than one evidence-qualified
    algorithm for mechanism follow-up without weakening any receipt checks.
    """
    output_root = Path(output_root).resolve()
    authority_path = Path(authority_path).resolve()
    if not authority_path.is_file() or not authority_path.is_relative_to(output_root):
        raise RuntimeError("ablation authority escaped the run root")
    if parent_id not in WINNER_FAMILIES:
        raise RuntimeError("terminal parent has no registered ablation family")
    family_record = WINNER_FAMILIES[parent_id]
    family = str(family_record["family"])
    ids = dict(family_record["ids"])
    parent_receipt_path = (
        output_root / "operations" / "terminal_receipts" / f"{parent_id}.json"
    )
    parent_card_path = output_root / "derive" / "cards" / f"{parent_id}.json"
    if not parent_receipt_path.is_file() or not parent_card_path.is_file():
        raise RuntimeError("winner source-bound receipt/card is missing")
    parent_receipt = _read_json(parent_receipt_path)
    if (
        parent_receipt.get("candidate_id") != parent_id
        or parent_receipt.get("algorithm_fingerprint")
        != authority_algorithm_fingerprint
        or parent_receipt.get("derivation_card_sha256") != file_sha256(parent_card_path)
    ):
        raise RuntimeError("parent receipt/card/authority identity mismatch")
    parent = _read_json(parent_card_path)

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    frozen = []
    for role in ("proposal_only", "observable_only"):
        candidate_id = ids[role]
        card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
        implementation_path = (
            output_root / "derive" / "implementations" / f"{candidate_id}.json"
        )
        generated_card = _card(
            parent=parent, parent_id=parent_id,
            parent_receipt_sha256=file_sha256(parent_receipt_path),
            candidate_id=candidate_id, family=family, role=role,
            sibling_ids=ids,
        )
        if card_path.is_file() and _read_json(card_path) != generated_card:
            raise RuntimeError(f"winner ablation card already differs: {candidate_id}")
        write_json(card_path, generated_card)
        generated_implementation = _implementation(
            candidate_id, family, role, card_path,
        )
        if implementation_path.is_file() and _read_json(
            implementation_path
        ) != generated_implementation:
            raise RuntimeError(f"winner ablation implementation already differs: {candidate_id}")
        write_json(implementation_path, generated_implementation)

        records = [
            row for row in ledger.get("records", [])
            if isinstance(row, dict) and row.get("candidate_id") == candidate_id
        ]
        if not records:
            ledger["records"].append({
                "candidate_id": candidate_id,
                "generation": 0,
                "parent_candidate_id": parent_id,
                "parent_evidence": parent.get("parent_evidence"),
                "construction_route": "winner_mechanism_ablation",
                "ablation_role": role,
                "status": "DERIVATION_REQUIRED",
                "revision_count": 0,
                "experiments": [],
                "paired_controller_access": False,
                "confirmation20_opened": False,
            })
        elif len(records) != 1 or (
            records[0].get("parent_candidate_id") != parent_id
            or records[0].get("ablation_role") != role
        ):
            raise RuntimeError(f"winner ablation ledger slot conflicts: {candidate_id}")
        write_json(ledger_path, ledger)
        registration = freeze_candidate_derivation(output_root, candidate_id)
        ledger = _read_json(ledger_path)
        frozen.append(registration.to_dict())

    result = {
        "schema": SCHEMA,
        "status": "FROZEN_FOR_EXECUTABLE_GATES",
        "parent_candidate_id": parent_id,
        "parent_terminal_receipt_sha256": file_sha256(parent_receipt_path),
        "source_authority_kind": authority_kind,
        "source_authority_path": authority_path.relative_to(output_root).as_posix(),
        "source_authority_sha256": file_sha256(authority_path),
        # Preserve the established aliases for old final-delivery consumers.
        "source_cross_version_adjudication_sha256": file_sha256(authority_path),
        "source_e200_selection_path": authority_path.relative_to(output_root).as_posix(),
        "source_e200_selection_sha256": file_sha256(authority_path),
        "ablation_candidate_ids": ids,
        "registrations": frozen,
        "long_horizon_started": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "operations" / freeze_filename, result)
    return result


def materialize_winner_ablation_definitions(
    output_root: Path, *, selection_path: Path | None = None,
    freeze_filename: str = "WINNER_ABLATION_FREEZE.json",
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    cross_path = (
        resolve_e200_selection_path(output_root)
        if selection_path is None else Path(selection_path).resolve()
    )
    cross = _read_json(cross_path)
    validate_e200_selection(cross_path)
    if cross.get("status") not in TERMINAL_SELECTION_STATUSES:
        raise RuntimeError("winner ablations require a terminal source-bound e200 selection")
    parent_id = str(cross["selected_candidate_id"])
    return materialize_parent_ablation_definitions(
        output_root,
        parent_id=parent_id,
        authority_path=cross_path,
        authority_algorithm_fingerprint=str(cross["selected_algorithm_fingerprint"]),
        freeze_filename=freeze_filename,
        authority_kind="terminal_e200_selection",
    )
