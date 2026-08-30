"""Evidence-bound registration for route-1 algorithms discovered after Phase C.

This module deliberately does not contain a candidate formula.  It freezes the
link from a completed causal atlas to a derivation card and then to arbitrary
model code.  The long runner can therefore support a genuinely discovered
algorithm without turning the repository into a list of prewritten lanes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .causal_audit import training_core_fingerprint
from .protocol import ROOT, ProbeSpec, file_sha256, load_protocol, object_sha256
from .runtime import write_json


CARD_SCHEMA = "final-unsb-route1-derivation-card-v1"
CARD_REQUIRED_FIELDS = (
    "parent_evidence",
    "lineage_evidence",
    "prior_equivalence_audit",
    "unsb_object",
    "formula",
    "identity_or_unbiased_condition",
    "objective_change",
    "estimator_change",
    "coordinate_change",
    "endpoint_law_change",
    "target_inaccessibility_proof",
    "expected_applicable_state",
    "falsifying_experiment",
    "compute_cost",
    "memory_cost",
    "recovery_state_cost",
    "algorithm_hyperparameters",
    "algorithm_state_variables",
    "ablation_definitions",
    "historical_evidence_index_sha256",
    "mechanism_object_map_sha256",
    "reuse_boundary_sha256",
)
IMPLEMENTATION_SCHEMA = "final-unsb-route1-candidate-implementation-v1"
GATE_SCHEMA = "final-unsb-route1-candidate-gate-v1"
REGISTRATION_SCHEMA = "final-unsb-route1-candidate-registration-v1"
REVISION_REQUEST_SCHEMA = "final-unsb-route1-causal-revision-request-v1"
DEFECT_ADJUDICATION_SCHEMA = "final-unsb-route1-candidate-defect-adjudication-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class CandidateRegistration:
    candidate_id: str
    card_path: Path
    implementation_path: Path
    card: dict[str, Any]
    implementation: dict[str, Any]
    spec: ProbeSpec
    algorithm_fingerprint: str
    candidate_fingerprint: str
    causal_matrix_sha256: str
    reversal_atlas_sha256: str
    base_e0_scientific_state_sha256: str
    base_protocol_fingerprint: str
    candidate_training_core_fingerprint: str
    hypothesis_ledger_sha256: str | None
    hypothesis_freeze_sha256: str | None
    gate: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": REGISTRATION_SCHEMA,
            "candidate_id": self.candidate_id,
            "card": str(self.card_path.resolve()),
            "implementation": str(self.implementation_path.resolve()),
            "spec": self.spec.to_dict(),
            "algorithm_fingerprint": self.algorithm_fingerprint,
            "candidate_fingerprint": self.candidate_fingerprint,
            "causal_matrix_sha256": self.causal_matrix_sha256,
            "reversal_atlas_sha256": self.reversal_atlas_sha256,
            "base_e0_scientific_state_sha256": self.base_e0_scientific_state_sha256,
            "base_protocol_fingerprint": self.base_protocol_fingerprint,
            "candidate_training_core_fingerprint": self.candidate_training_core_fingerprint,
            "hypothesis_ledger_sha256": self.hypothesis_ledger_sha256,
            "hypothesis_freeze_sha256": self.hypothesis_freeze_sha256,
            "gate_status": None if self.gate is None else self.gate.get("status"),
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }


def validate_candidate_id(candidate_id: str) -> str:
    value = str(candidate_id)
    if not _SAFE_ID.fullmatch(value):
        raise ValueError(f"unsafe candidate id: {value!r}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def _inside_root(relative: str) -> Path:
    path = (ROOT / str(relative)).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"candidate source escapes repository root: {relative}") from error
    return path


def _validate_card(
    *, candidate_id: str, card_path: Path, card: dict[str, Any],
    matrix: dict[str, Any], matrix_sha256: str, atlas_sha256: str,
) -> None:
    if card.get("schema") != CARD_SCHEMA:
        raise RuntimeError("derivation card schema mismatch")
    if card.get("candidate_id") != candidate_id:
        raise RuntimeError("derivation card candidate_id mismatch")
    absent = [
        key for key in CARD_REQUIRED_FIELDS
        if key not in card or card[key] is None or card[key] == ""
    ]
    if absent:
        raise RuntimeError(f"incomplete derivation card: {absent}")
    for key in (
        "objective_change", "estimator_change", "coordinate_change",
        "endpoint_law_change",
    ):
        if not isinstance(card[key], bool):
            raise RuntimeError(f"derivation card {key} must be boolean")
    if not isinstance(card["lineage_evidence"], list) or not card["lineage_evidence"]:
        raise RuntimeError("derivation card requires non-empty lineage_evidence")
    if not isinstance(card["algorithm_hyperparameters"], dict):
        raise RuntimeError("algorithm_hyperparameters must be an object")
    if not isinstance(card["algorithm_state_variables"], list):
        raise RuntimeError("algorithm_state_variables must be a list")
    equivalence = card["prior_equivalence_audit"]
    if not isinstance(equivalence, dict):
        raise RuntimeError("prior_equivalence_audit must be an object")
    if equivalence.get("equivalent_rerun") is not False:
        raise RuntimeError("a new candidate may not be an equivalent rerun of a prior implementation")
    if not equivalence.get("compared_implementations") or not equivalence.get(
        "material_difference"
    ):
        raise RuntimeError(
            "prior_equivalence_audit requires compared implementations and a material difference"
        )
    ablations = card["ablation_definitions"]
    if not isinstance(ablations, dict):
        raise RuntimeError("ablation_definitions must be an object")
    missing_ablations = [
        key for key in ("proposal_only", "observable_only", "projected_or_full")
        if not ablations.get(key)
    ]
    if missing_ablations:
        raise RuntimeError(f"derivation card missing ablations: {missing_ablations}")
    historical_sources = {
        "historical_evidence_index_sha256": ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl",
        "mechanism_object_map_sha256": ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json",
        "reuse_boundary_sha256": ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json",
    }
    for field, source in historical_sources.items():
        if not source.is_file() or card[field] != file_sha256(source):
            raise RuntimeError(f"derivation card historical evidence binding mismatch: {field}")
    if card.get("causal_matrix_sha256") != matrix_sha256:
        raise RuntimeError("derivation card is not frozen to the current causal matrix")
    if card.get("reversal_atlas_sha256") != atlas_sha256:
        raise RuntimeError("derivation card is not frozen to the current reversal atlas")
    authority = card.get("construction_authority")
    if authority not in (
        "eligible_target_blind_signal",
        "eligible_method_specific_signal",
        "independent_unbiased_reparameterization",
    ):
        raise RuntimeError(
            "construction_authority must be an eligible target-blind signal or an "
            "independently justified unbiased reparameterization"
        )
    if authority == "independent_unbiased_reparameterization" and not card.get("unbiased_proof"):
        raise RuntimeError("an independent unbiased route requires unbiased_proof")
    if authority in (
        "eligible_target_blind_signal", "eligible_method_specific_signal",
    ):
        parent = card.get("parent_evidence")
        if not isinstance(parent, dict) or not parent.get("failure_type"):
            raise RuntimeError("signal-driven card requires a parent failure_type")
        eligible_mechanisms = {
            row.get("failure_type"): row
            for row in matrix.get("ranked_failure_mechanisms", [])
            if row.get("candidate_generation_eligible") is True
        }
        if parent["failure_type"] not in eligible_mechanisms:
            raise RuntimeError("card parent mechanism is not eligible in the causal matrix")
        parent_mechanism = eligible_mechanisms[parent["failure_type"]]
        driver = card.get("target_blind_driver_signal")
        screen = matrix.get("target_blind_signal_screen") or {}
        if authority == "eligible_target_blind_signal":
            eligible_signals = set(screen.get(
                "eligible_shared_driver_signals",
                screen.get("eligible_driver_signals", []),
            ))
            if not driver or driver not in eligible_signals:
                raise RuntimeError(
                    "card driver is not an eligible target-blind signal shared across probes"
                )
            mechanism_drivers = parent_mechanism.get(
                "eligible_target_blind_driver_signals"
            )
            if mechanism_drivers is not None and driver not in set(mechanism_drivers):
                raise RuntimeError(
                    "card driver is not evidence-linked to the declared parent mechanism"
                )
        else:
            driver_probe = card.get("target_blind_driver_probe")
            method_specific = screen.get(
                "eligible_method_specific_driver_signals", {}
            )
            if (
                not driver_probe
                or driver not in set(method_specific.get(driver_probe, []))
            ):
                raise RuntimeError(
                    "card driver is not eligible for the declared method-specific probe"
                )
            supporting = set(parent_mechanism.get("supporting_probes", []))
            if driver_probe not in supporting:
                raise RuntimeError(
                    "method-specific driver probe does not support the parent mechanism"
                )
            mechanism_drivers = parent_mechanism.get(
                "eligible_method_specific_driver_signals_by_probe"
            )
            if (
                mechanism_drivers is not None
                and driver not in set(mechanism_drivers.get(driver_probe, []))
            ):
                raise RuntimeError(
                    "method-specific driver is not evidence-linked to the declared parent mechanism"
                )
    if card.get("paired_target_available_to_training") is not False:
        raise RuntimeError("derivation card must explicitly deny paired target access")
    ablation_role = card.get("ablation_role")
    if ablation_role is not None:
        if ablation_role not in ("proposal_only", "observable_only"):
            raise RuntimeError("winner ablation role must be proposal_only or observable_only")
        parent_id = validate_candidate_id(str(card.get("parent_candidate_id", "")))
        output_root = card_path.resolve().parents[2]
        receipt_path = (
            output_root / "operations" / "terminal_receipts" / f"{parent_id}.json"
        )
        if (
            not receipt_path.is_file()
            or card.get("parent_terminal_receipt_sha256") != file_sha256(receipt_path)
        ):
            raise RuntimeError("winner ablation is not bound to its source-bound parent receipt")
    replacement_for = card.get("engineering_replacement_for")
    if replacement_for is not None:
        validate_candidate_id(str(replacement_for))
        incident_path = (
            ROOT / "evidence" / "remote_route1_offload"
            / "RSMG_PLAYER_STATE_SEMANTIC_INCIDENT_20260830.json"
        )
        if (
            not incident_path.is_file()
            or card.get("engineering_incident_sha256") != file_sha256(incident_path)
        ):
            raise RuntimeError("engineering replacement is not bound to its incident")
        if not card.get("finite_step_coupling_change"):
            raise RuntimeError(
                "engineering replacement must disclose its finite-step coupling change"
            )
    if card_path.suffix.lower() != ".json":
        raise RuntimeError("derivation card must be canonical JSON")


def _validate_implementation(
    *, candidate_id: str, implementation: dict[str, Any], card_sha256: str,
) -> list[dict[str, str]]:
    if implementation.get("schema") != IMPLEMENTATION_SCHEMA:
        raise RuntimeError("candidate implementation schema mismatch")
    if implementation.get("candidate_id") != candidate_id:
        raise RuntimeError("candidate implementation id mismatch")
    if implementation.get("status") != "FROZEN_FOR_GATES":
        raise RuntimeError("candidate implementation is not frozen for gates")
    if implementation.get("derivation_card_sha256") != card_sha256:
        raise RuntimeError("candidate implementation is not bound to its derivation card")
    if not isinstance(implementation.get("model"), str) or not implementation["model"]:
        raise RuntimeError("candidate implementation requires a registered model name")
    if not isinstance(implementation.get("method"), dict):
        raise RuntimeError("candidate implementation method options must be an object")
    if implementation.get("training_target_access") != "unpaired_only":
        raise RuntimeError("candidate implementation may only access unpaired training data")
    if implementation.get("paired_controller_access") is not False:
        raise RuntimeError("candidate implementation must deny paired controller access")
    state = implementation.get("state_contract", {})
    for key in (
        "full_state_restorable",
        "zero_intervention_identity_test",
        "parent_state_isolation_test",
    ):
        if state.get(key) is not True:
            raise RuntimeError(f"candidate state contract missing {key}=true")

    registered_sources = implementation.get("source_files")
    if not isinstance(registered_sources, list) or not registered_sources:
        raise RuntimeError("candidate implementation requires frozen source_files")
    sources = []
    seen = set()
    for row in registered_sources:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise RuntimeError("every source_files row requires path and sha256")
        path = _inside_root(row["path"])
        if not path.is_file():
            raise RuntimeError(f"candidate source missing: {path}")
        relative = path.relative_to(ROOT.resolve()).as_posix()
        if relative in seen:
            raise RuntimeError(f"duplicate candidate source registration: {relative}")
        seen.add(relative)
        actual = file_sha256(path)
        if actual != str(row["sha256"]).lower():
            raise RuntimeError(f"candidate source hash mismatch: {relative}")
        sources.append({"path": relative, "sha256": actual})
    model_registration = ROOT / "src" / "models" / f"{implementation['model']}_model.py"
    if not model_registration.is_file():
        raise RuntimeError(f"registered model module missing: {model_registration}")
    if model_registration.relative_to(ROOT).as_posix() not in seen:
        raise RuntimeError("model registration module must be included in source_files")
    return sorted(sources, key=lambda row: row["path"])


def _base_e0_identity(output_root: Path) -> tuple[str, str, str]:
    e0_path = output_root / "shared_e0" / "e0.pt"
    sidecar_path = Path(str(e0_path) + ".json")
    if not e0_path.is_file() or not sidecar_path.is_file():
        raise RuntimeError("shared e0 and its sidecar are required")
    sidecar = _read_json(sidecar_path)
    actual_checkpoint_hash = file_sha256(e0_path)
    if sidecar.get("checkpoint_sha256") != actual_checkpoint_hash:
        raise RuntimeError("shared e0 checkpoint hash mismatch")
    scientific_hash = str(sidecar.get("scientific_state_sha256", ""))
    metadata = sidecar.get("metadata", {})
    protocol_fingerprint = str(metadata.get("protocol_fingerprint", ""))
    # Older accepted e0 sidecars store metadata at top level only in the torch
    # payload.  Read just that payload when necessary; the long runner later
    # recomputes and verifies the full scientific-state hash before training.
    if not protocol_fingerprint:
        import torch

        payload = torch.load(e0_path, map_location="cpu", weights_only=False)
        protocol_fingerprint = str(payload.get("metadata", {}).get("protocol_fingerprint", ""))
    if not protocol_fingerprint:
        raise RuntimeError("shared e0 has no base protocol fingerprint")
    if not scientific_hash:
        raise RuntimeError("shared e0 sidecar has no scientific state hash")
    return actual_checkpoint_hash, scientific_hash, protocol_fingerprint


def _ledger_evidence_identity(
    *, matrix_sha256: str, atlas_sha256: str,
) -> dict[str, str]:
    return {
        "causal_matrix_sha256": matrix_sha256,
        "reversal_atlas_sha256": atlas_sha256,
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


def _validate_hypothesis_ledger(
    *, output_root: Path, candidate_id: str, matrix_sha256: str,
    atlas_sha256: str, card_sha256: str, implementation_sha256: str,
    algorithm_fingerprint: str,
) -> tuple[str, str]:
    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    if not ledger_path.is_file():
        raise RuntimeError("candidate is not registered in the hypothesis ledger")
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
        raise RuntimeError("hypothesis ledger schema mismatch")
    if ledger.get("evidence_identity") != _ledger_evidence_identity(
        matrix_sha256=matrix_sha256, atlas_sha256=atlas_sha256,
    ):
        raise RuntimeError("hypothesis ledger is stale for the causal evidence")
    if ledger.get("paired_controller_access") is not False:
        raise RuntimeError("hypothesis ledger must deny paired controller access")
    if ledger.get("confirmation20_opened") is not False:
        raise RuntimeError("hypothesis ledger must keep confirmation20 locked")
    records = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(records) != 1:
        raise RuntimeError("candidate must have exactly one hypothesis ledger record")
    record = records[0]
    if record.get("status") != "FROZEN_FOR_GATES":
        raise RuntimeError("candidate hypothesis is not frozen for gates")
    expected = {
        "derivation_card_sha256": card_sha256,
        "implementation_sha256": implementation_sha256,
        "algorithm_fingerprint": algorithm_fingerprint,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise RuntimeError(f"candidate hypothesis ledger binding mismatch: {key}")
    if record.get("paired_controller_access") is not False:
        raise RuntimeError("candidate ledger record must deny paired controller access")
    if record.get("confirmation20_opened") is not False:
        raise RuntimeError("candidate ledger record must keep confirmation20 locked")
    freeze_identity = {
        "schema": "final-unsb-route1-hypothesis-freeze-v1",
        "evidence_identity": ledger["evidence_identity"],
        "candidate_id": candidate_id,
        "generation": record.get("generation"),
        "parent_candidate_id": record.get("parent_candidate_id"),
        "parent_evidence": record.get("parent_evidence"),
        "construction_route": record.get("construction_route"),
        "revision_count": record.get("revision_count"),
        "engineering_replacement": record.get("engineering_replacement"),
        **expected,
        "freeze_event": record.get("freeze_event"),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    return file_sha256(ledger_path), object_sha256(freeze_identity)


def load_candidate_registration(
    output_root: Path, candidate_id: str, *, require_gate: bool = False,
    require_hypothesis_ledger: bool = True,
) -> CandidateRegistration:
    """Validate an evidence/card/code chain and optionally its long-run gate."""
    candidate_id = validate_candidate_id(candidate_id)
    output_root = Path(output_root).resolve()
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    if not matrix_path.is_file() or not atlas_path.is_file():
        raise RuntimeError("completed causal matrix and reversal atlas are required")
    matrix = _read_json(matrix_path)
    if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
        raise RuntimeError("candidate registration requires a complete causal audit")
    matrix_sha256 = file_sha256(matrix_path)
    atlas_sha256 = file_sha256(atlas_path)

    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = output_root / "derive" / "implementations" / f"{candidate_id}.json"
    if not card_path.is_file() or not implementation_path.is_file():
        raise RuntimeError("candidate derivation card and implementation manifest are required")
    card = _read_json(card_path)
    implementation = _read_json(implementation_path)
    _validate_card(
        candidate_id=candidate_id, card_path=card_path, card=card,
        matrix=matrix, matrix_sha256=matrix_sha256, atlas_sha256=atlas_sha256,
    )
    card_sha256 = file_sha256(card_path)
    sources = _validate_implementation(
        candidate_id=candidate_id, implementation=implementation,
        card_sha256=card_sha256,
    )
    e0_file_sha256, e0_state_sha256, base_protocol_fingerprint = _base_e0_identity(output_root)
    protocol = load_protocol()
    candidate_training_core = training_core_fingerprint(ROOT)
    definition_card_fields = (
        "candidate_id", "lineage_evidence", "prior_equivalence_audit",
        "unsb_object", "formula", "identity_or_unbiased_condition",
        "target_inaccessibility_proof", "construction_authority", "unbiased_proof",
        "target_blind_driver_signal", "target_blind_driver_probe",
        "parent_candidate_id", "revision_request_sha256", "causal_revision_reason",
        "engineering_replacement_for", "engineering_incident_sha256",
        "finite_step_coupling_change",
        "ablation_role", "parent_terminal_receipt_sha256",
        "objective_change", "estimator_change",
        "coordinate_change", "endpoint_law_change", "algorithm_hyperparameters",
        "algorithm_state_variables", "expected_applicable_state",
        "falsifying_experiment", "compute_cost", "memory_cost",
        "recovery_state_cost", "ablation_definitions",
        "historical_evidence_index_sha256", "mechanism_object_map_sha256",
        "reuse_boundary_sha256",
    )
    definition_implementation_fields = (
        "schema", "candidate_id", "model", "method", "training_target_access",
        "paired_controller_access", "state_contract", "zero_intervention",
        "gate_hook",
    )
    algorithm_fingerprint_payload = {
        "schema": "final-unsb-route1-algorithm-fingerprint-v1",
        "definition_card": {
            key: card.get(key) for key in definition_card_fields if key in card
        },
        "definition_implementation": {
            key: implementation.get(key)
            for key in definition_implementation_fields if key in implementation
        },
        "sources": sources,
        "common_training_protocol": protocol["common"],
        "manifest_sha256": protocol["manifest"]["sha256"],
    }
    algorithm_fingerprint = object_sha256(algorithm_fingerprint_payload)
    implementation_sha256 = file_sha256(implementation_path)
    hypothesis_ledger_sha256 = None
    hypothesis_freeze_sha256 = None
    if require_hypothesis_ledger:
        hypothesis_ledger_sha256, hypothesis_freeze_sha256 = _validate_hypothesis_ledger(
            output_root=output_root,
            candidate_id=candidate_id,
            matrix_sha256=matrix_sha256,
            atlas_sha256=atlas_sha256,
            card_sha256=card_sha256,
            implementation_sha256=implementation_sha256,
            algorithm_fingerprint=algorithm_fingerprint,
        )
    fingerprint_payload = {
        "schema": REGISTRATION_SCHEMA,
        "algorithm_fingerprint": algorithm_fingerprint,
        "card_sha256": card_sha256,
        "implementation_sha256": implementation_sha256,
        # Bind execution to this candidate's immutable freeze record.  The
        # whole ledger may legitimately grow as sibling candidates are frozen.
        "hypothesis_freeze_sha256": hypothesis_freeze_sha256,
        "causal_matrix_sha256": matrix_sha256,
        "reversal_atlas_sha256": atlas_sha256,
        "candidate_training_core_fingerprint": candidate_training_core,
        "base_e0_file_sha256": e0_file_sha256,
        "base_e0_scientific_state_sha256": e0_state_sha256,
        "base_protocol_fingerprint": base_protocol_fingerprint,
        "candidate_orchestration_sources": [
            {
                "path": relative,
                "sha256": file_sha256(ROOT / relative),
            }
            for relative in (
                "research/local_route1/candidates.py",
                "research/local_route1/candidate_gate.py",
                "research/local_route1/candidate_runner.py",
            )
        ],
        "local_view": protocol["local_view"],
    }
    candidate_fingerprint = object_sha256(fingerprint_payload)
    gate_path = output_root / "derive" / "gates" / f"{candidate_id}.json"
    gate = _read_json(gate_path) if gate_path.is_file() else None
    if require_gate:
        if gate is None:
            raise RuntimeError("candidate long run is blocked until its gate record exists")
        if gate.get("schema") != GATE_SCHEMA or gate.get("status") != "PASS_LONG_RUN":
            raise RuntimeError("candidate long-run gate has not passed")
        if gate.get("candidate_fingerprint") != candidate_fingerprint:
            raise RuntimeError("candidate gate is stale for the registered candidate")
        if gate.get("algorithm_fingerprint") != algorithm_fingerprint:
            raise RuntimeError("candidate gate is stale for the frozen algorithm")
        checks = gate.get("checks", {})
        for key in (
            "mathematical_invariants",
            "zero_intervention_identity",
            "resume_exact",
            "cross_state_counterfactual",
            "target_blind_observable",
            "micro_engineering",
            "base_unsb_semantics_preserved",
            "shared_e0_load_exact",
        ):
            if checks.get(key) is not True:
                raise RuntimeError(f"candidate long-run gate missing {key}=true")
        if gate.get("paired_metric_used_for_promotion") is not False:
            raise RuntimeError("paired metrics may not promote a candidate through the gate")
        if gate.get("confirmation20_opened") is not False:
            raise RuntimeError("confirmation20 must remain locked")
        if implementation.get("model") in (
            "route1_pcrsmg", "route1_amtnc", "route1_pcnr",
        ):
            expected_schedule = (
                ["DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_VIEW", "GF_COMMIT"]
                if implementation.get("model") == "route1_pcnr" else
                ["DE_BUNDLE", "D_COMMIT", "E_COMMIT", "GF_BUNDLE", "GF_COMMIT"]
            )
            player_evidence = gate.get("evidence", {}).get(
                "player_conditional_execution_evidence"
            )
            if not isinstance(player_evidence, dict):
                raise RuntimeError("replicated gate lacks player-conditional execution evidence")
            if (
                player_evidence.get("all_de_and_gf_counts_equal_updates") is not True
                or player_evidence.get("all_bundle_serials_equal_twice_updates") is not True
                or player_evidence.get("expected_schedule") != expected_schedule
            ):
                raise RuntimeError("replicated gate has invalid player-bundle provenance")
        if implementation.get("model") == "route1_pcammcrb":
            parent = str(
                implementation.get("method", {}).get(
                    "pcammcrb_sampling_parent", "pcnr"
                )
            )
            expected_schedule = (
                ["DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_VIEW", "GF_COMMIT"]
                if parent == "pcnr" else
                [
                    "NATIVE_DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_BUNDLE",
                    "GF_BARRIER_COMMIT",
                ]
            )
            evidence = gate.get("evidence", {})
            player_evidence = evidence.get("player_conditional_execution_evidence")
            if not isinstance(player_evidence, dict) or (
                player_evidence.get("sampling_parent") != parent
                or player_evidence.get("expected_schedule") != expected_schedule
                or player_evidence.get("all_sampling_and_barrier_counts_equal_updates") is not True
            ):
                raise RuntimeError("PC-AMMCRB gate has invalid player/barrier provenance")
            compatibility = evidence.get("component_compatibility_evidence")
            if not isinstance(compatibility, dict) or (
                compatibility.get("data_epochs") != [20, 100, 200]
                or compatibility.get("branch_updates") != [1, 8, 32]
                or compatibility.get("all_parent_state_hashes_preserved") is not True
                or float(
                    compatibility.get(
                        "minimum_observed_component_correction_cosine", -2.0
                    )
                ) < -0.2
            ):
                raise RuntimeError("PC-AMMCRB component compatibility evidence is invalid")

    spec = ProbeSpec(
        id=candidate_id,
        contract_id=str(card.get("contract_id") or candidate_id),
        model=str(implementation["model"]),
        role="route1_candidate",
        method=dict(implementation["method"]),
        historical_fact=None,
    )
    return CandidateRegistration(
        candidate_id=candidate_id,
        card_path=card_path,
        implementation_path=implementation_path,
        card=card,
        implementation=implementation,
        spec=spec,
        algorithm_fingerprint=algorithm_fingerprint,
        candidate_fingerprint=candidate_fingerprint,
        causal_matrix_sha256=matrix_sha256,
        reversal_atlas_sha256=atlas_sha256,
        base_e0_scientific_state_sha256=e0_state_sha256,
        base_protocol_fingerprint=base_protocol_fingerprint,
        candidate_training_core_fingerprint=candidate_training_core,
        hypothesis_ledger_sha256=hypothesis_ledger_sha256,
        hypothesis_freeze_sha256=hypothesis_freeze_sha256,
        gate=gate,
    )


def freeze_candidate_derivation(
    output_root: Path, candidate_id: str,
) -> CandidateRegistration:
    """Freeze one derived algorithm into its pre-created causal ledger slot.

    This is intentionally idempotent only for an unchanged card, implementation
    and algorithm fingerprint.  A revised mechanism must use the one allowed
    revision record rather than silently rewriting a Generation-1 hypothesis.
    """
    output_root = Path(output_root).resolve()
    candidate_id = validate_candidate_id(candidate_id)
    registration = load_candidate_registration(
        output_root, candidate_id, require_gate=False,
        require_hypothesis_ledger=False,
    )
    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    if not ledger_path.is_file():
        raise RuntimeError("derive stage must create the hypothesis ledger first")
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
        raise RuntimeError("hypothesis ledger schema mismatch")
    expected_identity = _ledger_evidence_identity(
        matrix_sha256=registration.causal_matrix_sha256,
        atlas_sha256=registration.reversal_atlas_sha256,
    )
    if ledger.get("evidence_identity") != expected_identity:
        raise RuntimeError("hypothesis ledger is stale for the causal evidence")
    records = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if len(records) != 1:
        raise RuntimeError("candidate must occupy exactly one pre-created ledger slot")
    record = records[0]
    if int(record.get("generation", 1)) == 2:
        authorization = record.get("revision_authorization") or {}
        required_revision_bindings = {
            "parent_candidate_id": record.get("parent_candidate_id"),
            "revision_request_sha256": authorization.get("revision_request_sha256"),
            "causal_revision_reason": authorization.get("new_causal_failure_reason"),
        }
        for key, value in required_revision_bindings.items():
            if not value or registration.card.get(key) != value:
                raise RuntimeError(
                    f"revision derivation card is not bound to its causal authorization: {key}"
                )
    engineering_replacement = record.get("engineering_replacement")
    if engineering_replacement is not None:
        required_replacement_bindings = {
            "engineering_replacement_for": engineering_replacement.get(
                "parent_candidate_id"
            ),
            "engineering_incident_sha256": engineering_replacement.get(
                "incident_sha256"
            ),
        }
        for key, value in required_replacement_bindings.items():
            if not value or registration.card.get(key) != value:
                raise RuntimeError(
                    f"engineering replacement card binding mismatch: {key}"
                )
    bindings = {
        "derivation_card_sha256": file_sha256(registration.card_path),
        "implementation_sha256": file_sha256(registration.implementation_path),
        "algorithm_fingerprint": registration.algorithm_fingerprint,
    }
    if record.get("status") == "FROZEN_FOR_GATES":
        for key, value in bindings.items():
            if record.get(key) != value:
                raise RuntimeError(
                    f"frozen candidate hypothesis may not be silently rewritten: {key}"
                )
        return load_candidate_registration(output_root, candidate_id)
    if record.get("status") != "DERIVATION_REQUIRED":
        raise RuntimeError(
            "candidate ledger slot must be DERIVATION_REQUIRED before freezing"
        )
    record.update({
        "status": "FROZEN_FOR_GATES",
        **bindings,
        "freeze_event": {
            "event": "DERIVATION_AND_IMPLEMENTATION_FROZEN",
            "causal_matrix_sha256": registration.causal_matrix_sha256,
            "reversal_atlas_sha256": registration.reversal_atlas_sha256,
            "paired_metric_used_for_selection": False,
        },
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    write_json(ledger_path, ledger)
    return load_candidate_registration(output_root, candidate_id)


def register_engineering_replacement(
    output_root: Path, parent_candidate_id: str, replacement_candidate_id: str,
) -> dict:
    """Replace an implementation-invalid hypothesis without a scientific revision.

    This path cannot rescue a negative result.  It requires an immutable
    semantic incident that forbids scientific adjudication and requires an e0
    restart under a new identity.  The original record remains in the ledger.
    """
    output_root = Path(output_root).resolve()
    parent_candidate_id = validate_candidate_id(parent_candidate_id)
    replacement_candidate_id = validate_candidate_id(replacement_candidate_id)
    if parent_candidate_id == replacement_candidate_id:
        raise RuntimeError("engineering replacement requires a new candidate id")

    incident_path = (
        ROOT / "evidence" / "remote_route1_offload"
        / "RSMG_PLAYER_STATE_SEMANTIC_INCIDENT_20260830.json"
    )
    if not incident_path.is_file():
        raise RuntimeError("engineering incident evidence is missing")
    incident = _read_json(incident_path)
    if (
        incident.get("schema") != "final-unsb-route1-semantic-incident-v1"
        or incident.get("candidate_id") != parent_candidate_id
        or incident.get("classification") != "implementation_failure"
        or incident.get("scientific_conclusion_allowed") is not False
        or incident.get("parent_mechanism_falsified") is not False
        or incident.get("required_repair", {}).get("new_candidate_identity") is not True
        or incident.get("required_repair", {}).get("restart_from_common_e0") is not True
    ):
        raise RuntimeError("incident does not authorize an engineering replacement")

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    if not ledger_path.is_file():
        raise RuntimeError("derive stage must create the hypothesis ledger first")
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
        raise RuntimeError("hypothesis ledger schema mismatch")
    parents = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == parent_candidate_id
    ]
    if len(parents) != 1:
        raise RuntimeError("engineering replacement parent is not unique")
    parent = parents[0]
    invalid_algorithm = incident.get("invalid_identity", {}).get(
        "algorithm_fingerprint"
    )
    if (
        int(parent.get("generation", 0)) != 1
        or parent.get("algorithm_fingerprint") != invalid_algorithm
        or parent.get("status") not in ("FROZEN_FOR_GATES", "IMPLEMENTATION_INVALID")
    ):
        raise RuntimeError("incident does not match the frozen Generation-1 parent")

    incident_sha256 = file_sha256(incident_path)
    replacements = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == replacement_candidate_id
    ]
    if replacements:
        replacement = replacements[0]
        binding = replacement.get("engineering_replacement") or {}
        if (
            len(replacements) != 1
            or binding.get("parent_candidate_id") != parent_candidate_id
            or binding.get("incident_sha256") != incident_sha256
        ):
            raise RuntimeError("replacement id is already bound to different evidence")
        return {
            "schema": "final-unsb-route1-engineering-replacement-registration-v1",
            "status": replacement["status"],
            "record": replacement,
            "hypothesis_ledger_sha256": file_sha256(ledger_path),
            "confirmation20_opened": False,
        }

    already_replaced = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict)
        and (row.get("engineering_replacement") or {}).get("parent_candidate_id")
        == parent_candidate_id
    ]
    if already_replaced:
        raise RuntimeError("implementation-invalid parent already has a replacement")

    parent["status"] = "IMPLEMENTATION_INVALID"
    parent["scientific_result_admissible"] = False
    parent["engineering_incident"] = {
        "path": incident_path.relative_to(ROOT).as_posix(),
        "sha256": incident_sha256,
    }
    record = {
        "candidate_id": replacement_candidate_id,
        "generation": 1,
        "parent_candidate_id": parent_candidate_id,
        "parent_evidence": parent.get("parent_evidence"),
        "construction_route": parent.get("construction_route"),
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "engineering_replacement": {
            "parent_candidate_id": parent_candidate_id,
            "incident_sha256": incident_sha256,
            "consumes_generation1_scientific_slot": False,
            "consumes_causal_revision": False,
            "restart_from_common_e0": True,
        },
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    ledger["records"].append(record)
    write_json(ledger_path, ledger)
    return {
        "schema": "final-unsb-route1-engineering-replacement-registration-v1",
        "status": "DERIVATION_REQUIRED",
        "record": record,
        "hypothesis_ledger_sha256": file_sha256(ledger_path),
        "confirmation20_opened": False,
    }


def register_candidate_revision(
    output_root: Path, parent_candidate_id: str, revision_candidate_id: str,
) -> dict:
    """Append the single evidence-authorized Generation-2 slot for a mechanism."""
    output_root = Path(output_root).resolve()
    parent_candidate_id = validate_candidate_id(parent_candidate_id)
    revision_candidate_id = validate_candidate_id(revision_candidate_id)
    if parent_candidate_id == revision_candidate_id:
        raise RuntimeError("revision candidate id must differ from its parent")
    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    if not ledger_path.is_file():
        raise RuntimeError("derive stage must create the hypothesis ledger first")
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
        raise RuntimeError("hypothesis ledger schema mismatch")
    if int(ledger.get("generation_policy", {}).get(
        "maximum_revisions_per_mechanism", -1
    )) != 1:
        raise RuntimeError("hypothesis ledger does not authorize exactly one revision")
    parents = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == parent_candidate_id
    ]
    if len(parents) != 1 or parents[0].get("status") != "FROZEN_FOR_GATES":
        raise RuntimeError("revision parent must be one frozen Generation-1 candidate")
    parent = parents[0]
    if int(parent.get("generation", 0)) != 1:
        raise RuntimeError("a Generation-2 candidate cannot spawn another revision")
    trajectory_path = (
        output_root / "candidates" / parent_candidate_id / "CANDIDATE_TRAJECTORY.json"
    )
    if not trajectory_path.is_file():
        raise RuntimeError("causal revision requires the parent's complete e200 trajectory")
    trajectory = _read_json(trajectory_path)
    if (
        trajectory.get("candidate_id") != parent_candidate_id
        or trajectory.get("status") != "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION"
        or not any(int(row.get("epoch", -1)) == 200 for row in trajectory.get("trajectory", []))
    ):
        raise RuntimeError(
            "causal revision is allowed only after a negative complete e200 adjudication"
        )
    request_path = (
        output_root / "derive" / "revisions" / f"{revision_candidate_id}.json"
    )
    if not request_path.is_file():
        raise RuntimeError(f"causal revision request missing: {request_path}")
    request = _read_json(request_path)
    required = {
        "schema": REVISION_REQUEST_SCHEMA,
        "parent_candidate_id": parent_candidate_id,
        "revision_candidate_id": revision_candidate_id,
        "source_candidate_trajectory_sha256": file_sha256(trajectory_path),
        "fixed_window_or_handoff": False,
        "hyperparameter_grid_search": False,
        "paired_target_available_to_revision": False,
        "confirmation20_opened": False,
    }
    for key, value in required.items():
        if request.get(key) != value:
            raise RuntimeError(f"invalid causal revision request field: {key}")
    for key in (
        "new_causal_failure_reason", "mathematical_change_from_parent",
        "construction_route", "defect_evidence_path", "defect_evidence_sha256",
    ):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise RuntimeError(f"causal revision request requires non-empty {key}")
    evidence_relative = Path(request["defect_evidence_path"])
    if evidence_relative.is_absolute():
        raise RuntimeError("revision defect evidence path must be relative to the run root")
    evidence_path = (output_root / evidence_relative).resolve()
    if not evidence_path.is_relative_to(output_root) or not evidence_path.is_file():
        raise RuntimeError("revision defect evidence must exist inside the run root")
    if file_sha256(evidence_path) != request["defect_evidence_sha256"]:
        raise RuntimeError("revision defect evidence hash mismatch")
    defect = _read_json(evidence_path)
    defect_required = {
        "schema": DEFECT_ADJUDICATION_SCHEMA,
        "candidate_id": parent_candidate_id,
        "data_epoch_adjudicated": 200,
        "target_blind_defect_reduced": True,
        "long_horizon_benefit_reversed": True,
        "paired_target_used_to_compute_defect": False,
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    for key, value in defect_required.items():
        if defect.get(key) != value:
            raise RuntimeError(f"invalid candidate defect adjudication field: {key}")
    measurement = defect.get("target_blind_defect_measurement")
    if not isinstance(measurement, dict):
        raise RuntimeError("defect adjudication requires a target-blind measurement")
    for key in ("observable", "reference_value", "candidate_value"):
        if key not in measurement:
            raise RuntimeError(f"target-blind defect measurement missing {key}")
    try:
        reference_value = float(measurement["reference_value"])
        candidate_value = float(measurement["candidate_value"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("target-blind defect values must be finite numbers") from error
    if not np.isfinite(reference_value) or not np.isfinite(candidate_value):
        raise RuntimeError("target-blind defect values must be finite numbers")
    desired = measurement.get("desired_direction")
    reduced = (
        candidate_value < reference_value if desired == "decrease" else
        candidate_value > reference_value if desired == "increase" else False
    )
    if not reduced:
        raise RuntimeError("target-blind defect measurement does not demonstrate reduction")
    if defect.get("new_causal_failure_reason") != request["new_causal_failure_reason"]:
        raise RuntimeError("revision request and defect adjudication causal reasons differ")
    request_sha256 = file_sha256(request_path)
    existing_id = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == revision_candidate_id
    ]
    if existing_id:
        if (
            len(existing_id) == 1
            and existing_id[0].get("revision_authorization", {}).get(
                "revision_request_sha256"
            ) == request_sha256
        ):
            return {
                "schema": "final-unsb-route1-causal-revision-registration-v1",
                "status": existing_id[0]["status"],
                "record": existing_id[0],
                "hypothesis_ledger_sha256": file_sha256(ledger_path),
                "confirmation20_opened": False,
            }
        raise RuntimeError("revision candidate id is already bound to different evidence")
    parent_failure = (parent.get("parent_evidence") or {}).get("failure_type")
    if not parent_failure:
        raise RuntimeError("revision parent has no causal failure mechanism")
    revisions_for_mechanism = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict)
        and int(row.get("generation", 0)) == 2
        and (row.get("parent_evidence") or {}).get("failure_type") == parent_failure
    ]
    if revisions_for_mechanism:
        raise RuntimeError("the parent failure mechanism already used its one revision")
    record = {
        "candidate_id": revision_candidate_id,
        "generation": 2,
        "parent_candidate_id": parent_candidate_id,
        "parent_evidence": parent.get("parent_evidence"),
        "construction_route": request["construction_route"],
        "status": "DERIVATION_REQUIRED",
        "revision_count": 1,
        "revision_authorization": {
            "revision_request_sha256": request_sha256,
            "source_candidate_trajectory_sha256": file_sha256(trajectory_path),
            "defect_evidence_sha256": file_sha256(evidence_path),
            "new_causal_failure_reason": request["new_causal_failure_reason"],
            "mathematical_change_from_parent": request["mathematical_change_from_parent"],
            "paired_metric_used_for_training_or_control": False,
        },
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    ledger["records"].append(record)
    write_json(ledger_path, ledger)
    return {
        "schema": "final-unsb-route1-causal-revision-registration-v1",
        "status": "DERIVATION_REQUIRED",
        "record": record,
        "hypothesis_ledger_sha256": file_sha256(ledger_path),
        "confirmation20_opened": False,
    }
