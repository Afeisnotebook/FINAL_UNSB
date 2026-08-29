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

from .causal_audit import training_core_fingerprint
from .protocol import ROOT, ProbeSpec, file_sha256, load_protocol, object_sha256


CARD_REQUIRED_FIELDS = (
    "parent_evidence",
    "unsb_object",
    "formula",
    "identity_or_unbiased_condition",
    "target_inaccessibility_proof",
    "falsifying_experiment",
    "compute_cost",
)
IMPLEMENTATION_SCHEMA = "final-unsb-route1-candidate-implementation-v1"
GATE_SCHEMA = "final-unsb-route1-candidate-gate-v1"
REGISTRATION_SCHEMA = "final-unsb-route1-candidate-registration-v1"
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
    if card.get("candidate_id") != candidate_id:
        raise RuntimeError("derivation card candidate_id mismatch")
    absent = [key for key in CARD_REQUIRED_FIELDS if not card.get(key)]
    if absent:
        raise RuntimeError(f"incomplete derivation card: {absent}")
    if card.get("causal_matrix_sha256") != matrix_sha256:
        raise RuntimeError("derivation card is not frozen to the current causal matrix")
    if card.get("reversal_atlas_sha256") != atlas_sha256:
        raise RuntimeError("derivation card is not frozen to the current reversal atlas")
    authority = card.get("construction_authority")
    if authority not in (
        "eligible_target_blind_signal",
        "independent_unbiased_reparameterization",
    ):
        raise RuntimeError(
            "construction_authority must be an eligible target-blind signal or an "
            "independently justified unbiased reparameterization"
        )
    if authority == "independent_unbiased_reparameterization" and not card.get("unbiased_proof"):
        raise RuntimeError("an independent unbiased route requires unbiased_proof")
    if authority == "eligible_target_blind_signal":
        parent = card.get("parent_evidence")
        if not isinstance(parent, dict) or not parent.get("failure_type"):
            raise RuntimeError("signal-driven card requires a parent failure_type")
        eligible_mechanisms = {
            row.get("failure_type")
            for row in matrix.get("ranked_failure_mechanisms", [])
            if row.get("candidate_generation_eligible") is True
        }
        if parent["failure_type"] not in eligible_mechanisms:
            raise RuntimeError("card parent mechanism is not eligible in the causal matrix")
        driver = card.get("target_blind_driver_signal")
        eligible_signals = set(
            (matrix.get("target_blind_signal_screen") or {}).get(
                "eligible_driver_signals", []
            )
        )
        if not driver or driver not in eligible_signals:
            raise RuntimeError("card driver is not an eligible target-blind signal")
    if card.get("paired_target_available_to_training") is not False:
        raise RuntimeError("derivation card must explicitly deny paired target access")
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


def load_candidate_registration(
    output_root: Path, candidate_id: str, *, require_gate: bool = False,
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
    algorithm_fingerprint_payload = {
        "schema": "final-unsb-route1-algorithm-fingerprint-v1",
        "candidate_id": candidate_id,
        "card_sha256": card_sha256,
        "implementation_sha256": file_sha256(implementation_path),
        "sources": sources,
        "causal_matrix_sha256": matrix_sha256,
        "reversal_atlas_sha256": atlas_sha256,
        "candidate_training_core_fingerprint": candidate_training_core,
        "common_training_protocol": protocol["common"],
        "manifest_sha256": protocol["manifest"]["sha256"],
    }
    algorithm_fingerprint = object_sha256(algorithm_fingerprint_payload)
    fingerprint_payload = {
        "schema": REGISTRATION_SCHEMA,
        "algorithm_fingerprint": algorithm_fingerprint,
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
        gate=gate,
    )
