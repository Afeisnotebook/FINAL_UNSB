"""Route one source-bound parent ablation in the evidence-qualified second wave."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research.local_route1.candidates import freeze_candidate_derivation
from research.local_route1.frontier_adjudication import FRONTIER_IDS, SCHEMA as FRONTIER_SCHEMA
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json
from research.local_route1.winner_ablations import (
    WINNER_FAMILIES,
    _card,
    _implementation,
)


PCNR_ID, AMMCRB_ID = FRONTIER_IDS
ROLE_BY_PARENT = {
    PCNR_ID: "observable_only",
    AMMCRB_ID: "proposal_only",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def select_second_wave_parent_ablation(
    adjudication: dict[str, Any], advancement: dict[str, Any],
) -> dict[str, Any]:
    if adjudication.get("schema") != FRONTIER_SCHEMA:
        raise RuntimeError("second-wave ablation requires canonical frontier adjudication")
    if advancement.get("schema") != (
        "final-unsb-route1-frontier-advancement-classification-v1"
    ):
        raise RuntimeError("second-wave ablation requires canonical advancement classification")
    strict = list(adjudication.get("strict_gate_pass_candidate_ids", []))
    classified_strict = list(advancement.get("strict_candidate_ids", []))
    if set(strict) != set(classified_strict):
        raise RuntimeError("strict parent identities differ between terminal decisions")
    near = list(advancement.get("near_boundary_pending_target_blind_audit_ids", []))
    if near:
        return {
            "eligible": False,
            "reason": "SECOND_WAVE_SLOT_RESERVED_FOR_TARGET_BLIND_NEAR_BOUNDARY_AUDIT",
            "near_boundary_candidate_ids": near,
            "parent_candidate_id": None,
            "ablation_role": None,
            "ablation_candidate_id": None,
        }
    if not strict:
        return {
            "eligible": False,
            "reason": "NO_STRICT_PARENT_AND_NO_NEAR_BOUNDARY_REVISION_ROUTE",
            "near_boundary_candidate_ids": [],
            "parent_candidate_id": None,
            "ablation_role": None,
            "ablation_candidate_id": None,
        }
    recommended = adjudication.get("recommended_4090_replay_candidate_id")
    alternatives = [candidate_id for candidate_id in strict if candidate_id != recommended]
    parent_id = alternatives[0] if alternatives else strict[0]
    if parent_id not in ROLE_BY_PARENT or parent_id not in WINNER_FAMILIES:
        raise RuntimeError("strict frontier parent lacks a fixed second-wave ablation")
    role = ROLE_BY_PARENT[parent_id]
    candidate_id = WINNER_FAMILIES[parent_id]["ids"][role]
    return {
        "eligible": True,
        "reason": (
            "UNSELECTED_STRICT_PARENT_SOURCE_ABLATION"
            if alternatives else "SOLE_STRICT_PARENT_SOURCE_ABLATION"
        ),
        "near_boundary_candidate_ids": [],
        "parent_candidate_id": parent_id,
        "ablation_role": role,
        "ablation_candidate_id": candidate_id,
        "recommended_4090_replay_candidate_id": recommended,
    }


def _write_exact(path: Path, payload: dict[str, Any]) -> None:
    if path.is_file():
        if _read_json(path) != payload:
            raise RuntimeError(f"second-wave frozen artifact changed: {path}")
        return
    write_json(path, payload)


def materialize_second_wave_parent_ablation(
    output_root: Path, *, adjudication_path: Path, advancement_path: Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    adjudication_path = Path(adjudication_path).resolve()
    advancement_path = Path(advancement_path).resolve()
    adjudication = _read_json(adjudication_path)
    advancement = _read_json(advancement_path)
    route = select_second_wave_parent_ablation(adjudication, advancement)
    result_path = output_root / "operations" / "SECOND_WAVE_PARENT_ABLATION_FREEZE.json"
    common = {
        "schema": "final-unsb-route1-second-wave-parent-ablation-freeze-v1",
        "route": route,
        "frontier_adjudication_sha256": file_sha256(adjudication_path),
        "frontier_advancement_sha256": file_sha256(advancement_path),
        "maximum_parent_ablations": 1,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if route["eligible"] is not True:
        result = {**common, "status": "SECOND_WAVE_PARENT_ABLATION_INAPPLICABLE"}
        write_json(result_path, result)
        return result

    parent_id = str(route["parent_candidate_id"])
    role = str(route["ablation_role"])
    candidate_id = str(route["ablation_candidate_id"])
    family_record = WINNER_FAMILIES[parent_id]
    family = str(family_record["family"])
    ids = dict(family_record["ids"])
    parent_receipt_path = (
        output_root / "operations" / "terminal_receipts" / f"{parent_id}.json"
    )
    parent_card_path = output_root / "derive" / "cards" / f"{parent_id}.json"
    if not parent_receipt_path.is_file() or not parent_card_path.is_file():
        raise RuntimeError("second-wave parent receipt/card is missing")
    parent_receipt = _read_json(parent_receipt_path)
    ranking = next(
        (row for row in adjudication.get("ranking", []) if row.get("candidate_id") == parent_id),
        None,
    )
    if not isinstance(ranking, dict) or (
        parent_receipt.get("candidate_id") != parent_id
        or parent_receipt.get("algorithm_fingerprint") != ranking.get("algorithm_fingerprint")
        or file_sha256(parent_receipt_path) != ranking.get("receipt_sha256")
        or parent_receipt.get("derivation_card_sha256") != file_sha256(parent_card_path)
    ):
        raise RuntimeError("second-wave parent terminal identity changed")

    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = output_root / "derive" / "implementations" / f"{candidate_id}.json"
    card = _card(
        parent=_read_json(parent_card_path),
        parent_id=parent_id,
        parent_receipt_sha256=file_sha256(parent_receipt_path),
        candidate_id=candidate_id,
        family=family,
        role=role,
        sibling_ids=ids,
    )
    _write_exact(card_path, card)
    implementation = _implementation(candidate_id, family, role, card_path)
    _write_exact(implementation_path, implementation)

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    matches = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    expected = {
        "candidate_id": candidate_id,
        "generation": 0,
        "parent_candidate_id": parent_id,
        "parent_evidence": card.get("parent_evidence"),
        "construction_route": "evidence_qualified_second_wave_parent_ablation",
        "ablation_role": role,
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if not matches:
        ledger["records"].append(expected)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("second-wave parent ablation ledger identity is not unique")
    elif matches[0].get("status") != "FROZEN_FOR_GATES" and matches[0] != expected:
        raise RuntimeError("second-wave parent ablation ledger slot changed")
    registration = freeze_candidate_derivation(output_root, candidate_id)
    result = {
        **common,
        "status": "SECOND_WAVE_PARENT_ABLATION_FROZEN_FOR_GATE",
        "registration": registration.to_dict(),
    }
    write_json(result_path, result)
    return result

