"""Freeze the explicitly authorized two-candidate route-1 frontier expansion."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from research.local_route1.candidates import (
    IMPLEMENTATION_SCHEMA,
    freeze_candidate_derivation,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


DECISION = "decisions/DEC-20260831-ROUTE1-FRONTIER-EXPANSION.md"
SPECS = {
    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING": {
        "parent_candidate_id": "G1-02B-PLAYER-CONDITIONAL-RSMG",
        "parent_evidence": "evidence/remote_route1_offload/PCRSMG_E200_TERMINAL_AND_CROSS_ADJUDICATION_20260830.json",
        "model": "route1_pcnr",
        "method": {"pcnr_enable": True},
        "gate_callable": "run_pcnr_gate",
        "sources": [
            "src/models/sb_model.py",
            "src/models/route1/__init__.py",
            "src/models/route1/pcrsmg.py",
            "src/models/route1/pcnr.py",
            "src/models/route1_pcnr_model.py",
            "research/local_route1/generation1_gates.py",
        ],
    },
    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER": {
        "parent_candidate_id": "G1-03-STATE-FEEDBACK-MISSING",
        "parent_evidence": "evidence/remote_route1_offload/MCRB_E200_TERMINAL_20260831.json",
        "model": "route1_ammcrb",
        "method": {
            "ammcrb_enable": True,
            "mcrb_m": 4,
            "mcrb_region_patch": 32,
            "mcrb_u_floor": 1e-30,
            "mcrb_teacher_half_life_updates": 150,
            "ammcrb_projection_epsilon": 1e-24,
        },
        "gate_callable": "run_ammcrb_gate",
        "sources": [
            "src/models/sb_model.py",
            "src/models/dtcov/dtcovmatch.py",
            "src/models/route1/__init__.py",
            "src/models/route1/mcrb.py",
            "src/models/route1/ammcrb.py",
            "src/models/route1_ammcrb_model.py",
            "research/local_route1/generation1_gates.py",
        ],
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"non-identical frontier artifact exists: {destination}")
        return
    shutil.copyfile(source, destination)


def _authorize_slots(output_root: Path) -> dict[str, Any]:
    decision = ROOT / DECISION
    if not decision.is_file():
        raise RuntimeError("frontier expansion decision is missing")
    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
        raise RuntimeError("hypothesis ledger schema mismatch")
    records = ledger.setdefault("records", [])
    authorization = {
        "decision": DECISION,
        "decision_sha256": file_sha256(decision),
        "maximum_frontier_candidates": 2,
        "selection_seeds": [2026],
        "deferred_seeds": [2027, 2028],
        "requires_true_e200": True,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    existing_policy = ledger.get("user_authorized_frontier_expansion")
    if existing_policy is not None and existing_policy != authorization:
        raise RuntimeError("frontier expansion authorization changed")
    ledger["user_authorized_frontier_expansion"] = authorization
    for candidate_id, spec in SPECS.items():
        card = _read_json(
            ROOT / "research" / "local_route1" / "derivation_cards" / f"{candidate_id}.json"
        )
        parent_evidence = ROOT / spec["parent_evidence"]
        if not parent_evidence.is_file():
            raise RuntimeError(f"frontier parent evidence is missing: {parent_evidence}")
        expected = {
            "candidate_id": candidate_id,
            "generation": 3,
            "parent_candidate_id": spec["parent_candidate_id"],
            "parent_evidence": card["parent_evidence"],
            "construction_route": "user_authorized_near_miss_frontier_expansion",
            "status": "DERIVATION_REQUIRED",
            "revision_count": 0,
            "frontier_authorization": {
                "decision_sha256": authorization["decision_sha256"],
                "parent_evidence_path": spec["parent_evidence"],
                "parent_evidence_sha256": file_sha256(parent_evidence),
                "consumes_legacy_generation_slot": False,
                "consumes_repeated_seed_budget": False,
                "restart_from_common_e0": True,
            },
            "experiments": [],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        matches = [
            row for row in records
            if isinstance(row, dict) and row.get("candidate_id") == candidate_id
        ]
        if not matches:
            records.append(expected)
        elif len(matches) != 1:
            raise RuntimeError(f"frontier candidate ledger id is not unique: {candidate_id}")
        else:
            frozen = matches[0]
            for key, value in expected.items():
                if key in ("status", "experiments"):
                    continue
                if frozen.get(key) != value:
                    raise RuntimeError(f"frontier candidate ledger binding changed: {candidate_id}:{key}")
    write_json(ledger_path, ledger)
    return authorization


def materialize(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    authorization = _authorize_slots(output_root)
    registrations = []
    for candidate_id, spec in SPECS.items():
        card_source = (
            ROOT / "research" / "local_route1" / "derivation_cards" / f"{candidate_id}.json"
        )
        card_destination = output_root / "derive" / "cards" / card_source.name
        _copy_exact(card_source, card_destination)
        implementation = {
            "schema": IMPLEMENTATION_SCHEMA,
            "candidate_id": candidate_id,
            "status": "FROZEN_FOR_GATES",
            "derivation_card_sha256": file_sha256(card_destination),
            "model": spec["model"],
            "method": spec["method"],
            "training_target_access": "unpaired_only",
            "paired_controller_access": False,
            "state_contract": {
                "full_state_restorable": True,
                "zero_intervention_identity_test": True,
                "parent_state_isolation_test": True,
            },
            "gate_hook": {
                "module": "research.local_route1.generation1_gates",
                "callable": spec["gate_callable"],
            },
            "source_files": [
                {"path": relative, "sha256": file_sha256(ROOT / relative)}
                for relative in spec["sources"]
            ],
        }
        implementation_path = (
            output_root / "derive" / "implementations" / f"{candidate_id}.json"
        )
        if implementation_path.is_file():
            if _read_json(implementation_path) != implementation:
                raise RuntimeError(f"non-identical frontier implementation exists: {candidate_id}")
        else:
            write_json(implementation_path, implementation)
        registrations.append(
            freeze_candidate_derivation(output_root, candidate_id).to_dict()
        )
    return {
        "schema": "final-unsb-route1-frontier-expansion-freeze-v1",
        "status": "TWO_FRONTIER_CANDIDATES_FROZEN_FOR_GATES",
        "authorization": authorization,
        "candidates": registrations,
        "long_run_policy": "host_matched_seed2026_small25_batch1_true_e200",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
