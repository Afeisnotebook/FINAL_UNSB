"""Register and freeze the engineering-corrected PC-RSMG candidate."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from research.local_route1.candidates import (
    IMPLEMENTATION_SCHEMA,
    freeze_candidate_derivation,
    register_engineering_replacement,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


PARENT_ID = "G1-02-SAMPLING-VARIANCE"
CANDIDATE_ID = "G1-02B-PLAYER-CONDITIONAL-RSMG"
SPEC = {
    "model": "route1_pcrsmg",
    "method": {"pcrsmg_replicates": 2},
    "gate_callable": "run_pcrsmg_gate",
    "sources": [
        "src/models/sb_model.py",
        "src/models/route1/__init__.py",
        "src/models/route1/pcrsmg.py",
        "src/models/route1_pcrsmg_model.py",
        "research/local_route1/generation1_gates.py",
    ],
}


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"non-identical frozen replacement exists: {destination}")
        return
    shutil.copyfile(source, destination)


def materialize(output_root: Path) -> dict:
    output_root = Path(output_root).resolve()
    authorization = register_engineering_replacement(
        output_root, PARENT_ID, CANDIDATE_ID,
    )
    card_source = (
        ROOT / "research" / "local_route1" / "derivation_cards"
        / f"{CANDIDATE_ID}.json"
    )
    card_destination = output_root / "derive" / "cards" / card_source.name
    _copy_exact(card_source, card_destination)
    implementation = {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": "FROZEN_FOR_GATES",
        "derivation_card_sha256": file_sha256(card_destination),
        "model": SPEC["model"],
        "method": SPEC["method"],
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "gate_hook": {
            "module": "research.local_route1.generation1_gates",
            "callable": SPEC["gate_callable"],
        },
        "source_files": [
            {"path": relative, "sha256": file_sha256(ROOT / relative)}
            for relative in SPEC["sources"]
        ],
    }
    implementation_path = (
        output_root / "derive" / "implementations" / f"{CANDIDATE_ID}.json"
    )
    if implementation_path.is_file():
        existing = json.loads(implementation_path.read_text(encoding="utf-8"))
        if existing != implementation:
            raise RuntimeError("non-identical PC-RSMG implementation already exists")
    else:
        write_json(implementation_path, implementation)
    registration = freeze_candidate_derivation(output_root, CANDIDATE_ID)
    return {
        "schema": "final-unsb-route1-engineering-replacement-materialization-v1",
        "status": "PCRSMG_REPLACEMENT_FROZEN_FOR_GATES",
        "authorization": authorization,
        "candidate": registration.to_dict(),
        "old_trajectory_rankable": False,
        "restart_from_common_e0": True,
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
