"""Materialize the evidence-authorized third Generation-1 mechanism."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from research.local_route1.candidates import (
    IMPLEMENTATION_SCHEMA,
    freeze_candidate_derivation,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json


CANDIDATE_ID = "G1-03-STATE-FEEDBACK-MISSING"
SPEC = {
    "model": "route1_mcrb",
    "method": {
        "mcrb_enable": True,
        "mcrb_m": 4,
        "mcrb_region_patch": 32,
        "mcrb_u_floor": 1e-30,
        "mcrb_teacher_half_life_updates": 150,
        "mcrb_projection_epsilon": 1e-24,
    },
    "gate_callable": "run_mcrb_gate",
    "sources": [
        "src/models/sb_model.py",
        "src/models/dtcov/dtcovmatch.py",
        "src/models/route1/__init__.py",
        "src/models/route1/mcrb.py",
        "src/models/route1_mcrb_model.py",
        "research/local_route1/generation1_gates.py",
    ],
}


def materialize(output_root: Path) -> dict:
    output_root = Path(output_root).resolve()
    source = ROOT / "research" / "local_route1" / "derivation_cards" / f"{CANDIDATE_ID}.json"
    destination = output_root / "derive" / "cards" / source.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError("non-identical MCRB derivation card already exists")
    else:
        shutil.copyfile(source, destination)
    implementation = {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": CANDIDATE_ID,
        "status": "FROZEN_FOR_GATES",
        "derivation_card_sha256": file_sha256(destination),
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
    implementation_path = output_root / "derive" / "implementations" / f"{CANDIDATE_ID}.json"
    if implementation_path.is_file():
        existing = json.loads(implementation_path.read_text(encoding="utf-8"))
        if existing != implementation:
            raise RuntimeError("non-identical MCRB implementation already exists")
    else:
        write_json(implementation_path, implementation)
    registration = freeze_candidate_derivation(output_root, CANDIDATE_ID)
    return {
        "schema": "final-unsb-route1-generation1-third-materialization-v1",
        "status": "THIRD_CANDIDATE_FROZEN_FOR_GATES",
        "candidate": registration.to_dict(),
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
