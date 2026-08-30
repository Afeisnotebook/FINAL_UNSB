"""Materialize and freeze the two evidence-derived Generation-1 candidates."""

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


SPECS = {
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED": {
        "model": "route1_bvcp",
        "method": {"bvcp_enable": True, "bvcp_root_epsilon": 1e-12},
        "gate_callable": "run_bvcp_gate",
        "sources": [
            "src/models/sb_model.py",
            "src/models/route1/__init__.py",
            "src/models/route1/bvcp.py",
            "src/models/route1_bvcp_model.py",
            "research/local_route1/generation1_gates.py",
        ],
    },
    "G1-02-SAMPLING-VARIANCE": {
        "model": "route1_rsmg",
        "method": {"rsmg_replicates": 2},
        "gate_callable": "run_rsmg_gate",
        "sources": [
            "src/models/sb_model.py",
            "src/models/route1/__init__.py",
            "src/models/route1/rsmg.py",
            "src/models/route1_rsmg_model.py",
            "research/local_route1/generation1_gates.py",
        ],
    },
}


def _copy_exact(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != source.read_bytes():
            raise RuntimeError(f"non-identical frozen candidate file already exists: {destination}")
        return
    shutil.copyfile(source, destination)


def materialize(output_root: Path) -> dict:
    output_root = Path(output_root).resolve()
    results = []
    for candidate_id, spec in SPECS.items():
        card_source = ROOT / "research" / "local_route1" / "derivation_cards" / f"{candidate_id}.json"
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
        implementation_path = output_root / "derive" / "implementations" / f"{candidate_id}.json"
        if implementation_path.is_file():
            existing = json.loads(implementation_path.read_text(encoding="utf-8"))
            if existing != implementation:
                raise RuntimeError(f"non-identical implementation already exists: {candidate_id}")
        else:
            write_json(implementation_path, implementation)
        results.append(freeze_candidate_derivation(output_root, candidate_id))
    return {
        "schema": "final-unsb-route1-generation1-materialization-v1",
        "status": "TWO_CANDIDATES_FROZEN_FOR_GATES",
        "candidates": results,
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

