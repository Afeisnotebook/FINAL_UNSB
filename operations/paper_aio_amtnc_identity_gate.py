"""Prove that frozen one-replica AM-TNC is byte-identical to plain UNSB.

This is an engineering-only gate.  It reads no paired target or metric and
does not authorize a scientific claim.  The caller must run it with the
scientific training checkout first on ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from research.local_route1.runtime import full_state_hash, write_json
from research.paper_aio.gates import transition_core
from research.paper_aio.protocol import (
    ROOT,
    LaneSpec,
    file_sha256,
    lane_spec,
    load_protocol,
    protocol_fingerprint,
)
from research.paper_aio.runtime import create_e0, optimizer_step, prepare_lane


def run_gate(
    *, output: Path, manifest: Path, train_view: Path, gpu: int, updates: int,
) -> dict:
    if updates < 1:
        raise ValueError("updates must be positive")
    plain = lane_spec("plain")
    disabled = LaneSpec(
        id="amtnc_zero",
        backend="internal",
        family="unsb",
        model="route1_amtnc",
        role="engineering-only one-replica identity witness",
        method={"amtnc_replicates": 1},
    )
    e0 = create_e0(
        output_root=output,
        train_view=train_view,
        manifest_path=manifest,
        spec=plain,
        gpu=gpu,
    )
    hashes: dict[str, str] = {}
    target_updates = int(load_protocol()["training"]["target_updates"])
    for label, spec in (("plain", plain), ("amtnc_zero", disabled)):
        model, primary, secondary, _ = prepare_lane(
            output_root=output,
            train_view=train_view,
            manifest_path=manifest,
            spec=spec,
            gpu=gpu,
            e0=e0,
        )
        for step in range(updates):
            model.set_train_epoch(1)
            model.set_search_step(step, target_updates)
            optimizer_step(model, spec, primary, secondary)
        hashes[label] = full_state_hash(
            transition_core(model=model, primary=primary, secondary=secondary)
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    receipt = {
        "schema": "final-unsb-paper-amtnc-zero-intervention-v1",
        "status": "PASS" if hashes["plain"] == hashes["amtnc_zero"] else "FAIL",
        "updates": updates,
        "plain_transition_sha256": hashes["plain"],
        "amtnc_one_replica_transition_sha256": hashes["amtnc_zero"],
        "operator_sha256": file_sha256(ROOT / "src/models/route1/amtnc.py"),
        "model_registry_sha256": file_sha256(
            ROOT / "src/models/route1_amtnc_model.py"
        ),
        "protocol_fingerprint": protocol_fingerprint(manifest),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(output / "gates" / "ZERO_INTERVENTION_AMTNC.json", receipt)
    if receipt["status"] != "PASS":
        raise RuntimeError("AM-TNC one-replica identity differs from plain UNSB")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--updates", type=int, default=10)
    args = parser.parse_args()
    print(
        run_gate(
            output=args.output.resolve(),
            manifest=args.manifest.resolve(),
            train_view=args.train_view.resolve(),
            gpu=args.gpu,
            updates=args.updates,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
