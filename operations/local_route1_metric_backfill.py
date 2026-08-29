"""Deterministically recover a missing registered milestone evaluation.

This tool deliberately imports the scientific code from an immutable executor
worktree.  It never writes a checkpoint.  The evaluation is run twice and is
accepted only when the complete JSON payload is identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--executor-repo", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--lane", choices=("plain", "hj", "hnek", "dt"), required=True)
    value.add_argument("--epoch", type=int, required=True)
    value.add_argument("--gpu", type=int, default=0)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    executor = args.executor_repo.resolve()
    if not executor.is_dir():
        raise RuntimeError(f"executor repository missing: {executor}")
    sys.path.insert(0, str(executor))

    # Imported only after the immutable executor is first on sys.path.
    from research.local_route1.anchors import prepare_probe  # noqa: PLC0415
    from research.local_route1.evaluate import evaluate_model  # noqa: PLC0415
    from research.local_route1.protocol import (  # noqa: PLC0415
        load_protocol,
        probe_spec,
    )
    from research.local_route1.runtime import (  # noqa: PLC0415
        capture_rng,
        full_state_hash,
        load_full_state,
        model_state,
    )

    protocol = load_protocol()
    registered = {int(item) for item in protocol["local_view"]["trajectory_epochs"]}
    if args.epoch not in registered:
        raise RuntimeError(f"epoch {args.epoch} is not a registered milestone")
    include_lpips = args.epoch in {
        int(item) for item in protocol["local_view"]["lpips_epochs"]
    }

    run_root = args.run_root.resolve()
    lane_root = run_root / "anchors" / args.lane
    checkpoint = lane_root / "milestones" / f"e{args.epoch:03d}.pt"
    checkpoint_sidecar = Path(str(checkpoint) + ".json")
    metric_path = lane_root / "metrics" / f"e{args.epoch:03d}.json"
    if not checkpoint.is_file() or not checkpoint_sidecar.is_file():
        raise RuntimeError(f"registered checkpoint is incomplete: {checkpoint}")
    sidecar = json.loads(checkpoint_sidecar.read_text(encoding="utf-8"))
    if int(sidecar["physical_epoch_completed"]) != args.epoch:
        raise RuntimeError("checkpoint sidecar epoch mismatch")
    actual_checkpoint_hash = file_sha256(checkpoint)
    if actual_checkpoint_hash != sidecar["full_state_sha256"]:
        raise RuntimeError("checkpoint file hash mismatch")
    if sidecar["metadata"].get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 lock violated in source checkpoint")

    spec = probe_spec(args.lane, protocol)
    model, primary, secondary, rows = prepare_probe(
        spec=spec,
        output_root=run_root,
        train_view=args.train_view.resolve(),
        manifest_path=args.manifest.resolve(),
        gpu=args.gpu,
    )
    restored = load_full_state(
        checkpoint,
        model=model,
        spec=spec,
        primary=primary,
        secondary=secondary,
        expected_metadata={
            "project_id": sidecar["metadata"]["project_id"],
            "probe_id": args.lane,
            "seed": int(protocol["seed"]),
            "manifest_sha256": sidecar["metadata"]["manifest_sha256"],
            "protocol_fingerprint": sidecar["metadata"]["protocol_fingerprint"],
            "git_commit": sidecar["metadata"]["git_commit"],
        },
    )
    before = full_state_hash({"model": model_state(model), "rng": capture_rng()})

    evaluations = []
    for _ in range(2):
        result = evaluate_model(
            model,
            rows=rows,
            data_root=args.data_root.resolve(),
            protocol_hash=restored["metadata"]["protocol_fingerprint"],
            include_lpips=include_lpips,
        )
        result.update(
            {
                "probe_id": args.lane,
                "epoch": args.epoch,
                "updates": args.epoch * 150,
                "data_epoch": args.epoch,
            }
        )
        evaluations.append(result)
    after = full_state_hash({"model": model_state(model), "rng": capture_rng()})
    if before != after:
        raise RuntimeError("milestone evaluation changed model or RNG state")
    first_hash, second_hash = map(object_sha256, evaluations)
    if first_hash != second_hash:
        raise RuntimeError("repeated milestone evaluation was not exactly identical")

    if metric_path.is_file():
        existing = json.loads(metric_path.read_text(encoding="utf-8"))
        if object_sha256(existing) != first_hash:
            raise RuntimeError(f"existing metric differs; refusing overwrite: {metric_path}")
        disposition = "EXISTING_IDENTICAL"
    else:
        atomic_json(metric_path, evaluations[0])
        disposition = "RECOVERED_MISSING_METRIC"

    evidence = {
        "schema": "final-unsb-route1-milestone-recovery-v1",
        "status": "PASS",
        "disposition": disposition,
        "lane": args.lane,
        "epoch": args.epoch,
        "updates": args.epoch * 150,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": actual_checkpoint_hash,
        "source_scientific_state_sha256": sidecar["scientific_state_sha256"],
        "metric": str(metric_path),
        "metric_payload_sha256": first_hash,
        "repeat_payload_sha256": second_hash,
        "parent_state_hash_before": before,
        "parent_state_hash_after": after,
        "lpips_requested": include_lpips,
        "confirmation20_opened": False,
        "training_git_commit": restored["metadata"]["git_commit"],
        "training_protocol_fingerprint": restored["metadata"]["protocol_fingerprint"],
    }
    evidence_path = (
        run_root
        / "operations"
        / "milestone_recovery"
        / f"{args.lane}_e{args.epoch:03d}.json"
    )
    atomic_json(evidence_path, evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
