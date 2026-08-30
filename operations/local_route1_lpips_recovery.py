"""Recover missing plain LPIPS fields without changing scientific trajectories.

Some host-matched plain milestones were evaluated while the optional LPIPS
runtime was unavailable.  PSNR/SSIM and the 420 CRN outputs are valid, but a
``None`` LPIPS value would make every later candidate fail the registered late
guardrail mechanically.  This recovery loads the immutable training checkpoint,
evaluates it twice with LPIPS available, requires every non-LPIPS field to be
exactly identical to the frozen metric, proves parent state isolation, preserves
the original metric outside Git, and atomically installs the completed payload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-route1-lpips-recovery-v1"
LPIPS_EPOCHS = (100, 125, 150, 175, 200)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def object_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def without_lpips(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the exact scientific/evaluation identity excluding LPIPS values."""
    result = json.loads(json.dumps(payload, ensure_ascii=False))
    result.pop("macro_lpips", None)
    result.pop("lpips_available", None)
    for row in result.get("domains", {}).values():
        if isinstance(row, dict):
            row.pop("lpips", None)
    for row in result.get("images", []):
        if isinstance(row, dict):
            row.pop("lpips", None)
    return result


def validate_incomplete_lpips(metric: dict[str, Any], *, lane: str, epoch: int) -> None:
    expected = {
        "schema": "local-route1-discovery70-crn-single-rollout-v1",
        "split": "discovery",
        "count_per_domain": 70,
        "replicates": 1,
        "probe_id": lane,
        "epoch": int(epoch),
        "updates": int(epoch) * 150,
        "data_epoch": int(epoch),
        "lpips_requested": True,
        "lpips_available": False,
        "macro_lpips": None,
        "confirmation20_opened": False,
    }
    for key, value in expected.items():
        if metric.get(key) != value:
            raise RuntimeError(f"incomplete LPIPS metric identity mismatch: {key}")
    images = metric.get("images", [])
    if len(images) != 420 or any(row.get("lpips") is not None for row in images):
        raise RuntimeError("incomplete LPIPS image payload is not the expected 420 nulls")
    domains = metric.get("domains", {})
    if len(domains) != 6 or any(row.get("lpips") is not None for row in domains.values()):
        raise RuntimeError("incomplete LPIPS domain payload is not the expected six nulls")


def validate_complete_lpips(metric: dict[str, Any]) -> None:
    if (
        metric.get("lpips_requested") is not True
        or metric.get("lpips_available") is not True
        or metric.get("macro_lpips") is None
    ):
        raise RuntimeError("recomputed LPIPS payload is incomplete")
    images = metric.get("images", [])
    if len(images) != 420 or any(row.get("lpips") is None for row in images):
        raise RuntimeError("recomputed LPIPS image payload is incomplete")
    domains = metric.get("domains", {})
    if len(domains) != 6 or any(row.get("lpips") is None for row in domains.values()):
        raise RuntimeError("recomputed LPIPS domain payload is incomplete")


def recover(
    *, executor_repo: Path, run_root: Path, train_view: Path, data_root: Path,
    manifest: Path, lane: str, epoch: int, gpu: int,
) -> dict[str, Any]:
    if int(epoch) not in LPIPS_EPOCHS:
        raise RuntimeError(f"e{epoch} is not a registered LPIPS milestone")
    executor_repo = Path(executor_repo).resolve()
    run_root = Path(run_root).resolve()
    sys.path.insert(0, str(executor_repo))

    from research.local_route1.anchors import prepare_probe  # noqa: PLC0415
    from research.local_route1.evaluate import evaluate_model  # noqa: PLC0415
    from research.local_route1.protocol import load_protocol, probe_spec  # noqa: PLC0415
    from research.local_route1.runtime import (  # noqa: PLC0415
        capture_rng,
        full_state_hash,
        load_full_state,
        model_state,
    )

    protocol = load_protocol()
    if int(epoch) not in {
        int(value) for value in protocol["local_view"]["lpips_epochs"]
    }:
        raise RuntimeError("epoch is not LPIPS-enabled by the immutable protocol")
    lane_root = run_root / "anchors" / lane
    checkpoint = lane_root / "milestones" / f"e{epoch:03d}.pt"
    sidecar_path = Path(str(checkpoint) + ".json")
    metric_path = lane_root / "metrics" / f"e{epoch:03d}.json"
    for path in (checkpoint, sidecar_path, metric_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar = read_json(sidecar_path)
    if int(sidecar.get("physical_epoch_completed", -1)) != int(epoch):
        raise RuntimeError("checkpoint sidecar epoch mismatch")
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != sidecar.get("full_state_sha256"):
        raise RuntimeError("checkpoint file hash mismatch")
    if sidecar.get("metadata", {}).get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 lock violated in source checkpoint")

    existing = read_json(metric_path)
    if existing.get("lpips_available") is True:
        existing_complete = True
        validate_complete_lpips(existing)
    else:
        existing_complete = False
        validate_incomplete_lpips(existing, lane=lane, epoch=epoch)
    original_hash = file_sha256(metric_path)

    spec = probe_spec(lane, protocol)
    model, primary, secondary, rows = prepare_probe(
        spec=spec, output_root=run_root, train_view=Path(train_view).resolve(),
        manifest_path=Path(manifest).resolve(), gpu=int(gpu),
    )
    restored = load_full_state(
        checkpoint, model=model, spec=spec, primary=primary, secondary=secondary,
        expected_metadata={
            "project_id": sidecar["metadata"]["project_id"],
            "probe_id": lane,
            "seed": int(protocol["seed"]),
            "manifest_sha256": sidecar["metadata"]["manifest_sha256"],
            "protocol_fingerprint": sidecar["metadata"]["protocol_fingerprint"],
            "git_commit": sidecar["metadata"]["git_commit"],
        },
    )
    before = full_state_hash({"model": model_state(model), "rng": capture_rng()})
    evaluations = []
    for _ in range(2):
        value = evaluate_model(
            model, rows=rows, data_root=Path(data_root).resolve(),
            protocol_hash=restored["metadata"]["protocol_fingerprint"],
            include_lpips=True,
        )
        value.update({
            "probe_id": lane, "epoch": int(epoch), "updates": int(epoch) * 150,
            "data_epoch": int(epoch),
        })
        validate_complete_lpips(value)
        evaluations.append(value)
    after = full_state_hash({"model": model_state(model), "rng": capture_rng()})
    if before != after:
        raise RuntimeError("LPIPS recovery changed model or RNG state")
    first_hash, second_hash = (object_sha256(value) for value in evaluations)
    if first_hash != second_hash:
        raise RuntimeError("repeated LPIPS recovery evaluation was not exactly identical")
    if without_lpips(existing) != without_lpips(evaluations[0]):
        raise RuntimeError("LPIPS recovery changed a non-LPIPS metric field")
    if existing_complete and object_sha256(existing) != first_hash:
        raise RuntimeError("complete existing LPIPS metric differs from deterministic replay")

    recovery_root = run_root / "operations" / "lpips_recovery"
    backup = recovery_root / "original_metrics" / lane / f"e{epoch:03d}.json"
    if not backup.is_file():
        backup.parent.mkdir(parents=True, exist_ok=True)
        temporary_backup = backup.with_suffix(backup.suffix + ".tmp")
        shutil.copyfile(metric_path, temporary_backup)
        os.replace(temporary_backup, backup)
    elif file_sha256(backup) != original_hash and not existing_complete:
        raise RuntimeError("original LPIPS recovery backup identity changed")
    if not existing_complete:
        atomic_json(metric_path, evaluations[0])
    completed_file_hash = file_sha256(metric_path)
    if object_sha256(read_json(metric_path)) != first_hash:
        raise RuntimeError("atomically installed LPIPS metric differs from replay")
    result = {
        "schema": SCHEMA,
        "status": "EXISTING_COMPLETE_IDENTICAL" if existing_complete else "RECOVERED_MISSING_LPIPS",
        "lane": lane,
        "data_epoch": int(epoch),
        "updates": int(epoch) * 150,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_hash,
        "source_scientific_state_sha256": sidecar["scientific_state_sha256"],
        "original_metric_backup": str(backup),
        "original_metric_file_sha256": original_hash,
        "recovered_metric": str(metric_path),
        "recovered_metric_file_sha256": completed_file_hash,
        "repeated_payload_sha256": [first_hash, second_hash],
        "all_non_lpips_fields_exactly_unchanged": True,
        "parent_state_hash_before": before,
        "parent_state_hash_after": after,
        "training_git_commit": restored["metadata"]["git_commit"],
        "training_protocol_fingerprint": restored["metadata"]["protocol_fingerprint"],
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    evidence_path = recovery_root / f"{lane}_e{epoch:03d}.json"
    atomic_json(evidence_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--executor-repo", type=Path, required=True)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--lane", default="plain")
    value.add_argument("--epoch", type=int, required=True)
    value.add_argument("--gpu", type=int, default=0)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = recover(
        executor_repo=args.executor_repo, run_root=args.run_root,
        train_view=args.train_view, data_root=args.data_root, manifest=args.manifest,
        lane=args.lane, epoch=args.epoch, gpu=args.gpu,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
