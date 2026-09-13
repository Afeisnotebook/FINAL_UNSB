"""Capture one transient full-state checkpoint without touching training.

The paper runner overwrites ``full_state_latest.pt`` at every completed data
epoch.  This metric-blind control-plane watcher copies a requested epoch after
its sidecar is atomically published, verifies the source remained unchanged
during the copy, and publishes the receipt last.  It never loads model state,
evaluates a checkpoint, or signals a training process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any


SCHEMA = "final-unsb-paper-transient-checkpoint-capture-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "FROZEN",
        "source_checkpoint": str(args.source_checkpoint.resolve()),
        "source_sidecar": str(Path(str(args.source_checkpoint.resolve()) + ".json")),
        "destination_checkpoint": str(args.destination_checkpoint.resolve()),
        "lane_id": args.lane_id,
        "required_epoch": int(args.required_epoch),
        "required_step": int(args.required_step),
        "required_git_commit": args.required_git_commit,
        "required_protocol_fingerprint": args.required_protocol_fingerprint,
        "signals_training": False,
        "loads_checkpoint": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _freeze(path: Path, proposed: dict[str, Any]) -> None:
    if path.is_file() and _read(path) != proposed:
        raise RuntimeError("refusing to replace a different capture contract")
    if not path.is_file():
        _atomic_json(path, proposed)


def capture_once(args: argparse.Namespace) -> dict[str, Any] | None:
    source = args.source_checkpoint.resolve()
    sidecar_path = Path(str(source) + ".json")
    if not source.is_file() or not sidecar_path.is_file():
        return None
    sidecar_before_bytes = sidecar_path.read_bytes()
    sidecar = json.loads(sidecar_before_bytes.decode("utf-8"))
    epoch = int(sidecar.get("physical_epoch_completed", -1))
    if epoch < int(args.required_epoch):
        return None
    if epoch > int(args.required_epoch):
        raise RuntimeError("requested transient checkpoint was overwritten")
    metadata = sidecar.get("metadata") or {}
    if (
        sidecar.get("schema") != "final-unsb-paper-aio-full-state-v1"
        or sidecar.get("lane_id") != args.lane_id
        or int(sidecar.get("step", -1)) != int(args.required_step)
        or metadata.get("git_commit") != args.required_git_commit
        or metadata.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
        or metadata.get("paired_controller_access") is not False
        or metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("transient checkpoint sidecar identity mismatch")
    expected = str(sidecar.get("full_state_sha256"))
    if _sha256(source) != expected:
        raise RuntimeError("source checkpoint does not match published sidecar")

    destination = args.destination_checkpoint.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or Path(str(destination) + ".json").exists():
        raise RuntimeError("fresh capture destination required")
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    if _sha256(temporary) != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("captured checkpoint hash differs from source")
    if (
        sidecar_path.read_bytes() != sidecar_before_bytes
        or _sha256(source) != expected
    ):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("source changed during checkpoint capture")
    os.replace(temporary, destination)
    destination_sidecar = Path(str(destination) + ".json")
    sidecar_temporary = destination_sidecar.with_suffix(
        destination_sidecar.suffix + f".{os.getpid()}.tmp"
    )
    sidecar_temporary.write_bytes(sidecar_before_bytes)
    os.replace(sidecar_temporary, destination_sidecar)
    return {
        "schema": SCHEMA,
        "status": "COMPLETE_HASH_BOUND_CAPTURE",
        "lane_id": args.lane_id,
        "epoch": int(args.required_epoch),
        "step": int(args.required_step),
        "source_checkpoint": str(source),
        "destination_checkpoint": str(destination),
        "checkpoint_sha256": expected,
        "sidecar_sha256": _sha256(destination_sidecar),
        "scientific_state_sha256": sidecar.get("scientific_state_sha256"),
        "source_unchanged_during_capture": True,
        "signals_training": False,
        "loads_checkpoint": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = _contract(args)
    contract_path = output / "CHECKPOINT_CAPTURE_CONTRACT.json"
    state_path = output / "CHECKPOINT_CAPTURE_STATE.json"
    receipt_path = output / "CHECKPOINT_CAPTURE_RECEIPT.json"
    _freeze(contract_path, contract)
    started = time.monotonic()
    while time.monotonic() - started < float(args.timeout_hours) * 3600:
        result = capture_once(args)
        if result is not None:
            _atomic_json(receipt_path, result)
            _atomic_json(state_path, result)
            return 0
        _atomic_json(state_path, {
            **contract,
            "status": "WAITING_FOR_REQUIRED_EPOCH",
            "pid": os.getpid(),
        })
        time.sleep(float(args.poll_seconds))
    raise RuntimeError("checkpoint capture timed out")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-checkpoint", type=Path, required=True)
    value.add_argument("--destination-checkpoint", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--lane-id", required=True)
    value.add_argument("--required-epoch", type=int, required=True)
    value.add_argument("--required-step", type=int, required=True)
    value.add_argument("--required-git-commit", required=True)
    value.add_argument("--required-protocol-fingerprint", required=True)
    value.add_argument("--poll-seconds", type=float, default=1.0)
    value.add_argument("--timeout-hours", type=float, default=24.0)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
