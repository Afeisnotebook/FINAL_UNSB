"""Create source-bound DCLGAN milestone receipts after target-host e200.

This control process reads no metric files and copies no model weights.  It
waits for the durable long supervisor, verifies every selected full state, and
publishes immutable hashes for later evaluation import.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from operations import paper_aio_dclgan_adapter as adapter  # noqa: E402
from research.local_route1.runtime import full_state_hash  # noqa: E402


SCHEMA = "final-unsb-paper-dclgan-export-successor-v1"
EXPORT_SCHEMA = "final-unsb-paper-dclgan-checkpoint-export-v1"
EPOCHS = (100, 125, 150, 175, 200)
BLOCKED_PREFIXES = ("BLOCKED", "FAIL", "FATAL")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def repo_identity(repo: Path) -> tuple[str, str]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=repo,
        text=True,
    ).strip()
    return commit, dirty


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if handle.read(1) == b"":
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def export_one(
    *, checkpoint: Path, epoch: int, source_host_label: str,
    required_git_commit: str, required_adapter_fingerprint: str,
    destination: Path,
) -> dict[str, Any]:
    sidecar_path = Path(str(checkpoint) + ".json")
    if not checkpoint.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"DCLGAN e{epoch} checkpoint or sidecar is missing")
    sidecar = read_json(sidecar_path)
    expected_step = int(epoch) * 8553
    if (
        sidecar.get("schema") != adapter.FULL_STATE_SCHEMA
        or sidecar.get("lane_id") != adapter.LANE_ID
        or int(sidecar.get("step", -1)) != expected_step
        or int(sidecar.get("physical_epoch_completed", -1)) != int(epoch)
        or sidecar.get("full_state_sha256") != adapter.file_sha256(checkpoint)
    ):
        raise RuntimeError(f"DCLGAN e{epoch} sidecar identity differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    if (
        payload.get("schema") != adapter.FULL_STATE_SCHEMA
        or payload.get("lane_id") != adapter.LANE_ID
        or int(payload.get("step", -1)) != expected_step
        or full_state_hash(payload) != sidecar.get("scientific_state_sha256")
        or metadata.get("adapter_git_commit") != required_git_commit
        or metadata.get("adapter_fingerprint") != required_adapter_fingerprint
        or metadata.get("paired_controller_access") is not False
        or metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"DCLGAN e{epoch} scientific state differs")
    receipt = {
        "schema": EXPORT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_DCLGAN_CHECKPOINT",
        "lane_id": adapter.LANE_ID,
        "epoch": int(epoch),
        "updates": expected_step,
        "source_host_label": source_host_label,
        "source_checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sidecar["full_state_sha256"],
        "source_sidecar": str(sidecar_path.resolve()),
        "sidecar_sha256": adapter.file_sha256(sidecar_path),
        "scientific_state_sha256": sidecar["scientific_state_sha256"],
        "training_git_commit": required_git_commit,
        "training_protocol_fingerprint": required_adapter_fingerprint,
        "manifest_sha256": metadata.get("manifest_sha256"),
        "upstream_commit": metadata.get("upstream_commit"),
        "performance_values_read": False,
        "checkpoint_copy_performed": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    atomic_json(destination, receipt)
    return receipt


def state_payload(args: argparse.Namespace, *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "lane_id": adapter.LANE_ID,
        "source_host_label": args.source_host_label,
        "required_git_commit": args.required_git_commit,
        "required_adapter_fingerprint": args.required_adapter_fingerprint,
        "performance_values_read": False,
        "checkpoint_copy_performed": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--required-git-commit", required=True)
    parser.add_argument("--required-adapter-fingerprint", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    for name in ("repo", "source_output", "destination"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.poll_seconds < 30 or args.timeout_hours < 24:
        raise RuntimeError("DCLGAN exporter waiting policy is unsafe")
    commit, dirty = repo_identity(args.repo)
    if commit != args.required_git_commit or dirty:
        raise RuntimeError("DCLGAN exporter checkout is not frozen and clean")
    state_path = args.source_output / "operations" / "DCLGAN_EXPORT_STATE.json"
    lock = acquire_lock(args.source_output / "operations" / "DCLGAN_EXPORT.lock")
    started = time.time()
    try:
        while True:
            commit, dirty = repo_identity(args.repo)
            if commit != args.required_git_commit or dirty:
                raise RuntimeError("DCLGAN exporter checkout changed")
            supervisor_path = args.source_output / "gates" / "SUPERVISOR_dclgan.json"
            supervisor = read_json(supervisor_path) if supervisor_path.is_file() else {}
            status = str(supervisor.get("status", ""))
            if status == "COMPLETE_E200":
                break
            if status.startswith(BLOCKED_PREFIXES):
                atomic_json(
                    state_path,
                    state_payload(
                        args,
                        status="BLOCKED_SOURCE_LANE_ENGINEERING_FAILURE",
                        supervisor_status=status,
                    ),
                )
                return 2
            if time.time() - started > args.timeout_hours * 3600:
                atomic_json(
                    state_path,
                    state_payload(args, status="BLOCKED_TIMEOUT_WAITING_E200"),
                )
                return 3
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status="WAITING_FOR_COMPLETE_E200",
                    supervisor_status=status or None,
                    elapsed_seconds=time.time() - started,
                ),
            )
            time.sleep(args.poll_seconds)

        receipts = []
        lane = args.source_output / "lanes" / adapter.LANE_ID / "milestones"
        for epoch in EPOCHS:
            destination = args.destination / adapter.LANE_ID / f"e{epoch:03d}.export.json"
            receipt = export_one(
                checkpoint=lane / f"e{epoch:03d}.pt",
                epoch=epoch,
                source_host_label=args.source_host_label,
                required_git_commit=args.required_git_commit,
                required_adapter_fingerprint=args.required_adapter_fingerprint,
                destination=destination,
            )
            receipts.append({
                "epoch": epoch,
                "receipt": str(destination),
                "receipt_sha256": adapter.file_sha256(destination),
                "checkpoint_sha256": receipt["checkpoint_sha256"],
                "scientific_state_sha256": receipt["scientific_state_sha256"],
            })
        export_set = {
            "schema": "final-unsb-paper-dclgan-source-export-set-v1",
            "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
            "lane_id": adapter.LANE_ID,
            "source_host_label": args.source_host_label,
            "epochs": list(EPOCHS),
            "exports": receipts,
            "performance_values_read": False,
            "checkpoint_copy_performed": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        }
        export_set_path = args.destination / adapter.LANE_ID / "EXPORT_SET.json"
        atomic_json(export_set_path, export_set)
        atomic_json(
            state_path,
            state_payload(
                args,
                status="COMPLETE_SOURCE_BOUND_EXPORT_SET",
                export_set=str(export_set_path),
                export_set_sha256=adapter.file_sha256(export_set_path),
            ),
        )
        return 0
    finally:
        lock.close()


def main() -> int:
    return run(arguments())


if __name__ == "__main__":
    raise SystemExit(main())
