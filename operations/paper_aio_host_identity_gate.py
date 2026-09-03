"""Fail-closed physical GPU identity gate for paper host onboarding.

An SSH endpoint is not a compute resource identity.  This gate uses the NVIDIA
GPU UUID as the primary stable identity and treats hostname/machine-id as
supporting evidence.  It never reads training metrics or credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


REGISTRY_SCHEMA = "final-unsb-paper-host-identity-registry-v1"
RECEIPT_SCHEMA = "final-unsb-paper-host-identity-gate-v1"
SAFE_OUTCOMES = {
    "REGISTERED_HOST_MATCH",
    "NEW_PHYSICAL_GPU_CANDIDATE",
    "DUPLICATE_ENDPOINT_OF_REGISTERED_HOST",
    "LABEL_COLLISION_DIFFERENT_GPU",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_registry(registry: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if registry.get("schema") != REGISTRY_SCHEMA:
        raise RuntimeError("paper host identity registry schema mismatch")
    hosts = registry.get("hosts")
    if not isinstance(hosts, dict) or not hosts:
        raise RuntimeError("paper host identity registry has no hosts")
    seen: dict[str, str] = {}
    normalized: dict[str, dict[str, Any]] = {}
    for label, row in hosts.items():
        if not isinstance(label, str) or not label or not isinstance(row, dict):
            raise RuntimeError("paper host identity registry row is malformed")
        gpu_uuid = str(row.get("gpu_uuid", ""))
        if not gpu_uuid.startswith("GPU-"):
            raise RuntimeError(f"paper host {label} lacks a valid GPU UUID")
        if gpu_uuid in seen:
            raise RuntimeError(
                f"paper host registry duplicates GPU UUID {gpu_uuid}: "
                f"{seen[gpu_uuid]} and {label}"
            )
        seen[gpu_uuid] = label
        normalized[label] = dict(row)
    return normalized


def classify_identity(
    *, requested_label: str, identity: dict[str, Any], registry: dict[str, Any],
) -> dict[str, Any]:
    hosts = validate_registry(registry)
    gpu_uuid = str(identity.get("gpu_uuid", ""))
    if not gpu_uuid.startswith("GPU-"):
        raise RuntimeError("observed host lacks a valid GPU UUID")
    matching_labels = [
        label for label, row in hosts.items() if row["gpu_uuid"] == gpu_uuid
    ]
    registered_row = hosts.get(requested_label)
    if matching_labels:
        registered_label = matching_labels[0]
        outcome = (
            "REGISTERED_HOST_MATCH"
            if registered_label == requested_label
            else "DUPLICATE_ENDPOINT_OF_REGISTERED_HOST"
        )
    elif registered_row is not None:
        registered_label = None
        outcome = "LABEL_COLLISION_DIFFERENT_GPU"
    else:
        registered_label = None
        outcome = "NEW_PHYSICAL_GPU_CANDIDATE"
    if outcome not in SAFE_OUTCOMES:  # pragma: no cover - defensive invariant
        raise RuntimeError(f"unknown host identity outcome: {outcome}")
    return {
        "outcome": outcome,
        "requested_label": requested_label,
        "registered_label": registered_label,
        "gpu_uuid": gpu_uuid,
        "physical_gpu_count_may_increase": outcome == "NEW_PHYSICAL_GPU_CANDIDATE",
        "long_training_launch_allowed_by_identity_gate": outcome in {
            "REGISTERED_HOST_MATCH", "NEW_PHYSICAL_GPU_CANDIDATE",
        },
    }


def _machine_id() -> str | None:
    candidates = (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id"))
    for path in candidates:
        if path.is_file():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return None


def _gpu_row() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError(
            "paper host identity gate requires exactly one visible physical GPU"
        )
    fields = [field.strip() for field in output[0].split(",")]
    if len(fields) != 4:
        raise RuntimeError("unexpected nvidia-smi identity output")
    return {
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "gpu_memory_total_mib": int(fields[2]),
        "driver_version": fields[3],
    }


def collect_identity(*, repo: Path | None, data_root: Path | None) -> dict[str, Any]:
    identity = {
        "hostname": socket.gethostname(),
        "machine_id": _machine_id(),
        "platform": platform.platform(),
        **_gpu_row(),
    }
    if repo is not None:
        repo = repo.resolve()
        identity["repo"] = str(repo)
        identity["repo_git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
        ).strip()
        identity["repo_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True,
        ).strip())
    if data_root is not None:
        data_root = data_root.resolve()
        stat = data_root.stat()
        space = os.statvfs(data_root) if hasattr(os, "statvfs") else None
        identity["data_root"] = str(data_root)
        identity["data_root_device"] = int(stat.st_dev)
        identity["data_root_inode"] = int(stat.st_ino)
        if space is not None:
            identity["data_root_free_bytes"] = int(space.f_bavail * space.f_frsize)
    return identity


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--requested-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--require-new", action="store_true",
        help="return nonzero unless this is a previously unseen physical GPU",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    registry = _read_json(args.registry.resolve())
    identity = collect_identity(repo=args.repo, data_root=args.data_root)
    classification = classify_identity(
        requested_label=str(args.requested_label), identity=identity,
        registry=registry,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "captured_unix_time": time.time(),
        "status": classification["outcome"],
        "identity": identity,
        "classification": classification,
        "registry": str(args.registry.resolve()),
        "credentials_recorded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    if classification["outcome"] in {
        "DUPLICATE_ENDPOINT_OF_REGISTERED_HOST",
        "LABEL_COLLISION_DIFFERENT_GPU",
    }:
        return 3
    if args.require_new and classification["outcome"] != "NEW_PHYSICAL_GPU_CANDIDATE":
        return 4
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
