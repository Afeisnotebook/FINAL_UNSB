"""Fail-closed gate for continuing an interrupted lane on a provider clone.

This is an emergency infrastructure gate, not a same-physical-host claim.  It
accepts a source checkpoint only when the replacement GPU has independently
reproduced the frozen 2000-update runtime core and two isolated one-update
resume branches agree exactly.  The source run is read-only throughout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA = "final-unsb-paper-clone-continuation-gate-v1"
RUNTIME_SCHEMA = "final-unsb-paper-runtime-twin-receipt-v1"
HOST_SCHEMA = "final-unsb-paper-host-identity-gate-v1"
SIDECAR_SCHEMA = "final-unsb-paper-aio-full-state-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True,
    ).strip()


def _runtime_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    required = (
        "updates", "e0_core_sha256", "step_core_sha256",
        "protocol_fingerprint", "manifest_sha256",
    )
    if receipt.get("schema") != RUNTIME_SCHEMA:
        raise RuntimeError("runtime twin schema mismatch")
    return {key: receipt.get(key) for key in required}


def validate_static_inputs(args: argparse.Namespace) -> dict[str, Any]:
    host = _read_json(args.host_identity_receipt.resolve())
    source_twin = _read_json(args.source_runtime_twin.resolve())
    replacement_twin = _read_json(args.replacement_runtime_twin.resolve())
    sidecar = _read_json(args.source_sidecar.resolve())

    if (
        host.get("schema") != HOST_SCHEMA
        or host.get("status") != "NEW_PHYSICAL_GPU_CANDIDATE"
        or host.get("classification", {}).get("requested_label")
        != args.replacement_host_label
        or host.get("classification", {}).get("long_training_launch_allowed_by_identity_gate")
        is not True
    ):
        raise RuntimeError("replacement physical-host identity gate did not pass")
    if (
        replacement_twin.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or replacement_twin.get("exact_runtime_equivalence") is not True
        or replacement_twin.get("host_label") != args.replacement_host_label
        or replacement_twin.get("differences") != {}
    ):
        raise RuntimeError("replacement runtime twin is not exact")
    source_identity = _runtime_identity(source_twin)
    replacement_identity = _runtime_identity(replacement_twin)
    if source_identity != replacement_identity:
        raise RuntimeError("source and replacement runtime cores differ")
    if source_twin.get("host_label") != args.source_host_label:
        raise RuntimeError("source runtime twin host label mismatch")

    training_repo = args.training_repo.resolve()
    if (
        _git(training_repo, "rev-parse", "HEAD") != args.training_git_commit
        or _git(training_repo, "status", "--porcelain")
    ):
        raise RuntimeError("frozen training checkout identity changed")
    checkpoint = args.source_checkpoint.resolve()
    if (
        sidecar.get("schema") != SIDECAR_SCHEMA
        or sidecar.get("lane_id") != args.lane
        or sidecar.get("metadata", {}).get("git_commit") != args.training_git_commit
        or sidecar.get("metadata", {}).get("protocol_fingerprint")
        != args.protocol_fingerprint
        or sidecar.get("metadata", {}).get("manifest_sha256")
        != replacement_identity["manifest_sha256"]
        or sidecar.get("full_state_sha256") != _sha256(checkpoint)
    ):
        raise RuntimeError("source full-state identity is invalid")
    step = int(sidecar.get("step", -1))
    target = int(sidecar.get("target_steps", -1))
    if step < 1 or step >= target:
        raise RuntimeError("source checkpoint is not a resumable partial trajectory")
    if host.get("identity", {}).get("gpu_uuid") == args.source_gpu_uuid:
        raise RuntimeError("clone gate requires a different physical GPU UUID")
    return {
        "host": host,
        "source_twin": source_twin,
        "replacement_twin": replacement_twin,
        "sidecar": sidecar,
        "runtime_identity": replacement_identity,
        "source_checkpoint_sha256": sidecar["full_state_sha256"],
        "source_step": step,
        "target_steps": target,
    }


def _copy_probe_source(args: argparse.Namespace, destination: Path) -> None:
    if destination.exists():
        raise RuntimeError(f"resume probe destination already exists: {destination}")
    source_output = args.source_output.resolve()
    copies = (
        (source_output / "PAPER_PROTOCOL.json", destination / "PAPER_PROTOCOL.json"),
        (
            source_output / "gates" / f"LANE_AUTHORIZATION_{args.lane}.json",
            destination / "gates" / f"LANE_AUTHORIZATION_{args.lane}.json",
        ),
        (
            source_output / "shared_e0" / "unsb_common" / "e0.pt",
            destination / "shared_e0" / "unsb_common" / "e0.pt",
        ),
        (
            source_output / "shared_e0" / "unsb_common" / "e0.pt.json",
            destination / "shared_e0" / "unsb_common" / "e0.pt.json",
        ),
        (
            args.source_checkpoint.resolve(),
            destination / "lanes" / args.lane / "full_state_latest.pt",
        ),
        (
            args.source_sidecar.resolve(),
            destination / "lanes" / args.lane / "full_state_latest.pt.json",
        ),
    )
    for source, target in copies:
        if not source.is_file():
            raise RuntimeError(f"required resume probe source is missing: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _run_probe(
    args: argparse.Namespace, *, destination: Path, stop_step: int,
) -> dict[str, Any]:
    _copy_probe_source(args, destination)
    command = [
        str(args.python.resolve()), "-m", "research.paper_aio.run",
        "--stage", "train", "--lane", args.lane, "--resume",
        "--engineering-stop-after-updates", str(stop_step),
        "--output", str(destination),
        "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--gpu", str(args.gpu),
    ]
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        command,
        cwd=args.training_repo.resolve(),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log = destination / "CLONE_RESUME_PROBE.log"
    log.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        raise RuntimeError(
            f"isolated clone resume probe failed ({destination.name}); see {log}"
        )
    checkpoint = destination / "lanes" / args.lane / "full_state_latest.pt"
    sidecar_path = checkpoint.with_suffix(checkpoint.suffix + ".json")
    sidecar = _read_json(sidecar_path)
    if (
        int(sidecar.get("step", -1)) != stop_step
        or sidecar.get("full_state_sha256") != _sha256(checkpoint)
    ):
        raise RuntimeError("isolated resume probe output failed full-state validation")
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sidecar["full_state_sha256"],
        "sidecar": str(sidecar_path),
        "sidecar_sha256": _sha256(sidecar_path),
        "scientific_state_sha256": sidecar.get("scientific_state_sha256"),
        "step": stop_step,
        "returncode": completed.returncode,
        "log": str(log),
        "log_sha256": _sha256(log),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    validated = validate_static_inputs(args)
    source_checkpoint = args.source_checkpoint.resolve()
    source_before = _sha256(source_checkpoint)
    probe_root = args.probe_root.resolve()
    stop_step = validated["source_step"] + int(args.probe_updates)
    if stop_step > validated["target_steps"]:
        raise RuntimeError("resume probe would exceed the frozen target")
    branch_a = _run_probe(args, destination=probe_root / "branch_a", stop_step=stop_step)
    branch_b = _run_probe(args, destination=probe_root / "branch_b", stop_step=stop_step)
    source_after = _sha256(source_checkpoint)
    exact = (
        branch_a["checkpoint_sha256"] == branch_b["checkpoint_sha256"]
        and branch_a["scientific_state_sha256"]
        == branch_b["scientific_state_sha256"]
    )
    source_preserved = source_before == source_after == validated["source_checkpoint_sha256"]
    if not exact or not source_preserved:
        raise RuntimeError("clone resume branches differ or source checkpoint changed")
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_ENGINEERING_CLONE_CONTINUATION_GATE",
        "captured_unix_time": time.time(),
        "lane_id": args.lane,
        "source_host_label": args.source_host_label,
        "source_gpu_uuid": args.source_gpu_uuid,
        "replacement_host_label": args.replacement_host_label,
        "replacement_gpu_uuid": validated["host"]["identity"]["gpu_uuid"],
        "same_physical_host_claimed": False,
        "source_checkpoint_step": validated["source_step"],
        "source_checkpoint_sha256": source_before,
        "source_checkpoint_preserved": source_preserved,
        "runtime_identity": validated["runtime_identity"],
        "source_runtime_twin_sha256": _sha256(args.source_runtime_twin.resolve()),
        "replacement_runtime_twin_sha256": _sha256(args.replacement_runtime_twin.resolve()),
        "host_identity_receipt_sha256": _sha256(args.host_identity_receipt.resolve()),
        "resume_probe_updates": int(args.probe_updates),
        "resume_probe_branches_exact": exact,
        "resume_probe_a": branch_a,
        "resume_probe_b": branch_b,
        "training_repo": str(args.training_repo.resolve()),
        "training_git_commit": args.training_git_commit,
        "protocol_fingerprint": args.protocol_fingerprint,
        "comparison_disposition": (
            "SEGMENTED_EXACT_RUNTIME_COHORT_REQUIRES_EXPLICIT_GIT_ADMISSION_"
            "AND_MANUSCRIPT_DISCLOSURE"
        ),
        "automatic_same_host_relabel_allowed": False,
        "paired_metric_control": False,
        "performance_values_read": False,
        "confirmation20_opened": False,
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-identity-receipt", type=Path, required=True)
    parser.add_argument("--source-runtime-twin", type=Path, required=True)
    parser.add_argument("--replacement-runtime-twin", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sidecar", type=Path, required=True)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--probe-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", default="plain")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--probe-updates", type=int, default=1)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--source-gpu-uuid", required=True)
    parser.add_argument("--replacement-host-label", required=True)
    parser.add_argument("--training-git-commit", required=True)
    parser.add_argument("--protocol-fingerprint", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    if args.probe_updates < 1 or args.probe_updates > 32:
        raise SystemExit("probe-updates must be in [1,32]")
    receipt = run(args)
    _atomic_json(args.output.resolve(), receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
