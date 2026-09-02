"""Build a fail-closed local audit node after the read-only data mirror lands.

This successor never evaluates a model, reads a performance value, starts a
training lane, or opens confirmation20.  Its only transition is from a verified
mirror receipt to the canonical full-manifest/content-hash preflight and a
read-only materialized view.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-paper-local-evaluation-node-successor-v1"
STATE_SCHEMA = "final-unsb-paper-local-evaluation-node-successor-state-v1"
SOURCE_RELATIVES = (
    "operations/paper_aio_local_evaluation_node_successor.py",
    "research/paper_aio/run.py",
    "research/paper_aio/gates.py",
    "research/paper_aio/protocol.py",
    "research/paper_aio/runtime.py",
    "tools/materialize_views.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _run_text(command: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mirror_decision(value: dict[str, Any]) -> str:
    if value.get("schema") != "final-unsb-paper-local-data-mirror-state-v1":
        raise RuntimeError("local data mirror state schema mismatch")
    if value.get("source_mutated") is not False:
        raise RuntimeError("local data mirror did not preserve its source")
    if value.get("confirmation20_evaluated") is not False:
        raise RuntimeError("local data mirror crossed the confirmation boundary")
    status = value.get("status")
    if status == "MIRROR_COMPLETE_AWAITING_MANIFEST_HASH_GATE":
        return "READY_FOR_MANIFEST_GATE"
    if status == "MIRROR_FAILED_REVIEW_REQUIRED":
        return "BLOCKED_MIRROR_FAILED"
    if status == "TRANSFERRING_IDLE_IO_PRIORITY":
        return "WAITING_FOR_MIRROR"
    raise RuntimeError(f"unknown local data mirror status: {status!r}")


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if _run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("local evaluation successor checkout must be clean")
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if commit != args.required_git_commit:
        raise RuntimeError("local evaluation successor checkout moved")
    return {
        "schema": SCHEMA,
        "status": "FROZEN_WAITING",
        "repo": str(repo),
        "git_commit": commit,
        "source_sha256": {
            relative: _sha256(repo / relative) for relative in SOURCE_RELATIVES
        },
        "python": str(args.python.resolve()),
        "mirror_state": str(args.mirror_state.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "output": str(args.output.resolve()),
        "manifest": str((repo / "manifests" / "FULL_DATA_MANIFEST.csv").resolve()),
        "host_label": args.host_label,
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "training_authorized": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify_contract(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    if _run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract["git_commit"]:
        raise RuntimeError("local evaluation successor checkout moved")
    if _run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("local evaluation successor checkout became dirty")
    for relative, expected in contract["source_sha256"].items():
        if _sha256(repo / relative) != expected:
            raise RuntimeError(f"local evaluation successor source changed: {relative}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.poll_seconds <= 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise ValueError("timeout must be at least 24 hours")
    contract = _contract(args)
    output = Path(contract["output"])
    operations = output / "operations"
    contract_path = operations / "LOCAL_EVALUATION_NODE_SUCCESSOR_CONTRACT.json"
    state_path = operations / "LOCAL_EVALUATION_NODE_SUCCESSOR_STATE.json"
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("local evaluation node successor contract changed")
    else:
        _write_json(contract_path, contract)

    started = time.time()
    while True:
        _verify_contract(contract)
        mirror_state_path = Path(contract["mirror_state"])
        if not mirror_state_path.is_file():
            decision = "WAITING_FOR_MIRROR_STATE"
        else:
            decision = mirror_decision(_read_json(mirror_state_path))
        _write_json(state_path, {
            "schema": STATE_SCHEMA,
            "status": decision,
            "pid": os.getpid(),
            "contract": str(contract_path),
            "contract_sha256": _sha256(contract_path),
            "elapsed_seconds": time.time() - started,
            "training_authorized": False,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        if decision == "BLOCKED_MIRROR_FAILED":
            raise RuntimeError("local data mirror failed; review required")
        if decision == "READY_FOR_MANIFEST_GATE":
            break
        if time.time() - started > contract["timeout_hours"] * 3600:
            raise TimeoutError("local evaluation node successor timed out")
        time.sleep(contract["poll_seconds"])

    command = [
        contract["python"], "-m", "research.paper_aio.run",
        "--stage", "materialize",
        "--output", contract["output"],
        "--manifest", contract["manifest"],
        "--data-root", contract["data_root"],
        "--train-view", contract["train_view"],
        "--host-label", contract["host_label"],
        "--node-role", "audit_only",
        "--gpu", "0",
    ]
    subprocess.run(command, cwd=contract["repo"], check=True)
    preflight = output / "gates" / "PREFLIGHT.json"
    view_summary = Path(contract["train_view"]) / "VIEW_SUMMARY.json"
    preflight_value = _read_json(preflight)
    if (
        preflight_value.get("status") != "PASS"
        or preflight_value.get("node_role") != "audit_only"
        or preflight_value.get("confirmation20_opened") is not False
        or preflight_value.get("paired_controller_access") is not False
    ):
        raise RuntimeError("local audit-only preflight did not pass its frozen boundary")
    result = {
        "schema": STATE_SCHEMA,
        "status": "LOCAL_AUDIT_NODE_PREFLIGHT_PASS",
        "pid": os.getpid(),
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "preflight": str(preflight),
        "preflight_sha256": _sha256(preflight),
        "view_summary": str(view_summary),
        "view_summary_sha256": _sha256(view_summary),
        "training_authorized": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json(state_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--mirror-state", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--host-label", default="local1060_audit")
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=72)
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
