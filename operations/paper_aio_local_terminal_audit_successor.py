"""Durably run fixed full-data terminal audits on the local evaluation GPU.

The successor consumes only verified imported checkpoints and an audit-only
preflight receipt.  It never reads paired performance, chooses a checkpoint,
or authorizes training.  The fixed e100/e150/e200 audits run independently of
matched-delta availability.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from operations.paper_aio_unified_evaluation_successor import (
    import_lane_path,
    imports_ready,
    release_decision,
    validate_import_lane,
)
from research.paper_aio.protocol import file_sha256, protocol_fingerprint


STATE_SCHEMA = "final-unsb-paper-local-terminal-audit-successor-state-v1"
CONTRACT_SCHEMA = "final-unsb-paper-local-terminal-audit-successor-contract-v1"
RECEIPT_SCHEMA = "final-unsb-paper-local-terminal-audit-receipt-v1"
AUDIT_SCHEMA = "final-unsb-paper-terminal-spectrum-audit-v1"
AUDIT_EPOCHS = (100, 150, 200)
STCGR = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"
SOURCE_RELATIVES = (
    "operations/paper_aio_local_terminal_audit_successor.py",
    "operations/paper_aio_unified_evaluation_successor.py",
    "research/paper_aio/run.py",
    "research/paper_aio/terminal_audit.py",
    "research/paper_aio/runtime.py",
    "research/paper_aio/protocol.py",
)

PROBES = (
    {
        "probe_id": "4090A_plain",
        "host_label": "4090A",
        "import_lane": "plain",
        "cli_lane": "plain",
        "candidate_id": None,
    },
    {
        "probe_id": "5090C_proposal",
        "host_label": "5090C",
        "import_lane": "proposal",
        "cli_lane": "proposal",
        "candidate_id": None,
    },
    {
        "probe_id": "4090A_amtnc",
        "host_label": "4090A",
        "import_lane": "amtnc",
        "cli_lane": "amtnc",
        "candidate_id": None,
    },
    {
        "probe_id": "5090A_stcgr",
        "host_label": "5090A",
        "import_lane": STCGR,
        "cli_lane": "candidate",
        "candidate_id": STCGR,
    },
)


try:
    import fcntl as _fcntl
except ImportError:  # Windows local audit node.
    _fcntl = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True,
    ).strip()


def _audit_result(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"terminal audit output missing: {path}")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(lines) != 1:
        raise RuntimeError(f"terminal audit output must contain exactly one row: {path}")
    value = json.loads(lines[0])
    if (
        value.get("schema") != AUDIT_SCHEMA
        or value.get("status") != "TARGET_BLIND_AUDIT_COMPLETE"
        or value.get("parent_state_sha256_before")
        != value.get("parent_state_sha256_after")
        or value.get("parent_rng_sha256_before")
        != value.get("parent_rng_sha256_after")
        or value.get("paired_labels_attached") is not False
        or value.get("terminal_pathology_confirmed") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"terminal audit boundary failed: {path}")
    return value


def _completed_receipt(path: Path, *, row: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    value = _read_json(path)
    audit = Path(str(value.get("audit", "")))
    return bool(
        value.get("schema") == RECEIPT_SCHEMA
        and value.get("status") == "PASS_FIXED_TARGET_BLIND_TERMINAL_AUDIT"
        and value.get("checkpoint_sha256") == row["checkpoint_sha256"]
        and value.get("export_receipt_sha256") == row["export_receipt_sha256"]
        and audit.is_file()
        and file_sha256(audit) == value.get("audit_sha256")
        and value.get("performance_values_read") is False
        and value.get("paired_metric_control") is False
        and value.get("confirmation20_opened") is False
    )


def _acquire_lock(handle) -> bool:
    if _fcntl is None:
        import msvcrt

        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        _fcntl.flock(handle.fileno(), _fcntl.LOCK_EX | _fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("local terminal audit control checkout is not frozen")
    authority = args.candidate_authority.resolve()
    if not authority.is_file():
        raise RuntimeError("local terminal audit lacks ST-CGR portable authority")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "repo": str(repo),
        "control_git_commit": commit,
        "control_source_sha256": {
            relative: file_sha256(repo / relative) for relative in SOURCE_RELATIVES
        },
        "paper_protocol_fingerprint": protocol_fingerprint(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "preflight_state": str(args.preflight_state.resolve()),
        "preflight_required_status": "LOCAL_AUDIT_NODE_PREFLIGHT_PASS",
        "import_root": str(args.import_root.resolve()),
        "output": str(args.output.resolve()),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "candidate_authority": str(authority),
        "candidate_authority_sha256": file_sha256(authority),
        "gpu_lock": str(args.gpu_lock.resolve()),
        "gpu": int(args.gpu),
        "audit_epochs": list(AUDIT_EPOCHS),
        "probes": list(PROBES),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
    ):
        raise RuntimeError("local terminal audit control checkout moved")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"local terminal audit source changed: {relative}")
    authority = Path(contract["candidate_authority"])
    if file_sha256(authority) != contract["candidate_authority_sha256"]:
        raise RuntimeError("ST-CGR portable evaluation authority changed")


def _ready_rows(contract: dict[str, Any], probe: dict[str, Any]) -> dict[int, dict[str, Any]]:
    import_root = Path(contract["import_root"])
    if not imports_ready(
        import_root, {probe["import_lane"]: probe["host_label"]},
    ):
        return {}
    path = import_lane_path(
        import_root, probe["import_lane"], probe["host_label"],
    )
    if not path.is_file():
        return {}
    rows = validate_import_lane(
        path, import_root=import_root, lane_id=probe["import_lane"],
        host_label=probe["host_label"],
    )
    return {int(row["epoch"]): row for row in rows if int(row["epoch"]) in AUDIT_EPOCHS}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.poll_seconds <= 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise ValueError("timeout must be at least 24 hours")
    contract = _contract(args)
    output = Path(contract["output"])
    operations = output / "operations"
    contract_path = operations / "LOCAL_TERMINAL_AUDIT_SUCCESSOR_CONTRACT.json"
    state_path = operations / "LOCAL_TERMINAL_AUDIT_SUCCESSOR_STATE.json"
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("local terminal audit successor contract changed")
    else:
        _write_json(contract_path, contract)

    started = time.time()
    complete: list[str] = []
    while len(complete) < len(PROBES) * len(AUDIT_EPOCHS):
        _verify_control(contract)
        release = release_decision(
            Path(contract["preflight_state"]), contract["preflight_required_status"],
        )
        if release == "BLOCKED":
            raise RuntimeError("local audit-only preflight failed")
        made_progress = False
        if release == "READY":
            preflight = _read_json(Path(contract["preflight_state"]))
            if (
                preflight.get("training_authorized") is not False
                or preflight.get("performance_values_read") is not False
                or preflight.get("paired_metric_control") is not False
                or preflight.get("confirmation20_opened") is not False
            ):
                raise RuntimeError("local audit-only preflight crossed its boundary")
            for probe in contract["probes"]:
                rows = _ready_rows(contract, probe)
                for epoch in contract["audit_epochs"]:
                    key = f"{probe['probe_id']}:e{epoch}"
                    if key in complete or epoch not in rows:
                        continue
                    row = rows[epoch]
                    audit_root = output / "probes" / probe["probe_id"] / f"e{epoch:03d}"
                    receipt_path = audit_root / "AUDIT_RECEIPT.json"
                    if _completed_receipt(receipt_path, row=row):
                        complete.append(key)
                        made_progress = True
                        continue
                    gpu_lock = Path(contract["gpu_lock"])
                    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
                    with gpu_lock.open("a+", encoding="utf-8") as handle:
                        if not _acquire_lock(handle):
                            continue
                        audit_root.mkdir(parents=True, exist_ok=True)
                        audit_path = audit_root / "TERMINAL_AUDIT.jsonl"
                        if audit_path.is_file():
                            _audit_result(audit_path)
                            receipt = {
                                "schema": RECEIPT_SCHEMA,
                                "status": "PASS_FIXED_TARGET_BLIND_TERMINAL_AUDIT",
                                "probe_id": probe["probe_id"],
                                "host_label": probe["host_label"],
                                "lane_id": probe["import_lane"],
                                "epoch": epoch,
                                "checkpoint_sha256": row["checkpoint_sha256"],
                                "export_receipt_sha256": row["export_receipt_sha256"],
                                "audit": str(audit_path.resolve()),
                                "audit_sha256": file_sha256(audit_path),
                                "parent_state_and_rng_unchanged": True,
                                "performance_values_read": False,
                                "paired_metric_control": False,
                                "confirmation20_opened": False,
                            }
                            _write_json(receipt_path, receipt)
                            complete.append(key)
                            made_progress = True
                            continue
                        command = [
                            contract["python"], "-m", "research.paper_aio.run",
                            "--stage", "terminal-audit",
                            "--lane", probe["cli_lane"],
                            "--checkpoint", str(row["checkpoint"]),
                            "--output", str(audit_root),
                            "--manifest", contract["manifest"],
                            "--data-root", contract["data_root"],
                            "--train-view", contract["train_view"],
                            "--gpu", str(contract["gpu"]),
                        ]
                        if probe["candidate_id"] is not None:
                            command.extend([
                                "--candidate-id", probe["candidate_id"],
                                "--candidate-authority", contract["candidate_authority"],
                            ])
                        _write_json(state_path, {
                            "schema": STATE_SCHEMA,
                            "status": "RUNNING_FIXED_TARGET_BLIND_AUDIT",
                            "pid": os.getpid(),
                            "current_probe": probe["probe_id"],
                            "current_epoch": epoch,
                            "completed": sorted(complete),
                            "performance_values_read": False,
                            "paired_metric_control": False,
                            "confirmation20_opened": False,
                        })
                        process = subprocess.run(
                            command, cwd=contract["repo"], text=True,
                            capture_output=True, check=False,
                        )
                        (audit_root / "STDOUT.log").write_text(
                            process.stdout, encoding="utf-8",
                        )
                        (audit_root / "STDERR.log").write_text(
                            process.stderr, encoding="utf-8",
                        )
                        if process.returncode:
                            raise RuntimeError(
                                f"terminal audit failed for {key}: exit {process.returncode}"
                            )
                        _audit_result(audit_path)
                        receipt = {
                            "schema": RECEIPT_SCHEMA,
                            "status": "PASS_FIXED_TARGET_BLIND_TERMINAL_AUDIT",
                            "probe_id": probe["probe_id"],
                            "host_label": probe["host_label"],
                            "lane_id": probe["import_lane"],
                            "epoch": epoch,
                            "checkpoint_sha256": row["checkpoint_sha256"],
                            "export_receipt_sha256": row["export_receipt_sha256"],
                            "audit": str(audit_path.resolve()),
                            "audit_sha256": file_sha256(audit_path),
                            "parent_state_and_rng_unchanged": True,
                            "performance_values_read": False,
                            "paired_metric_control": False,
                            "confirmation20_opened": False,
                        }
                        _write_json(receipt_path, receipt)
                        complete.append(key)
                        made_progress = True
        if len(complete) == len(PROBES) * len(AUDIT_EPOCHS):
            break
        if time.time() - started > contract["timeout_hours"] * 3600:
            raise TimeoutError("local terminal audit successor timed out")
        _write_json(state_path, {
            "schema": STATE_SCHEMA,
            "status": "WAITING_FOR_PREFLIGHT_IMPORTS_OR_GPU",
            "pid": os.getpid(),
            "preflight_release": release,
            "completed": sorted(complete),
            "elapsed_seconds": time.time() - started,
            "made_progress_last_poll": made_progress,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        })
        time.sleep(contract["poll_seconds"])

    result = {
        "schema": STATE_SCHEMA,
        "status": "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS",
        "pid": os.getpid(),
        "completed": sorted(complete),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json(state_path, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--preflight-state", type=Path, required=True)
    value.add_argument("--import-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--candidate-authority", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
