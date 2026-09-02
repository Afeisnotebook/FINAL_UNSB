"""Durably adjudicate terminal pathology after all target-blind audits.

The successor does not parse the paired metric-binding file until the complete
fixed audit set has passed.  Its output can authorize derivation, never training.
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


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from research.paper_aio.protocol import file_sha256  # noqa: E402
from research.paper_aio.terminal_adjudicate import (  # noqa: E402
    AUDIT_EPOCHS,
    BINDING_SCHEMA,
    PROBES,
    adjudicate_terminal_pathology,
)
from operations.paper_aio_local_terminal_audit_successor import (  # noqa: E402
    _acquire_lock,
    _ready_rows,
)
from research.paper_aio.unified import (  # noqa: E402
    UNIFIED_RECEIPT_SCHEMA,
    evaluate_imported_checkpoint,
)


CONTRACT_SCHEMA = "final-unsb-paper-terminal-pathology-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-terminal-pathology-successor-state-v1"
SOURCE_RELATIVES = (
    "operations/paper_aio_terminal_pathology_successor.py",
    "operations/paper_aio_local_terminal_audit_successor.py",
    "research/paper_aio/terminal_adjudicate.py",
    "research/paper_aio/terminal_audit.py",
    "research/paper_aio/protocol.py",
    "research/paper_aio/unified.py",
    "research/paper_aio/evaluate.py",
    "research/paper_aio/runtime.py",
    "research/local_route1/runtime.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def audit_release(path: Path) -> str:
    if not Path(path).is_file():
        return "WAIT"
    value = _read_json(path)
    if (
        value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        return "BLOCKED"
    status = str(value.get("status", ""))
    if "FAIL" in status or "BLOCK" in status:
        return "BLOCKED"
    if status != "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS":
        return "WAIT"
    expected = {f"{probe_id}:e{epoch}" for probe_id in PROBES for epoch in AUDIT_EPOCHS}
    return "READY" if set(value.get("completed", [])) == expected else "BLOCKED"


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("terminal pathology successor checkout is not frozen")
    auto = bool(args.auto_evaluate_after_audits)
    auto_paths = (
        args.import_root,
        args.evaluation_root,
        args.manifest,
        args.data_root,
        args.train_view,
        args.candidate_authority,
        args.gpu_lock,
    )
    if auto and any(path is None for path in auto_paths):
        raise RuntimeError("automatic posthoc evaluation paths are incomplete")
    candidate_authority = (
        None if args.candidate_authority is None else args.candidate_authority.resolve()
    )
    if auto and not candidate_authority.is_file():
        raise RuntimeError("automatic posthoc evaluation lacks candidate authority")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "repo": str(repo),
        "control_git_commit": commit,
        "control_source_sha256": {
            relative: file_sha256(repo / relative) for relative in SOURCE_RELATIVES
        },
        "audit_successor_state": str(args.audit_successor_state.resolve()),
        "audit_root": str(args.audit_root.resolve()),
        "metric_bindings": str(args.metric_bindings.resolve()),
        "destination": str(args.destination.resolve()),
        "auto_evaluate_after_audits": auto,
        "evaluation_python": str(Path(sys.executable).resolve()),
        "import_root": None
        if args.import_root is None
        else str(args.import_root.resolve()),
        "evaluation_root": (
            None
            if args.evaluation_root is None
            else str(args.evaluation_root.resolve())
        ),
        "manifest": None if args.manifest is None else str(args.manifest.resolve()),
        "data_root": None if args.data_root is None else str(args.data_root.resolve()),
        "train_view": None
        if args.train_view is None
        else str(args.train_view.resolve()),
        "candidate_authority": (
            None if candidate_authority is None else str(candidate_authority)
        ),
        "candidate_authority_sha256": (
            None if candidate_authority is None else file_sha256(candidate_authority)
        ),
        "gpu_lock": None if args.gpu_lock is None else str(args.gpu_lock.resolve()),
        "gpu": int(args.gpu),
        "fixed_epochs": list(AUDIT_EPOCHS),
        "fixed_probes": list(PROBES),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "paired_files_parsed_before_all_audits": False,
        "paired_metric_control": False,
        "algorithm_or_module_auto_start": False,
        "confirmation20_opened": False,
    }


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    if _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"] or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("terminal pathology successor checkout moved")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"terminal pathology source changed: {relative}")
    if (
        contract["candidate_authority"] is not None
        and file_sha256(Path(contract["candidate_authority"]))
        != contract["candidate_authority_sha256"]
    ):
        raise RuntimeError("terminal pathology candidate authority changed")


def _metric_paths(evaluation_root: Path, lane_id: str, epoch: int) -> dict[str, Path]:
    return {
        "receipt": evaluation_root
        / "gates"
        / f"UNIFIED_EVALUATION_{lane_id}_e{epoch:03d}.json",
        "metric": evaluation_root
        / "lanes"
        / lane_id
        / "metrics"
        / f"e{epoch:03d}.json",
    }


def _completed_metric(
    paths: dict[str, Path], *, probe: dict[str, str], epoch: int
) -> bool:
    receipt_path, metric_path = paths["receipt"], paths["metric"]
    if not receipt_path.is_file() and not metric_path.is_file():
        return False
    # The evaluator writes the metric before its receipt. A process interruption
    # in that narrow interval is recoverable: reevaluation verifies the existing
    # metric byte-for-byte and then materializes the missing receipt.
    if metric_path.is_file() and not receipt_path.is_file():
        return False
    if not metric_path.is_file():
        raise RuntimeError("partial automatic terminal metric cell exists")
    value = _read_json(receipt_path)
    if (
        value.get("schema") != UNIFIED_RECEIPT_SCHEMA
        or value.get("status") != "PASS_UNIFIED_READ_ONLY_EVALUATION"
        or value.get("lane_id") != probe["lane_id"]
        or value.get("source_host_label") != probe["source_host_label"]
        or int(value.get("epoch", -1)) != epoch
        or Path(str(value.get("metric", ""))).resolve() != metric_path.resolve()
        or value.get("metric_sha256") != file_sha256(metric_path)
        or value.get("training_checkpoint_read_only") is not True
        or value.get("paired_metric_control") is not False
        or value.get("cross_host_training_delta_merged") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("existing automatic terminal metric cell is invalid")
    return True


def _write_wait_state(
    state_path: Path,
    *,
    status: str,
    audit_release_value: str,
    completed_evaluations: int = 0,
    current_probe: str | None = None,
    current_epoch: int | None = None,
) -> None:
    _write_json(
        state_path,
        {
            "schema": STATE_SCHEMA,
            "status": status,
            "pid": os.getpid(),
            "audit_release": audit_release_value,
            "current_probe": current_probe,
            "current_epoch": current_epoch,
            "completed_evaluations": completed_evaluations,
            "paired_files_parsed": False,
            "performance_values_read": completed_evaluations > 0,
            "performance_values_generated_posthoc": completed_evaluations > 0,
            "metric_used_for_training_or_scheduling": False,
            "paired_metric_control": False,
            "algorithm_or_module_auto_started": False,
            "confirmation20_opened": False,
        },
    )


def _automatic_metrics(contract: dict[str, Any], state_path: Path) -> Path:
    evaluation_root = Path(contract["evaluation_root"])
    audit_contract = {
        "import_root": contract["import_root"],
    }
    ready_rows = {}
    for probe_id, probe in PROBES.items():
        lookup = {
            "import_lane": probe["lane_id"],
            "host_label": probe["source_host_label"],
        }
        rows = _ready_rows(audit_contract, lookup)
        if set(rows) != set(AUDIT_EPOCHS):
            raise RuntimeError(f"automatic metric imports incomplete: {probe_id}")
        ready_rows[probe_id] = rows

    completed = 0
    gpu_lock = Path(contract["gpu_lock"])
    gpu_lock.parent.mkdir(parents=True, exist_ok=True)
    with gpu_lock.open("a+", encoding="utf-8") as handle:
        while not _acquire_lock(handle):
            _verify_control(contract)
            if audit_release(Path(contract["audit_successor_state"])) != "READY":
                raise RuntimeError(
                    "target-blind audit release changed while waiting for GPU"
                )
            _write_wait_state(
                state_path,
                status="WAITING_FOR_SHARED_EVALUATION_GPU_AFTER_TARGET_BLIND_AUDITS",
                audit_release_value="READY",
                completed_evaluations=completed,
            )
            time.sleep(contract["poll_seconds"])
        for probe_id, probe in PROBES.items():
            for epoch in AUDIT_EPOCHS:
                paths = _metric_paths(evaluation_root, probe["lane_id"], epoch)
                if not _completed_metric(paths, probe=probe, epoch=epoch):
                    _write_wait_state(
                        state_path,
                        status="RUNNING_POSTHOC_UNIFIED_METRIC_AFTER_TARGET_BLIND_AUDITS",
                        audit_release_value="READY",
                        completed_evaluations=completed,
                        current_probe=probe_id,
                        current_epoch=epoch,
                    )
                    row = ready_rows[probe_id][epoch]
                    evaluate_imported_checkpoint(
                        output_root=evaluation_root,
                        export_receipt=Path(row["export_receipt"]),
                        copied_checkpoint=Path(row["checkpoint"]),
                        train_view=Path(contract["train_view"]),
                        data_root=Path(contract["data_root"]),
                        manifest_path=Path(contract["manifest"]),
                        gpu=contract["gpu"],
                        candidate_id=(
                            probe["lane_id"] if probe_id == "5090A_stcgr" else None
                        ),
                        candidate_authority=(
                            Path(contract["candidate_authority"])
                            if probe_id == "5090A_stcgr"
                            else None
                        ),
                    )
                completed += 1

    binding = {"schema": BINDING_SCHEMA, "probes": {}}
    for probe_id, probe in PROBES.items():
        binding["probes"][probe_id] = {}
        for epoch in AUDIT_EPOCHS:
            paths = _metric_paths(evaluation_root, probe["lane_id"], epoch)
            if not _completed_metric(paths, probe=probe, epoch=epoch):
                raise RuntimeError(
                    "automatic terminal metric disappeared after evaluation"
                )
            binding["probes"][probe_id][f"e{epoch:03d}"] = {
                key: str(path.resolve()) for key, path in paths.items()
            }
    binding_path = Path(contract["metric_bindings"])
    if binding_path.is_file():
        if _read_json(binding_path) != binding:
            raise RuntimeError("automatic terminal metric binding changed")
    else:
        _write_json(binding_path, binding)
    return binding_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.poll_seconds <= 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise ValueError("timeout must be at least 24 hours")
    contract = _contract(args)
    output = args.output.resolve()
    contract_path = output / "TERMINAL_PATHOLOGY_SUCCESSOR_CONTRACT.json"
    state_path = output / "TERMINAL_PATHOLOGY_SUCCESSOR_STATE.json"
    if contract_path.is_file():
        if _read_json(contract_path) != contract:
            raise RuntimeError("terminal pathology successor contract changed")
    else:
        _write_json(contract_path, contract)

    started = time.time()
    while True:
        _verify_control(contract)
        release = audit_release(Path(contract["audit_successor_state"]))
        if release == "BLOCKED":
            raise RuntimeError("target-blind terminal audit release is invalid")
        if release == "READY":
            binding_path = (
                _automatic_metrics(contract, state_path)
                if contract["auto_evaluate_after_audits"]
                else Path(contract["metric_bindings"])
            )
            if binding_path.is_file():
                result = adjudicate_terminal_pathology(
                    audit_root=Path(contract["audit_root"]),
                    metric_bindings=binding_path,
                    destination=Path(contract["destination"]),
                )
                state = {
                    "schema": STATE_SCHEMA,
                    "status": "COMPLETE_POSTHOC_TERMINAL_PATHOLOGY_ADJUDICATION",
                    "pid": os.getpid(),
                    "decision_status": result["status"],
                    "terminal_pathology_confirmed": result[
                        "terminal_pathology_confirmed"
                    ],
                    "decision_sha256": file_sha256(Path(contract["destination"])),
                    "all_target_blind_audits_validated_before_paired_metric_read": True,
                    "performance_values_read_posthoc": True,
                    "automatic_unified_evaluations": contract[
                        "auto_evaluate_after_audits"
                    ],
                    "paired_metric_control": False,
                    "algorithm_or_module_auto_started": False,
                    "confirmation20_opened": False,
                }
                _write_json(state_path, state)
                return state
            status = "WAITING_FOR_POSTHOC_UNIFIED_METRIC_BINDINGS"
        else:
            status = "WAITING_FOR_COMPLETE_TARGET_BLIND_AUDITS"
        if time.time() - started > contract["timeout_hours"] * 3600:
            raise TimeoutError("terminal pathology successor timed out")
        _write_json(
            state_path,
            {
                "schema": STATE_SCHEMA,
                "status": status,
                "pid": os.getpid(),
                "audit_release": release,
                "metric_binding_exists": Path(contract["metric_bindings"]).is_file(),
                "automatic_unified_evaluations": contract["auto_evaluate_after_audits"],
                "elapsed_seconds": time.time() - started,
                "paired_files_parsed": False,
                "performance_values_read": False,
                "paired_metric_control": False,
                "algorithm_or_module_auto_started": False,
                "confirmation20_opened": False,
            },
        )
        time.sleep(contract["poll_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--audit-successor-state", type=Path, required=True)
    value.add_argument("--audit-root", type=Path, required=True)
    value.add_argument("--metric-bindings", type=Path, required=True)
    value.add_argument("--destination", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--auto-evaluate-after-audits", action="store_true")
    value.add_argument("--import-root", type=Path)
    value.add_argument("--evaluation-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--candidate-authority", type=Path)
    value.add_argument("--gpu-lock", type=Path)
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
