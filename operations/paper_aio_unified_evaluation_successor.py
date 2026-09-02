"""Durably evaluate fixed paper checkpoints after verified imports arrive.

This successor closes the metric-blind control-plane gap between source-host
checkpoint exporters/import relays and the one-container paper evaluator.  It
waits only on fixed completion artifacts and an optional GPU-release state. It
never inspects a metric value to decide what to run: every first-wave lane and
every frozen epoch is evaluated, then the unified cohort is locked and the
post-hoc paper adjudicator runs once.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from research.paper_aio.protocol import (
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    REQUIRED_FIRST_WAVE_TRAINED,
    file_sha256,
    protocol_fingerprint,
)


CONTRACT_SCHEMA = "final-unsb-paper-unified-evaluation-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-unified-evaluation-successor-state-v1"
IMPORT_LANE_SCHEMA = "final-unsb-paper-imported-lane-v1"
IMPORT_SET_SCHEMA = "final-unsb-paper-import-set-v1"
INPUT_RECEIPT_SCHEMA = "final-unsb-paper-unified-input-evaluation-v1"
UNIFIED_RECEIPT_SCHEMA = "final-unsb-paper-unified-evaluation-receipt-v1"
UNIFIED_EPOCHS = (100, 125, 150, 175, 200)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")

try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - deployment is Linux, tests may be Windows.
    _fcntl = None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def parse_lane_source(value: str) -> tuple[str, str]:
    lane, separator, host = str(value).partition("=")
    if not separator or not _SAFE_ID.fullmatch(lane) or not _SAFE_ID.fullmatch(host):
        raise ValueError("--lane-source requires SAFE_LANE=SAFE_HOST")
    return lane, host


def release_decision(path: Path | None, required_status: str) -> str:
    """Classify a control-state dependency without reading scientific values."""
    if path is None:
        return "READY"
    path = Path(path)
    if not path.is_file():
        return "WAIT"
    status = str(_read_json(path).get("status", ""))
    if status == required_status:
        return "READY"
    if status.startswith(("BLOCKED", "FAIL", "ERROR")):
        return "BLOCKED"
    return "WAIT"


def import_lane_path(import_root: Path, lane_id: str, host_label: str) -> Path:
    return (
        Path(import_root).resolve() / "sources" / host_label / lane_id
        / "IMPORT_LANE.json"
    )


def imports_ready(import_root: Path, lane_sources: dict[str, str]) -> bool:
    return all(
        import_lane_path(import_root, lane_id, host).is_file()
        and bool(_matching_import_sets(import_root, lane_id, host))
        for lane_id, host in lane_sources.items()
    )


def _inside(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes verified import root: {path}") from error
    return path


def _matching_import_sets(
    import_root: Path, lane_id: str, host_label: str,
) -> list[dict[str, Any]]:
    """Find completed relay sets that bind this host/lane import receipt."""
    import_root = Path(import_root).resolve()
    lane_receipt = import_lane_path(import_root, lane_id, host_label)
    if not lane_receipt.is_file():
        return []
    lane_digest = file_sha256(lane_receipt)
    matched = []
    for path in sorted((import_root / "operations").glob("IMPORT_SET_*.json")):
        value = _read_json(path)
        row = (value.get("lane_imports") or {}).get(lane_id) or {}
        advertised = Path(str(row.get("receipt", "")))
        try:
            advertised = _inside(advertised, import_root, "advertised import lane")
        except RuntimeError:
            continue
        if (
            value.get("schema") == IMPORT_SET_SCHEMA
            and value.get("status") == "COMPLETE_VERIFIED_IMPORT_SET"
            and value.get("source_host_label") == host_label
            and lane_id in value.get("lanes", [])
            and value.get("epochs") == list(UNIFIED_EPOCHS)
            and advertised == lane_receipt.resolve()
            and row.get("receipt_sha256") == lane_digest
            and value.get("checkpoint_copy_performed") is True
            and value.get("source_checkpoint_mutation") is False
            and value.get("performance_values_read") is False
            and value.get("paired_metric_control") is False
            and value.get("confirmation20_opened") is False
        ):
            matched.append({"path": path, "sha256": file_sha256(path)})
    return matched


def validate_import_lane(
    path: Path, *, import_root: Path, lane_id: str, host_label: str,
) -> list[dict[str, Any]]:
    """Bind every imported checkpoint/receipt/sidecar before GPU evaluation."""
    path = _inside(path, import_root, "import lane receipt")
    value = _read_json(path)
    memberships = _matching_import_sets(import_root, lane_id, host_label)
    if not memberships:
        raise RuntimeError(f"imported lane lacks a completed relay-set binding: {path}")
    if (
        value.get("schema") != IMPORT_LANE_SCHEMA
        or value.get("status") != "COMPLETE_VERIFIED_IMPORTED_LANE"
        or value.get("source_host_label") != host_label
        or value.get("lane_id") != lane_id
        or value.get("epochs") != list(UNIFIED_EPOCHS)
        or value.get("checkpoint_copy_performed") is not True
        or value.get("source_checkpoint_mutation") is not False
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"invalid imported lane receipt: {path}")
    rows = value.get("imports")
    if not isinstance(rows, list) or len(rows) != len(UNIFIED_EPOCHS):
        raise RuntimeError(f"incomplete imported lane: {lane_id}")
    by_epoch: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise RuntimeError(f"invalid imported checkpoint row: {lane_id}")
        epoch = int(row.get("epoch", -1))
        if epoch in by_epoch or epoch not in UNIFIED_EPOCHS:
            raise RuntimeError(f"duplicate or unexpected imported epoch: {lane_id}")
        receipt = _inside(Path(str(row.get("export_receipt", ""))), import_root, "export receipt")
        checkpoint = _inside(Path(str(row.get("checkpoint", ""))), import_root, "checkpoint")
        sidecar = _inside(Path(str(row.get("sidecar", ""))), import_root, "sidecar")
        for item, key in (
            (receipt, "export_receipt_sha256"),
            (checkpoint, "checkpoint_sha256"),
            (sidecar, "sidecar_sha256"),
        ):
            if not item.is_file() or file_sha256(item) != row.get(key):
                raise RuntimeError(f"imported file hash mismatch: {item}")
        by_epoch[epoch] = {
            "epoch": epoch,
            "export_receipt": receipt,
            "checkpoint": checkpoint,
            "source_host_label": host_label,
            "export_receipt_sha256": row["export_receipt_sha256"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "import_set_receipts": memberships,
        }
    if tuple(sorted(by_epoch)) != tuple(UNIFIED_EPOCHS):
        raise RuntimeError(f"imported epochs changed: {lane_id}")
    return [by_epoch[epoch] for epoch in UNIFIED_EPOCHS]


def existing_evaluation_status(
    *, output_root: Path, lane_id: str, row: dict[str, Any],
) -> str:
    """Validate an existing receipt without parsing its performance payload."""
    epoch = int(row["epoch"])
    receipt_path = (
        Path(output_root) / "gates"
        / f"UNIFIED_EVALUATION_{lane_id}_e{epoch:03d}.json"
    )
    if not receipt_path.is_file():
        return "MISSING"
    receipt = _read_json(receipt_path)
    metric = Path(str(receipt.get("metric", "")))
    if (
        receipt.get("schema") != UNIFIED_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS_UNIFIED_READ_ONLY_EVALUATION"
        or receipt.get("lane_id") != lane_id
        or int(receipt.get("epoch", -1)) != epoch
        or receipt.get("source_host_label") != row["source_host_label"]
        or receipt.get("source_export_receipt_sha256")
        != row["export_receipt_sha256"]
        or receipt.get("source_checkpoint_sha256") != row["checkpoint_sha256"]
        or receipt.get("evaluation_bundle_fingerprint")
        != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
        or receipt.get("training_checkpoint_read_only") is not True
        or receipt.get("paired_metric_control") is not False
        or receipt.get("cross_host_training_delta_merged") is not False
        or receipt.get("confirmation20_opened") is not False
        or not metric.is_file()
        or file_sha256(metric) != receipt.get("metric_sha256")
    ):
        raise RuntimeError(f"existing unified evaluation is stale or invalid: {receipt_path}")
    return "COMPLETE"


def input_evaluation_status(output_root: Path) -> str:
    receipt_path = Path(output_root) / "gates" / "UNIFIED_EVALUATION_input_e200.json"
    if not receipt_path.is_file():
        return "MISSING"
    receipt = _read_json(receipt_path)
    metric = Path(str(receipt.get("metric", "")))
    if (
        receipt.get("schema") != INPUT_RECEIPT_SCHEMA
        or receipt.get("status") != "PASS_UNIFIED_INPUT_EVALUATION"
        or receipt.get("lane_id") != "input"
        or int(receipt.get("epoch", -1)) != 200
        or receipt.get("evaluation_bundle_fingerprint")
        != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
        or receipt.get("evaluation_only_reference") is not True
        or receipt.get("training_checkpoint_read_only") is not True
        or receipt.get("paired_metric_control") is not False
        or receipt.get("cross_host_training_delta_merged") is not False
        or receipt.get("confirmation20_opened") is not False
        or not metric.is_file()
        or file_sha256(metric) != receipt.get("metric_sha256")
    ):
        raise RuntimeError("existing unified Input evaluation is stale or invalid")
    return "COMPLETE"


class StateHeartbeat:
    def __init__(self, path: Path, base: dict[str, Any], poll_seconds: int):
        self.path = Path(path)
        self.base = dict(base)
        self.current: dict[str, Any] = {}
        self.poll_seconds = max(30, int(poll_seconds))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.current.update(values)
            payload = {**self.base, **self.current, "updated_unix_time": time.time()}
            _write_json(self.path, payload)

    def _loop(self) -> None:
        while not self.stop.wait(self.poll_seconds):
            with self.lock:
                payload = {**self.base, **self.current, "updated_unix_time": time.time()}
                _write_json(self.path, payload)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=5)


def _git_output(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, text=True, capture_output=True, check=True,
    ).stdout.strip()


def _build_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    manifest = args.manifest.resolve()
    lane_sources = dict(parse_lane_source(value) for value in args.lane_source)
    if set(lane_sources) != set(REQUIRED_FIRST_WAVE_TRAINED):
        raise ValueError(
            "first-wave successor requires exactly plain, proposal, cut, cyclegan"
        )
    if len(args.lane_source) != len(lane_sources):
        raise ValueError("--lane-source lanes must be unique")
    if int(args.poll_seconds) < 30 or int(args.poll_seconds) > 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if float(args.timeout_hours) < 24:
        raise ValueError("timeout must be at least 24 hours")
    if not args.required_control_git_commit:
        raise ValueError("required control Git commit is mandatory")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(repo),
        "control_git_commit": str(args.required_control_git_commit),
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__).resolve()),
        "paper_protocol_fingerprint": protocol_fingerprint(manifest),
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "output_root": str(args.output.resolve()),
        "import_root": str(args.import_root.resolve()),
        "manifest": str(manifest),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "lane_sources": lane_sources,
        "epochs": list(UNIFIED_EPOCHS),
        "gpu": int(args.gpu),
        "gpu_release_state": (
            None if args.gpu_release_state is None
            else str(args.gpu_release_state.resolve())
        ),
        "gpu_release_status": str(args.gpu_release_status),
        "gpu_lock": str(args.gpu_lock.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "fixed_complete_evaluation_set": True,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }


def _freeze_contract(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    proposed = _build_contract(args)
    path = Path(proposed["output_root"]) / "operations" / "UNIFIED_EVALUATION_SUCCESSOR_CONTRACT.json"
    if path.is_file():
        current = _read_json(path)
        if current != proposed:
            raise RuntimeError("unified evaluation successor contract changed")
        return path, current
    _write_json(path, proposed)
    return path, proposed


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["control_repo"])
    if _git_output(repo, "rev-parse", "HEAD") != contract["control_git_commit"]:
        raise RuntimeError("unified evaluator control checkout moved")
    if _git_output(repo, "status", "--porcelain"):
        raise RuntimeError("unified evaluator control checkout is dirty")
    script = Path(contract["control_script"])
    if not script.is_file() or file_sha256(script) != contract["control_script_sha256"]:
        raise RuntimeError("unified evaluation successor source changed")
    if protocol_fingerprint(Path(contract["manifest"])) != contract["paper_protocol_fingerprint"]:
        raise RuntimeError("unified evaluator protocol fingerprint changed")


def _acquire_lock(handle, *, blocking: bool) -> bool:
    if _fcntl is None:  # pragma: no cover - Linux deployment uses flock.
        import msvcrt
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    flags = _fcntl.LOCK_EX | (0 if blocking else _fcntl.LOCK_NB)
    try:
        _fcntl.flock(handle.fileno(), flags)
        return True
    except BlockingIOError:
        return False


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_path, contract = _freeze_contract(args)
    _verify_control(contract)
    output = Path(contract["output_root"])
    state_path = output / "operations" / "UNIFIED_EVALUATION_SUCCESSOR_STATE.json"
    process_lock = output / "operations" / "UNIFIED_EVALUATION_SUCCESSOR.lock"
    process_lock.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    base = {
        "schema": STATE_SCHEMA,
        "pid": os.getpid(),
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "control_git_commit": contract["control_git_commit"],
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }
    heartbeat = StateHeartbeat(state_path, base, contract["poll_seconds"])
    with process_lock.open("a+", encoding="utf-8") as lock_handle:
        if not _acquire_lock(lock_handle, blocking=False):
            raise RuntimeError("unified evaluation successor is already running")
        heartbeat.start()
        try:
            while True:
                _verify_control(contract)
                release = release_decision(
                    None if contract["gpu_release_state"] is None
                    else Path(contract["gpu_release_state"]),
                    contract["gpu_release_status"],
                )
                ready = imports_ready(Path(contract["import_root"]), contract["lane_sources"])
                if release == "BLOCKED":
                    raise RuntimeError("GPU release dependency is blocked")
                if release == "READY" and ready:
                    break
                if time.time() - started > contract["timeout_hours"] * 3600:
                    raise TimeoutError("unified evaluation successor timed out")
                heartbeat.update(
                    status="WAITING_FOR_FIXED_IMPORTS_OR_GPU_RELEASE",
                    imports_ready=ready,
                    gpu_release_decision=release,
                    completed_evaluations=0,
                    paired_performance_generated=False,
                    elapsed_seconds=time.time() - started,
                )
                time.sleep(contract["poll_seconds"])

            rows: dict[str, list[dict[str, Any]]] = {}
            for lane_id, host in contract["lane_sources"].items():
                path = import_lane_path(Path(contract["import_root"]), lane_id, host)
                rows[lane_id] = validate_import_lane(
                    path, import_root=Path(contract["import_root"]),
                    lane_id=lane_id, host_label=host,
                )

            shared_lock = Path(contract["gpu_lock"])
            shared_lock.parent.mkdir(parents=True, exist_ok=True)
            with shared_lock.open("a+", encoding="utf-8") as gpu_handle:
                while not _acquire_lock(gpu_handle, blocking=False):
                    if time.time() - started > contract["timeout_hours"] * 3600:
                        raise TimeoutError("unified evaluator GPU lock timed out")
                    heartbeat.update(
                        status="WAITING_FOR_SHARED_EVALUATION_GPU",
                        imports_ready=True,
                        gpu_release_decision="READY",
                        completed_evaluations=0,
                        paired_performance_generated=False,
                        elapsed_seconds=time.time() - started,
                    )
                    time.sleep(contract["poll_seconds"])

                # Keep the metric-blind waiting process lightweight.  Heavy torch/model
                # imports begin only after every immutable input and the evaluation GPU
                # are ready; the heartbeat thread remains live during those imports.
                from research.paper_aio.adjudicate import adjudicate
                from research.paper_aio.unified import (
                    evaluate_imported_checkpoint,
                    evaluate_input_reference,
                    lock_unified_evaluation_cohort,
                )

                completed = 0
                if input_evaluation_status(output) == "MISSING":
                    heartbeat.update(
                        status="EVALUATING_FIXED_INPUT_REFERENCE",
                        current_lane="input", current_epoch=200,
                        completed_evaluations=completed,
                        paired_performance_generated=False,
                    )
                    evaluate_input_reference(
                        output_root=output, data_root=Path(contract["data_root"]),
                        manifest_path=Path(contract["manifest"]), gpu=contract["gpu"],
                    )
                completed += 1

                for lane_id in REQUIRED_FIRST_WAVE_TRAINED:
                    for row in rows[lane_id]:
                        if existing_evaluation_status(
                            output_root=output, lane_id=lane_id, row=row,
                        ) == "MISSING":
                            heartbeat.update(
                                status="EVALUATING_FIXED_CHECKPOINT",
                                current_lane=lane_id, current_epoch=row["epoch"],
                                completed_evaluations=completed,
                                total_evaluations=(
                                    1 + len(REQUIRED_FIRST_WAVE_TRAINED)
                                    * len(UNIFIED_EPOCHS)
                                ),
                                paired_performance_generated=completed > 0,
                                checkpoint_loaded_by_evaluator=True,
                                metric_used_for_training_or_scheduling=False,
                            )
                            evaluate_imported_checkpoint(
                                output_root=output,
                                export_receipt=row["export_receipt"],
                                copied_checkpoint=row["checkpoint"],
                                train_view=Path(contract["train_view"]),
                                data_root=Path(contract["data_root"]),
                                manifest_path=Path(contract["manifest"]),
                                gpu=contract["gpu"],
                            )
                        completed += 1

                heartbeat.update(
                    status="LOCKING_FIXED_FIRST_WAVE_COHORT",
                    current_lane=None, current_epoch=None,
                    completed_evaluations=completed,
                    paired_performance_generated=True,
                    metric_used_for_training_or_scheduling=False,
                )
                cohort = lock_unified_evaluation_cohort(output)
                result = adjudicate(output)
                final = {
                    "schema": STATE_SCHEMA,
                    "status": "COMPLETE_FIRST_WAVE_UNIFIED_EVALUATION_AND_ADJUDICATION",
                    "pid": os.getpid(),
                    "contract": str(contract_path),
                    "contract_sha256": file_sha256(contract_path),
                    "control_git_commit": contract["control_git_commit"],
                    "completed_evaluations": completed,
                    "cohort_status": cohort.get("status"),
                    "paper_results_status": result.get("results", {}).get("status"),
                    "algorithm_set_status": result.get("algorithm_set", {}).get("status"),
                    "paired_performance_generated": True,
                    "metric_used_for_training_or_scheduling": False,
                    "performance_values_in_control_state": False,
                    "paired_metric_control": False,
                    "best_checkpoint_selection": False,
                    "confirmation20_opened": False,
                }
                heartbeat.update(**final)
                return final
        except Exception as error:
            heartbeat.update(
                status="FAIL_CLOSED_REQUIRES_CODEX_AUDIT",
                error_type=type(error).__name__, error_message=str(error),
                metric_used_for_training_or_scheduling=False,
                paired_metric_control=False, confirmation20_opened=False,
            )
            raise
        finally:
            heartbeat.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--import-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--lane-source", action="append", required=True)
    value.add_argument("--gpu-release-state", type=Path)
    value.add_argument("--gpu-release-status", default="COMPLETE_E200")
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
