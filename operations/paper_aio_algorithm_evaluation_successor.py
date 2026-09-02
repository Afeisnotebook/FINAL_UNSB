"""Durably evaluate a fixed e200 algorithm against its legal matched plain.

Two modes are supported without changing scientific semantics:

``static_pair`` evaluates a same-host plain and static UNSB lane (currently
4090A plain versus AM-TNC) into a separate comparison root.

``dynamic_candidate`` adds one evidence-locked candidate (currently ST-CGR)
to a completed first-wave unified cohort whose plain came from the same host.

Every e100/e125/e150/e175/e200 checkpoint is fixed in advance.  Waiting and
scheduling use only completion receipts, hashes and GPU locks; metric values
are consumed only by the terminal posthoc adjudicator.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from operations.paper_aio_export_relay import (
    validate_export_receipt,
    validate_export_set,
)
from operations.paper_aio_unified_evaluation_successor import (
    StateHeartbeat,
    _acquire_lock,
    _read_json,
    _write_json,
    existing_evaluation_status,
    import_lane_path,
    imports_ready,
    validate_import_lane,
)
from research.paper_aio.protocol import (
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    file_sha256,
    lane_spec,
    protocol_fingerprint,
)


CONTRACT_SCHEMA = "final-unsb-paper-algorithm-evaluation-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-algorithm-evaluation-successor-state-v1"
DISPOSITION_SCHEMA = "final-unsb-paper-algorithm-disposition-v1"
METADATA_SCHEMA = "final-unsb-paper-candidate-metadata-import-v1"
UNIFIED_EPOCHS = (100, 125, 150, 175, 200)
COMPLETE_STATUS = "COMPLETE_SUCCESSOR_E200_ALGORITHM_EVALUATION_AND_DISPOSITION"


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _inside(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve()
    try:
        path.relative_to(Path(root).resolve())
    except ValueError as error:
        raise RuntimeError(f"{label} escapes frozen root: {path}") from error
    return path


def local_export_ready(export_root: Path, lane_id: str) -> bool:
    return (Path(export_root) / lane_id / "EXPORT_SET.json").is_file()


def validate_local_export_lane(
    export_root: Path, *, lane_id: str, source_host_label: str,
) -> list[dict[str, Any]]:
    export_root = Path(export_root).resolve()
    set_path = export_root / lane_id / "EXPORT_SET.json"
    value = _read_json(set_path)
    rows = validate_export_set(
        value, lane_id=lane_id, source_host_label=source_host_label,
    )
    validated = []
    for row in rows:
        epoch = int(row["epoch"])
        receipt = _inside(Path(row["receipt"]), export_root, "export receipt")
        if not receipt.is_file() or file_sha256(receipt) != row["receipt_sha256"]:
            raise RuntimeError(f"local export receipt hash differs: {receipt}")
        receipt_value = _read_json(receipt)
        validate_export_receipt(
            receipt_value, lane_id=lane_id, epoch=epoch,
            source_host_label=source_host_label,
        )
        checkpoint = Path(receipt_value["source_checkpoint"]).resolve()
        sidecar = Path(receipt_value["source_sidecar"]).resolve()
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != receipt_value["checkpoint_sha256"]
            or not sidecar.is_file()
            or file_sha256(sidecar) != receipt_value["sidecar_sha256"]
        ):
            raise RuntimeError(f"local exported checkpoint changed: {lane_id} e{epoch}")
        validated.append({
            "epoch": epoch,
            "export_receipt": receipt,
            "checkpoint": checkpoint,
            "source_host_label": source_host_label,
            "export_receipt_sha256": row["receipt_sha256"],
            "checkpoint_sha256": receipt_value["checkpoint_sha256"],
        })
    return validated


def cohort_decision(path: Path | None) -> str:
    if path is None:
        return "READY"
    if not Path(path).is_file():
        return "WAIT"
    value = _read_json(path)
    if (
        value.get("schema") == "final-unsb-paper-unified-evaluation-cohort-v1"
        and value.get("status") == "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT"
        and value.get("cross_host_training_delta_merged") is False
        and value.get("paired_metric_control") is False
        and value.get("confirmation20_opened") is False
    ):
        return "READY"
    return "BLOCKED"


def validate_metadata_receipt(
    path: Path, *, candidate_id: str, authority: Path,
) -> dict[str, Any]:
    value = _read_json(path)
    if (
        value.get("schema") != METADATA_SCHEMA
        or value.get("status") != "COMPLETE_VERIFIED_CANDIDATE_METADATA_IMPORT"
        or value.get("candidate_id") != candidate_id
        or value.get("authority_sha256") != file_sha256(authority)
        or value.get("training_authorized_or_scheduled") is not False
        or value.get("paired_performance_used_for_training_or_scheduling") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate metadata import receipt is invalid")
    for path_key, hash_key in (
        ("candidate_lock", "candidate_lock_sha256"),
        ("authorization", "authorization_sha256"),
        ("runtime_gate", "runtime_gate_sha256"),
    ):
        artifact = Path(value.get(path_key, ""))
        if not artifact.is_file() or file_sha256(artifact) != value.get(hash_key):
            raise RuntimeError(f"candidate metadata artifact changed: {path_key}")
    return value


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("algorithm evaluator checkout must be clean")
    head = _git(repo, "rev-parse", "HEAD")
    if head != args.required_control_git_commit:
        raise RuntimeError("algorithm evaluator checkout moved")
    if args.mode == "static_pair":
        lane_spec(args.method_lane)
        if not args.plain_source_root or not args.plain_source_host:
            raise ValueError("static_pair requires a frozen plain export source")
        if args.method_source_host != args.plain_source_host:
            raise ValueError("static_pair requires same-host method and plain")
        if args.candidate_authority or args.candidate_metadata_receipt:
            raise ValueError("static_pair cannot use dynamic-candidate metadata")
    else:
        if not args.candidate_authority or not args.candidate_metadata_receipt:
            raise ValueError("dynamic_candidate requires authority and metadata")
        if not args.first_wave_cohort:
            raise ValueError("dynamic_candidate requires completed first-wave cohort")
        if args.plain_source_root or args.plain_source_host:
            raise ValueError("dynamic candidate reuses the locked same-host cohort plain")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(repo),
        "control_git_commit": head,
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__).resolve()),
        "paper_protocol_fingerprint": protocol_fingerprint(args.manifest),
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "mode": args.mode,
        "method_lane": args.method_lane,
        "method_source_root": str(args.method_source_root.resolve()),
        "method_source_host": args.method_source_host,
        "plain_source_root": (
            None if args.plain_source_root is None
            else str(args.plain_source_root.resolve())
        ),
        "plain_source_host": args.plain_source_host,
        "output_root": str(args.output.resolve()),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "candidate_authority": (
            None if args.candidate_authority is None
            else str(args.candidate_authority.resolve())
        ),
        "candidate_metadata_receipt": (
            None if args.candidate_metadata_receipt is None
            else str(args.candidate_metadata_receipt.resolve())
        ),
        "first_wave_cohort": (
            None if args.first_wave_cohort is None
            else str(args.first_wave_cohort.resolve())
        ),
        "gpu_lock": str(args.gpu_lock.resolve()),
        "gpu": int(args.gpu),
        "epochs": list(UNIFIED_EPOCHS),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "fixed_complete_evaluation_set": True,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }


def _freeze_contract(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if int(args.poll_seconds) < 30 or int(args.poll_seconds) > 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if float(args.timeout_hours) < 24:
        raise ValueError("timeout must be at least 24 hours")
    proposed = _contract(args)
    output = Path(proposed["output_root"])
    path = output / "operations" / f"ALGORITHM_EVALUATION_SUCCESSOR_{args.method_lane}_CONTRACT.json"
    if path.is_file():
        if _read_json(path) != proposed:
            raise RuntimeError("algorithm evaluation successor contract changed")
    else:
        _write_json(path, proposed)
    return path, proposed


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["control_repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
        or file_sha256(Path(contract["control_script"]))
        != contract["control_script_sha256"]
        or protocol_fingerprint(Path(contract["manifest"]))
        != contract["paper_protocol_fingerprint"]
    ):
        raise RuntimeError("algorithm evaluator control identity changed")


def _ready(contract: dict[str, Any]) -> tuple[bool, str]:
    if contract["mode"] == "static_pair":
        plain_ready = local_export_ready(
            Path(contract["plain_source_root"]), "plain",
        )
        method_ready = local_export_ready(
            Path(contract["method_source_root"]), contract["method_lane"],
        )
        return plain_ready and method_ready, "WAITING_FOR_STATIC_PAIR_EXPORTS"
    imported = imports_ready(
        Path(contract["method_source_root"]),
        {contract["method_lane"]: contract["method_source_host"]},
    )
    metadata = Path(contract["candidate_metadata_receipt"]).is_file()
    cohort = cohort_decision(Path(contract["first_wave_cohort"]))
    if cohort == "BLOCKED":
        raise RuntimeError("first-wave cohort is invalid")
    return imported and metadata and cohort == "READY", "WAITING_FOR_CANDIDATE_IMPORT_METADATA_OR_COHORT"


def _rows(contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    if contract["mode"] == "static_pair":
        return {
            "plain": validate_local_export_lane(
                Path(contract["plain_source_root"]), lane_id="plain",
                source_host_label=contract["plain_source_host"],
            ),
            contract["method_lane"]: validate_local_export_lane(
                Path(contract["method_source_root"]),
                lane_id=contract["method_lane"],
                source_host_label=contract["method_source_host"],
            ),
        }
    authority = Path(contract["candidate_authority"])
    validate_metadata_receipt(
        Path(contract["candidate_metadata_receipt"]),
        candidate_id=contract["method_lane"], authority=authority,
    )
    lane_path = import_lane_path(
        Path(contract["method_source_root"]),
        contract["method_lane"], contract["method_source_host"],
    )
    return {
        contract["method_lane"]: validate_import_lane(
            lane_path, import_root=Path(contract["method_source_root"]),
            lane_id=contract["method_lane"],
            host_label=contract["method_source_host"],
        )
    }


def _disposition(
    *, output: Path, method_lane: str, result: dict[str, Any],
) -> dict[str, Any]:
    entries = [
        row for row in result["results"]["lanes"]
        if row["lane_id"] == method_lane
    ]
    if len(entries) != 1:
        raise RuntimeError("posthoc adjudicator omitted the fixed algorithm")
    entry = entries[0]
    if entry.get("status") != "COMPLETE_E200" or "scientific_gate" not in entry:
        raise RuntimeError("fixed algorithm lacks complete terminal adjudication")
    receipts = []
    for epoch in UNIFIED_EPOCHS:
        path = output / "gates" / f"UNIFIED_EVALUATION_{method_lane}_e{epoch:03d}.json"
        if not path.is_file():
            raise RuntimeError(f"fixed algorithm receipt missing: {path}")
        receipts.append({
            "epoch": epoch, "path": str(path.resolve()),
            "sha256": file_sha256(path),
        })
    return {
        "schema": DISPOSITION_SCHEMA,
        "status": "COMPLETE_POSTHOC_ALGORITHM_DISPOSITION",
        "method_lane": method_lane,
        "primary_epoch": 200,
        "fixed_epochs": list(UNIFIED_EPOCHS),
        "entry": entry,
        "evaluation_receipts": receipts,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_path, contract = _freeze_contract(args)
    output = Path(contract["output_root"])
    state_path = output / "operations" / f"ALGORITHM_EVALUATION_SUCCESSOR_{args.method_lane}_STATE.json"
    process_lock = output / "operations" / f"ALGORITHM_EVALUATION_SUCCESSOR_{args.method_lane}.lock"
    process_lock.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "schema": STATE_SCHEMA,
        "pid": os.getpid(),
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "control_git_commit": contract["control_git_commit"],
        "method_lane": contract["method_lane"],
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }
    heartbeat = StateHeartbeat(state_path, base, contract["poll_seconds"])
    started = time.time()
    with process_lock.open("a+", encoding="utf-8") as process_handle:
        if not _acquire_lock(process_handle, blocking=False):
            raise RuntimeError("algorithm evaluation successor is already running")
        heartbeat.start()
        try:
            while True:
                _verify_control(contract)
                ready, waiting_status = _ready(contract)
                if ready:
                    break
                if time.time() - started > contract["timeout_hours"] * 3600:
                    raise TimeoutError("algorithm evaluation successor timed out")
                heartbeat.update(
                    status=waiting_status, completed_evaluations=0,
                    performance_values_generated=False,
                    metric_used_for_training_or_scheduling=False,
                )
                time.sleep(contract["poll_seconds"])
            rows = _rows(contract)
            gpu_lock = Path(contract["gpu_lock"])
            gpu_lock.parent.mkdir(parents=True, exist_ok=True)
            with gpu_lock.open("a+", encoding="utf-8") as gpu_handle:
                while not _acquire_lock(gpu_handle, blocking=False):
                    heartbeat.update(
                        status="WAITING_FOR_SHARED_EVALUATION_GPU",
                        completed_evaluations=0,
                        performance_values_generated=False,
                        metric_used_for_training_or_scheduling=False,
                    )
                    time.sleep(contract["poll_seconds"])
                from research.paper_aio.unified import evaluate_imported_checkpoint

                completed = 0
                lane_order = (
                    ("plain", contract["method_lane"])
                    if contract["mode"] == "static_pair"
                    else (contract["method_lane"],)
                )
                for lane_id in lane_order:
                    for row in rows[lane_id]:
                        if existing_evaluation_status(
                            output_root=output, lane_id=lane_id, row=row,
                        ) == "MISSING":
                            heartbeat.update(
                                status="EVALUATING_FIXED_ALGORITHM_CHECKPOINT",
                                current_lane=lane_id, current_epoch=row["epoch"],
                                completed_evaluations=completed,
                                performance_values_generated=completed > 0,
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
                                candidate_id=(
                                    contract["method_lane"]
                                    if contract["mode"] == "dynamic_candidate" else None
                                ),
                                candidate_authority=(
                                    Path(contract["candidate_authority"])
                                    if contract["mode"] == "dynamic_candidate" else None
                                ),
                            )
                        completed += 1

            heartbeat.update(
                status="RUNNING_FIXED_POSTHOC_ADJUDICATION",
                current_lane=None, current_epoch=None,
                completed_evaluations=completed,
                performance_values_generated=True,
                metric_used_for_training_or_scheduling=False,
            )
            from research.paper_aio.adjudicate import adjudicate

            adjudication_lock = output / "operations" / "PAPER_ADJUDICATE.lock"
            with adjudication_lock.open("a+", encoding="utf-8") as handle:
                _acquire_lock(handle, blocking=True)
                adjudicated = adjudicate(output)
                disposition = _disposition(
                    output=output, method_lane=contract["method_lane"],
                    result=adjudicated,
                )
                disposition_path = (
                    output / "algorithm_dispositions"
                    / f"{contract['method_lane']}.json"
                )
                _write_json(disposition_path, disposition)
            final = {
                **base,
                "status": COMPLETE_STATUS,
                "completed_evaluations": completed,
                "scientific_gate_status": disposition["entry"]["scientific_gate"]["status"],
                "disposition": str(disposition_path.resolve()),
                "disposition_sha256": file_sha256(disposition_path),
                "performance_values_generated": True,
                "performance_values_in_control_state": False,
                "metric_used_for_training_or_scheduling": False,
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
    value.add_argument("--mode", choices=("static_pair", "dynamic_candidate"), required=True)
    value.add_argument("--method-lane", required=True)
    value.add_argument("--method-source-root", type=Path, required=True)
    value.add_argument("--method-source-host", required=True)
    value.add_argument("--plain-source-root", type=Path)
    value.add_argument("--plain-source-host")
    value.add_argument("--candidate-authority", type=Path)
    value.add_argument("--candidate-metadata-receipt", type=Path)
    value.add_argument("--first-wave-cohort", type=Path)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
