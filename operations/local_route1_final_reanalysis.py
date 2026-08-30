"""Freeze the completed route-1 atlas and rebuild its final causal authority.

The immutable primary collector intentionally uses an older analysis commit.
This executor waits for the append-only HJ reversal supplement, archives the
collector's matrix and derivation artifacts, and reanalyzes the unchanged raw
rows with a separately frozen current analyzer.  It never trains a candidate.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from operations.local_route1_hj_reversal_supplement import (
    EXPECTED_AUDIT_COMMIT,
    EXPECTED_FINAL,
    EXPECTED_MANIFEST,
    EXPECTED_TRAINING_CORE,
    atomic_json,
    exclusive_lock,
    file_sha256,
    now,
    verify_atlases,
)


SCHEMA = "final-unsb-route1-final-reanalysis-contract-v1"
TERMINAL = "FINAL_REANALYSIS_COMPLETE_DERIVATION_REQUIRED"


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed for {repo}: {result.stderr}")
    return result.stdout.strip()


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("final reanalysis contract schema mismatch")
    required = (
        "run_root", "analysis_repo", "python", "analysis_git_commit",
        "audit_git_commit", "training_core_fingerprint", "manifest_sha256",
    )
    missing = [name for name in required if not contract.get(name)]
    if missing:
        raise RuntimeError(f"final reanalysis contract missing: {missing}")
    if contract["audit_git_commit"] != EXPECTED_AUDIT_COMMIT:
        raise RuntimeError("final reanalysis audit identity mismatch")
    if contract["training_core_fingerprint"] != EXPECTED_TRAINING_CORE:
        raise RuntimeError("final reanalysis training-core identity mismatch")
    if contract["manifest_sha256"] != EXPECTED_MANIFEST:
        raise RuntimeError("final reanalysis manifest identity mismatch")
    if contract.get("paired_controller_access") is not False:
        raise RuntimeError("paired controller access is forbidden")
    if contract.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 must remain locked")
    repo = Path(contract["analysis_repo"])
    if _git(repo, "rev-parse", "HEAD") != contract["analysis_git_commit"]:
        raise RuntimeError("final analysis worktree moved")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("final analysis worktree is dirty")
    if not Path(contract["python"]).is_file():
        raise RuntimeError("final reanalysis Python runtime is missing")


def wait_for_supplement(run_root: Path, poll_seconds: int) -> None:
    state_path = run_root / "operations" / "HJ_REVERSAL_SUPPLEMENT_STATE.json"
    while True:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status = str(state.get("status"))
        if status == "SUPPLEMENT_COMPLETE_REANALYSIS_REQUIRED":
            return
        if status == "FAILED":
            raise RuntimeError("HJ reversal supplement failed")
        time.sleep(poll_seconds)


def verify_queue(run_root: Path) -> dict[str, Any]:
    path = run_root / "audit" / "AUDIT_QUEUE.json"
    queue = json.loads(path.read_text(encoding="utf-8"))
    jobs = list(queue.get("jobs", []))
    cells = {(str(row["probe"]), int(row["data_epoch"])) for row in jobs}
    if len(jobs) != 28 or len(cells) != 28:
        raise RuntimeError("final audit queue must contain 28 unique cells")
    if not all(("hj", epoch) in cells for epoch in (40, 60, 80)):
        raise RuntimeError("final audit queue omits the repaired HJ reversal cells")
    if queue.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 opened in final audit queue")
    return {"jobs": 28, "queue_sha256": file_sha256(path)}


def freeze_raw_inputs(run_root: Path) -> dict[str, Any]:
    atlas = verify_atlases(
        run_root, expected=EXPECTED_FINAL, require_supplement=True,
    )
    queue = verify_queue(run_root)
    result = {
        "schema": "final-unsb-route1-final-raw-freeze-v1",
        "created": now(),
        **atlas,
        **queue,
        "audit_git_commit": EXPECTED_AUDIT_COMMIT,
        "training_core_fingerprint": EXPECTED_TRAINING_CORE,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    atomic_json(run_root / "operations" / "FINAL_RAW_EVIDENCE_FREEZE.json", result)
    return result


def _tree_hashes(root: Path) -> dict[str, str]:
    if not root.is_dir():
        return {}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _verify_archive_payload(archive: Path, intent: dict[str, Any]) -> None:
    matrix = archive / "LONG_CAUSAL_MATRIX.json"
    expected_matrix = intent.get("matrix_sha256")
    if expected_matrix and (
        not matrix.is_file() or file_sha256(matrix) != expected_matrix
    ):
        raise RuntimeError("archived collector matrix hash mismatch")
    expected_derive = dict(intent.get("derive_files", {}))
    if expected_derive != _tree_hashes(archive / "derive"):
        raise RuntimeError("archived collector derive tree hash mismatch")


def archive_collector_outputs(run_root: Path, analysis_commit: str) -> dict[str, Any]:
    operations = run_root / "operations"
    archive = operations / "final_reanalysis_archive" / f"pre_{analysis_commit[:12]}"
    intent_path = archive / "ARCHIVE_INTENT.json"
    result_path = archive / "ARCHIVE_RESULT.json"
    source_matrix = run_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    source_derive = run_root / "derive"
    archive.mkdir(parents=True, exist_ok=True)
    if intent_path.is_file():
        intent = json.loads(intent_path.read_text(encoding="utf-8"))
    else:
        intent = {
            "schema": "final-unsb-route1-pre-reanalysis-archive-intent-v1",
            "created": now(),
            "analysis_git_commit": analysis_commit,
            "matrix_sha256": (
                file_sha256(source_matrix) if source_matrix.is_file() else None
            ),
            "derive_files": _tree_hashes(source_derive),
            "deletion_permitted": False,
        }
        atomic_json(intent_path, intent)
    archived_matrix = archive / "LONG_CAUSAL_MATRIX.json"
    archived_derive = archive / "derive"
    if result_path.is_file():
        _verify_archive_payload(archive, intent)
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        return {**existing, "result_sha256": file_sha256(result_path)}
    if intent.get("matrix_sha256"):
        if source_matrix.exists() and not archived_matrix.exists():
            os.replace(source_matrix, archived_matrix)
        elif source_matrix.exists() and archived_matrix.exists():
            raise RuntimeError("both source and archived collector matrices exist")
    if intent.get("derive_files"):
        if source_derive.exists() and not archived_derive.exists():
            os.replace(source_derive, archived_derive)
        elif source_derive.exists() and archived_derive.exists():
            raise RuntimeError("both source and archived collector derive trees exist")
    _verify_archive_payload(archive, intent)
    result = {
        "schema": "final-unsb-route1-pre-reanalysis-archive-result-v1",
        "created": now(),
        "archive": str(archive),
        "intent_sha256": file_sha256(intent_path),
        "matrix_archived": bool(intent.get("matrix_sha256")),
        "derive_files_archived": len(intent.get("derive_files", {})),
        "deletion_permitted": False,
    }
    atomic_json(result_path, result)
    return {**result, "result_sha256": file_sha256(result_path)}


def archive_retry_outputs(run_root: Path) -> dict[str, Any] | None:
    matrix = run_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    derive = run_root / "derive"
    if not matrix.exists() and not derive.exists():
        return None
    parent = run_root / "operations" / "final_reanalysis_archive"
    index = 1
    while (parent / f"retry_{index:02d}").exists():
        index += 1
    target = parent / f"retry_{index:02d}"
    target.mkdir(parents=True)
    record = {
        "schema": "final-unsb-route1-reanalysis-retry-archive-v1",
        "created": now(),
        "matrix_sha256": file_sha256(matrix) if matrix.is_file() else None,
        "derive_files": _tree_hashes(derive),
        "deletion_permitted": False,
    }
    if matrix.exists():
        os.replace(matrix, target / "LONG_CAUSAL_MATRIX.json")
    if derive.exists():
        os.replace(derive, target / "derive")
    atomic_json(target / "ARCHIVE_RESULT.json", record)
    return {"archive": str(target), **record}


def run_reanalysis(contract: dict[str, Any], operations: Path) -> None:
    stdout = operations / "FINAL_REANALYSIS.stdout.log"
    stderr = operations / "FINAL_REANALYSIS.stderr.log"
    with stdout.open("a", encoding="utf-8") as out, stderr.open(
        "a", encoding="utf-8"
    ) as err:
        result = subprocess.run(
            [
                str(contract["python"]), "-m", "research.local_route1.run",
                "--stage", "reanalyze", "--output", str(contract["run_root"]),
            ],
            cwd=Path(contract["analysis_repo"]), stdout=out, stderr=err,
            text=True, check=False,
        )
    if result.returncode:
        raise RuntimeError(
            f"latest causal reanalysis failed with {result.returncode}: {stderr}"
        )


def verify_final_outputs(
    run_root: Path, contract: dict[str, Any], raw: dict[str, Any],
) -> dict[str, Any]:
    post = freeze_raw_inputs(run_root)
    for key in (
        "reversal_sha256", "variance_sha256", "queue_sha256",
        "reversal_rows", "variance_rows", "jobs",
    ):
        if post[key] != raw[key]:
            raise RuntimeError(f"raw evidence changed during final reanalysis: {key}")
    matrix_path = run_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
        raise RuntimeError("latest causal matrix is not complete")
    if (
        matrix.get("rows") != EXPECTED_FINAL[0]
        or matrix.get("expected_rows") != EXPECTED_FINAL[0]
        or matrix.get("sampling_variance_rows") != EXPECTED_FINAL[1]
        or matrix.get("expected_sampling_variance_rows") != EXPECTED_FINAL[1]
        or matrix.get("missing_rows")
        or matrix.get("missing_sampling_variance_rows")
    ):
        raise RuntimeError("latest causal matrix does not cover the 474/140 design")
    identity = matrix.get("analysis_identity", {})
    expected_identity = {
        "analysis_git_commit": contract["analysis_git_commit"],
        "reversal_atlas_sha256": raw["reversal_sha256"],
        "sampling_variance_atlas_sha256": raw["variance_sha256"],
        "audit_queue_sha256": raw["queue_sha256"],
        "branch_rows_modified_by_analysis": False,
        "paired_metrics_accessed_by_controller": False,
        "confirmation20_opened": False,
    }
    for key, value in expected_identity.items():
        if identity.get(key) != value:
            raise RuntimeError(f"latest causal matrix identity mismatch: {key}")
    if (
        matrix.get("paired_metrics_accessed_by_controller") is not False
        or matrix.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("latest causal matrix violated access guards")
    queue_path = run_root / "derive" / "DERIVATION_QUEUE.json"
    ledger_path = run_root / "derive" / "HYPOTHESIS_LEDGER.json"
    derivation = json.loads(queue_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    if derivation.get("status") not in (
        "DERIVATION_CARDS_REQUIRED", "NO_ELIGIBLE_DRIVER_OR_UNBIASED_ROUTE",
    ):
        raise RuntimeError("latest derivation queue has an invalid state")
    if len(derivation.get("cards", [])) > 3:
        raise RuntimeError("latest derivation queue exceeds the Generation-1 cap")
    evidence = ledger.get("evidence_identity", {})
    if (
        evidence.get("causal_matrix_sha256") != file_sha256(matrix_path)
        or evidence.get("reversal_atlas_sha256") != raw["reversal_sha256"]
    ):
        raise RuntimeError("hypothesis ledger is not bound to latest causal evidence")
    if (
        derivation.get("confirmation20_opened") is not False
        or ledger.get("confirmation20_opened") is not False
        or ledger.get("paired_controller_access") is not False
    ):
        raise RuntimeError("latest derivation artifacts violated access guards")
    return {
        "matrix_sha256": file_sha256(matrix_path),
        "derivation_queue_sha256": file_sha256(queue_path),
        "hypothesis_ledger_sha256": file_sha256(ledger_path),
        "ranked_failure_mechanisms": len(
            matrix.get("ranked_failure_mechanisms", [])
        ),
        "generation1_routes": len(derivation.get("cards", [])),
        "derivation_status": derivation["status"],
    }


def execute(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    run_root = Path(contract["run_root"])
    operations = run_root / "operations"
    state_path = operations / "FINAL_REANALYSIS_STATE.json"
    result_path = operations / "FINAL_REANALYSIS_RESULT.json"

    def state(status: str, **fields: Any) -> None:
        atomic_json(state_path, {
            "schema": "final-unsb-route1-final-reanalysis-state-v1",
            "updated": now(), "status": status, "pid": os.getpid(),
            "paired_controller_access": False, "confirmation20_opened": False,
            **fields,
        })

    if result_path.is_file():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == TERMINAL:
            return existing
    state("WAITING_FOR_HJ_REVERSAL_SUPPLEMENT")
    wait_for_supplement(run_root, int(contract.get("poll_seconds", 30)))
    raw = freeze_raw_inputs(run_root)
    state("ARCHIVING_COLLECTOR_ANALYSIS", raw_evidence=raw)
    archive = archive_collector_outputs(
        run_root, str(contract["analysis_git_commit"]),
    )
    retry = archive_retry_outputs(run_root)
    state("FINAL_REANALYSIS_RUNNING", raw_evidence=raw, archive=archive, retry=retry)
    run_reanalysis(contract, operations)
    outputs = verify_final_outputs(run_root, contract, raw)
    result = {
        "schema": "final-unsb-route1-final-reanalysis-result-v1",
        "status": TERMINAL,
        "completed": now(),
        "analysis_git_commit": contract["analysis_git_commit"],
        "raw_evidence": raw,
        "collector_archive": archive,
        "retry_archive": retry,
        "outputs": outputs,
        "candidate_training_started": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    atomic_json(result_path, result)
    state(TERMINAL, result_sha256=file_sha256(result_path), outputs=outputs)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", required=True, type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    lock = args.contract.with_suffix(args.contract.suffix + ".lock")
    try:
        with exclusive_lock(lock):
            print(json.dumps(execute(args.contract), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        print(json.dumps({
            "status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "paired_controller_access": False,
            "confirmation20_opened": False,
        }, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
