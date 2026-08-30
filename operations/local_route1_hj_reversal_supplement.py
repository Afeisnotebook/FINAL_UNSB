"""Durably append the three HJ states omitted by the original audit selector.

The primary immutable collector must finish first.  This executor then uses the
same frozen audit worktree for HJ e40/e60/e80, verifies the append-only atlases,
and asks a separately frozen current-main analyzer to regenerate only the audit
queue.  It deliberately stops before final reanalysis or algorithm derivation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-route1-hj-reversal-supplement-contract-v1"
EXPECTED_PRIMARY = (432, 128)
EXPECTED_FINAL = (474, 140)
SUPPLEMENT_EPOCHS = (40, 60, 80)
EXPECTED_AUDIT_COMMIT = "729826f55f2cdbd59fbc51cd64b437bb392ea21c"
EXPECTED_AUDIT_SOURCE = (
    "41434187402e9f3c2226931e1e3c4e0474dbc84f0c2f17e603b084ac54005e1a"
)
EXPECTED_TRAINING_CORE = (
    "0a67135dffcf87f31ea1534f10b34c3e218961186f85dfb7487f10916a200714"
)
EXPECTED_MANIFEST = (
    "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def exclusive_lock(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": now()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if path.exists():
            path.unlink()


def _git_head(repo: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(f"cannot read git head for {repo}: {result.stderr}")
    return result.stdout.strip()


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("HJ supplement contract schema mismatch")
    required = (
        "run_root", "audit_repo", "analysis_repo", "training_repo",
        "train_view", "data_root", "manifest", "python",
        "analysis_git_commit", "audit_git_commit",
    )
    missing = [name for name in required if not contract.get(name)]
    if missing:
        raise RuntimeError(f"HJ supplement contract missing: {missing}")
    if contract["audit_git_commit"] != EXPECTED_AUDIT_COMMIT:
        raise RuntimeError("HJ supplement audit commit differs from frozen collector")
    _allowed_audit_identities(contract)
    if contract.get("paired_controller_access") is not False:
        raise RuntimeError("paired controller access is forbidden")
    if contract.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 must remain locked")
    if _git_head(Path(contract["audit_repo"])) != EXPECTED_AUDIT_COMMIT:
        raise RuntimeError("frozen audit worktree moved")
    if _git_head(Path(contract["analysis_repo"])) != contract["analysis_git_commit"]:
        raise RuntimeError("analysis worktree moved")
    for label in ("audit_repo", "analysis_repo"):
        if subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(contract[label]),
            capture_output=True, text=True, check=False,
        ).stdout.strip():
            raise RuntimeError(f"frozen {label} worktree is dirty")
    if file_sha256(Path(contract["manifest"])) != EXPECTED_MANIFEST:
        raise RuntimeError("HJ supplement manifest identity mismatch")
    if not Path(contract["python"]).is_file():
        raise RuntimeError("HJ supplement Python runtime is missing")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _allowed_audit_identities(
    contract: dict[str, Any] | None = None,
) -> set[tuple[str, str]]:
    rows = None if contract is None else contract.get("allowed_audit_identities")
    if rows is None:
        return {(EXPECTED_AUDIT_COMMIT, EXPECTED_AUDIT_SOURCE)}
    if not isinstance(rows, list) or not rows:
        raise RuntimeError("allowed_audit_identities must be a non-empty list")
    identities: set[tuple[str, str]] = set()
    for row in rows:
        if not isinstance(row, dict) or not row.get("audit_git_commit") or not row.get(
            "audit_source_fingerprint"
        ):
            raise RuntimeError("allowed audit identity requires commit and source fingerprint")
        identities.add((
            str(row["audit_git_commit"]), str(row["audit_source_fingerprint"]),
        ))
    if (EXPECTED_AUDIT_COMMIT, EXPECTED_AUDIT_SOURCE) not in identities:
        raise RuntimeError("allowed audit identities omit the frozen supplement writer")
    return identities


def verify_atlases(
    run_root: Path, *, expected: tuple[int, int], require_supplement: bool,
    allowed_audit_identities: set[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    audit = run_root / "audit"
    reversal = _read_rows(audit / "LONG_REVERSAL_ATLAS.jsonl")
    variance = _read_rows(audit / "SAMPLING_VARIANCE_ATLAS.jsonl")
    if (len(reversal), len(variance)) != expected:
        raise RuntimeError(
            f"atlas count mismatch: {(len(reversal), len(variance))} != {expected}"
        )
    for name, rows in (("reversal", reversal), ("variance", variance)):
        ids = [str(row["row_id"]) for row in rows]
        if len(set(ids)) != len(ids):
            raise RuntimeError(f"duplicate {name} row id")
        if not all(_finite(row) for row in rows):
            raise RuntimeError(f"nonfinite {name} row")
        if not all(row.get("confirmation20_opened") is False for row in rows):
            raise RuntimeError(f"confirmation20 opened in {name} atlas")
    allowed = allowed_audit_identities or _allowed_audit_identities()
    if not all(
        row.get("paired_metrics_accessed_by_controller") is False
        and row["parent_state_sha256_before"] == row["parent_state_sha256_after"]
        and (
            row.get("audit_identity", {}).get("audit_git_commit"),
            row.get("audit_identity", {}).get("audit_source_fingerprint"),
        ) in allowed
        and row.get("audit_identity", {}).get("training_core_fingerprint")
        == EXPECTED_TRAINING_CORE
        for row in reversal
    ):
        raise RuntimeError("reversal atlas identity/isolation/access guard failed")
    if not all(
        row.get("paired_metrics_accessed_by_controller") is False
        and row["parent_state_sha256_before"] == row["parent_state_sha256_after"]
        and (
            row.get("audit_identity", {}).get("audit_git_commit"),
            row.get("audit_identity", {}).get("audit_source_fingerprint"),
        ) in allowed
        and row.get("audit_identity", {}).get("training_core_fingerprint")
        == EXPECTED_TRAINING_CORE
        for row in variance
    ):
        raise RuntimeError("variance atlas identity/isolation/access guard failed")
    if require_supplement:
        observed = {
            (str(row["probe"]), int(row["data_epoch"])) for row in reversal
        }
        missing = [("hj", epoch) for epoch in SUPPLEMENT_EPOCHS if ("hj", epoch) not in observed]
        if missing:
            raise RuntimeError(f"HJ reversal supplement cells missing: {missing}")
    observed: dict[tuple[str, str], int] = {}
    for row in [*reversal, *variance]:
        identity = (
            str(row["audit_identity"]["audit_git_commit"]),
            str(row["audit_identity"]["audit_source_fingerprint"]),
        )
        observed[identity] = observed.get(identity, 0) + 1
    return {
        "reversal_rows": len(reversal),
        "variance_rows": len(variance),
        "reversal_sha256": file_sha256(audit / "LONG_REVERSAL_ATLAS.jsonl"),
        "variance_sha256": file_sha256(audit / "SAMPLING_VARIANCE_ATLAS.jsonl"),
        "observed_audit_identities": [
            {
                "audit_git_commit": identity[0],
                "audit_source_fingerprint": identity[1],
                "rows": count,
            }
            for identity, count in sorted(observed.items())
        ],
    }


def wait_for_primary(run_root: Path, poll_seconds: int) -> None:
    state_path = run_root / "operations" / "AUDIT_EXECUTION_STATE.json"
    while True:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        status = str(state.get("status"))
        if status == "PHASE_C_COMPLETE_DERIVATION_REQUIRED":
            return
        if status == "FAILED":
            raise RuntimeError("primary audit executor failed")
        time.sleep(poll_seconds)


def run_job(contract: dict[str, Any], epoch: int, operations: Path) -> None:
    log = operations / "logs" / f"supplement_hj_e{epoch:03d}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        str(contract["python"]), "-m", "research.local_route1.run",
        "--output", str(contract["run_root"]),
        "--train-view", str(contract["train_view"]),
        "--data-root", str(contract["data_root"]),
        "--manifest", str(contract["manifest"]),
        "--training-worktree", str(contract["training_repo"]),
        "--gpu", "0", "--stage", "audit", "--audit-probe", "hj",
        "--audit-epoch", str(epoch), "--audit-horizons", "1,8,32,200",
        "--audit-label-horizons", "200",
    ]
    with log.open("a", encoding="utf-8") as handle:
        result = subprocess.run(
            argv, cwd=Path(contract["audit_repo"]), stdout=handle,
            stderr=subprocess.STDOUT, text=True, check=False,
        )
    if result.returncode:
        raise RuntimeError(f"HJ supplement e{epoch} failed with {result.returncode}: {log}")


def regenerate_queue(contract: dict[str, Any]) -> dict[str, Any]:
    result = subprocess.run(
        [
            str(contract["python"]), "-m", "research.local_route1.run",
            "--stage", "audit", "--output", str(contract["run_root"]),
        ],
        cwd=Path(contract["analysis_repo"]), capture_output=True, text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"final queue regeneration failed: {result.stderr}")
    queue = json.loads(
        (Path(contract["run_root"]) / "audit" / "AUDIT_QUEUE.json").read_text(
            encoding="utf-8"
        )
    )
    cells = {(str(row["probe"]), int(row["data_epoch"])) for row in queue["jobs"]}
    if len(queue["jobs"]) != 28 or not all(
        ("hj", epoch) in cells for epoch in SUPPLEMENT_EPOCHS
    ):
        raise RuntimeError("regenerated queue does not contain the 28-cell final design")
    return {
        "jobs": len(queue["jobs"]),
        "queue_sha256": file_sha256(
            Path(contract["run_root"]) / "audit" / "AUDIT_QUEUE.json"
        ),
    }


def execute(contract_path: Path) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract(contract)
    run_root = Path(contract["run_root"])
    operations = run_root / "operations"
    state_path = operations / "HJ_REVERSAL_SUPPLEMENT_STATE.json"
    events = operations / "HJ_REVERSAL_SUPPLEMENT_EVENTS.jsonl"

    def state(status: str, **fields: Any) -> None:
        atomic_json(state_path, {
            "schema": "final-unsb-route1-hj-reversal-supplement-state-v1",
            "updated": now(), "status": status, "pid": os.getpid(),
            "paired_controller_access": False, "confirmation20_opened": False,
            **fields,
        })

    state("WAITING_FOR_PRIMARY_AUDIT")
    wait_for_primary(run_root, int(contract.get("poll_seconds", 30)))
    allowed = _allowed_audit_identities(contract)
    primary = verify_atlases(
        run_root, expected=EXPECTED_PRIMARY, require_supplement=False,
        allowed_audit_identities=allowed,
    )
    append_event(events, {"time": now(), "event": "PRIMARY_VERIFIED", **primary})
    state("SUPPLEMENT_RUNNING", completed_epochs=[])
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda epoch: run_job(contract, epoch, operations), (40, 60)))
    state("SUPPLEMENT_RUNNING", completed_epochs=[40, 60])
    run_job(contract, 80, operations)
    final = verify_atlases(
        run_root, expected=EXPECTED_FINAL, require_supplement=True,
        allowed_audit_identities=allowed,
    )
    queue = regenerate_queue(contract)
    result = {
        "schema": "final-unsb-route1-hj-reversal-supplement-result-v1",
        "status": "SUPPLEMENT_COMPLETE_REANALYSIS_REQUIRED",
        "primary": primary, "final": final, "queue": queue,
        "analysis_git_commit": contract["analysis_git_commit"],
        "audit_git_commit": contract["audit_git_commit"],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    atomic_json(operations / "HJ_REVERSAL_SUPPLEMENT_RESULT.json", result)
    state(result["status"], result_sha256=file_sha256(
        operations / "HJ_REVERSAL_SUPPLEMENT_RESULT.json"
    ))
    append_event(events, {"time": now(), "event": result["status"], **final, **queue})
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", required=True, type=Path)
    return value


def write_failure_state(contract_path: Path, error: Exception) -> None:
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        run_root = Path(contract["run_root"])
        atomic_json(
            run_root / "operations" / "HJ_REVERSAL_SUPPLEMENT_STATE.json",
            {
                "schema": "final-unsb-route1-hj-reversal-supplement-state-v1",
                "updated": now(), "status": "FAILED", "pid": os.getpid(),
                "error_type": type(error).__name__, "error": str(error),
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
    except Exception:
        # The original exception remains authoritative when even the failure
        # destination cannot be recovered from the contract.
        return


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    lock = args.contract.with_suffix(args.contract.suffix + ".lock")
    try:
        with exclusive_lock(lock):
            print(json.dumps(execute(args.contract), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        write_failure_state(args.contract, error)
        print(json.dumps({
            "status": "FAILED", "error_type": type(error).__name__,
            "error": str(error), "confirmation20_opened": False,
        }, ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
