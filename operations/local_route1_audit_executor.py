"""Durable post-anchor executor for the local route-1 causal audit.

This operational supervisor waits until the frozen anchor executor reaches a
terminal scientific boundary, then runs the actual-update audit from an
immutable audit worktree.  It never overlaps an active anchor lane and stops
after producing the causal atlas/matrix plus the evidence-driven derivation
queue; mathematical candidate construction remains a separate research step.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TASK_NAME = "FINAL_UNSB_ROUTE1_AUDITOR"
ANCHOR_TERMINAL = {"PAUSED_PROXY_NOT_CALIBRATED", "ANCHOR_PHASE_COMPLETE"}
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"


def post_audit_terminal_state(anchor_status: str) -> str:
    if anchor_status == "PAUSED_PROXY_NOT_CALIBRATED":
        return "PHASE_C_COMPLETE_PROXY_ADJUDICATION_REQUIRED"
    if anchor_status == "ANCHOR_PHASE_COMPLETE":
        return "PHASE_C_COMPLETE_DERIVATION_REQUIRED"
    raise ValueError(f"unsupported terminal anchor status: {anchor_status}")


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
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_text(argv: list[str], *, cwd: Path, timeout: int = 180) -> str:
    result = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # type: ignore[attr-defined]
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def executor_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            owner_pid = int(json.loads(path.read_text(encoding="utf-8")).get("pid", -1))
        except Exception:
            owner_pid = -1
        if process_exists(owner_pid):
            raise RuntimeError(f"audit executor already owns lock with PID {owner_pid}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": now()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if path.exists():
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                owner = {}
            if int(owner.get("pid", -1)) == os.getpid():
                path.unlink()


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    audit_head = run_text(["git", "rev-parse", "HEAD"], cwd=args.audit_repo)
    script = Path(__file__).resolve()
    return {
        "schema": "final-unsb-route1-audit-executor-contract-v1",
        "created": now(),
        "task_name": TASK_NAME,
        "main_repo": str(args.main_repo.resolve()),
        "audit_repo": str(args.audit_repo.resolve()),
        "training_repo": str(args.training_repo.resolve()),
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "audit_git_commit": audit_head,
        "supervisor_script": str(script),
        "supervisor_sha256": file_sha256(script),
        "manifest_sha256": EXPECTED_MANIFEST,
        "poll_seconds": 30,
        "child_stall_seconds": 2700,
        "maximum_job_failures": 3,
        "horizons": [1, 8, 32, 200],
        "label_horizons": [200],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != "final-unsb-route1-audit-executor-contract-v1":
        raise RuntimeError("audit executor contract schema mismatch")
    required = {
        "audit_repo", "training_repo", "run_root", "train_view", "data_root",
        "manifest", "python", "audit_git_commit", "supervisor_script",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise RuntimeError(f"audit contract missing fields: {missing}")
    if contract.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("audit contract manifest identity mismatch")
    if file_sha256(Path(contract["manifest"])) != EXPECTED_MANIFEST:
        raise RuntimeError("audit manifest changed")
    if file_sha256(Path(contract["supervisor_script"])) != contract.get("supervisor_sha256"):
        raise RuntimeError("audit supervisor changed after contract initialization")
    if contract.get("paired_controller_access") is not False:
        raise RuntimeError("paired controller access is forbidden")
    if contract.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 lock violated")


def verify_audit_worktree(contract: dict[str, Any]) -> dict[str, str]:
    repo = Path(contract["audit_repo"])
    head = run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != contract["audit_git_commit"]:
        raise RuntimeError(f"audit worktree moved to {head}")
    dirty = run_text(["git", "status", "--porcelain"], cwd=repo)
    if dirty:
        raise RuntimeError(f"audit worktree is dirty:\n{dirty}")
    python = Path(contract["python"])
    training = json.dumps(str(Path(contract["training_repo"]).resolve()))
    code = (
        "from pathlib import Path; "
        "from research.local_route1.causal_audit import audit_identity; "
        f"import json; print(json.dumps(audit_identity(Path({training})), sort_keys=True))"
    )
    identity = json.loads(run_text([str(python), "-c", code], cwd=repo, timeout=300))
    return {
        "audit_git_commit": head,
        "training_core_fingerprint": identity["training_core_fingerprint"],
        "audit_source_fingerprint": identity["audit_source_fingerprint"],
    }


class AuditExecutor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        validate_contract(self.contract)
        self.identity = verify_audit_worktree(self.contract)
        self.repo = Path(self.contract["audit_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.events = self.operations / "AUDIT_EXECUTOR_EVENTS.jsonl"
        self.state_path = self.operations / "AUDIT_EXECUTION_STATE.json"
        self.python = Path(self.contract["python"])

    def event(self, event: str, **fields: Any) -> None:
        append_jsonl(self.events, {
            "schema": "final-unsb-route1-audit-executor-event-v1",
            "time": now(), "event": event, "executor_pid": os.getpid(),
            **self.identity, "confirmation20_opened": False, **fields,
        })

    def state(self, status: str, **fields: Any) -> None:
        atomic_json(self.state_path, {
            "schema": "final-unsb-route1-audit-execution-state-v1",
            "updated": now(), "status": status, "executor_pid": os.getpid(),
            **self.identity, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def anchor_state(self) -> dict[str, Any]:
        path = self.operations / "EXECUTION_STATE.json"
        if not path.is_file():
            return {"status": "ANCHOR_EXECUTOR_NOT_INITIALIZED"}
        return json.loads(path.read_text(encoding="utf-8"))

    def wait_for_anchors(self) -> dict[str, Any]:
        last_status = None
        while True:
            anchor = self.anchor_state()
            status = anchor.get("status")
            if status in ANCHOR_TERMINAL:
                self.event("ANCHOR_TERMINAL_OBSERVED", anchor_status=status)
                return anchor
            if status == "FAILED":
                raise RuntimeError("anchor executor failed; audit may not overtake it")
            if status != last_status:
                self.event("WAITING_FOR_ANCHORS", anchor_status=status)
                last_status = status
            self.state("WAITING_FOR_ANCHORS", anchor_status=status)
            time.sleep(int(self.contract["poll_seconds"]))

    def _base_command(self) -> list[str]:
        return [
            str(self.python), "-m", "research.local_route1.run",
            "--output", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--training-worktree", self.contract["training_repo"],
            "--gpu", "0",
        ]

    def run_short(self, stage: str, *, log_name: str) -> subprocess.CompletedProcess:
        log = self.operations / "logs" / log_name
        log.parent.mkdir(parents=True, exist_ok=True)
        argv = self._base_command() + ["--stage", stage]
        with log.open("w", encoding="utf-8") as handle:
            return subprocess.run(argv, cwd=self.repo, stdout=handle, stderr=subprocess.STDOUT, check=False)

    @staticmethod
    def _job_key(probe: str, epoch: int) -> str:
        return f"{probe}:e{int(epoch):03d}"

    def _failures_path(self) -> Path:
        return self.operations / "AUDIT_FAILURE_COUNTS.json"

    def failure_count(self, key: str, *, success: bool) -> int:
        path = self._failures_path()
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        payload[key] = 0 if success else int(payload.get(key, 0)) + 1
        atomic_json(path, payload)
        return int(payload[key])

    def run_job(self, probe: str, epoch: int) -> None:
        key = self._job_key(probe, epoch)
        while True:
            attempts = json.loads(self._failures_path().read_text(encoding="utf-8")) if self._failures_path().is_file() else {}
            attempt = int(attempts.get(key, 0)) + 1
            log_root = self.operations / "logs"
            log_root.mkdir(parents=True, exist_ok=True)
            stdout = log_root / f"audit_{probe}_e{epoch:03d}_a{attempt}.stdout.log"
            stderr = log_root / f"audit_{probe}_e{epoch:03d}_a{attempt}.stderr.log"
            argv = self._base_command() + [
                "--stage", "audit", "--audit-probe", probe,
                "--audit-epoch", str(epoch),
                "--audit-horizons", ",".join(str(value) for value in self.contract["horizons"]),
                "--audit-label-horizons", ",".join(str(value) for value in self.contract["label_horizons"]),
            ]
            with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
                process = subprocess.Popen(argv, cwd=self.repo, stdout=out, stderr=err)
            self.event("AUDIT_JOB_START", probe=probe, data_epoch=epoch, child_pid=process.pid, attempt=attempt)
            last_progress = time.time()
            atlas = self.run_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
            last_mtime = atlas.stat().st_mtime if atlas.is_file() else 0.0
            while process.poll() is None:
                current_mtime = atlas.stat().st_mtime if atlas.is_file() else 0.0
                if current_mtime > last_mtime:
                    last_mtime = current_mtime
                    last_progress = time.time()
                self.state(
                    "AUDIT_JOB_RUNNING", probe=probe, data_epoch=epoch,
                    child_pid=process.pid, seconds_since_row_progress=time.time() - last_progress,
                )
                if time.time() - last_progress > int(self.contract["child_stall_seconds"]):
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(30)
            returncode = int(process.wait())
            success = returncode == 0
            count = self.failure_count(key, success=success)
            self.event(
                "AUDIT_JOB_COMPLETE" if success else "AUDIT_JOB_FAILED",
                probe=probe, data_epoch=epoch, child_pid=process.pid,
                exit_code=returncode, attempt=attempt, failure_count=count,
                stdout=str(stdout), stderr=str(stderr),
            )
            if success:
                return
            if count >= int(self.contract["maximum_job_failures"]):
                raise RuntimeError(f"audit job failed {count} times: {key}")

    def prepare_queue(self) -> list[tuple[str, int]]:
        result = self.run_short("audit", log_name="audit_prepare.log")
        if result.returncode not in (0, 4):
            raise RuntimeError("audit queue preparation failed")
        path = self.run_root / "audit" / "AUDIT_QUEUE.json"
        if not path.is_file():
            raise RuntimeError("audit queue was not written")
        queue = json.loads(path.read_text(encoding="utf-8"))
        jobs = sorted({(str(row["probe"]), int(row["data_epoch"])) for row in queue.get("jobs", [])})
        if not jobs:
            raise RuntimeError(f"audit queue has no runnable jobs: {queue.get('status')}")
        self.event("AUDIT_QUEUE_READY", jobs=len(jobs), queue_status=queue.get("status"))
        return jobs

    def disable_task(self, reason: str) -> None:
        if os.name != "nt":
            return
        result = subprocess.run(
            ["schtasks.exe", "/Change", "/TN", TASK_NAME, "/Disable"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )
        self.event("SCHEDULED_TASK_DISABLE", reason=reason, exit_code=result.returncode)

    def run(self) -> int:
        self.event("AUDIT_EXECUTOR_START", contract=str(self.contract_path))
        anchor = self.wait_for_anchors()
        jobs = self.prepare_queue()
        self.event("AUDIT_EXECUTION_START", anchor_status=anchor.get("status"), jobs=len(jobs))
        for index, (probe, epoch) in enumerate(jobs, 1):
            self.state("AUDIT_QUEUE_RUNNING", job_index=index, jobs=len(jobs), probe=probe, data_epoch=epoch)
            self.run_job(probe, epoch)
        matrix_path = self.run_root / "audit" / "LONG_CAUSAL_MATRIX.json"
        matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
            raise RuntimeError("causal matrix did not reach COMPLETE_CAUSAL_AUDIT")
        terminal = post_audit_terminal_state(str(anchor.get("status")))
        if terminal == "PHASE_C_COMPLETE_PROXY_ADJUDICATION_REQUIRED":
            self.state(
                terminal,
                atlas_rows=matrix.get("rows"),
                ranked_failure_mechanisms=matrix.get("ranked_failure_mechanisms", []),
                derivation_started=False,
                reason=(
                    "HJ/HNEK did not calibrate the proxy. Diagnose lineage/proxy distortion "
                    "and obtain user adjudication before DT or candidate generation."
                ),
            )
            self.event(
                "AUDIT_EXECUTOR_PROXY_ADJUDICATION_REQUIRED",
                atlas_rows=matrix.get("rows"), derivation_started=False,
            )
            self.disable_task("phase_c_complete_proxy_adjudication_required")
            return 0
        derive = self.run_short("derive", log_name="audit_derive_queue.log")
        if derive.returncode != 0:
            raise RuntimeError("completed causal matrix did not produce a derivation queue")
        self.state(
            terminal,
            atlas_rows=matrix.get("rows"),
            ranked_failure_mechanisms=matrix.get("ranked_failure_mechanisms", []),
            derivation_started=True,
        )
        self.event("AUDIT_EXECUTOR_COMPLETE", atlas_rows=matrix.get("rows"))
        self.disable_task("phase_c_complete_derivation_required")
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--main-repo", type=Path)
    value.add_argument("--audit-repo", type=Path)
    value.add_argument("--training-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "contract", "main_repo", "audit_repo", "training_repo", "run_root",
            "train_view", "data_root", "manifest", "python",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    if args.contract is None:
        raise SystemExit("--contract is required")
    contract_path = args.contract.resolve()
    try:
        with executor_lock(contract_path.with_suffix(".lock")):
            return AuditExecutor(contract_path).run()
    except Exception as exc:
        contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
        run_root = Path(contract.get("run_root", contract_path.parent.parent))
        failure = {
            "schema": "final-unsb-route1-audit-executor-fatal-v1",
            "time": now(), "status": "FAILED", "exception_type": type(exc).__name__,
            "exception": str(exc), "traceback": traceback.format_exc(),
            "pid": os.getpid(), "confirmation20_opened": False,
        }
        atomic_json(run_root / "operations" / "AUDIT_EXECUTION_STATE.json", failure)
        append_jsonl(run_root / "operations" / "AUDIT_EXECUTOR_EVENTS.jsonl", failure)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
