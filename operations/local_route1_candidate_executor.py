"""Durable e200 executor for one evidence-frozen route-1 candidate.

The supervisor has no candidate-selection logic.  A candidate must already be
bound to a complete causal atlas, derivation card, source manifest and PASS
gate.  Once started, intermediate paired metrics cannot stop or alter it.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


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


def safe_candidate_id(value: str) -> str:
    candidate_id = str(value)
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate id: {candidate_id!r}")
    return candidate_id


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
            raise RuntimeError(f"candidate executor lock is owned by PID {owner_pid}")
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


def run_text(argv: list[str], *, cwd: Path, timeout: int = 300) -> str:
    result = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=timeout, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def candidate_status_command(contract: dict[str, Any]) -> list[str]:
    return [
        str(Path(contract["python"])), "-m", "research.local_route1.run",
        "--stage", "candidate", "--candidate-action", "status",
        "--candidate-id", contract["candidate_id"],
        "--output", str(Path(contract["run_root"])),
        "--train-view", str(Path(contract["train_view"])),
        "--data-root", str(Path(contract["data_root"])),
        "--manifest", str(Path(contract["manifest"])),
        "--gpu", "0",
    ]


def candidate_train_command(contract: dict[str, Any], target_epoch: int) -> list[str]:
    return [
        str(Path(contract["python"])), "-m", "research.local_route1.run",
        "--stage", "candidate", "--candidate-action", "train",
        "--candidate-id", contract["candidate_id"], "--resume",
        "--output", str(Path(contract["run_root"])),
        "--train-view", str(Path(contract["train_view"])),
        "--data-root", str(Path(contract["data_root"])),
        "--manifest", str(Path(contract["manifest"])),
        "--gpu", "0", "--engineering-stop-after-epoch", str(int(target_epoch)),
    ]


def _parse_status(text: str) -> dict[str, Any]:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as error:
        raise RuntimeError("candidate status command did not emit canonical JSON") from error
    if payload.get("status") != "READY_FOR_MATCHED_E200":
        raise RuntimeError(f"candidate is not ready for e200: {payload.get('status')}")
    if payload.get("paired_controller_access") is not False:
        raise RuntimeError("candidate status does not prove paired controller access is closed")
    if payload.get("confirmation20_opened") is not False:
        raise RuntimeError("candidate status does not prove confirmation20 is closed")
    if not payload.get("candidate_fingerprint"):
        raise RuntimeError("candidate status has no frozen fingerprint")
    return payload


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    candidate_id = safe_candidate_id(args.candidate_id)
    candidate_repo = args.candidate_repo.resolve()
    head = run_text(["git", "rev-parse", "HEAD"], cwd=candidate_repo)
    dirty = run_text(["git", "status", "--porcelain"], cwd=candidate_repo)
    if dirty:
        raise RuntimeError("candidate worktree must be clean before contract initialization")
    script = Path(__file__).resolve()
    provisional = {
        "python": str(args.python.resolve()),
        "candidate_id": candidate_id,
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    status = _parse_status(run_text(
        candidate_status_command(provisional), cwd=candidate_repo, timeout=600,
    ))
    return {
        "schema": "final-unsb-route1-candidate-executor-contract-v1",
        "created": now(),
        "main_repo": str(args.main_repo.resolve()),
        "candidate_repo": str(candidate_repo),
        "candidate_git_commit": head,
        "candidate_id": candidate_id,
        "candidate_fingerprint": status["candidate_fingerprint"],
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": EXPECTED_MANIFEST,
        "python": str(args.python.resolve()),
        "supervisor_script": str(script),
        "supervisor_sha256": file_sha256(script),
        "chunk_data_epochs_max": 5,
        "target_data_epochs": 200,
        "stall_seconds": 900,
        "maximum_same_chunk_failures": 3,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != "final-unsb-route1-candidate-executor-contract-v1":
        raise RuntimeError("candidate executor contract schema mismatch")
    safe_candidate_id(contract.get("candidate_id", ""))
    required = {
        "candidate_repo", "candidate_git_commit", "candidate_fingerprint", "run_root",
        "train_view", "data_root", "manifest", "python", "supervisor_script",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise RuntimeError(f"candidate executor contract missing fields: {missing}")
    if contract.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("candidate executor manifest identity mismatch")
    if file_sha256(Path(contract["manifest"])) != EXPECTED_MANIFEST:
        raise RuntimeError("candidate executor manifest changed")
    if file_sha256(Path(contract["supervisor_script"])) != contract.get("supervisor_sha256"):
        raise RuntimeError("candidate executor supervisor changed after contract initialization")
    if int(contract.get("chunk_data_epochs_max", 0)) > 5:
        raise RuntimeError("candidate chunks may not exceed five data epochs")
    if int(contract.get("target_data_epochs", 0)) != 200:
        raise RuntimeError("candidate target must be exactly 200 data epochs")
    for key in ("paired_metric_early_stop", "paired_controller_access", "confirmation20_opened"):
        if contract.get(key) is not False:
            raise RuntimeError(f"candidate contract requires {key}=false")


def scientific_identity(contract: dict[str, Any]) -> dict[str, str]:
    repo = Path(contract["candidate_repo"])
    head = run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != contract["candidate_git_commit"]:
        raise RuntimeError(f"candidate worktree moved to {head}")
    dirty = run_text(["git", "status", "--porcelain"], cwd=repo)
    if dirty:
        raise RuntimeError(f"candidate worktree is dirty:\n{dirty}")
    status = _parse_status(run_text(candidate_status_command(contract), cwd=repo, timeout=600))
    if status["candidate_fingerprint"] != contract["candidate_fingerprint"]:
        raise RuntimeError("candidate fingerprint changed after executor contract freeze")
    return {
        "candidate_id": contract["candidate_id"],
        "candidate_git_commit": head,
        "candidate_fingerprint": status["candidate_fingerprint"],
        "manifest_sha256": EXPECTED_MANIFEST,
    }


def latest_sidecar(run_root: Path, candidate_id: str) -> dict[str, Any] | None:
    path = run_root / "candidates" / candidate_id / "full_state_latest.pt.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def current_epoch(run_root: Path, candidate_id: str) -> int:
    sidecar = latest_sidecar(run_root, candidate_id)
    return int(sidecar["physical_epoch_completed"]) if sidecar else 0


class CandidateExecutor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        validate_contract(self.contract)
        self.identity = scientific_identity(self.contract)
        self.repo = Path(self.contract["candidate_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.candidate_id = self.contract["candidate_id"]
        self.operations = self.run_root / "operations"
        suffix = self.candidate_id
        self.events = self.operations / f"CANDIDATE_EXECUTOR_EVENTS_{suffix}.jsonl"
        self.state_path = self.operations / f"CANDIDATE_EXECUTION_STATE_{suffix}.json"
        self.failures_path = self.operations / f"CANDIDATE_FAILURE_COUNTS_{suffix}.json"

    def event(self, event: str, **fields: Any) -> None:
        append_jsonl(self.events, {
            "schema": "final-unsb-route1-candidate-executor-event-v1",
            "time": now(), "event": event, "executor_pid": os.getpid(),
            **self.identity, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def state(self, status: str, **fields: Any) -> None:
        atomic_json(self.state_path, {
            "schema": "final-unsb-route1-candidate-execution-state-v1",
            "updated": now(), "status": status, "executor_pid": os.getpid(),
            **self.identity, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def failure_counts(self) -> dict[str, int]:
        if not self.failures_path.is_file():
            return {}
        return {str(k): int(v) for k, v in json.loads(
            self.failures_path.read_text(encoding="utf-8")
        ).items()}

    def set_failure(self, key: str, *, success: bool) -> int:
        values = self.failure_counts()
        if success:
            values.pop(key, None)
            count = 0
        else:
            count = int(values.get(key, 0)) + 1
            values[key] = count
        atomic_json(self.failures_path, values)
        return count

    def run_chunk(self, target_epoch: int) -> None:
        start_epoch = current_epoch(self.run_root, self.candidate_id)
        if not (start_epoch < target_epoch <= min(200, start_epoch + 5)):
            raise RuntimeError(f"invalid candidate chunk {start_epoch}->{target_epoch}")
        version = str(self.contract["supervisor_sha256"])[:12]
        key = f"{version}:{self.candidate_id}:{start_epoch}:{target_epoch}"
        attempt = self.failure_counts().get(key, 0) + 1
        stem = f"candidate_{self.candidate_id}_e{start_epoch:03d}_to_e{target_epoch:03d}_a{attempt}"
        stdout = self.operations / "logs" / f"{stem}.stdout.log"
        stderr = self.operations / "logs" / f"{stem}.stderr.log"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        argv = candidate_train_command(self.contract, target_epoch)
        started = time.time()
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.Popen(argv, cwd=self.repo, stdout=out, stderr=err)
            self.event(
                "CANDIDATE_CHUNK_START", start_data_epoch=start_epoch,
                target_data_epoch=target_epoch, child_pid=process.pid, attempt=attempt,
                stdout=str(stdout), stderr=str(stderr),
            )
            reason = "PROCESS_EXIT"
            while process.poll() is None:
                heartbeat = self.run_root / "candidates" / self.candidate_id / "HEARTBEAT.json"
                latest_activity = max(started, heartbeat.stat().st_mtime if heartbeat.is_file() else 0.0)
                idle = time.time() - latest_activity
                self.state(
                    "CANDIDATE_CHUNK_RUNNING", start_data_epoch=start_epoch,
                    target_data_epoch=target_epoch,
                    current_data_epoch=current_epoch(self.run_root, self.candidate_id),
                    child_pid=process.pid, seconds_since_heartbeat=idle,
                )
                if idle > int(self.contract["stall_seconds"]):
                    reason = "STALL_TIMEOUT"
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    break
                time.sleep(15)
            returncode = int(process.wait())
        output_epoch = current_epoch(self.run_root, self.candidate_id)
        success = returncode == 0 and output_epoch == target_epoch
        count = self.set_failure(key, success=success)
        self.event(
            "CANDIDATE_CHUNK_COMPLETE" if success else "CANDIDATE_CHUNK_FAILED",
            start_data_epoch=start_epoch, target_data_epoch=target_epoch,
            final_data_epoch=output_epoch, child_pid=process.pid,
            exit_code=returncode, reason=reason, attempt=attempt,
            same_chunk_failure_count=count, wall_seconds=time.time() - started,
            stdout=str(stdout), stderr=str(stderr),
        )
        if not success:
            if count >= int(self.contract["maximum_same_chunk_failures"]):
                raise RuntimeError(f"candidate chunk failed {count} times: {key}")
            raise ChildProcessError(f"retryable candidate chunk failure: {key}")

    def recover_current_boundary(self) -> None:
        epoch = current_epoch(self.run_root, self.candidate_id)
        if epoch <= 0:
            return
        command = candidate_train_command(self.contract, epoch)
        # Calling the idempotent runner at its current boundary backfills a
        # missing milestone metric and, at e200, writes final trajectory state.
        result = subprocess.run(command, cwd=self.repo, capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"candidate boundary recovery failed at e{epoch}:\n{result.stdout}\n{result.stderr}"
            )

    def run(self) -> int:
        self.event("CANDIDATE_EXECUTOR_START", contract=str(self.contract_path))
        while current_epoch(self.run_root, self.candidate_id) < 200:
            start = current_epoch(self.run_root, self.candidate_id)
            target = min(200, start + int(self.contract["chunk_data_epochs_max"]))
            try:
                self.run_chunk(target)
            except ChildProcessError:
                # If a process died after an epoch boundary was committed, the
                # next loop advances from that state instead of replaying it.
                continue
        self.recover_current_boundary()
        trajectory = self.run_root / "candidates" / self.candidate_id / "CANDIDATE_TRAJECTORY.json"
        if not trajectory.is_file():
            raise RuntimeError("candidate reached e200 without a final trajectory record")
        result = json.loads(trajectory.read_text(encoding="utf-8"))
        self.state(
            "CANDIDATE_E200_COMPLETE_ADJUDICATION_REQUIRED",
            data_epoch=200, updates=30000,
            trajectory_status=result.get("status"), trajectory=str(trajectory),
        )
        self.event(
            "CANDIDATE_EXECUTOR_COMPLETE", data_epoch=200, updates=30000,
            trajectory_status=result.get("status"),
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--main-repo", type=Path)
    value.add_argument("--candidate-repo", type=Path)
    value.add_argument("--candidate-id")
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
            "contract", "main_repo", "candidate_repo", "candidate_id", "run_root",
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
            return CandidateExecutor(contract_path).run()
    except Exception as exc:
        contract = json.loads(contract_path.read_text(encoding="utf-8")) if contract_path.is_file() else {}
        run_root = Path(contract.get("run_root", contract_path.parent.parent))
        candidate_id = str(contract.get("candidate_id", "UNKNOWN"))
        failure = {
            "schema": "final-unsb-route1-candidate-executor-fatal-v1",
            "time": now(), "status": "FAILED",
            "candidate_id": candidate_id,
            "exception_type": type(exc).__name__, "exception": str(exc),
            "traceback": traceback.format_exc(), "pid": os.getpid(),
            "paired_controller_access": False, "confirmation20_opened": False,
        }
        atomic_json(
            run_root / "operations" / f"CANDIDATE_EXECUTOR_FATAL_{candidate_id}.json",
            failure,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
