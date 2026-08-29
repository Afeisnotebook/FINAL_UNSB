"""Durable, chunked executor for the frozen FINAL_UNSB route-1 anchors.

The executor is intentionally operational code.  Scientific training always
runs from the immutable detached worktree recorded in EXECUTOR_CONTRACT.json.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TRAINING_COMMIT = "0da2a37086cca5bc4ad4488bb07c53096a7152ed"
EXPECTED_PROTOCOL = "b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2"
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
MILESTONES = (1, 5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200)
LANES = ("plain", "hj", "hnek")
TASK_NAME = "FINAL_UNSB_ROUTE1_EXECUTOR"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def run_text(argv: list[str], *, cwd: Path, timeout: int = 120) -> str:
    result = subprocess.run(
        argv,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        # os.kill(pid, 0) is not a harmless existence probe on Windows.
        query_limited_information = 0x1000
        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            query_limited_information, False, pid
        )
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
            owner = json.loads(path.read_text(encoding="utf-8"))
            owner_pid = int(owner.get("pid", -1))
        except Exception:
            owner_pid = -1
        if process_exists(owner_pid):
            raise RuntimeError(f"executor already owns lock with PID {owner_pid}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": utc_now()}, handle)
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
    script = Path(__file__).resolve()
    backfill = script.with_name("local_route1_metric_backfill.py")
    return {
        "schema": "final-unsb-route1-executor-contract-v1",
        "task_name": TASK_NAME,
        "created": utc_now(),
        "main_repo": str(args.main_repo.resolve()),
        "executor_repo": str(args.executor_repo.resolve()),
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "supervisor_script": str(script),
        "supervisor_sha256": file_sha256(script),
        "backfill_script": str(backfill),
        "backfill_sha256": file_sha256(backfill),
        "training_git_commit": EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "authorized_lanes_before_proxy": list(LANES),
        "dt_requires_calibrated_proxy": True,
        "chunk_data_epochs_max": 5,
        "stall_seconds": 900,
        "maximum_same_chunk_failures": 3,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != "final-unsb-route1-executor-contract-v1":
        raise RuntimeError("executor contract schema mismatch")
    required = {
        "executor_repo",
        "run_root",
        "train_view",
        "data_root",
        "manifest",
        "python",
        "supervisor_script",
        "backfill_script",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise RuntimeError(f"executor contract missing fields: {missing}")
    if contract.get("training_git_commit") != EXPECTED_TRAINING_COMMIT:
        raise RuntimeError("executor contract training commit mismatch")
    if contract.get("training_protocol_fingerprint") != EXPECTED_PROTOCOL:
        raise RuntimeError("executor contract protocol mismatch")
    if contract.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("executor contract manifest mismatch")
    if contract.get("confirmation20_opened") is not False:
        raise RuntimeError("executor contract confirmation lock violated")
    if file_sha256(Path(contract["supervisor_script"])) != contract["supervisor_sha256"]:
        raise RuntimeError("supervisor script changed after contract initialization")
    if file_sha256(Path(contract["backfill_script"])) != contract["backfill_sha256"]:
        raise RuntimeError("backfill script changed after contract initialization")


def scientific_identity(contract: dict[str, Any]) -> dict[str, str]:
    repo = Path(contract["executor_repo"])
    python = Path(contract["python"])
    head = run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != EXPECTED_TRAINING_COMMIT:
        raise RuntimeError(f"frozen executor moved to {head}")
    dirty = run_text(["git", "status", "--porcelain"], cwd=repo)
    if dirty:
        raise RuntimeError(f"frozen executor worktree is dirty:\n{dirty}")
    manifest_literal = json.dumps(str(Path(contract["manifest"]).resolve()))
    code = (
        "from pathlib import Path; "
        "from research.local_route1.protocol import protocol_fingerprint; "
        f"print(protocol_fingerprint(Path({manifest_literal})))"
    )
    fingerprint = run_text([str(python), "-c", code], cwd=repo, timeout=180)
    manifest_hash = file_sha256(Path(contract["manifest"]))
    if fingerprint != EXPECTED_PROTOCOL:
        raise RuntimeError(f"frozen protocol fingerprint changed: {fingerprint}")
    if manifest_hash != EXPECTED_MANIFEST:
        raise RuntimeError(f"frozen manifest changed: {manifest_hash}")
    return {
        "git_commit": head,
        "protocol_fingerprint": fingerprint,
        "manifest_sha256": manifest_hash,
    }


def latest_sidecar(run_root: Path, lane: str) -> dict[str, Any] | None:
    path = run_root / "anchors" / lane / "full_state_latest.pt.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def current_epoch(run_root: Path, lane: str) -> int:
    sidecar = latest_sidecar(run_root, lane)
    return int(sidecar["physical_epoch_completed"]) if sidecar else 0


def anchor_command(contract: dict[str, Any], lane: str, target_epoch: int) -> list[str]:
    return [
        str(Path(contract["python"])),
        "-m",
        "research.local_route1.run",
        "--stage",
        "anchors",
        "--lane",
        lane,
        "--resume",
        "--output",
        str(Path(contract["run_root"])),
        "--train-view",
        str(Path(contract["train_view"])),
        "--data-root",
        str(Path(contract["data_root"])),
        "--manifest",
        str(Path(contract["manifest"])),
        "--gpu",
        "0",
        "--engineering-stop-after-epoch",
        str(target_epoch),
    ]


def validate_lane_sidecar(sidecar: dict[str, Any], identity: dict[str, str], lane: str) -> None:
    metadata = sidecar.get("metadata", {})
    expected = {
        "probe_id": lane,
        "git_commit": identity["git_commit"],
        "protocol_fingerprint": identity["protocol_fingerprint"],
        "manifest_sha256": identity["manifest_sha256"],
        "confirmation20_opened": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"{lane} checkpoint metadata mismatch for {key}")


class DurableExecutor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        validate_contract(self.contract)
        self.repo = Path(self.contract["executor_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.python = Path(self.contract["python"])
        self.operations = self.run_root / "operations"
        self.events = self.operations / "EXECUTOR_EVENTS.jsonl"
        self.state_path = self.operations / "EXECUTION_STATE.json"
        self.failures_path = self.operations / "FAILURE_COUNTS.json"
        self.identity = scientific_identity(self.contract)

    def event(self, kind: str, **payload: Any) -> None:
        append_jsonl(
            self.events,
            {
                "schema": "final-unsb-route1-executor-event-v1",
                "time": utc_now(),
                "event": kind,
                "executor_pid": os.getpid(),
                **self.identity,
                "confirmation20_opened": False,
                **payload,
            },
        )

    def state(self, status: str, **payload: Any) -> None:
        atomic_json(
            self.state_path,
            {
                "schema": "final-unsb-route1-execution-state-v1",
                "updated": utc_now(),
                "status": status,
                "executor_pid": os.getpid(),
                **self.identity,
                "confirmation20_opened": False,
                **payload,
            },
        )

    def failure_counts(self) -> dict[str, int]:
        if not self.failures_path.is_file():
            return {}
        return {
            str(key): int(value)
            for key, value in json.loads(self.failures_path.read_text(encoding="utf-8")).items()
        }

    def set_failure(self, key: str, *, success: bool) -> int:
        values = self.failure_counts()
        if success:
            values.pop(key, None)
            count = 0
        else:
            count = values.get(key, 0) + 1
            values[key] = count
        atomic_json(self.failures_path, values)
        return count

    def recover_missing_metrics(self, lane: str) -> None:
        completed = current_epoch(self.run_root, lane)
        for epoch in MILESTONES:
            if epoch > completed:
                continue
            lane_root = self.run_root / "anchors" / lane
            checkpoint = lane_root / "milestones" / f"e{epoch:03d}.pt"
            metric = lane_root / "metrics" / f"e{epoch:03d}.json"
            if checkpoint.is_file() and not metric.is_file():
                self.event("MILESTONE_RECOVERY_START", lane=lane, data_epoch=epoch, updates=epoch * 150)
                argv = [
                    str(self.python),
                    self.contract["backfill_script"],
                    "--executor-repo",
                    str(self.repo),
                    "--run-root",
                    str(self.run_root),
                    "--train-view",
                    self.contract["train_view"],
                    "--data-root",
                    self.contract["data_root"],
                    "--manifest",
                    self.contract["manifest"],
                    "--lane",
                    lane,
                    "--epoch",
                    str(epoch),
                    "--gpu",
                    "0",
                ]
                log = self.operations / "logs" / f"recover_{lane}_e{epoch:03d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                result = subprocess.run(
                    argv,
                    cwd=self.repo,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                )
                log.write_text(result.stdout + "\n--- STDERR ---\n" + result.stderr, encoding="utf-8")
                if result.returncode != 0 or not metric.is_file():
                    raise RuntimeError(f"milestone recovery failed for {lane} e{epoch}; see {log}")
                self.event("MILESTONE_RECOVERY_COMPLETE", lane=lane, data_epoch=epoch, updates=epoch * 150)

    def _terminate(self, process: subprocess.Popen[Any]) -> None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)

    def run_chunk(self, lane: str, target_epoch: int) -> None:
        start_epoch = current_epoch(self.run_root, lane)
        if not (start_epoch < target_epoch <= min(200, start_epoch + 5)):
            raise RuntimeError(f"invalid chunk boundary {lane}: {start_epoch}->{target_epoch}")
        sidecar = latest_sidecar(self.run_root, lane)
        if sidecar:
            validate_lane_sidecar(sidecar, self.identity, lane)
        input_hash = sidecar.get("full_state_sha256") if sidecar else None
        executor_version = str(self.contract["supervisor_sha256"])[:12]
        key = f"{executor_version}:{lane}:{start_epoch}:{target_epoch}"
        attempt = self.failure_counts().get(key, 0) + 1
        log_stem = f"{lane}_e{start_epoch:03d}_to_e{target_epoch:03d}_a{attempt}"
        stdout_log = self.operations / "logs" / f"{log_stem}.stdout.log"
        stderr_log = self.operations / "logs" / f"{log_stem}.stderr.log"
        stdout_log.parent.mkdir(parents=True, exist_ok=True)
        argv = anchor_command(self.contract, lane, target_epoch)
        started = time.time()
        with (
            stdout_log.open("w", encoding="utf-8") as stdout_handle,
            stderr_log.open("w", encoding="utf-8") as stderr_handle,
        ):
            process = subprocess.Popen(
                argv,
                cwd=self.repo,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
            )
            self.event(
                "CHUNK_START",
                lane=lane,
                start_data_epoch=start_epoch,
                start_updates=start_epoch * 150,
                target_data_epoch=target_epoch,
                target_updates=target_epoch * 150,
                child_pid=process.pid,
                child_parent_pid=os.getpid(),
                attempt=attempt,
                input_checkpoint_sha256=input_hash,
                stdout_log=str(stdout_log),
                stderr_log=str(stderr_log),
            )
            reason = "PROCESS_EXIT"
            while process.poll() is None:
                heartbeat = self.run_root / "anchors" / lane / "HEARTBEAT.json"
                latest_activity = started
                if heartbeat.is_file():
                    latest_activity = max(latest_activity, heartbeat.stat().st_mtime)
                idle = time.time() - latest_activity
                self.state(
                    "CHUNK_RUNNING",
                    lane=lane,
                    start_data_epoch=start_epoch,
                    target_data_epoch=target_epoch,
                    current_data_epoch=current_epoch(self.run_root, lane),
                    child_pid=process.pid,
                    seconds_since_heartbeat=idle,
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                )
                if idle > int(self.contract["stall_seconds"]):
                    reason = "STALL_TIMEOUT"
                    self._terminate(process)
                    break
                time.sleep(15)
            returncode = int(process.wait())
        output = latest_sidecar(self.run_root, lane)
        output_epoch = int(output["physical_epoch_completed"]) if output else 0
        output_hash = output.get("full_state_sha256") if output else None
        success = returncode == 0 and output_epoch == target_epoch
        count = self.set_failure(key, success=success)
        self.event(
            "CHUNK_COMPLETE" if success else "CHUNK_FAILED",
            lane=lane,
            start_data_epoch=start_epoch,
            target_data_epoch=target_epoch,
            final_data_epoch=output_epoch,
            final_updates=output_epoch * 150,
            child_pid=process.pid,
            exit_code=returncode,
            reason=reason,
            attempt=attempt,
            same_chunk_failure_count=count,
            input_checkpoint_sha256=input_hash,
            output_checkpoint_sha256=output_hash,
            wall_seconds=time.time() - started,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
        )
        if not success:
            if count >= int(self.contract["maximum_same_chunk_failures"]):
                raise RuntimeError(f"same chunk failed {count} times: {key}")
            raise ChildProcessError(f"retryable chunk failure {key}, attempt {count}")

    def complete_lane(self, lane: str) -> None:
        while current_epoch(self.run_root, lane) < 200:
            self.recover_missing_metrics(lane)
            start = current_epoch(self.run_root, lane)
            target = min(200, start + int(self.contract["chunk_data_epochs_max"]))
            try:
                self.run_chunk(lane, target)
            except ChildProcessError:
                continue
        self.recover_missing_metrics(lane)
        sidecar = latest_sidecar(self.run_root, lane)
        if not sidecar or int(sidecar["physical_epoch_completed"]) != 200:
            raise RuntimeError(f"{lane} failed to reach e200")
        validate_lane_sidecar(sidecar, self.identity, lane)
        state = self.run_root / "anchors" / lane / "RUN_STATE.json"
        payload = json.loads(state.read_text(encoding="utf-8")) if state.is_file() else {}
        if payload.get("status") != "COMPLETE_E200" or int(payload.get("final_data_epoch", -1)) != 200:
            metric = self.run_root / "anchors" / lane / "metrics" / "e200.json"
            if not metric.is_file():
                raise RuntimeError(f"{lane} e200 completion cannot be recovered without e200 metric")
            payload = {
                "status": "COMPLETE_E200",
                "probe_id": lane,
                "start_updates": 30_000,
                "final_updates": 30_000,
                "final_data_epoch": 200,
                "target_updates": 30_000,
                "target_data_epochs": 200,
                "wall_seconds_this_call": 0.0,
                "metadata": sidecar["metadata"],
                "confirmation20_opened": False,
                "recovered_completion_after_process_loss": True,
            }
            atomic_json(state, payload)
            self.event(
                "RUN_STATE_RECOVERED",
                lane=lane,
                data_epoch=200,
                updates=30_000,
                source_checkpoint_sha256=sidecar["full_state_sha256"],
                metric=str(metric),
            )
        self.event("LANE_COMPLETE_E200", lane=lane, data_epoch=200, updates=30_000)

    def disable_scheduled_task(self, reason: str) -> None:
        if os.name != "nt":
            return
        result = subprocess.run(
            ["schtasks.exe", "/Change", "/TN", TASK_NAME, "/Disable"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        self.event(
            "SCHEDULED_TASK_DISABLE",
            reason=reason,
            exit_code=result.returncode,
            stdout=result.stdout.strip(),
            stderr=result.stderr.strip(),
        )

    def evaluate_proxy(self) -> dict[str, Any]:
        log = self.operations / "logs" / "proxy_evaluate.log"
        argv = [
            str(self.python),
            "-m",
            "research.local_route1.run",
            "--stage",
            "evaluate",
            "--output",
            str(self.run_root),
            "--manifest",
            self.contract["manifest"],
        ]
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run(argv, cwd=self.repo, stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"proxy evaluation failed; see {log}")
        path = self.run_root / "evidence" / "PROXY_CALIBRATION.json"
        if not path.is_file():
            raise RuntimeError("proxy evaluation did not produce PROXY_CALIBRATION.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.event("PROXY_EVALUATED", proxy_status=payload.get("status"), evidence=str(path))
        return payload

    def run(self) -> int:
        self.event("EXECUTOR_START", contract=str(self.contract_path))
        for lane in LANES:
            self.complete_lane(lane)
        calibration = self.evaluate_proxy()
        if calibration.get("status") != "CALIBRATED":
            self.state("PAUSED_PROXY_NOT_CALIBRATED", proxy_status=calibration.get("status"))
            self.event("EXECUTOR_PAUSED_PROXY", proxy_status=calibration.get("status"))
            self.disable_scheduled_task("proxy_requires_scientific_adjudication")
            return 0
        self.complete_lane("dt")
        self.evaluate_proxy()
        self.state("ANCHOR_PHASE_COMPLETE", lanes=["plain", "hj", "hnek", "dt"])
        self.event("EXECUTOR_COMPLETE", lanes=["plain", "hj", "hnek", "dt"])
        self.disable_scheduled_task("anchor_phase_complete")
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--main-repo", type=Path)
    value.add_argument("--executor-repo", type=Path)
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
            "main_repo",
            "executor_repo",
            "run_root",
            "train_view",
            "data_root",
            "manifest",
            "python",
            "contract",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        atomic_json(args.contract.resolve(), contract)
        validate_contract(contract)
        identity = scientific_identity(contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **identity}, indent=2))
        return 0
    if args.contract is None:
        raise SystemExit("--contract is required")
    contract_path = args.contract.resolve()
    operations = contract_path.parent
    try:
        with executor_lock(operations / "EXECUTOR.lock"):
            return DurableExecutor(contract_path).run()
    except Exception as exc:
        payload = {
            "schema": "final-unsb-route1-executor-fatal-v1",
            "time": utc_now(),
            "status": "FAILED",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "pid": os.getpid(),
            "confirmation20_opened": False,
        }
        atomic_json(operations / "EXECUTOR_FATAL.json", payload)
        append_jsonl(operations / "EXECUTOR_EVENTS.jsonl", payload)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
