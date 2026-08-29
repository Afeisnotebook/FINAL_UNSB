"""Durable matched seed-validation executor for a frozen route-1 algorithm.

This supervisor runs one seed-specific plain trajectory to e200 and only then
runs the unchanged algorithm to e200 from the exact same e0.  It contains no
candidate selection, paired-metric early stop, or algorithm modification path.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from operations import local_route1_candidate_executor as support
except ModuleNotFoundError:  # direct ``python operations/...py`` execution
    import local_route1_candidate_executor as support  # type: ignore[no-redef]


EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
ALLOWED_SEEDS = (2027, 2028)
CONTRACT_SCHEMA = "final-unsb-route1-seed-executor-contract-v1"


def validation_status_command(contract: dict[str, Any]) -> list[str]:
    return [
        str(Path(contract["python"])), "-m", "research.local_route1.run",
        "--stage", "seed_validate", "--validation-action", "status",
        "--candidate-id", contract["candidate_id"],
        "--validation-seed", str(int(contract["validation_seed"])),
        "--output", str(Path(contract["run_root"])),
        "--train-view", str(Path(contract["train_view"])),
        "--data-root", str(Path(contract["data_root"])),
        "--manifest", str(Path(contract["manifest"])),
        "--gpu", "0",
    ]


def validation_train_command(
    contract: dict[str, Any], lane: str, target_epoch: int,
) -> list[str]:
    if lane not in ("plain", "candidate"):
        raise ValueError(f"invalid seed-validation lane: {lane}")
    return [
        str(Path(contract["python"])), "-m", "research.local_route1.run",
        "--stage", "seed_validate", "--validation-action", "train",
        "--candidate-id", contract["candidate_id"],
        "--validation-seed", str(int(contract["validation_seed"])),
        "--validation-lane", lane, "--resume",
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
        raise RuntimeError("seed validation status did not emit canonical JSON") from error
    if payload.get("status") != "READY_FOR_FROZEN_SEED_VALIDATION":
        raise RuntimeError(f"seed validation is not ready: {payload.get('status')}")
    if not payload.get("algorithm_fingerprint"):
        raise RuntimeError("seed validation status has no algorithm fingerprint")
    if not payload.get("seed_freeze_sha256"):
        raise RuntimeError("seed validation status has no freeze hash")
    if payload.get("paired_controller_access") is not False:
        raise RuntimeError("seed validation status permits paired controller access")
    if payload.get("confirmation20_opened") is not False:
        raise RuntimeError("seed validation status does not lock confirmation20")
    return payload


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    candidate_id = support.safe_candidate_id(args.candidate_id)
    validation_seed = int(args.validation_seed)
    if validation_seed not in ALLOWED_SEEDS:
        raise ValueError(f"seed validation is restricted to {ALLOWED_SEEDS}")
    seed_repo = args.seed_repo.resolve()
    head = support.run_text(["git", "rev-parse", "HEAD"], cwd=seed_repo)
    dirty = support.run_text(["git", "status", "--porcelain"], cwd=seed_repo)
    if dirty:
        raise RuntimeError("seed-validation worktree must be clean before freeze")
    script = Path(__file__).resolve()
    support_script = Path(support.__file__).resolve()
    provisional = {
        "python": str(args.python.resolve()),
        "candidate_id": candidate_id,
        "validation_seed": validation_seed,
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
    }
    status = _parse_status(support.run_text(
        validation_status_command(provisional), cwd=seed_repo, timeout=600,
    ))
    return {
        "schema": CONTRACT_SCHEMA,
        "created": support.now(),
        "main_repo": str(args.main_repo.resolve()),
        "seed_repo": str(seed_repo),
        "seed_git_commit": head,
        "candidate_id": candidate_id,
        "validation_seed": validation_seed,
        "algorithm_fingerprint": status["algorithm_fingerprint"],
        "seed2026_candidate_fingerprint": status["seed2026_candidate_fingerprint"],
        "seed_freeze_sha256": status["seed_freeze_sha256"],
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": EXPECTED_MANIFEST,
        "python": str(args.python.resolve()),
        "supervisor_script": str(script),
        "supervisor_sha256": support.file_sha256(script),
        "support_script": str(support_script),
        "support_sha256": support.file_sha256(support_script),
        "lane_order": ["plain", "candidate"],
        "chunk_data_epochs_max": 5,
        "target_data_epochs": 200,
        "stall_seconds": 900,
        "maximum_same_chunk_failures": 3,
        "algorithm_change_after_seed2026": False,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise RuntimeError("seed executor contract schema mismatch")
    support.safe_candidate_id(contract.get("candidate_id", ""))
    required = {
        "seed_repo", "seed_git_commit", "candidate_id", "validation_seed",
        "algorithm_fingerprint", "seed2026_candidate_fingerprint",
        "seed_freeze_sha256", "run_root", "train_view", "data_root",
        "manifest", "python", "supervisor_script", "support_script",
    }
    missing = sorted(required - set(contract))
    if missing:
        raise RuntimeError(f"seed executor contract missing fields: {missing}")
    if int(contract["validation_seed"]) not in ALLOWED_SEEDS:
        raise RuntimeError("seed executor uses an unauthorized seed")
    if contract.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("seed executor manifest identity mismatch")
    if support.file_sha256(Path(contract["manifest"])) != EXPECTED_MANIFEST:
        raise RuntimeError("seed executor manifest changed")
    if support.file_sha256(Path(contract["supervisor_script"])) != contract.get(
        "supervisor_sha256"
    ):
        raise RuntimeError("seed executor supervisor changed after contract freeze")
    if support.file_sha256(Path(contract["support_script"])) != contract.get("support_sha256"):
        raise RuntimeError("seed executor support code changed after contract freeze")
    if contract.get("lane_order") != ["plain", "candidate"]:
        raise RuntimeError("seed validation must run matched plain before candidate")
    if int(contract.get("chunk_data_epochs_max", 0)) > 5:
        raise RuntimeError("seed-validation chunks may not exceed five data epochs")
    if int(contract.get("target_data_epochs", 0)) != 200:
        raise RuntimeError("seed-validation target must be exactly 200 data epochs")
    for key in (
        "algorithm_change_after_seed2026", "paired_metric_early_stop",
        "paired_controller_access", "confirmation20_opened",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"seed executor requires {key}=false")


def scientific_identity(contract: dict[str, Any]) -> dict[str, Any]:
    repo = Path(contract["seed_repo"])
    head = support.run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != contract["seed_git_commit"]:
        raise RuntimeError(f"seed-validation worktree moved to {head}")
    dirty = support.run_text(["git", "status", "--porcelain"], cwd=repo)
    if dirty:
        raise RuntimeError(f"seed-validation worktree is dirty:\n{dirty}")
    status = _parse_status(support.run_text(
        validation_status_command(contract), cwd=repo, timeout=600,
    ))
    for key in (
        "algorithm_fingerprint", "seed2026_candidate_fingerprint",
        "seed_freeze_sha256",
    ):
        if status[key] != contract[key]:
            raise RuntimeError(f"seed validation {key} changed after contract freeze")
    return {
        "candidate_id": contract["candidate_id"],
        "validation_seed": int(contract["validation_seed"]),
        "seed_git_commit": head,
        "algorithm_fingerprint": status["algorithm_fingerprint"],
        "seed_freeze_sha256": status["seed_freeze_sha256"],
        "manifest_sha256": EXPECTED_MANIFEST,
    }


def latest_sidecar(run_root: Path, seed: int, lane: str) -> dict[str, Any] | None:
    path = (
        Path(run_root) / "seed_validation" / f"seed{int(seed)}" / lane /
        "full_state_latest.pt.json"
    )
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def current_epoch(run_root: Path, seed: int, lane: str) -> int:
    sidecar = latest_sidecar(run_root, seed, lane)
    return int(sidecar["physical_epoch_completed"]) if sidecar else 0


class SeedValidationExecutor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = json.loads(self.contract_path.read_text(encoding="utf-8"))
        validate_contract(self.contract)
        self.identity = scientific_identity(self.contract)
        self.repo = Path(self.contract["seed_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.seed = int(self.contract["validation_seed"])
        self.candidate_id = self.contract["candidate_id"]
        self.operations = self.run_root / "operations"
        suffix = f"{self.candidate_id}_s{self.seed}"
        self.events = self.operations / f"SEED_EXECUTOR_EVENTS_{suffix}.jsonl"
        self.state_path = self.operations / f"SEED_EXECUTION_STATE_{suffix}.json"
        self.failures_path = self.operations / f"SEED_FAILURE_COUNTS_{suffix}.json"

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events, {
            "schema": "final-unsb-route1-seed-executor-event-v1",
            "time": support.now(), "event": event, "executor_pid": os.getpid(),
            **self.identity, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-seed-execution-state-v1",
            "updated": support.now(), "status": status, "executor_pid": os.getpid(),
            **self.identity, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def failure_counts(self) -> dict[str, int]:
        if not self.failures_path.is_file():
            return {}
        return {str(key): int(value) for key, value in json.loads(
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
        support.atomic_json(self.failures_path, values)
        return count

    def run_chunk(self, lane: str, target_epoch: int) -> None:
        start_epoch = current_epoch(self.run_root, self.seed, lane)
        if not (start_epoch < target_epoch <= min(200, start_epoch + 5)):
            raise RuntimeError(f"invalid seed-validation chunk {lane} {start_epoch}->{target_epoch}")
        version = str(self.contract["supervisor_sha256"])[:12]
        key = f"{version}:{self.candidate_id}:s{self.seed}:{lane}:{start_epoch}:{target_epoch}"
        attempt = self.failure_counts().get(key, 0) + 1
        stem = (
            f"seed_{self.candidate_id}_s{self.seed}_{lane}_"
            f"e{start_epoch:03d}_to_e{target_epoch:03d}_a{attempt}"
        )
        stdout = self.operations / "logs" / f"{stem}.stdout.log"
        stderr = self.operations / "logs" / f"{stem}.stderr.log"
        stdout.parent.mkdir(parents=True, exist_ok=True)
        argv = validation_train_command(self.contract, lane, target_epoch)
        started = time.time()
        with stdout.open("w", encoding="utf-8") as out, stderr.open(
            "w", encoding="utf-8"
        ) as err:
            process = subprocess.Popen(argv, cwd=self.repo, stdout=out, stderr=err)
            self.event(
                "SEED_CHUNK_START", lane=lane, start_data_epoch=start_epoch,
                target_data_epoch=target_epoch, child_pid=process.pid,
                attempt=attempt, stdout=str(stdout), stderr=str(stderr),
            )
            reason = "PROCESS_EXIT"
            while process.poll() is None:
                heartbeat = (
                    self.run_root / "seed_validation" / f"seed{self.seed}" / lane /
                    "HEARTBEAT.json"
                )
                latest_activity = max(
                    started, heartbeat.stat().st_mtime if heartbeat.is_file() else 0.0,
                )
                idle = time.time() - latest_activity
                self.state(
                    "SEED_CHUNK_RUNNING", lane=lane,
                    start_data_epoch=start_epoch, target_data_epoch=target_epoch,
                    current_data_epoch=current_epoch(self.run_root, self.seed, lane),
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
        output_epoch = current_epoch(self.run_root, self.seed, lane)
        success = returncode == 0 and output_epoch == target_epoch
        count = self.set_failure(key, success=success)
        self.event(
            "SEED_CHUNK_COMPLETE" if success else "SEED_CHUNK_FAILED",
            lane=lane, start_data_epoch=start_epoch, target_data_epoch=target_epoch,
            final_data_epoch=output_epoch, child_pid=process.pid,
            exit_code=returncode, reason=reason, attempt=attempt,
            same_chunk_failure_count=count, wall_seconds=time.time() - started,
            stdout=str(stdout), stderr=str(stderr),
        )
        if not success:
            if count >= int(self.contract["maximum_same_chunk_failures"]):
                raise RuntimeError(f"seed-validation chunk failed {count} times: {key}")
            raise ChildProcessError(f"retryable seed-validation failure: {key}")

    def recover_boundary(self, lane: str) -> None:
        epoch = current_epoch(self.run_root, self.seed, lane)
        if epoch <= 0:
            return
        result = subprocess.run(
            validation_train_command(self.contract, lane, epoch), cwd=self.repo,
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"seed boundary recovery failed for {lane} e{epoch}:\n"
                f"{result.stdout}\n{result.stderr}"
            )

    def run(self) -> int:
        self.event("SEED_EXECUTOR_START", contract=str(self.contract_path))
        for lane in ("plain", "candidate"):
            if lane == "candidate" and current_epoch(self.run_root, self.seed, "plain") != 200:
                raise RuntimeError("candidate seed validation cannot precede matched plain e200")
            while current_epoch(self.run_root, self.seed, lane) < 200:
                start = current_epoch(self.run_root, self.seed, lane)
                target = min(200, start + int(self.contract["chunk_data_epochs_max"]))
                try:
                    self.run_chunk(lane, target)
                except ChildProcessError:
                    continue
            self.recover_boundary(lane)
        summary = (
            self.run_root / "seed_validation" / f"seed{self.seed}" /
            "SEED_VALIDATION_SUMMARY.json"
        )
        if not summary.is_file():
            raise RuntimeError("seed validation reached e200 without a summary")
        result = json.loads(summary.read_text(encoding="utf-8"))
        if result.get("status") != "COMPLETE":
            raise RuntimeError("seed validation summary is incomplete at e200")
        self.state(
            "SEED_VALIDATION_COMPLETE", plain_data_epoch=200,
            candidate_data_epoch=200, late_sign=result.get("late_sign"),
            summary=str(summary),
        )
        self.event(
            "SEED_EXECUTOR_COMPLETE", plain_data_epoch=200,
            candidate_data_epoch=200, late_sign=result.get("late_sign"),
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--main-repo", type=Path)
    value.add_argument("--seed-repo", type=Path)
    value.add_argument("--candidate-id")
    value.add_argument("--validation-seed", type=int, choices=ALLOWED_SEEDS)
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
            "contract", "main_repo", "seed_repo", "candidate_id",
            "validation_seed", "run_root", "train_view", "data_root",
            "manifest", "python",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    if args.contract is None:
        raise SystemExit("--contract is required")
    contract_path = args.contract.resolve()
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        validate_contract(contract)
        run_root = Path(contract["run_root"])
        lock = (
            run_root / "operations" /
            f"SEED_EXECUTOR_{contract['candidate_id']}_s{contract['validation_seed']}.lock"
        )
        with support.executor_lock(lock):
            return SeedValidationExecutor(contract_path).run()
    except Exception as error:
        try:
            if "contract" in locals():
                run_root = Path(contract["run_root"])
                suffix = f"{contract.get('candidate_id', 'unknown')}_s{contract.get('validation_seed', 'x')}"
                support.atomic_json(
                    run_root / "operations" / f"SEED_EXECUTION_STATE_{suffix}.json",
                    {
                        "schema": "final-unsb-route1-seed-execution-state-v1",
                        "updated": support.now(), "status": "FAILED",
                        "executor_pid": os.getpid(), "error": repr(error),
                        "traceback": traceback.format_exc(),
                        "paired_controller_access": False,
                        "confirmation20_opened": False,
                    },
                )
        finally:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
