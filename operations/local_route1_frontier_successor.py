"""Durably gate and execute the two user-authorized route-1 frontier candidates."""

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
except ModuleNotFoundError:
    import local_route1_candidate_executor as support  # type: ignore[no-redef]

from operations.local_route1_candidate_terminal_receipt import materialize_receipt
from research.local_route1.candidates import load_candidate_registration


SCHEMA = "final-unsb-route1-frontier-successor-contract-v1"
CANDIDATE_IDS = (
    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING",
    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER",
)
SOURCE_RELATIVES = (
    "operations/local_route1_frontier_successor.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "operations/local_route1_freeze_frontier_expansion.py",
    "research/local_route1/candidates.py",
    "research/local_route1/candidate_gate.py",
    "research/local_route1/candidate_runner.py",
    "research/local_route1/generation1_gates.py",
    "src/models/route1/pcnr.py",
    "src/models/route1/ammcrb.py",
    "src/models/route1_pcnr_model.py",
    "src/models/route1_ammcrb_model.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    candidate_repo = args.candidate_repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier successor worktree must be clean")
    if support.run_text(["git", "status", "--porcelain"], cwd=candidate_repo):
        raise RuntimeError("frontier candidate worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "candidate_repo": str(candidate_repo),
        "candidate_git_commit": support.run_text(
            ["git", "rev-parse", "HEAD"], cwd=candidate_repo
        ),
        "candidate_ids": list(CANDIDATE_IDS),
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "baseline_environment_record": str(args.baseline_environment_record.resolve()),
        "gate_poll_seconds": int(args.gate_poll_seconds),
        "gate_timeout_seconds": int(args.gate_timeout_seconds),
        "training_poll_seconds": int(args.training_poll_seconds),
        "training_timeout_seconds": int(args.training_timeout_seconds),
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parallel_e200_executors": 2,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("frontier successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("frontier successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier successor worktree is dirty")
    candidate_repo = Path(contract["candidate_repo"])
    if support.run_text(
        ["git", "rev-parse", "HEAD"], cwd=candidate_repo
    ) != contract.get("candidate_git_commit"):
        raise RuntimeError("frontier candidate worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=candidate_repo):
        raise RuntimeError("frontier candidate worktree is dirty")
    if tuple(contract.get("candidate_ids", [])) != CANDIDATE_IDS:
        raise RuntimeError("frontier successor candidate set changed")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"frontier successor source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("frontier successor manifest changed")
    if int(contract.get("batch_size", 0)) != 1:
        raise RuntimeError("frontier candidates must retain batch1")
    if int(contract.get("target_data_epochs", 0)) != 200:
        raise RuntimeError("frontier candidates must run true e200")
    if int(contract.get("maximum_parallel_e200_executors", 0)) != 2:
        raise RuntimeError("frontier successor must preserve the two-stream 5090 policy")
    if contract.get("selection_seeds") != [2026]:
        raise RuntimeError("frontier successor requires seed2026 only")
    if contract.get("deferred_seed_validation") != [2027, 2028]:
        raise RuntimeError("frontier deferred seed set changed")
    if int(contract.get("gate_poll_seconds", 0)) < 15:
        raise RuntimeError("frontier gate polling is too frequent")
    if int(contract.get("gate_timeout_seconds", 0)) < 3600:
        raise RuntimeError("frontier gate timeout is too short")
    if int(contract.get("training_poll_seconds", 0)) < 15:
        raise RuntimeError("frontier training polling is too frequent")
    if int(contract.get("training_timeout_seconds", 0)) < 43200:
        raise RuntimeError("frontier training timeout is too short")
    for key in ("paired_metric_scheduling", "paired_controller_access", "confirmation20_opened"):
        if contract.get(key) is not False:
            raise RuntimeError(f"frontier successor requires {key}=false")


class FrontierSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"])
        self.candidate_repo = Path(self.contract["candidate_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "FRONTIER_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "FRONTIER_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-frontier-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_ids": list(CANDIDATE_IDS),
            "batch_size": 1,
            "target_data_epochs": 200,
            "maximum_parallel_e200_executors": 2,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-frontier-successor-event-v1",
            "time": support.now(),
            "event": event,
            "supervisor_pid": os.getpid(),
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_for_gates(self) -> None:
        started = time.time()
        while True:
            ready = []
            pending = {}
            for candidate_id in CANDIDATE_IDS:
                try:
                    registration = load_candidate_registration(
                        self.run_root, candidate_id, require_gate=True,
                    )
                    ready.append(candidate_id)
                    pending[candidate_id] = registration.gate.get("status")
                except Exception as error:
                    pending[candidate_id] = repr(error)
            self.state(
                "WAITING_FOR_FRONTIER_GPU_GATES",
                gate_status=pending,
                ready_candidate_ids=ready,
                elapsed_seconds=time.time() - started,
            )
            if len(ready) == len(CANDIDATE_IDS):
                return
            if time.time() - started > int(self.contract["gate_timeout_seconds"]):
                raise TimeoutError("frontier successor timed out waiting for GPU gates")
            time.sleep(int(self.contract["gate_poll_seconds"]))

    def _run_checked(
        self, command: list[str], *, label: str, cwd: Path | None = None,
    ) -> None:
        run_cwd = self.repo if cwd is None else cwd
        result = subprocess.run(
            command, cwd=run_cwd, env=_env(run_cwd), capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")

    def _init_executor_contract(self, candidate_id: str) -> Path:
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
        if path.is_file():
            return path
        self._run_checked([
            self.contract["python"],
            str(self.candidate_repo / "operations/local_route1_candidate_executor.py"),
            "--init-contract", "--contract", str(path),
            "--main-repo", str(self.candidate_repo),
            "--candidate-repo", str(self.candidate_repo),
            "--candidate-id", candidate_id,
            "--run-root", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--python", self.contract["python"],
            "--baseline-environment-record", self.contract["baseline_environment_record"],
        ], label=f"initialize frontier executor {candidate_id}", cwd=self.candidate_repo)
        return path

    def run_e200(self) -> dict[str, Path]:
        processes: dict[str, subprocess.Popen] = {}
        streams = {}
        contracts = {}
        for candidate_id in CANDIDATE_IDS:
            contract = self._init_executor_contract(candidate_id)
            contracts[candidate_id] = contract
            stdout = (
                self.operations / f"FRONTIER_EXECUTOR_{candidate_id}.stdout.log"
            ).open("a", encoding="utf-8")
            stderr = (
                self.operations / f"FRONTIER_EXECUTOR_{candidate_id}.stderr.log"
            ).open("a", encoding="utf-8")
            streams[candidate_id] = (stdout, stderr)
            processes[candidate_id] = subprocess.Popen(
                [
                    self.contract["python"],
                    str(self.candidate_repo / "operations/local_route1_candidate_executor.py"),
                    "--contract", str(contract),
                ],
                cwd=self.candidate_repo,
                env=_env(self.candidate_repo),
                stdout=stdout,
                stderr=stderr,
            )
            self.event(
                "FRONTIER_E200_EXECUTOR_STARTED",
                candidate_id=candidate_id,
                child_pid=processes[candidate_id].pid,
                executor_contract=str(contract),
            )
        started = time.time()
        finished: dict[str, int] = {}
        try:
            while len(finished) < len(processes):
                for candidate_id, process in processes.items():
                    if candidate_id in finished:
                        continue
                    returncode = process.poll()
                    if returncode is not None:
                        finished[candidate_id] = int(returncode)
                        self.event(
                            "FRONTIER_E200_EXECUTOR_EXITED",
                            candidate_id=candidate_id,
                            exit_code=int(returncode),
                            data_epoch=support.current_epoch(self.run_root, candidate_id),
                        )
                self.state(
                    "FRONTIER_E200_RUNNING_TWO_STREAMS",
                    children={key: process.pid for key, process in processes.items()},
                    data_epochs={
                        key: support.current_epoch(self.run_root, key)
                        for key in CANDIDATE_IDS
                    },
                    exit_codes=dict(finished),
                    elapsed_seconds=time.time() - started,
                )
                if len(finished) == len(processes):
                    break
                if time.time() - started > int(self.contract["training_timeout_seconds"]):
                    raise TimeoutError("frontier e200 execution exceeded its durable timeout")
                time.sleep(int(self.contract["training_poll_seconds"]))
        finally:
            for stdout, stderr in streams.values():
                stdout.close()
                stderr.close()
        failures = {key: value for key, value in finished.items() if value != 0}
        if failures:
            raise RuntimeError(f"frontier e200 executor failures: {failures}")
        receipts = {}
        for candidate_id in CANDIDATE_IDS:
            path = self.operations / "terminal_receipts" / f"{candidate_id}.json"
            materialize_receipt(self.run_root, candidate_id, path)
            receipts[candidate_id] = path
        return receipts

    def run(self) -> int:
        self.event("FRONTIER_SUCCESSOR_START", contract=str(self.contract_path))
        self.wait_for_gates()
        self.event("FRONTIER_GPU_GATES_PASS", candidate_ids=list(CANDIDATE_IDS))
        receipts = self.run_e200()
        result = {
            "schema": "final-unsb-route1-frontier-e200-complete-v1",
            "status": "FRONTIER_E200_COMPLETE_ADJUDICATION_REQUIRED",
            "candidate_receipts": {
                candidate_id: {
                    "path": path.relative_to(self.run_root).as_posix(),
                    "sha256": support.file_sha256(path),
                }
                for candidate_id, path in receipts.items()
            },
            "selection_seeds": [2026],
            "cross_seed_stability_claimed": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        support.atomic_json(self.operations / "FRONTIER_E200_COMPLETE.json", result)
        self.state(
            "FRONTIER_E200_COMPLETE_ADJUDICATION_REQUIRED",
            completed_candidate_ids=list(CANDIDATE_IDS),
            result_sha256=support.file_sha256(
                self.operations / "FRONTIER_E200_COMPLETE.json"
            ),
        )
        self.event("FRONTIER_SUCCESSOR_COMPLETE", candidate_ids=list(CANDIDATE_IDS))
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--candidate-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--gate-poll-seconds", type=int, default=30)
    value.add_argument("--gate-timeout-seconds", type=int, default=7200)
    value.add_argument("--training-poll-seconds", type=int, default=30)
    value.add_argument("--training-timeout-seconds", type=int, default=172800)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "candidate_repo", "run_root", "train_view", "data_root", "manifest", "python",
            "baseline_environment_record",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    contract = _read_json(args.contract)
    run_root = Path(contract["run_root"])
    try:
        with support.executor_lock(
            run_root / "operations" / "FRONTIER_SUCCESSOR.lock"
        ):
            return FrontierSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(run_root / "operations" / "FRONTIER_SUCCESSOR_FATAL.json", {
            "schema": "final-unsb-route1-frontier-successor-fatal-v1",
            "time": support.now(),
            "status": "FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "supervisor_pid": os.getpid(),
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
