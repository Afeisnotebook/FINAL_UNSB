"""Durably execute every evidence-qualified repaired-frontier ablation stream."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from operations.local_route1_candidate_terminal_receipt import materialize_receipt
from operations.local_route1_cross_version_adjudicate import _validate_receipt
from research.local_route1.protocol import file_sha256
from research.local_route1.repaired_frontier_followups import (
    SCHEMA as FOLLOWUP_SCHEMA,
    materialize_repaired_frontier_followups,
)


SCHEMA = "final-unsb-route1-repaired-followup-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_repaired_followup_successor.py",
    "operations/local_route1_repaired_frontier_followups.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/repaired_frontier_followups.py",
    "research/local_route1/repaired_frontier_adjudication.py",
    "research/local_route1/winner_ablations.py",
    "research/local_route1/generation1_gates.py",
    "src/models/route1/rfammcrb.py",
    "src/models/route1/rfammcrb_ablation.py",
    "src/models/route1/rfmcrb.py",
    "src/models/route1/rfmcrb_ablation.py",
    "src/models/route1_rfammcrb_ablation_model.py",
    "src/models/route1_rfmcrb_ablation_model.py",
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
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("repaired follow-up successor worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "baseline_environment_record": str(
            args.baseline_environment_record.resolve()
        ),
        "baseline_environment_record_sha256": support.file_sha256(
            args.baseline_environment_record.resolve()
        ),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parallel_parent_streams": 2,
        "maximum_active_candidates_per_parent_stream": 1,
        "within_parent_order": ["proposal_only", "observable_only"],
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "action_priority_is_not_an_exclusivity_rule": True,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("repaired follow-up successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("repaired follow-up successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("repaired follow-up successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"repaired follow-up source changed: {relative}")
    for key in ("manifest", "baseline_environment_record"):
        if support.file_sha256(Path(contract[key])) != contract.get(f"{key}_sha256"):
            raise RuntimeError(f"repaired follow-up {key} changed")
    fixed = {
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parallel_parent_streams": 2,
        "maximum_active_candidates_per_parent_stream": 1,
        "within_parent_order": ["proposal_only", "observable_only"],
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "action_priority_is_not_an_exclusivity_rule": True,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"repaired follow-up contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("repaired follow-up polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("repaired follow-up timeout is too short")


class RepairedFollowupSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "REPAIRED_FOLLOWUP_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-repaired-followup-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "batch_size": 1,
            "target_data_epochs": 200,
            "maximum_parallel_parent_streams": 2,
            "action_priority_is_not_an_exclusivity_rule": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_for_adjudication(self) -> Path:
        path = self.operations / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        fatal = self.operations / "REPAIRED_FRONTIER_SUCCESSOR_FATAL.json"
        while not path.is_file():
            if fatal.is_file():
                raise RuntimeError(f"repaired frontier successor failed: {fatal}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for repaired frontier adjudication")
            upstream = self.operations / "REPAIRED_FRONTIER_SUCCESSOR_STATE.json"
            value = _read_json(upstream) if upstream.is_file() else {}
            self.state(
                "WAITING_FOR_COMPLETE_REPAIRED_FRONTIER_E200",
                upstream_status=value.get("status"),
                upstream_repair_results_ready=value.get("repair_results_ready"),
            )
            time.sleep(int(self.contract["poll_seconds"]))
        return path

    def _worker_state(self, parent_id: str) -> dict[str, Any]:
        path = self.operations / f"REPAIRED_FOLLOWUP_WORKER_STATE_{parent_id}.json"
        return _read_json(path) if path.is_file() else {}

    def run_supervisor(self) -> int:
        adjudication_path = self.wait_for_adjudication()
        plan_path = self.operations / "REPAIRED_FRONTIER_FOLLOWUPS.json"
        plan = materialize_repaired_frontier_followups(
            self.run_root,
            adjudication_path=adjudication_path,
            output_path=plan_path,
        )
        if plan.get("schema") != FOLLOWUP_SCHEMA:
            raise RuntimeError("repaired follow-up plan schema mismatch")
        streams = list(plan["eligible_parent_streams"])
        if not streams:
            result = {
                "schema": "final-unsb-route1-repaired-followup-execution-v1",
                "status": "NO_EVIDENCE_QUALIFIED_REPAIRED_ABLATION_STREAM",
                "source_plan_sha256": file_sha256(plan_path),
                "parent_results": [],
                "algorithm_discovery_collapsed_to_single_candidate": False,
                "paired_metrics_used_for_formula_or_training_control": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
            result_path = self.operations / "REPAIRED_FOLLOWUP_EXECUTION_RESULT.json"
            support.atomic_json(result_path, result)
            self.state(result["status"], result_sha256=file_sha256(result_path))
            return 0

        processes = []
        for stream in streams:
            parent_id = str(stream["parent_candidate_id"])
            stdout_path = self.operations / f"REPAIRED_FOLLOWUP_WORKER_{parent_id}.stdout.log"
            stderr_path = self.operations / f"REPAIRED_FOLLOWUP_WORKER_{parent_id}.stderr.log"
            stdout = stdout_path.open("a", encoding="utf-8")
            stderr = stderr_path.open("a", encoding="utf-8")
            process = subprocess.Popen(
                [
                    self.contract["python"],
                    str(self.repo / "operations/local_route1_repaired_followup_successor.py"),
                    "--contract", str(self.contract_path),
                    "--worker-parent", parent_id,
                ],
                cwd=self.repo,
                env=_env(self.repo),
                stdout=stdout,
                stderr=stderr,
            )
            processes.append((parent_id, process, stdout, stderr))
        try:
            while any(process.poll() is None for _, process, _, _ in processes):
                if time.time() - self.started > int(self.contract["timeout_seconds"]):
                    raise TimeoutError("repaired follow-up streams exceeded durable timeout")
                self.state(
                    "REPAIRED_MULTI_PARENT_ABLATION_STREAMS_RUNNING",
                    workers={
                        parent_id: {
                            "pid": process.pid,
                            **self._worker_state(parent_id),
                        }
                        for parent_id, process, _, _ in processes
                    },
                )
                time.sleep(int(self.contract["poll_seconds"]))
        finally:
            for _, _, stdout, stderr in processes:
                stdout.close()
                stderr.close()
        failures = [
            parent_id for parent_id, process, _, _ in processes
            if int(process.wait()) != 0
        ]
        if failures:
            raise RuntimeError(f"repaired follow-up parent streams failed: {failures}")
        parent_results = []
        for stream in streams:
            parent_id = str(stream["parent_candidate_id"])
            state = self._worker_state(parent_id)
            if state.get("status") != "PARENT_ABLATION_STREAM_COMPLETE_E200":
                raise RuntimeError(f"repaired follow-up worker incomplete: {parent_id}")
            parent_results.append(state)
        result = {
            "schema": "final-unsb-route1-repaired-followup-execution-v1",
            "status": "ALL_EVIDENCE_QUALIFIED_REPAIRED_ABLATIONS_COMPLETE_E200",
            "source_plan_sha256": file_sha256(plan_path),
            "parent_results": parent_results,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "paired_metrics_used_for_formula_or_training_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        result_path = self.operations / "REPAIRED_FOLLOWUP_EXECUTION_RESULT.json"
        support.atomic_json(result_path, result)
        self.state(result["status"], result_sha256=file_sha256(result_path))
        return 0

    def _run_checked(self, command: list[str], *, label: str) -> None:
        result = subprocess.run(
            command,
            cwd=self.repo,
            env=_env(self.repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")

    def _init_executor_contract(self, candidate_id: str) -> Path:
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
        if path.is_file():
            return path
        self._run_checked([
            self.contract["python"],
            str(self.repo / "operations/local_route1_candidate_executor.py"),
            "--init-contract", "--contract", str(path),
            "--main-repo", str(self.repo),
            "--candidate-repo", str(self.repo),
            "--candidate-id", candidate_id,
            "--run-root", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--python", self.contract["python"],
            "--baseline-environment-record", self.contract[
                "baseline_environment_record"
            ],
        ], label=f"initialize repaired follow-up executor {candidate_id}")
        return path

    def _run_candidate(self, parent_id: str, candidate_id: str) -> Path:
        receipt_path = self.operations / "terminal_receipts" / f"{candidate_id}.json"
        if receipt_path.is_file() and Path(str(receipt_path) + ".sha256.json").is_file():
            receipt = _validate_receipt(receipt_path)
            if receipt.get("candidate_id") != candidate_id:
                raise RuntimeError("repaired follow-up receipt identity changed")
            return receipt_path
        self._run_checked([
            self.contract["python"], "-m", "research.local_route1.run",
            "--stage", "candidate", "--candidate-action", "gate",
            "--candidate-id", candidate_id,
            "--output", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--gpu", "0",
        ], label=f"repaired follow-up gate {candidate_id}")
        contract_path = self._init_executor_contract(candidate_id)
        stdout_path = self.operations / f"REPAIRED_FOLLOWUP_EXECUTOR_{candidate_id}.stdout.log"
        stderr_path = self.operations / f"REPAIRED_FOLLOWUP_EXECUTOR_{candidate_id}.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [
                    self.contract["python"],
                    str(self.repo / "operations/local_route1_candidate_executor.py"),
                    "--contract", str(contract_path),
                ],
                cwd=self.repo,
                env=_env(self.repo),
                stdout=stdout,
                stderr=stderr,
            )
            while process.poll() is None:
                self._write_worker_state(
                    parent_id,
                    "PARENT_ABLATION_CANDIDATE_E200_RUNNING",
                    active_candidate_id=candidate_id,
                    active_child_pid=process.pid,
                    active_data_epoch=support.current_epoch(
                        self.run_root, candidate_id,
                    ),
                )
                time.sleep(int(self.contract["poll_seconds"]))
            if int(process.wait()) != 0:
                raise RuntimeError(f"repaired follow-up executor failed: {candidate_id}")
        materialize_receipt(self.run_root, candidate_id, receipt_path)
        return receipt_path

    def _write_worker_state(
        self, parent_id: str, status: str, **fields: Any,
    ) -> None:
        support.atomic_json(
            self.operations / f"REPAIRED_FOLLOWUP_WORKER_STATE_{parent_id}.json",
            {
                "schema": "final-unsb-route1-repaired-followup-worker-state-v1",
                "updated": support.now(),
                "status": status,
                "worker_pid": os.getpid(),
                "parent_candidate_id": parent_id,
                "batch_size": 1,
                "target_data_epochs": 200,
                "paired_metric_scheduling": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
                **fields,
            },
        )

    def run_worker(self, parent_id: str) -> int:
        plan_path = self.operations / "REPAIRED_FRONTIER_FOLLOWUPS.json"
        plan = _read_json(plan_path)
        stream = next((
            row for row in plan.get("eligible_parent_streams", [])
            if row.get("parent_candidate_id") == parent_id
        ), None)
        if stream is None:
            raise RuntimeError("worker parent is absent from the frozen follow-up plan")
        completed = []
        receipt_rows = []
        for candidate_id in stream["execution_order_within_stream"]:
            receipt_path = self._run_candidate(parent_id, str(candidate_id))
            completed.append(str(candidate_id))
            receipt_rows.append({
                "candidate_id": str(candidate_id),
                "receipt_path": str(receipt_path),
                "receipt_sha256": file_sha256(receipt_path),
            })
            self._write_worker_state(
                parent_id,
                "PARENT_ABLATION_CANDIDATE_COMPLETE_E200",
                completed_candidate_ids=list(completed),
                receipts=list(receipt_rows),
            )
        self._write_worker_state(
            parent_id,
            "PARENT_ABLATION_STREAM_COMPLETE_E200",
            completed_candidate_ids=completed,
            receipts=receipt_rows,
            source_plan_sha256=file_sha256(plan_path),
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--worker-parent")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "run_root", "train_view", "data_root", "manifest", "python",
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
        successor = RepairedFollowupSuccessor(args.contract)
        if args.worker_parent:
            lock = run_root / "operations" / f"REPAIRED_FOLLOWUP_WORKER_{args.worker_parent}.lock"
            with support.executor_lock(lock):
                return successor.run_worker(args.worker_parent)
        with support.executor_lock(
            run_root / "operations" / "REPAIRED_FOLLOWUP_SUCCESSOR.lock"
        ):
            return successor.run_supervisor()
    except Exception as error:
        suffix = f"_{args.worker_parent}" if args.worker_parent else ""
        support.atomic_json(
            run_root / "operations" / f"REPAIRED_FOLLOWUP_SUCCESSOR_FATAL{suffix}.json",
            {
                "schema": "final-unsb-route1-repaired-followup-successor-fatal-v1",
                "updated": support.now(),
                "status": "FAILED",
                "worker_parent": args.worker_parent,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
