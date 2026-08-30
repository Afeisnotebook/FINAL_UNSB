"""Durably route, gate and execute the preregistered Generation-3 synthesis."""

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
from operations.local_route1_freeze_generation3_synthesis import (
    CANDIDATE_ID,
    materialize,
)
from research.local_route1.candidate_gate import run_candidate_gate


SCHEMA = "final-unsb-route1-generation3-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_generation3_successor.py",
    "operations/local_route1_freeze_generation3_synthesis.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/candidates.py",
    "research/local_route1/candidate_gate.py",
    "research/local_route1/candidate_runner.py",
    "research/local_route1/generation1_gates.py",
    "src/models/route1/pcammcrb.py",
    "src/models/route1_pcammcrb_model.py",
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
        raise RuntimeError("Generation-3 successor worktree must be clean")
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
        "baseline_environment_record": str(args.baseline_environment_record.resolve()),
        "baseline_environment_record_sha256": support.file_sha256(
            args.baseline_environment_record.resolve()
        ),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_generation3_candidates": 1,
        "maximum_components": 2,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("Generation-3 successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("Generation-3 successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("Generation-3 successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"Generation-3 source changed: {relative}")
    for key in ("manifest", "baseline_environment_record"):
        if support.file_sha256(Path(contract[key])) != contract.get(f"{key}_sha256"):
            raise RuntimeError(f"Generation-3 {key} changed")
    fixed = {
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_generation3_candidates": 1,
        "maximum_components": 2,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Generation-3 successor contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("Generation-3 polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("Generation-3 timeout is too short")


class Generation3Successor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "GENERATION3_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "GENERATION3_SUCCESSOR_EVENTS.jsonl"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-generation3-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_id": CANDIDATE_ID,
            "batch_size": 1,
            "target_data_epochs": 200,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "intermediate_metric_routing": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-generation3-successor-event-v1",
            "time": support.now(),
            "event": event,
            "supervisor_pid": os.getpid(),
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_terminal_adjudication(self) -> Path:
        path = self.operations / "FRONTIER_E200_ADJUDICATION.json"
        fatal = self.operations / "FRONTIER_TERMINAL_SUCCESSOR_FATAL.json"
        while not path.is_file():
            if fatal.is_file():
                raise RuntimeError(f"frontier terminal successor failed: {fatal}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for terminal frontier adjudication")
            epochs = {
                candidate_id: support.current_epoch(self.run_root, candidate_id)
                for candidate_id in (
                    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING",
                    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER",
                )
            }
            self.state(
                "WAITING_FOR_COMPLETE_FRONTIER_ADJUDICATION",
                data_epochs=epochs,
                elapsed_seconds=time.time() - self.started,
            )
            time.sleep(int(self.contract["poll_seconds"]))
        return path

    def run_gate(self) -> dict[str, Any]:
        self.state("RUNNING_TARGET_BLIND_GENERATION3_COMPATIBILITY_GATE")
        return run_candidate_gate(
            output_root=self.run_root,
            candidate_id=CANDIDATE_ID,
            train_view=Path(self.contract["train_view"]),
            data_root=Path(self.contract["data_root"]),
            manifest_path=Path(self.contract["manifest"]),
            gpu=0,
        )

    def init_executor_contract(self) -> Path:
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{CANDIDATE_ID}.json"
        if path.is_file():
            return path
        command = [
            self.contract["python"],
            str(self.repo / "operations/local_route1_candidate_executor.py"),
            "--init-contract", "--contract", str(path),
            "--main-repo", str(self.repo),
            "--candidate-repo", str(self.repo),
            "--candidate-id", CANDIDATE_ID,
            "--run-root", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--python", self.contract["python"],
            "--baseline-environment-record", self.contract["baseline_environment_record"],
        ]
        result = subprocess.run(
            command, cwd=self.repo, env=_env(self.repo), capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"Generation-3 executor contract initialization failed:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return path

    def run_e200(self, executor_contract: Path) -> Path:
        stdout_path = self.operations / "GENERATION3_EXECUTOR.stdout.log"
        stderr_path = self.operations / "GENERATION3_EXECUTOR.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [
                    self.contract["python"],
                    str(self.repo / "operations/local_route1_candidate_executor.py"),
                    "--contract", str(executor_contract),
                ],
                cwd=self.repo,
                env=_env(self.repo),
                stdout=stdout,
                stderr=stderr,
            )
            self.event("GENERATION3_E200_EXECUTOR_STARTED", child_pid=process.pid)
            while process.poll() is None:
                if time.time() - self.started > int(self.contract["timeout_seconds"]):
                    raise TimeoutError("Generation-3 e200 execution exceeded durable timeout")
                self.state(
                    "GENERATION3_E200_RUNNING",
                    child_pid=process.pid,
                    data_epoch=support.current_epoch(self.run_root, CANDIDATE_ID),
                    elapsed_seconds=time.time() - self.started,
                )
                time.sleep(int(self.contract["poll_seconds"]))
            if process.returncode:
                raise RuntimeError(f"Generation-3 executor exited {process.returncode}")
        receipt_path = self.operations / "terminal_receipts" / f"{CANDIDATE_ID}.json"
        materialize_receipt(self.run_root, CANDIDATE_ID, receipt_path)
        return receipt_path

    def run(self) -> int:
        self.event("GENERATION3_SUCCESSOR_START", contract=str(self.contract_path))
        adjudication_path = self.wait_terminal_adjudication()
        frozen = materialize(self.run_root, adjudication_path)
        if frozen["status"] == "SYNTHESIS_INAPPLICABLE":
            self.state(
                "SYNTHESIS_INAPPLICABLE",
                reason=frozen["route"]["reason"],
                freeze_sha256=support.file_sha256(
                    self.operations / "GENERATION3_SYNTHESIS_FREEZE.json"
                ),
            )
            self.event("GENERATION3_INAPPLICABLE", reason=frozen["route"]["reason"])
            return 0
        self.event(
            "GENERATION3_SOURCE_FROZEN",
            sampling_parent=frozen["route"]["sampling_parent"],
        )
        try:
            gate = self.run_gate()
        except RuntimeError as error:
            if "component corrections violate the preregistered cosine floor" not in str(error):
                raise
            result = {
                "schema": "final-unsb-route1-generation3-compatibility-inapplicable-v1",
                "status": "SYNTHESIS_INAPPLICABLE_COMPONENT_COSINE",
                "candidate_id": CANDIDATE_ID,
                "error": str(error),
                "paired_metric_computed": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
            write_path = self.operations / "GENERATION3_COMPATIBILITY_INAPPLICABLE.json"
            support.atomic_json(write_path, result)
            self.state(result["status"], evidence_sha256=support.file_sha256(write_path))
            return 0
        self.event("GENERATION3_COMPATIBILITY_GATE_PASS", gate_sha256=support.file_sha256(
            self.run_root / "derive/gates" / f"{CANDIDATE_ID}.json"
        ))
        executor_contract = self.init_executor_contract()
        receipt_path = self.run_e200(executor_contract)
        receipt = _read_json(receipt_path)
        result = {
            "schema": "final-unsb-route1-generation3-e200-result-v1",
            "status": "GENERATION3_COMPLETE_E200",
            "candidate_id": CANDIDATE_ID,
            "sampling_parent": frozen["route"]["sampling_parent"],
            "trajectory_status": receipt["trajectory_status"],
            "ranking_fields": receipt["ranking_fields"],
            "terminal_receipt_path": str(receipt_path),
            "terminal_receipt_sha256": support.file_sha256(receipt_path),
            "gate_sha256": support.file_sha256(
                self.run_root / "derive/gates" / f"{CANDIDATE_ID}.json"
            ),
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        result_path = self.operations / "GENERATION3_E200_RESULT.json"
        support.atomic_json(result_path, result)
        self.state(
            "GENERATION3_COMPLETE_E200",
            data_epoch=200,
            trajectory_status=receipt["trajectory_status"],
            result_sha256=support.file_sha256(result_path),
        )
        self.event("GENERATION3_SUCCESSOR_COMPLETE", trajectory_status=receipt["trajectory_status"])
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--timeout-seconds", type=int, default=345600)
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
        with support.executor_lock(
            run_root / "operations" / "GENERATION3_SUCCESSOR.lock"
        ):
            return Generation3Successor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "GENERATION3_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-generation3-successor-fatal-v1",
                "updated": support.now(),
                "status": "FAILED",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "supervisor_pid": os.getpid(),
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

