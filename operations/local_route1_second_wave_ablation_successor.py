"""Durably execute the one evidence-routed 5090 second-wave parent ablation."""

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
from research.local_route1.candidate_gate import run_candidate_gate
from research.local_route1.frontier_second_wave import (
    materialize_second_wave_parent_ablation,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-second-wave-ablation-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_second_wave_ablation_successor.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/frontier_second_wave.py",
    "research/local_route1/winner_ablations.py",
    "research/local_route1/candidate_gate.py",
    "research/local_route1/generation1_gates.py",
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
        raise RuntimeError("second-wave ablation successor worktree must be clean")
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
        "maximum_parent_ablations": 1,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("second-wave ablation successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get("git_commit"):
        raise RuntimeError("second-wave ablation successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("second-wave ablation successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"second-wave ablation source changed: {relative}")
    for key in ("manifest", "baseline_environment_record"):
        if support.file_sha256(Path(contract[key])) != contract.get(f"{key}_sha256"):
            raise RuntimeError(f"second-wave ablation {key} changed")
    fixed = {
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parent_ablations": 1,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"second-wave ablation contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15 or int(
        contract.get("timeout_seconds", 0)
    ) < 43200:
        raise RuntimeError("second-wave ablation polling/timeout is unsafe")


class SecondWaveAblationSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "SECOND_WAVE_ABLATION_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-second-wave-ablation-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(), "batch_size": 1,
            "target_data_epochs": 200, "maximum_parent_ablations": 1,
            "selection_seeds": [2026], "deferred_seed_validation": [2027, 2028],
            "paired_metric_scheduling": False, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def wait_decisions(self) -> tuple[Path, Path]:
        adjudication = self.operations / "FRONTIER_E200_ADJUDICATION.json"
        advancement = self.operations / "FRONTIER_ADVANCEMENT_CLASSIFICATION.json"
        while not (adjudication.is_file() and advancement.is_file()):
            for fatal_name in (
                "FRONTIER_TERMINAL_SUCCESSOR_FATAL.json",
                "FRONTIER_ADVANCEMENT_SUCCESSOR_FATAL.json",
            ):
                fatal = self.operations / fatal_name
                if fatal.is_file():
                    raise RuntimeError(f"second-wave prerequisite failed: {fatal}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for second-wave decisions")
            self.state("WAITING_FOR_COMPLETE_E200_ADVANCEMENT_CLASSIFICATION")
            time.sleep(int(self.contract["poll_seconds"]))
        return adjudication, advancement

    def init_executor_contract(self, candidate_id: str) -> Path:
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
        if path.is_file():
            return path
        command = [
            self.contract["python"], str(self.repo / "operations/local_route1_candidate_executor.py"),
            "--init-contract", "--contract", str(path), "--main-repo", str(self.repo),
            "--candidate-repo", str(self.repo), "--candidate-id", candidate_id,
            "--run-root", str(self.run_root), "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"], "--manifest", self.contract["manifest"],
            "--python", self.contract["python"], "--baseline-environment-record",
            self.contract["baseline_environment_record"],
        ]
        result = subprocess.run(
            command, cwd=self.repo, env=_env(self.repo), capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise RuntimeError(f"second-wave executor contract failed:\n{result.stdout}\n{result.stderr}")
        return path

    def run(self) -> int:
        adjudication, advancement = self.wait_decisions()
        frozen = materialize_second_wave_parent_ablation(
            self.run_root, adjudication_path=adjudication, advancement_path=advancement,
        )
        if frozen["status"] == "SECOND_WAVE_PARENT_ABLATION_INAPPLICABLE":
            self.state(
                frozen["status"], reason=frozen["route"]["reason"],
                near_boundary_candidate_ids=frozen["route"]["near_boundary_candidate_ids"],
            )
            return 0
        candidate_id = str(frozen["route"]["ablation_candidate_id"])
        self.state("RUNNING_SECOND_WAVE_PARENT_ABLATION_GATE", candidate_id=candidate_id)
        run_candidate_gate(
            output_root=self.run_root, candidate_id=candidate_id,
            train_view=Path(self.contract["train_view"]),
            data_root=Path(self.contract["data_root"]),
            manifest_path=Path(self.contract["manifest"]), gpu=0,
        )
        contract = self.init_executor_contract(candidate_id)
        stdout_path = self.operations / "SECOND_WAVE_ABLATION_EXECUTOR.stdout.log"
        stderr_path = self.operations / "SECOND_WAVE_ABLATION_EXECUTOR.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [self.contract["python"], str(self.repo / "operations/local_route1_candidate_executor.py"), "--contract", str(contract)],
                cwd=self.repo, env=_env(self.repo), stdout=stdout, stderr=stderr,
            )
            while process.poll() is None:
                if time.time() - self.started > int(self.contract["timeout_seconds"]):
                    raise TimeoutError("second-wave parent ablation exceeded durable timeout")
                self.state(
                    "SECOND_WAVE_PARENT_ABLATION_E200_RUNNING",
                    candidate_id=candidate_id, child_pid=process.pid,
                    data_epoch=support.current_epoch(self.run_root, candidate_id),
                )
                time.sleep(int(self.contract["poll_seconds"]))
            if process.returncode:
                raise RuntimeError(f"second-wave parent ablation exited {process.returncode}")
        receipt_path = self.operations / "terminal_receipts" / f"{candidate_id}.json"
        receipt = materialize_receipt(self.run_root, candidate_id, receipt_path)
        result = {
            "schema": "final-unsb-route1-second-wave-parent-ablation-e200-v1",
            "status": "SECOND_WAVE_PARENT_ABLATION_COMPLETE_E200",
            "candidate_id": candidate_id,
            "parent_candidate_id": frozen["route"]["parent_candidate_id"],
            "ablation_role": frozen["route"]["ablation_role"],
            "trajectory_status": receipt["trajectory_status"],
            "ranking_fields": receipt["ranking_fields"],
            "terminal_receipt_path": str(receipt_path),
            "terminal_receipt_sha256": file_sha256(receipt_path),
            "selection_seeds": [2026], "deferred_seed_validation": [2027, 2028],
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False, "confirmation20_opened": False,
        }
        result_path = self.operations / "SECOND_WAVE_PARENT_ABLATION_E200_RESULT.json"
        support.atomic_json(result_path, result)
        self.state(
            result["status"], candidate_id=candidate_id, data_epoch=200,
            result_sha256=file_sha256(result_path),
        )
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
        with support.executor_lock(run_root / "operations" / "SECOND_WAVE_ABLATION_SUCCESSOR.lock"):
            return SecondWaveAblationSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "SECOND_WAVE_ABLATION_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-second-wave-ablation-successor-fatal-v1",
                "updated": support.now(), "status": "FAILED", "error": repr(error),
                "traceback": traceback.format_exc(), "supervisor_pid": os.getpid(),
                "paired_controller_access": False, "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

