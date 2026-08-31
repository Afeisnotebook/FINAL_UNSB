"""Durably run G3-02 after its two strict same-host 4090 parents exist."""

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
from operations.local_route1_cross_version_adjudicate import _validate_receipt
from operations.local_route1_freeze_residual_synthesis import (
    CANDIDATE_ID,
    PCRSMG_PROPOSAL_ID,
    RFAMMCRB_ID,
    materialize,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-residual-synthesis-4090-successor-contract-v1"
PORTFOLIO_SCHEMA = "final-unsb-route1-repaired-portfolio-4090-result-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_residual_synthesis_4090_successor.py",
    "operations/local_route1_freeze_residual_synthesis.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
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


def _git_identity(repo: Path) -> dict[str, str]:
    repo = repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("residual synthesis worktree is dirty")
    return {
        "path": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
    }


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    manifest = args.manifest.resolve()
    environment = args.baseline_environment_record.resolve()
    sampling = args.sampling_receipt.resolve()
    for path, label in (
        (manifest, "manifest"),
        (environment, "baseline environment"),
        (sampling, "sampling parent receipt"),
    ):
        if not path.is_file():
            raise RuntimeError(f"residual synthesis {label} is missing: {path}")
    receipt = _validate_receipt(sampling)
    if receipt.get("candidate_id") != PCRSMG_PROPOSAL_ID:
        raise RuntimeError("4090 residual synthesis requires PC-RSMG proposal parent")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": _git_identity(repo),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "run_root": str(args.run_root.resolve()),
        "portfolio_result": str(args.portfolio_result.resolve()),
        "sampling_receipt": str(sampling),
        "sampling_receipt_sha256": support.file_sha256(sampling),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": support.file_sha256(manifest),
        "python": str(args.python.resolve()),
        "baseline_environment_record": str(environment),
        "baseline_environment_record_sha256": support.file_sha256(environment),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "candidate_id": CANDIDATE_ID,
        "required_sampling_parent_id": PCRSMG_PROPOSAL_ID,
        "required_barrier_parent_id": RFAMMCRB_ID,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("residual synthesis successor contract schema mismatch")
    repo = Path(str(contract.get("repo", {}).get("path", ""))).resolve()
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract[
        "repo"
    ].get("git_commit"):
        raise RuntimeError("residual synthesis worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("residual synthesis worktree became dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"residual synthesis source changed: {relative}")
    for key in ("sampling_receipt", "manifest", "baseline_environment_record"):
        if support.file_sha256(Path(contract[key])) != contract.get(f"{key}_sha256"):
            raise RuntimeError(f"residual synthesis authority changed: {key}")
    fixed = {
        "candidate_id": CANDIDATE_ID,
        "required_sampling_parent_id": PCRSMG_PROPOSAL_ID,
        "required_barrier_parent_id": RFAMMCRB_ID,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"residual synthesis contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("residual synthesis polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("residual synthesis timeout is too short")


class ResidualSynthesis4090Successor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"]["path"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "RESIDUAL_SYNTHESIS_4090_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-residual-synthesis-4090-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_id": CANDIDATE_ID,
            "cross_host_deltas_merged": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_portfolio(self) -> dict[str, Any]:
        path = Path(self.contract["portfolio_result"])
        while not path.is_file():
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for repaired 4090 portfolio")
            self.state("WAITING_FOR_REPAIRED_4090_PORTFOLIO_E200")
            time.sleep(int(self.contract["poll_seconds"]))
        value = _read_json(path)
        if value.get("schema") != PORTFOLIO_SCHEMA:
            raise RuntimeError("repaired 4090 portfolio schema changed")
        if value.get("cross_host_deltas_merged") is not False or value.get(
            "paired_controller_access"
        ) is not False or value.get("confirmation20_opened") is not False:
            raise RuntimeError("repaired 4090 portfolio violates route-1 scope")
        return value

    def _run_checked(
        self, command: list[str], *, label: str, timeout: int = 7200,
    ) -> None:
        result = subprocess.run(
            command,
            cwd=self.repo,
            env=_env(self.repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")

    def run(self) -> int:
        portfolio = self.wait_portfolio()
        rows = {
            str(row.get("candidate_id", "")): row
            for row in portfolio.get("candidate_results", [])
            if isinstance(row, dict)
        }
        if RFAMMCRB_ID not in rows:
            result = {
                "schema": "final-unsb-route1-residual-synthesis-4090-result-v1",
                "status": "SYNTHESIS_INAPPLICABLE_RFAMMCRB_NOT_REPLAYED",
                "candidate_id": None,
                "old_g3_run": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
            path = self.operations / "RESIDUAL_SYNTHESIS_4090_RESULT.json"
            support.atomic_json(path, result)
            self.state(result["status"], result_sha256=support.file_sha256(path))
            return 0
        barrier_row = rows[RFAMMCRB_ID]
        barrier_receipt = Path(str(barrier_row.get("receipt_path", ""))).resolve()
        if (
            not barrier_receipt.is_file()
            or support.file_sha256(barrier_receipt) != barrier_row.get("receipt_sha256")
        ):
            raise RuntimeError("RF-AMMCRB 4090 parent receipt changed")

        freeze = materialize(
            self.run_root,
            sampling_receipt_path=Path(self.contract["sampling_receipt"]),
            barrier_receipt_path=barrier_receipt,
        )
        if freeze["status"] == "SYNTHESIS_INAPPLICABLE":
            path = self.operations / "RESIDUAL_SYNTHESIS_4090_RESULT.json"
            result = {
                "schema": "final-unsb-route1-residual-synthesis-4090-result-v1",
                "status": "SYNTHESIS_INAPPLICABLE_ONE_OR_MORE_PARENTS_NOT_STRICT",
                "candidate_id": None,
                "freeze_sha256": support.file_sha256(
                    self.operations / "RESIDUAL_SYNTHESIS_FREEZE.json"
                ),
                "old_g3_run": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
            support.atomic_json(path, result)
            self.state(result["status"], result_sha256=support.file_sha256(path))
            return 0
        if freeze["status"] != "SYNTHESIS_FROZEN_FOR_COMPATIBILITY_GATE":
            raise RuntimeError("unexpected residual synthesis freeze outcome")

        self.state("RUNNING_TARGET_BLIND_COMPATIBILITY_AND_ENGINEERING_GATE")
        self._run_checked([
            self.contract["python"], "-m", "research.local_route1.run",
            "--stage", "candidate", "--candidate-action", "gate",
            "--candidate-id", CANDIDATE_ID,
            "--output", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--gpu", "0",
        ], label="residual synthesis compatibility gate", timeout=21600)
        executor = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{CANDIDATE_ID}_4090.json"
        if not executor.is_file():
            self._run_checked([
                self.contract["python"],
                str(self.repo / "operations/local_route1_candidate_executor.py"),
                "--init-contract", "--contract", str(executor),
                "--main-repo", str(self.repo),
                "--candidate-repo", str(self.repo),
                "--candidate-id", CANDIDATE_ID,
                "--run-root", str(self.run_root),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", self.contract["python"],
                "--baseline-environment-record", self.contract[
                    "baseline_environment_record"
                ],
            ], label="residual synthesis executor freeze")
        self.state("RESIDUAL_SYNTHESIS_E200_RUNNING")
        self._run_checked([
            self.contract["python"],
            str(self.repo / "operations/local_route1_candidate_executor.py"),
            "--contract", str(executor),
        ], label="residual synthesis e200", timeout=int(self.contract["timeout_seconds"]))
        receipt_path = (
            self.operations / "terminal_receipts" / f"{CANDIDATE_ID}_4090.json"
        )
        self._run_checked([
            self.contract["python"],
            str(self.repo / "operations/local_route1_candidate_terminal_receipt.py"),
            "--output", str(self.run_root),
            "--candidate-id", CANDIDATE_ID,
            "--receipt", str(receipt_path),
        ], label="residual synthesis receipt", timeout=1800)
        receipt = _validate_receipt(receipt_path)
        result = {
            "schema": "final-unsb-route1-residual-synthesis-4090-result-v1",
            "status": "RESIDUAL_SYNTHESIS_4090_COMPLETE_E200",
            "candidate_id": CANDIDATE_ID,
            "trajectory_status": receipt["trajectory_status"],
            "ranking_fields": receipt["ranking_fields"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
            "old_g3_run": False,
            "cross_host_deltas_merged": False,
            "paired_metrics_used_only_after_complete_trajectory": True,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        path = self.operations / "RESIDUAL_SYNTHESIS_4090_RESULT.json"
        support.atomic_json(path, result)
        self.state(result["status"], result_sha256=support.file_sha256(path))
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--portfolio-result", type=Path)
    value.add_argument("--sampling-receipt", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "run_root", "portfolio_result", "sampling_receipt",
            "train_view", "data_root", "manifest", "python",
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
            run_root / "operations" / "RESIDUAL_SYNTHESIS_4090_SUCCESSOR.lock"
        ):
            return ResidualSynthesis4090Successor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "RESIDUAL_SYNTHESIS_4090_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-residual-synthesis-4090-successor-fatal-v1",
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

