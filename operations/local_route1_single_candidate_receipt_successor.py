"""Publish one completed candidate's canonical receipt without portfolio coupling.

This successor is a scheduling bridge only.  It waits for the candidate's
completed e200 trajectory and then asks that candidate's frozen source worktree
to build the ordinary source-bound terminal receipt.  It does not inspect
rankings while waiting, select a checkpoint, alter training, or start/stop a
worker.  Downstream successors may therefore consume the freed GPU slot as
soon as this one candidate has genuinely completed, without waiting for an
unrelated portfolio peer.
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

from operations import local_route1_candidate_executor as support
from operations.local_route1_cross_version_adjudicate import _validate_receipt


SCHEMA = "final-unsb-route1-single-candidate-receipt-successor-contract-v1"
STATE_SCHEMA = "final-unsb-route1-single-candidate-receipt-successor-state-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_single_candidate_receipt_successor.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_cross_version_adjudicate.py",
)
CANDIDATE_RECEIPT_RELATIVES = (
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/generation1_adjudication.py",
    "research/local_route1/candidates.py",
    "research/local_route1/candidate_runner.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git_identity(repo: Path, *, label: str) -> dict[str, str]:
    repo = repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError(f"{label} worktree is dirty: {repo}")
    return {
        "path": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
    }


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    candidate_id = support.safe_candidate_id(args.candidate_id)
    repo = args.repo.resolve()
    candidate_repo = args.candidate_repo.resolve()
    run_root = args.run_root.resolve()
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError(f"receipt successor Python is missing: {python}")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": _git_identity(repo, label="receipt orchestration"),
        "candidate_repo": _git_identity(
            candidate_repo, label="candidate receipt source",
        ),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "candidate_receipt_source_sha256": {
            relative: support.file_sha256(candidate_repo / relative)
            for relative in CANDIDATE_RECEIPT_RELATIVES
        },
        "python": str(python),
        "run_root": str(run_root),
        "candidate_id": candidate_id,
        "trajectory_path": str(
            run_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        ),
        "canonical_receipt_path": str(
            run_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
        ),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "scheduling_bridge_only": True,
        "requires_complete_e200_trajectory": True,
        "checkpoint_transfer": False,
        "formula_changed": False,
        "ranking_changed": False,
        "paired_metrics_used_for_scheduling": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("single-candidate receipt contract schema mismatch")
    candidate_id = support.safe_candidate_id(str(contract.get("candidate_id", "")))
    for key, label in (("repo", "orchestration"), ("candidate_repo", "candidate")):
        identity = contract.get(key) or {}
        repo = Path(str(identity.get("path", "")))
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != identity.get(
            "git_commit"
        ):
            raise RuntimeError(f"single-candidate receipt {label} worktree moved")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"single-candidate receipt {label} worktree is dirty")
    repo = Path(contract["repo"]["path"])
    candidate_repo = Path(contract["candidate_repo"]["path"])
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"receipt successor source changed: {relative}")
    for relative, expected in contract.get(
        "candidate_receipt_source_sha256", {}
    ).items():
        if support.file_sha256(candidate_repo / relative) != expected:
            raise RuntimeError(f"candidate receipt source changed: {relative}")
    python = Path(str(contract.get("python", "")))
    if not python.is_file():
        raise RuntimeError("single-candidate receipt Python is missing")
    run_root = Path(str(contract.get("run_root", ""))).resolve()
    expected_trajectory = (
        run_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    ).resolve()
    expected_receipt = (
        run_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    ).resolve()
    if Path(str(contract.get("trajectory_path", ""))).resolve() != expected_trajectory:
        raise RuntimeError("single-candidate trajectory path changed")
    if Path(str(contract.get("canonical_receipt_path", ""))).resolve() != expected_receipt:
        raise RuntimeError("single-candidate canonical receipt path changed")
    fixed = {
        "scheduling_bridge_only": True,
        "requires_complete_e200_trajectory": True,
        "checkpoint_transfer": False,
        "formula_changed": False,
        "ranking_changed": False,
        "paired_metrics_used_for_scheduling": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"single-candidate receipt contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 30:
        raise RuntimeError("single-candidate receipt polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("single-candidate receipt timeout is too short")


class SingleCandidateReceiptSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.candidate_repo = Path(self.contract["candidate_repo"]["path"])
        self.candidate_id = str(self.contract["candidate_id"])
        self.state_path = self.operations / (
            f"SINGLE_RECEIPT_SUCCESSOR_{self.candidate_id}_STATE.json"
        )
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": STATE_SCHEMA,
            "updated": support.now(),
            "status": status,
            "pid": os.getpid(),
            "candidate_id": self.candidate_id,
            "scheduling_bridge_only": True,
            "paired_metrics_used_for_scheduling": False,
            "paired_metrics_used_for_formula_or_training_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _accepted_receipt(self, path: Path) -> dict[str, Any]:
        receipt = _validate_receipt(path)
        if (
            receipt.get("candidate_id") != self.candidate_id
            or receipt.get("training_git_commit")
            != self.contract["candidate_repo"]["git_commit"]
            or receipt.get("verification_git_commit")
            != self.contract["candidate_repo"]["git_commit"]
            or receipt.get("paired_metrics_used_for_training_or_control") is not False
            or receipt.get("confirmation20_opened") is not False
        ):
            raise RuntimeError("single-candidate canonical receipt identity changed")
        return receipt

    def run(self) -> int:
        trajectory_path = Path(self.contract["trajectory_path"])
        receipt_path = Path(self.contract["canonical_receipt_path"])
        if receipt_path.is_file():
            receipt = self._accepted_receipt(receipt_path)
            self.state(
                "CANONICAL_SOURCE_BOUND_RECEIPT_AVAILABLE",
                receipt_sha256=support.file_sha256(receipt_path),
                trajectory_status=receipt["trajectory_status"],
                resumed=True,
            )
            return 0
        while not trajectory_path.is_file():
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for completed candidate trajectory")
            self.state("WAITING_FOR_COMPLETE_E200_TRAJECTORY")
            time.sleep(int(self.contract["poll_seconds"]))
        self.state("COMPLETE_E200_TRAJECTORY_OBSERVED_BUILDING_RECEIPT")
        command = [
            self.contract["python"],
            str(
                self.candidate_repo
                / "operations/local_route1_candidate_terminal_receipt.py"
            ),
            "--output", str(self.run_root),
            "--candidate-id", self.candidate_id,
            "--receipt", str(receipt_path),
        ]
        result = subprocess.run(
            command,
            cwd=self.candidate_repo,
            env=_env(self.candidate_repo),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            timeout=1800,
        )
        if result.returncode:
            raise RuntimeError(
                "single-candidate terminal receipt failed:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        receipt = self._accepted_receipt(receipt_path)
        self.state(
            "CANONICAL_SOURCE_BOUND_RECEIPT_AVAILABLE",
            receipt_sha256=support.file_sha256(receipt_path),
            trajectory_status=receipt["trajectory_status"],
            resumed=False,
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--candidate-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--candidate-id")
    value.add_argument("--python", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = ("repo", "candidate_repo", "run_root", "candidate_id", "python")
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
    candidate_id = support.safe_candidate_id(str(contract["candidate_id"]))
    lock = run_root / "operations" / f"SINGLE_RECEIPT_SUCCESSOR_{candidate_id}.lock"
    try:
        with support.executor_lock(lock):
            return SingleCandidateReceiptSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / f"SINGLE_RECEIPT_SUCCESSOR_{candidate_id}_FATAL.json",
            {
                "schema": "final-unsb-route1-single-candidate-receipt-successor-fatal-v1",
                "updated": support.now(),
                "status": "FAILED",
                "candidate_id": candidate_id,
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
                "paired_metrics_used_for_scheduling": False,
                "paired_metrics_used_for_formula_or_training_control": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
