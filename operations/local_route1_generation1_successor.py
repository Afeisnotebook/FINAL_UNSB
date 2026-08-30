"""Durable fail-closed successor for Generation-1 e200 candidate runs.

The supervisor waits for both complete trajectories, performs the frozen
post-e200 adjudication, and starts matched seed validation only for a numeric
gate-passing winner.  It never reads intermediate quality to schedule work and
has no revision, handoff, checkpoint-selection, or algorithm-editing path.
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
except ModuleNotFoundError:  # direct script execution from operations/
    import local_route1_candidate_executor as support  # type: ignore[no-redef]

from research.local_route1.generation1_adjudication import adjudicate_generation1
from research.local_route1.candidate_defect_audit import (
    adjudicate_revision_need,
    audit_candidate_defect,
)
from research.local_route1.final_delivery import materialize_final_delivery
from research.local_route1.seed_validation import summarize_multi_seed_validation


SCHEMA = "final-unsb-route1-generation1-successor-contract-v1"
DEFAULT_IDS = (
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
    "G1-02-SAMPLING-VARIANCE",
)
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source_paths(repo: Path) -> tuple[Path, ...]:
    return tuple(repo / value for value in (
        "operations/local_route1_generation1_successor.py",
        "operations/local_route1_seed_executor.py",
        "operations/local_route1_candidate_executor.py",
        "research/local_route1/generation1_adjudication.py",
        "research/local_route1/candidate_defect_audit.py",
        "research/local_route1/candidate_runner.py",
        "research/local_route1/final_delivery.py",
        "research/local_route1/seed_validation.py",
    ))


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.successor_repo.resolve()
    head = support.run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    dirty = support.run_text(["git", "status", "--porcelain"], cwd=repo)
    if dirty:
        raise RuntimeError("successor worktree must be clean before contract freeze")
    candidates = [support.safe_candidate_id(value) for value in args.candidate_ids]
    if tuple(candidates) != DEFAULT_IDS:
        raise RuntimeError("Generation-1 successor requires the two frozen candidate ids")
    sources = _source_paths(repo)
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"successor source files are missing: {missing}")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "main_repo": str(args.main_repo.resolve()),
        "successor_repo": str(repo),
        "successor_git_commit": head,
        "candidate_ids": candidates,
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": EXPECTED_MANIFEST,
        "python": str(args.python.resolve()),
        "source_sha256": {
            path.relative_to(repo).as_posix(): support.file_sha256(path)
            for path in sources
        },
        "candidate_wait_poll_seconds": int(args.poll_seconds),
        "candidate_wait_timeout_seconds": int(args.timeout_seconds),
        "seed_order": [2027, 2028],
        "seed2028_requires_seed2027_sign_inconsistency": True,
        "freeze_only_numeric_gate_winner": True,
        "algorithm_revision_path": False,
        "handoff_or_window_path": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("Generation-1 successor contract schema mismatch")
    if tuple(contract.get("candidate_ids", [])) != DEFAULT_IDS:
        raise RuntimeError("successor candidate set changed")
    if contract.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("successor manifest identity mismatch")
    if support.file_sha256(Path(contract["manifest"])) != EXPECTED_MANIFEST:
        raise RuntimeError("successor manifest changed")
    repo = Path(contract["successor_repo"])
    head = support.run_text(["git", "rev-parse", "HEAD"], cwd=repo)
    if head != contract.get("successor_git_commit"):
        raise RuntimeError("successor worktree moved after contract freeze")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("successor worktree became dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"successor source changed after freeze: {relative}")
    if int(contract.get("candidate_wait_poll_seconds", 0)) < 15:
        raise RuntimeError("successor poll interval is too short")
    if int(contract.get("candidate_wait_timeout_seconds", 0)) < 3600:
        raise RuntimeError("successor timeout is too short for long-horizon runs")
    if contract.get("seed_order") != [2027, 2028]:
        raise RuntimeError("seed validation order changed")
    for key in (
        "algorithm_revision_path", "handoff_or_window_path",
        "paired_metric_scheduling", "paired_controller_access",
        "confirmation20_opened",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"successor requires {key}=false")
    if contract.get("freeze_only_numeric_gate_winner") is not True:
        raise RuntimeError("successor may only freeze a numeric-gate winner")
    if contract.get("seed2028_requires_seed2027_sign_inconsistency") is not True:
        raise RuntimeError("seed2028 authorization rule changed")


class Generation1Successor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["successor_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "GENERATION1_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "GENERATION1_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-generation1-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(),
            "successor_git_commit": self.contract["successor_git_commit"],
            "candidate_ids": self.contract["candidate_ids"],
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-generation1-successor-event-v1",
            "time": support.now(), "event": event,
            "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _trajectory_path(self, candidate_id: str) -> Path:
        return self.run_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"

    def wait_for_candidates(self) -> None:
        started = time.time()
        while True:
            completed = [
                value for value in self.contract["candidate_ids"]
                if self._trajectory_path(value).is_file()
            ]
            fatal = [
                value for value in self.contract["candidate_ids"]
                if (self.operations / f"CANDIDATE_EXECUTOR_FATAL_{value}.json").is_file()
            ]
            self.state(
                "WAITING_FOR_BOTH_MATCHED_E200_TRAJECTORIES",
                completed_candidate_ids=completed,
                pending_candidate_ids=[
                    value for value in self.contract["candidate_ids"] if value not in completed
                ],
                elapsed_seconds=time.time() - started,
            )
            if fatal:
                raise RuntimeError(f"candidate executor fatal record observed: {fatal}")
            if len(completed) == len(self.contract["candidate_ids"]):
                return
            if time.time() - started > int(self.contract["candidate_wait_timeout_seconds"]):
                raise TimeoutError("timed out waiting for both matched e200 trajectories")
            time.sleep(int(self.contract["candidate_wait_poll_seconds"]))

    def _seed_contract_path(self, candidate_id: str, seed: int) -> Path:
        return self.operations / f"SEED_EXECUTOR_CONTRACT_{candidate_id}_s{seed}.json"

    def run_seed(self, candidate_id: str, seed: int) -> None:
        contract_path = self._seed_contract_path(candidate_id, seed)
        if not contract_path.is_file():
            command = [
                self.contract["python"],
                "operations/local_route1_seed_executor.py", "--init-contract",
                "--contract", str(contract_path),
                "--main-repo", self.contract["main_repo"],
                "--seed-repo", self.contract["successor_repo"],
                "--candidate-id", candidate_id,
                "--validation-seed", str(seed),
                "--run-root", self.contract["run_root"],
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", self.contract["python"],
            ]
            result = subprocess.run(
                command, cwd=self.repo, capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"seed{seed} contract initialization failed:\n{result.stdout}\n{result.stderr}"
                )
        stdout = self.operations / f"SEED_EXECUTOR_{candidate_id}_s{seed}.stdout.log"
        stderr = self.operations / f"SEED_EXECUTOR_{candidate_id}_s{seed}.stderr.log"
        with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
            process = subprocess.Popen(
                [self.contract["python"], "operations/local_route1_seed_executor.py",
                 "--contract", str(contract_path)],
                cwd=self.repo, stdout=out, stderr=err,
            )
            self.event("SEED_EXECUTOR_START", candidate_id=candidate_id, seed=seed, child_pid=process.pid)
            while process.poll() is None:
                child_state_path = self.operations / f"SEED_EXECUTION_STATE_{candidate_id}_s{seed}.json"
                child_state = _read_json(child_state_path) if child_state_path.is_file() else {}
                self.state(
                    "FROZEN_SEED_VALIDATION_RUNNING",
                    candidate_id=candidate_id, seed=seed, child_pid=process.pid,
                    child_status=child_state.get("status"),
                    plain_data_epoch=child_state.get("plain_data_epoch"),
                    candidate_data_epoch=child_state.get("candidate_data_epoch"),
                )
                time.sleep(30)
            returncode = int(process.wait())
        self.event("SEED_EXECUTOR_COMPLETE", candidate_id=candidate_id, seed=seed, exit_code=returncode)
        if returncode:
            raise RuntimeError(f"seed{seed} executor failed with exit code {returncode}")

    def run(self) -> int:
        self.event("SUCCESSOR_START", contract=str(self.contract_path))
        self.wait_for_candidates()
        adjudication = adjudicate_generation1(
            self.run_root, self.contract["candidate_ids"], freeze_winner=True,
        )
        self.event(
            "GENERATION1_ADJUDICATED", status=adjudication["status"],
            selected_candidate_id=adjudication.get("selected_candidate_id"),
        )
        if adjudication.get("status") != "SEED2026_WINNER_READY_FOR_FROZEN_SEED2027":
            for candidate_id in self.contract["candidate_ids"]:
                self.state(
                    "TARGET_BLIND_E200_DEFECT_AUDIT_RUNNING",
                    candidate_id=candidate_id,
                    automatic_revision_started=False,
                )
                audit_candidate_defect(
                    output_root=self.run_root, candidate_id=candidate_id,
                    train_view=Path(self.contract["train_view"]),
                    manifest_path=Path(self.contract["manifest"]),
                    gpu=0, samples=16,
                )
            revision = adjudicate_revision_need(
                self.run_root, list(self.contract["candidate_ids"]),
            )
            if revision["status"] == "NO_REVISION_APPLICABLE_FINAL_FALLBACK":
                delivery = materialize_final_delivery(self.run_root)
                self.state(
                    "FINAL_NEGATIVE_FALLBACK_MATERIALIZED",
                    selected_fallback=revision["selected_candidate_id"],
                    final_candidate_status=delivery["status"],
                    final_candidate_path=str(self.run_root / "final" / "CANDIDATE.json"),
                    automatic_revision_started=False,
                )
            else:
                self.state(
                    "TARGET_BLIND_DEFECT_REDUCED_REVISION_DERIVATION_REQUIRED",
                    revision_applicable_candidate_ids=revision[
                        "revision_applicable_candidate_ids"
                    ],
                    automatic_revision_started=False,
                    fixed_window_or_handoff_started=False,
                )
            return 0
        winner = str(adjudication["selected_candidate_id"])
        if adjudication.get("winner_frozen_for_seed2027") is not True:
            raise RuntimeError("eligible Generation-1 winner was not frozen")
        self.run_seed(winner, 2027)
        aggregate = summarize_multi_seed_validation(self.run_root, winner)
        if aggregate["status"] == "WAITING_FOR_AUTHORIZED_SEED2028":
            self.run_seed(winner, 2028)
            aggregate = summarize_multi_seed_validation(self.run_root, winner)
        delivery = materialize_final_delivery(self.run_root)
        self.state(
            "MULTI_SEED_ADJUDICATION_COMPLETE",
            candidate_id=winner, final_status=aggregate["status"],
            classification=aggregate.get("classification"),
            included_seeds=aggregate.get("included_seeds"),
            failed_checks=aggregate.get("failed_checks"),
            final_candidate_status=delivery.get("status"),
            final_candidate_path=str(self.run_root / "final" / "CANDIDATE.json"),
        )
        self.event(
            "SUCCESSOR_COMPLETE", candidate_id=winner,
            final_status=aggregate["status"],
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--main-repo", type=Path)
    value.add_argument("--successor-repo", type=Path)
    value.add_argument("--candidate-id", action="append", dest="candidate_ids")
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=86400)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        args.candidate_ids = args.candidate_ids or list(DEFAULT_IDS)
        required = (
            "main_repo", "successor_repo", "run_root", "train_view",
            "data_root", "manifest", "python",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    contract_path = args.contract.resolve()
    try:
        contract = _read_json(contract_path)
        run_root = Path(contract["run_root"])
        with support.executor_lock(run_root / "operations" / "GENERATION1_SUCCESSOR.lock"):
            return Generation1Successor(contract_path).run()
    except Exception as error:
        try:
            run_root = Path(contract["run_root"]) if "contract" in locals() else contract_path.parent.parent
            support.atomic_json(run_root / "operations" / "GENERATION1_SUCCESSOR_FATAL.json", {
                "schema": "final-unsb-route1-generation1-successor-fatal-v1",
                "time": support.now(), "status": "FAILED",
                "error": repr(error), "traceback": traceback.format_exc(),
                "supervisor_pid": os.getpid(),
                "paired_metric_scheduling": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            })
        finally:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
