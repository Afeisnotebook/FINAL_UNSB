"""Wait for repaired/G3 terminal results and freeze the complete 4090 frontier."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.complete_frontier import (
    RESULT_FILES,
    materialize_complete_4090_frontier,
)


SCHEMA = "final-unsb-route1-complete-4090-frontier-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_complete_frontier_4090_successor.py",
    "operations/local_route1_complete_frontier_4090.py",
    "research/local_route1/complete_frontier.py",
    "research/local_route1/frontier_advancement.py",
    "operations/local_route1_cross_version_adjudicate.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("complete-frontier successor worktree must be clean")
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
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "requires_complete_repaired_portfolio": True,
        "requires_both_generation3_terminal_results_even_if_inapplicable": True,
        "canonical_candidate_is_action_priority_only": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("complete-frontier successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("complete-frontier successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("complete-frontier successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"complete-frontier successor source changed: {relative}")
    fixed = {
        "requires_complete_repaired_portfolio": True,
        "requires_both_generation3_terminal_results_even_if_inapplicable": True,
        "canonical_candidate_is_action_priority_only": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"complete-frontier successor changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("complete-frontier successor polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("complete-frontier successor timeout is too short")


class CompleteFrontier4090Successor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "COMPLETE_FRONTIER_4090_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-complete-4090-frontier-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "canonical_candidate_is_action_priority_only": True,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        while True:
            ready = {name: (self.operations / name).is_file() for name in RESULT_FILES}
            fatal = sorted(
                path.name for path in self.operations.glob("*SUCCESSOR_FATAL.json")
                if path.name.startswith(("REPAIRED_PORTFOLIO", "RESIDUAL_SYNTHESIS", "RESIDUAL_EUCLIDEAN"))
            )
            if fatal:
                raise RuntimeError(f"upstream complete-frontier successor failed: {fatal}")
            if all(ready.values()):
                break
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for repaired/G3 terminal results")
            self.state("WAITING_FOR_REPAIRED_AND_GENERATION3_TERMINAL_RESULTS", readiness=ready)
            time.sleep(int(self.contract["poll_seconds"]))
        result = materialize_complete_4090_frontier(self.run_root)
        output = self.operations / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
        self.state(
            result["status"],
            action_priority_candidate_id=result["action_priority_candidate_id"],
            priority_alternate_candidate_ids=result["priority_alternate_candidate_ids"],
            rankable_complete_e200_candidate_count=result[
                "rankable_complete_e200_candidate_count"
            ],
            result_sha256=support.file_sha256(output),
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        if args.repo is None or args.run_root is None:
            raise SystemExit("--init-contract requires --repo and --run-root")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    contract = _read_json(args.contract)
    run_root = Path(contract["run_root"])
    try:
        with support.executor_lock(
            run_root / "operations" / "COMPLETE_FRONTIER_4090_SUCCESSOR.lock"
        ):
            return CompleteFrontier4090Successor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "COMPLETE_FRONTIER_4090_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-complete-4090-frontier-successor-fatal-v1",
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
