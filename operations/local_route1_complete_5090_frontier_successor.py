"""Wait for cross-runtime replays and export the complete 5090 frontier."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.complete_5090_frontier import (
    export_portable_complete_5090,
    materialize_complete_5090_frontier,
)


SCHEMA = "final-unsb-route1-complete-5090-frontier-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_complete_5090_frontier_successor.py",
    "research/local_route1/complete_5090_frontier.py",
    "research/local_route1/cross_runtime_portfolio.py",
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
        raise RuntimeError("complete 5090 frontier worktree must be clean")
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
        "requires_cross_runtime_portfolio_complete_e200": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("complete 5090 frontier contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("complete 5090 frontier worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("complete 5090 frontier worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"complete 5090 frontier source changed: {relative}")
    fixed = {
        "requires_cross_runtime_portfolio_complete_e200": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"complete 5090 frontier contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("complete 5090 frontier polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("complete 5090 frontier timeout is too short")


class Complete5090FrontierSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "COMPLETE_5090_FRONTIER_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-complete-5090-frontier-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "canonical_candidate_is_action_priority_only": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        source = self.operations / "CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json"
        fatal = self.operations / "CROSS_RUNTIME_5090_SUCCESSOR_FATAL.json"
        while not source.is_file():
            if fatal.is_file():
                raise RuntimeError(f"cross-runtime 5090 successor failed: {fatal}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for 5090 cross-runtime e200")
            upstream = self.operations / "CROSS_RUNTIME_5090_SUCCESSOR_STATE.json"
            state = _read_json(upstream) if upstream.is_file() else {}
            self.state(
                "WAITING_FOR_CROSS_RUNTIME_5090_PORTFOLIO_E200",
                upstream_status=state.get("status"),
            )
            time.sleep(int(self.contract["poll_seconds"]))
        frontier = materialize_complete_5090_frontier(self.run_root)
        portable = export_portable_complete_5090(self.run_root)
        output = self.operations / "PORTABLE_COMPLETE_5090_FRONTIER.json"
        self.state(
            portable["status"],
            action_priority_candidate_id=frontier["action_priority_candidate_id"],
            priority_alternate_candidate_ids=frontier[
                "priority_alternate_candidate_ids"
            ],
            rankable_complete_e200_candidate_count=frontier[
                "rankable_complete_e200_candidate_count"
            ],
            output_sha256=support.file_sha256(output),
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
            run_root / "operations" / "COMPLETE_5090_FRONTIER_SUCCESSOR.lock"
        ):
            return Complete5090FrontierSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "COMPLETE_5090_FRONTIER_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-complete-5090-frontier-successor-fatal-v1",
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
