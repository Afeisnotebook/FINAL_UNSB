"""Wait for all repaired follow-ups and freeze the extended 5090 frontier."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.extended_repaired_frontier import (
    materialize_extended_repaired_frontier,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-extended-repaired-frontier-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_extended_repaired_frontier_successor.py",
    "operations/local_route1_extended_repaired_frontier.py",
    "operations/local_route1_winner_ablation_adjudicate.py",
    "research/local_route1/extended_repaired_frontier.py",
    "research/local_route1/repaired_frontier_followups.py",
    "research/local_route1/repaired_frontier_adjudication.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("extended repaired frontier worktree must be clean")
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
        "requires_complete_repaired_followup_execution": True,
        "observable_only_excluded_from_candidate_ranking": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("extended repaired frontier contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("extended repaired frontier worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("extended repaired frontier worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"extended repaired source changed: {relative}")
    fixed = {
        "requires_complete_repaired_followup_execution": True,
        "observable_only_excluded_from_candidate_ranking": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"extended repaired frontier contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("extended repaired frontier polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("extended repaired frontier timeout is too short")


class ExtendedRepairedFrontierSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = (
            self.operations / "EXTENDED_REPAIRED_FRONTIER_SUCCESSOR_STATE.json"
        )
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-extended-repaired-frontier-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "canonical_candidate_is_action_priority_only": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "cross_host_deltas_merged": False,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        result_path = self.operations / "REPAIRED_FOLLOWUP_EXECUTION_RESULT.json"
        fatal_path = self.operations / "REPAIRED_FOLLOWUP_SUCCESSOR_FATAL.json"
        while not result_path.is_file():
            if fatal_path.is_file():
                raise RuntimeError(f"repaired follow-up successor failed: {fatal_path}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for repaired follow-up execution")
            upstream = self.operations / "REPAIRED_FOLLOWUP_SUCCESSOR_STATE.json"
            value = _read_json(upstream) if upstream.is_file() else {}
            self.state(
                "WAITING_FOR_COMPLETE_REPAIRED_FOLLOWUP_EXECUTION",
                upstream_status=value.get("status"),
                upstream_workers=value.get("workers"),
            )
            time.sleep(int(self.contract["poll_seconds"]))
        output_path = (
            self.operations / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        )
        result = materialize_extended_repaired_frontier(
            self.run_root, output_path=output_path,
        )
        self.state(
            result["status"],
            action_priority_candidate_id=result["action_priority_candidate_id"],
            priority_alternate_candidate_ids=result[
                "priority_alternate_candidate_ids"
            ],
            strict_candidate_ids=result["strict_candidate_ids"],
            rankable_complete_e200_candidate_count=result[
                "rankable_complete_e200_candidate_count"
            ],
            result_sha256=file_sha256(output_path),
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
            run_root / "operations" / "EXTENDED_REPAIRED_FRONTIER_SUCCESSOR.lock"
        ):
            return ExtendedRepairedFrontierSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "EXTENDED_REPAIRED_FRONTIER_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-extended-repaired-frontier-successor-fatal-v1",
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
