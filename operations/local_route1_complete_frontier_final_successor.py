"""Wait for both host-separated frontiers and publish the terminal delivery."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.complete_frontier_final_delivery import (
    POINTER,
    materialize_complete_frontier_final_delivery,
)


SCHEMA = "final-unsb-route1-complete-frontier-final-successor-contract-v1"
REQUIRED_RESULTS = (
    "COMPLETE_FRONTIER_4090_ADJUDICATION.json",
    "PORTABLE_EXTENDED_REPAIRED_FRONTIER_5090.json",
)
SOURCE_RELATIVES = (
    "operations/local_route1_complete_frontier_final_successor.py",
    "operations/local_route1_complete_frontier_final_delivery.py",
    "research/local_route1/complete_frontier_final_delivery.py",
    "research/local_route1/complete_frontier.py",
    "research/local_route1/portable_extended_frontier.py",
    "research/local_route1/frontier_final_delivery.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("complete final successor worktree must be clean")
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
        "requires_complete_same_host_4090_frontier": True,
        "requires_complete_portable_5090_mechanism_frontier": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "all_mechanism_bearing_candidates_preserved": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("complete final successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("complete final successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("complete final successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"complete final successor source changed: {relative}")
    fixed = {
        "requires_complete_same_host_4090_frontier": True,
        "requires_complete_portable_5090_mechanism_frontier": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "all_mechanism_bearing_candidates_preserved": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"complete final successor changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("complete final successor polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("complete final successor timeout is too short")


class CompleteFrontierFinalSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = (
            self.operations / "COMPLETE_FRONTIER_FINAL_SUCCESSOR_STATE.json"
        )
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-complete-frontier-final-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "canonical_candidate_is_action_priority_only": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "all_mechanism_bearing_candidates_preserved": True,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        while True:
            ready = {name: (self.operations / name).is_file() for name in REQUIRED_RESULTS}
            fatal = self.operations / "COMPLETE_FRONTIER_4090_SUCCESSOR_FATAL.json"
            if fatal.is_file():
                raise RuntimeError(f"complete 4090 frontier failed: {fatal}")
            if all(ready.values()):
                break
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for both complete frontiers")
            self.state("WAITING_FOR_BOTH_HOST_SEPARATED_COMPLETE_FRONTIERS", readiness=ready)
            time.sleep(int(self.contract["poll_seconds"]))
        result = materialize_complete_frontier_final_delivery(self.run_root)
        pointer_path = self.operations / POINTER
        self.state(
            result["status"],
            selected_candidate_id=result["selected_candidate_id"],
            research_frontier_candidate_count=result[
                "research_frontier_candidate_count"
            ],
            pointer_sha256=support.file_sha256(pointer_path),
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
            run_root / "operations" / "COMPLETE_FRONTIER_FINAL_SUCCESSOR.lock"
        ):
            return CompleteFrontierFinalSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "COMPLETE_FRONTIER_FINAL_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-complete-frontier-final-successor-fatal-v1",
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
