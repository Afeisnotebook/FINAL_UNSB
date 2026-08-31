"""Wait for every related e200 branch and publish the multi-algorithm result."""

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
    materialize_complete_frontier_final_delivery,
)
from research.local_route1.related_multi_algorithm_final_delivery import (
    HJPCNR_RECEIPT,
    POINTER,
    materialize_related_multi_algorithm_final_delivery,
)


SCHEMA = "final-unsb-route1-related-multi-algorithm-final-successor-contract-v1"
REQUIRED_RESULTS = (
    "COMPLETE_FRONTIER_4090_ADJUDICATION.json",
    "PORTABLE_COMPLETE_5090_FRONTIER.json",
    "RELATED_4090_HOST_ADJUDICATION.json",
    "RELATED_5090_HOST_ADJUDICATION.json",
    "RELATED_MULTI_HOST_ADJUDICATION.json",
    HJPCNR_RECEIPT,
)
SOURCE_RELATIVES = (
    "operations/local_route1_related_multi_algorithm_final_successor.py",
    "research/local_route1/related_multi_algorithm_final_delivery.py",
    "research/local_route1/related_algorithm_adjudication.py",
    "research/local_route1/complete_frontier_final_delivery.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("related final successor worktree must be clean")
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
        "requires_all_related_e200_branches": True,
        "requires_host_separated_complete_frontiers": True,
        "requires_hj_specific_single_view_e200_control": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("related final successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("related final successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("related final successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"related final successor source changed: {relative}")
    fixed = {
        "requires_all_related_e200_branches": True,
        "requires_host_separated_complete_frontiers": True,
        "requires_hj_specific_single_view_e200_control": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"related final successor changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("related final successor polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("related final successor timeout is too short")


class RelatedMultiAlgorithmFinalSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "RELATED_MULTI_ALGORITHM_FINAL_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-related-multi-algorithm-final-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "action_priority_is_not_scientific_exclusivity": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "cross_host_deltas_merged": False,
            "cross_seed_stability_claimed": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        while True:
            readiness = {
                name: (self.operations / name).is_file()
                for name in REQUIRED_RESULTS
            }
            if all(readiness.values()):
                break
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for related e200 frontier")
            self.state("WAITING_FOR_ALL_RELATED_E200_RESULTS", readiness=readiness)
            time.sleep(int(self.contract["poll_seconds"]))
        compatibility = materialize_complete_frontier_final_delivery(self.run_root)
        result = materialize_related_multi_algorithm_final_delivery(self.run_root)
        self.state(
            result["status"],
            action_priority_candidate_id=result["action_priority_candidate_id"],
            algorithm_set_status=result["algorithm_set_status"],
            strict_viable_candidate_count=result["strict_viable_candidate_count"],
            compatibility_action_priority_candidate_id=compatibility[
                "selected_candidate_id"
            ],
            pointer_sha256=support.file_sha256(self.operations / POINTER),
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
            run_root / "operations" / "RELATED_MULTI_ALGORITHM_FINAL.lock"
        ):
            return RelatedMultiAlgorithmFinalSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "RELATED_MULTI_ALGORITHM_FINAL_FATAL.json",
            {
                "schema": "final-unsb-route1-related-multi-algorithm-final-fatal-v1",
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
