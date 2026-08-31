"""Wait for both local deliveries and run the terminal multi-algorithm audit."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.related_goal_completion_audit import (
    materialize_related_goal_completion_audit,
)


SCHEMA = "final-unsb-route1-related-goal-completion-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_related_goal_completion_successor.py",
    "research/local_route1/related_goal_completion_audit.py",
    "research/local_route1/goal_completion_audit.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("related Goal audit successor worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "compatibility_delivery": str(args.compatibility_delivery.resolve()),
        "related_delivery": str(args.related_delivery.resolve()),
        "output": str(args.output.resolve()),
        "state": str(args.state.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "requires_related_algorithm_set": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("related Goal audit successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("related Goal audit successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("related Goal audit successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"related Goal audit successor source changed: {relative}")
    fixed = {
        "requires_related_algorithm_set": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"related Goal audit successor changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 30:
        raise RuntimeError("related Goal audit successor polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("related Goal audit successor timeout is too short")


class RelatedGoalCompletionSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.compatibility = Path(self.contract["compatibility_delivery"])
        self.related = Path(self.contract["related_delivery"])
        self.output = Path(self.contract["output"])
        self.state_path = Path(self.contract["state"])
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-related-goal-completion-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "pid": os.getpid(),
            "action_priority_is_not_scientific_exclusivity": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        while True:
            readiness = {
                "compatibility_relay_manifest": (
                    self.compatibility / "RELAY_MANIFEST.json"
                ).is_file(),
                "related_relay_manifest": (
                    self.related / "RELAY_MANIFEST.json"
                ).is_file(),
            }
            if all(readiness.values()):
                break
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for both local final deliveries")
            self.state("WAITING_FOR_BOTH_LOCAL_FINAL_DELIVERIES", readiness=readiness)
            time.sleep(int(self.contract["poll_seconds"]))
        result = materialize_related_goal_completion_audit(
            self.compatibility, self.related, self.output,
        )
        self.state(
            result["status"],
            action_priority_candidate_id=result["action_priority_candidate_id"],
            algorithm_set_status=result["algorithm_set_status"],
            algorithm_member_count=result["algorithm_member_count"],
            audit_sha256=support.file_sha256(self.output),
            final_repository_commit_and_push_required=True,
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--compatibility-delivery", type=Path)
    value.add_argument("--related-delivery", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--state", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "compatibility_delivery", "related_delivery", "output", "state",
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
    try:
        return RelatedGoalCompletionSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(Path(contract["state"]), {
            "schema": "final-unsb-route1-related-goal-completion-successor-fatal-v1",
            "updated": support.now(),
            "status": "FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "pid": os.getpid(),
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

