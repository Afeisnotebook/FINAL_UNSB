"""Durably audit the exact local terminal delivery when its relay completes."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.goal_completion_audit import (
    materialize_goal_completion_audit,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-goal-completion-audit-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_goal_completion_audit_successor.py",
    "operations/local_route1_goal_completion_audit.py",
    "operations/local_route1_complete_final_result_relay.py",
    "research/local_route1/goal_completion_audit.py",
    "research/local_route1/complete_frontier_final_delivery.py",
    "research/local_route1/portable_extended_frontier.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("Goal-audit successor worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "delivery": str(args.delivery.resolve()),
        "output": str(args.output.resolve()),
        "state": str(args.state.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(value: dict[str, Any]) -> None:
    if value.get("schema") != SCHEMA:
        raise RuntimeError("Goal-audit successor contract schema mismatch")
    repo = Path(value["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != value.get("git_commit"):
        raise RuntimeError("Goal-audit successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("Goal-audit successor worktree is dirty")
    for relative, expected in value.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"Goal-audit successor source changed: {relative}")
    fixed = {
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise RuntimeError(f"Goal-audit successor boundary changed: {key}")
    if int(value.get("poll_seconds", 0)) < 30:
        raise RuntimeError("Goal-audit successor polling is too frequent")
    if int(value.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("Goal-audit successor timeout is too short")
    paths = [Path(value[key]).resolve() for key in ("delivery", "output", "state")]
    if len(set(paths)) != 3:
        raise RuntimeError("Goal-audit successor paths must be distinct")


class GoalCompletionAuditSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.delivery = Path(self.contract["delivery"])
        self.output = Path(self.contract["output"])
        self.state_path = Path(self.contract["state"])
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-goal-completion-audit-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "pid": os.getpid(),
            "contract_sha256": file_sha256(self.contract_path),
            "cross_seed_stability_claimed": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        manifest = self.delivery / "RELAY_MANIFEST.json"
        while not manifest.is_file():
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("Goal-audit successor timed out waiting for terminal relay")
            self.state("WAITING_FOR_EXACT_TERMINAL_DELIVERY_RELAY")
            time.sleep(int(self.contract["poll_seconds"]))
        result = materialize_goal_completion_audit(self.delivery, self.output)
        self.state(
            "TERMINAL_ARTIFACTS_PROVEN_FINAL_REPOSITORY_ADJUDICATION_REQUIRED",
            selected_candidate_id=result["selected_candidate_id"],
            goal_completion_audit_sha256=file_sha256(self.output),
            research_frontier_unique_candidate_count=result[
                "research_frontier_unique_candidate_count"
            ],
            terminal_artifact_requirements_proven=True,
            completion_claim_allowed=False,
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--delivery", type=Path)
    value.add_argument("--output", type=Path)
    value.add_argument("--state", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        missing = [
            name for name in ("repo", "delivery", "output", "state")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    try:
        successor = GoalCompletionAuditSuccessor(args.contract)
        state = Path(successor.contract["state"])
        with support.executor_lock(Path(str(state) + ".lock")):
            return successor.run()
    except Exception as error:
        contract = _read_json(args.contract)
        support.atomic_json(Path(contract["state"]), {
            "schema": "final-unsb-route1-goal-completion-audit-successor-fatal-v1",
            "updated": support.now(),
            "status": "FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "pid": os.getpid(),
            "cross_seed_stability_claimed": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
