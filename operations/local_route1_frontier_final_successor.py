"""Durably wait for frontier routing and freeze the canonical final delivery."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.frontier_final_delivery import (
    materialize_frontier_final_delivery,
)


SCHEMA = "final-unsb-route1-frontier-final-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_frontier_final_successor.py",
    "operations/local_route1_frontier_final_delivery.py",
    "operations/local_route1_frontier_cross_host_successor.py",
    "operations/local_route1_cross_version_adjudicate.py",
    "research/local_route1/frontier_final_delivery.py",
    "operations/local_route1_frontier_winner_ablation_successor.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier final successor worktree must be clean")
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
        "requires_terminal_cross_host_result": True,
        "requires_selected_algorithm_specific_ablations": True,
        "preserve_pre_frontier_delivery": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("frontier final successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("frontier final successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier final successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"frontier final successor source changed: {relative}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("frontier final polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("frontier final timeout is too short")
    fixed = {
        "requires_terminal_cross_host_result": True,
        "requires_selected_algorithm_specific_ablations": True,
        "preserve_pre_frontier_delivery": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"frontier final successor changed: {key}")


class FrontierFinalSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "FRONTIER_FINAL_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-frontier-final-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(),
            "cross_host_deltas_merged": False,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        result_path = self.operations / "FRONTIER_WINNER_ABLATION_RESULT.json"
        fatal_path = self.operations / "FRONTIER_WINNER_ABLATION_SUCCESSOR_FATAL.json"
        while not result_path.is_file():
            if fatal_path.is_file():
                raise RuntimeError(f"frontier winner ablation successor failed: {fatal_path}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for selected algorithm ablations")
            state_path = self.operations / "FRONTIER_WINNER_ABLATION_SUCCESSOR_STATE.json"
            ablation_state = _read_json(state_path) if state_path.is_file() else {}
            self.state(
                "WAITING_FOR_SELECTED_ALGORITHM_SPECIFIC_ABLATIONS",
                winner_ablation_status=ablation_state.get("status"),
                winner_ablation_data_epoch=ablation_state.get("active_data_epoch"),
            )
            time.sleep(int(self.contract["poll_seconds"]))
        pointer = materialize_frontier_final_delivery(self.run_root)
        self.state(
            "FRONTIER_FINAL_DELIVERY_COMPLETE",
            selected_candidate_id=pointer["selected_candidate_id"],
            final_delivery_pointer_sha256=support.file_sha256(
                self.operations / "ROUTE1_FINAL_DELIVERY_POINTER.json"
            ),
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
            run_root / "operations" / "FRONTIER_FINAL_SUCCESSOR.lock"
        ):
            return FrontierFinalSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "FRONTIER_FINAL_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-frontier-final-successor-fatal-v1",
                "updated": support.now(), "status": "FAILED",
                "error": repr(error), "traceback": traceback.format_exc(),
                "supervisor_pid": os.getpid(),
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
