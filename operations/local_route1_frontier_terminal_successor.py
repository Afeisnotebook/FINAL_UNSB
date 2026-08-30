"""Wait for both complete frontier trajectories and emit a replay decision."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.frontier_adjudication import (
    FRONTIER_IDS,
    REPLAY_STATUS,
    adjudicate_frontier,
)


SCHEMA = "final-unsb-route1-frontier-terminal-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_frontier_terminal_successor.py",
    "operations/local_route1_frontier_adjudicate.py",
    "operations/local_route1_cross_version_adjudicate.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/frontier_adjudication.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier terminal successor worktree must be clean")
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
        "candidate_ids": list(FRONTIER_IDS),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "complete_e200_only": True,
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("frontier terminal successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("frontier terminal successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier terminal successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"frontier terminal source changed: {relative}")
    if tuple(contract.get("candidate_ids", [])) != FRONTIER_IDS:
        raise RuntimeError("frontier terminal candidate identities changed")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("frontier terminal polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("frontier terminal timeout is too short")
    if contract.get("complete_e200_only") is not True:
        raise RuntimeError("frontier terminal routing must wait for complete e200")
    for key in (
        "intermediate_metric_routing", "cross_host_deltas_merged",
        "paired_controller_access", "confirmation20_opened",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"frontier terminal successor requires {key}=false")
    if contract.get("selection_seeds") != [2026]:
        raise RuntimeError("frontier terminal successor is seed2026-only")
    if contract.get("deferred_seed_validation") != [2027, 2028]:
        raise RuntimeError("frontier terminal deferred seed policy changed")


def validate_completion(run_root: Path, complete: dict[str, Any]) -> list[Path]:
    if complete.get("schema") != "final-unsb-route1-frontier-e200-complete-v1":
        raise RuntimeError("frontier completion schema mismatch")
    if complete.get("status") != "FRONTIER_E200_COMPLETE_ADJUDICATION_REQUIRED":
        raise RuntimeError("frontier completion is not terminal")
    if complete.get("selection_seeds") != [2026]:
        raise RuntimeError("frontier completion seed identity changed")
    for key in ("paired_metric_scheduling", "paired_controller_access", "confirmation20_opened"):
        if complete.get(key) is not False:
            raise RuntimeError(f"frontier completion requires {key}=false")
    receipt_map = complete.get("candidate_receipts", {})
    if set(receipt_map) != set(FRONTIER_IDS):
        raise RuntimeError("frontier completion receipt set changed")
    paths = []
    root = Path(run_root).resolve()
    for candidate_id in FRONTIER_IDS:
        record = receipt_map[candidate_id]
        path = (root / record["path"]).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RuntimeError("frontier receipt escapes run root") from error
        if not path.is_file() or support.file_sha256(path) != record.get("sha256"):
            raise RuntimeError(f"frontier receipt integrity failed: {candidate_id}")
        paths.append(path)
    return paths


class FrontierTerminalSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "FRONTIER_TERMINAL_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-frontier-terminal-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_ids": list(FRONTIER_IDS),
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "intermediate_metric_routing": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        complete_path = self.operations / "FRONTIER_E200_COMPLETE.json"
        fatal_path = self.operations / "FRONTIER_SUCCESSOR_FATAL.json"
        while not complete_path.is_file():
            if fatal_path.is_file():
                raise RuntimeError(f"frontier training successor failed: {fatal_path}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for complete frontier e200 receipts")
            epochs = {
                candidate_id: support.current_epoch(self.run_root, candidate_id)
                for candidate_id in FRONTIER_IDS
            }
            self.state(
                "WAITING_FOR_COMPLETE_FRONTIER_E200",
                data_epochs=epochs,
                elapsed_seconds=time.time() - self.started,
            )
            time.sleep(int(self.contract["poll_seconds"]))

        complete = _read_json(complete_path)
        receipt_paths = validate_completion(self.run_root, complete)
        adjudication_path = self.operations / "FRONTIER_E200_ADJUDICATION.json"
        result = adjudicate_frontier(receipt_paths, adjudication_path)
        decision = {
            "schema": "final-unsb-route1-frontier-4090-replay-decision-v1",
            "status": (
                "REPLAY_REQUEST_READY_REQUIRES_4090_SOURCE_BOUND_EXECUTOR"
                if result["status"] == REPLAY_STATUS else
                "NO_4090_REPLAY_FRONTIER_CURRENT_IMPLEMENTATIONS_NEGATIVE"
            ),
            "recommended_candidate_id": result[
                "recommended_4090_replay_candidate_id"
            ],
            "recommended_algorithm_fingerprint": result[
                "recommended_4090_replay_algorithm_fingerprint"
            ],
            "frontier_adjudication_path": str(adjudication_path),
            "frontier_adjudication_sha256": support.file_sha256(adjudication_path),
            "complete_frontier_sha256": support.file_sha256(complete_path),
            "complete_e200_only": True,
            "intermediate_metric_routing": False,
            "cross_host_deltas_merged": False,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        decision_path = self.operations / "FRONTIER_4090_REPLAY_DECISION.json"
        support.atomic_json(decision_path, decision)
        self.state(
            "FRONTIER_TERMINAL_ADJUDICATION_COMPLETE",
            adjudication_status=result["status"],
            adjudication_sha256=support.file_sha256(adjudication_path),
            replay_decision_sha256=support.file_sha256(decision_path),
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--timeout-seconds", type=int, default=172800)
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
            run_root / "operations" / "FRONTIER_TERMINAL_SUCCESSOR.lock"
        ):
            return FrontierTerminalSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "FRONTIER_TERMINAL_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-frontier-terminal-successor-fatal-v1",
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

