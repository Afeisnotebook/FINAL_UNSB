"""Wait for both numerical repairs and adjudicate the complete 5090 frontier."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.protocol import file_sha256
from research.local_route1.repaired_frontier_adjudication import (
    INVALID_DIAGNOSTIC_INCIDENTS,
    RANKABLE_IDS,
    adjudicate_repaired_frontier,
)


SCHEMA = "final-unsb-route1-repaired-frontier-successor-contract-v1"
RESULT_REQUIREMENTS = {
    "RFAMMCRB_E200_RESULT.json": (
        "final-unsb-route1-rfammcrb-e200-result-v1",
        "RFAMMCRB_SEMANTIC_REPAIR_COMPLETE_E200",
        "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER",
    ),
    "RFMCRB_E200_RESULT.json": (
        "final-unsb-route1-rfmcrb-e200-result-v1",
        "RFMCRB_SEMANTIC_REPAIR_COMPLETE_E200",
        "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER",
    ),
}
SOURCE_RELATIVES = (
    "operations/local_route1_repaired_frontier_successor.py",
    "operations/local_route1_repaired_frontier_adjudicate.py",
    "operations/local_route1_cross_version_adjudicate.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/repaired_frontier_adjudication.py",
    "research/local_route1/frontier_advancement.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("repaired frontier successor worktree must be clean")
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
        "rankable_candidate_ids": list(RANKABLE_IDS),
        "implementation_invalid_diagnostic_ids": sorted(
            INVALID_DIAGNOSTIC_INCIDENTS
        ),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "complete_e200_only": True,
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
        raise RuntimeError("repaired frontier successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("repaired frontier successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("repaired frontier successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"repaired frontier successor source changed: {relative}")
    fixed = {
        "rankable_candidate_ids": list(RANKABLE_IDS),
        "implementation_invalid_diagnostic_ids": sorted(
            INVALID_DIAGNOSTIC_INCIDENTS
        ),
        "complete_e200_only": True,
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
            raise RuntimeError(f"repaired frontier successor contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("repaired frontier polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("repaired frontier timeout is too short")


class RepairedFrontierSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.receipts = self.operations / "terminal_receipts"
        self.state_path = self.operations / "REPAIRED_FRONTIER_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-repaired-frontier-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "rankable_candidate_ids": list(RANKABLE_IDS),
            "implementation_invalid_diagnostic_ids": sorted(
                INVALID_DIAGNOSTIC_INCIDENTS
            ),
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _validate_result(self, filename: str) -> bool:
        path = self.operations / filename
        if not path.is_file():
            return False
        schema, status, candidate_id = RESULT_REQUIREMENTS[filename]
        result = _read_json(path)
        if (
            result.get("schema") != schema
            or result.get("status") != status
            or result.get("candidate_id") != candidate_id
            or result.get("paired_metrics_used_for_training_or_control") is not False
            or result.get("confirmation20_opened") is not False
        ):
            raise RuntimeError(f"invalid completed repair result: {filename}")
        receipt_path = Path(result["terminal_receipt_path"]).resolve()
        if (
            not receipt_path.is_file()
            or file_sha256(receipt_path) != result.get("terminal_receipt_sha256")
        ):
            raise RuntimeError(f"repair terminal receipt integrity failed: {candidate_id}")
        return True

    def _receipt_paths(self) -> tuple[list[Path], list[Path]]:
        return (
            [self.receipts / f"{candidate_id}.json" for candidate_id in RANKABLE_IDS],
            [
                self.receipts / f"{candidate_id}.json"
                for candidate_id in INVALID_DIAGNOSTIC_INCIDENTS
            ],
        )

    def wait_for_complete_frontier(self) -> tuple[list[Path], list[Path]]:
        fatal_paths = (
            self.operations / "RFAMMCRB_SUCCESSOR_FATAL.json",
            self.operations / "RFMCRB_SUCCESSOR_FATAL.json",
            self.operations / "FRONTIER_SUCCESSOR_FATAL.json",
        )
        while True:
            fatal = next((path for path in fatal_paths if path.is_file()), None)
            if fatal is not None:
                raise RuntimeError(f"repaired frontier prerequisite failed: {fatal}")
            result_ready = {
                filename: self._validate_result(filename)
                for filename in RESULT_REQUIREMENTS
            }
            rankable, invalid = self._receipt_paths()
            receipt_ready = {
                path.stem: path.is_file() and Path(str(path) + ".sha256.json").is_file()
                for path in [*rankable, *invalid]
            }
            self.state(
                "WAITING_FOR_COMPLETE_REPAIRED_FRONTIER_E200",
                repair_results_ready=result_ready,
                receipts_ready=receipt_ready,
                elapsed_seconds=time.time() - self.started,
            )
            if all(result_ready.values()) and all(receipt_ready.values()):
                return rankable, invalid
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for complete repaired frontier")
            time.sleep(int(self.contract["poll_seconds"]))

    def run(self) -> int:
        rankable, invalid = self.wait_for_complete_frontier()
        output_path = self.operations / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        result = adjudicate_repaired_frontier(rankable, invalid, output_path)
        self.state(
            result["status"],
            action_priority_candidate_id=result["action_priority_candidate_id"],
            priority_alternate_candidate_ids=result["priority_alternate_candidate_ids"],
            strict_candidate_ids=result["strict_candidate_ids"],
            recommended_4090_replay_queue=result["recommended_4090_replay_queue"],
            result_sha256=file_sha256(output_path),
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--timeout-seconds", type=int, default=345600)
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
            run_root / "operations" / "REPAIRED_FRONTIER_SUCCESSOR.lock"
        ):
            return RepairedFrontierSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "REPAIRED_FRONTIER_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-repaired-frontier-successor-fatal-v1",
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
