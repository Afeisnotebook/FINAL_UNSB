"""Wait for complete frontier e200 and write the second-wave classification."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from operations.local_route1_cross_version_adjudicate import _validate_receipt
from research.local_route1.frontier_advancement import classify_frontier
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-frontier-advancement-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_frontier_advancement_successor.py",
    "operations/local_route1_cross_version_adjudicate.py",
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
        raise RuntimeError("frontier advancement successor worktree must be clean")
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
        "complete_e200_only": True,
        "maximum_second_wave_e200_trajectories": 2,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("frontier advancement successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("frontier advancement successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier advancement successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"frontier advancement source changed: {relative}")
    fixed = {
        "complete_e200_only": True,
        "maximum_second_wave_e200_trajectories": 2,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"frontier advancement contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("frontier advancement polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("frontier advancement timeout is too short")


class FrontierAdvancementSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "FRONTIER_ADVANCEMENT_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-frontier-advancement-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "maximum_second_wave_e200_trajectories": 2,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_metrics_used_for_formula_or_training_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_adjudication(self) -> Path:
        path = self.operations / "FRONTIER_E200_ADJUDICATION.json"
        fatal = self.operations / "FRONTIER_TERMINAL_SUCCESSOR_FATAL.json"
        while not path.is_file():
            if fatal.is_file():
                raise RuntimeError(f"frontier terminal successor failed: {fatal}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for complete frontier adjudication")
            self.state("WAITING_FOR_COMPLETE_FRONTIER_E200")
            time.sleep(int(self.contract["poll_seconds"]))
        return path

    def run(self) -> int:
        adjudication_path = self.wait_adjudication()
        adjudication = _read_json(adjudication_path)
        rows = []
        for ranking in adjudication.get("ranking", []):
            receipt_path = Path(ranking["receipt_path"]).resolve()
            if not receipt_path.is_file() or file_sha256(receipt_path) != ranking.get(
                "receipt_sha256"
            ):
                raise RuntimeError("frontier advancement receipt integrity failed")
            receipt = _validate_receipt(receipt_path)
            trajectory_path = Path(receipt["trajectory_path"]).resolve()
            if not trajectory_path.is_file() or file_sha256(trajectory_path) != receipt.get(
                "trajectory_sha256"
            ):
                raise RuntimeError("frontier advancement trajectory integrity failed")
            rows.append((receipt, _read_json(trajectory_path)))
        if len(rows) != 2:
            raise RuntimeError("frontier advancement requires both complete candidates")
        result = classify_frontier(rows)
        result.update({
            "frontier_adjudication_path": str(adjudication_path),
            "frontier_adjudication_sha256": file_sha256(adjudication_path),
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
        })
        result_path = self.operations / "FRONTIER_ADVANCEMENT_CLASSIFICATION.json"
        support.atomic_json(result_path, result)
        self.state(
            result["status"],
            strict_candidate_ids=result["strict_candidate_ids"],
            near_boundary_pending_target_blind_audit_ids=result[
                "near_boundary_pending_target_blind_audit_ids"
            ],
            result_sha256=file_sha256(result_path),
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
            run_root / "operations" / "FRONTIER_ADVANCEMENT_SUCCESSOR.lock"
        ):
            return FrontierAdvancementSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "FRONTIER_ADVANCEMENT_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-frontier-advancement-successor-fatal-v1",
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

