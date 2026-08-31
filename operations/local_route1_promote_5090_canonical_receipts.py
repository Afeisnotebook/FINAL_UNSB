"""Promote validated 5090 replay receipts to canonical host-local names.

The cross-runtime executor intentionally writes ``<candidate>_5090.json``.
Downstream host-local consumers use the canonical ``<candidate>.json`` name.
This successor bridges only that naming boundary after the complete portfolio
result exists; it copies the already-validated receipt bytes and never touches
checkpoints, trajectories, metrics, formulas, or rankings.
"""

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
from research.local_route1.cross_runtime_portfolio import REPLAY_IDS, RESULT_SCHEMA
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-promote-5090-canonical-receipts-contract-v1"
RESULT_FILE = "CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json"
PROMOTION_FILE = "CROSS_RUNTIME_5090_CANONICAL_RECEIPTS.json"
SOURCE_RELATIVES = (
    "operations/local_route1_promote_5090_canonical_receipts.py",
    "operations/local_route1_cross_version_adjudicate.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_exact_copy(source: Path, destination: Path) -> None:
    payload = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"canonical 5090 receipt differs: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)


def promote_canonical_receipts(run_root: Path) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    operations = run_root / "operations"
    result_path = operations / RESULT_FILE
    if not result_path.is_file():
        raise RuntimeError("complete 5090 cross-runtime portfolio is absent")
    result = _read_json(result_path)
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != "CROSS_RUNTIME_5090_PORTFOLIO_COMPLETE_E200"
        or result.get("cross_host_deltas_merged") is not False
        or result.get("paired_metrics_used_for_formula_or_training_control") is not False
        or result.get("paired_controller_access") is not False
        or result.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("5090 cross-runtime portfolio is not admissible")
    rows = {
        str(row.get("candidate_id", "")): row
        for row in result.get("candidate_results", []) if isinstance(row, dict)
    }
    if set(rows) != set(REPLAY_IDS):
        raise RuntimeError("5090 cross-runtime portfolio candidate set changed")

    promoted = []
    for candidate_id in REPLAY_IDS:
        row = rows[candidate_id]
        source = Path(str(row.get("receipt_path", ""))).resolve()
        try:
            source.relative_to(run_root)
        except ValueError as error:
            raise RuntimeError("5090 replay receipt escapes the run root") from error
        expected_name = f"{candidate_id}_5090.json"
        if source.name != expected_name or not source.is_file():
            raise RuntimeError(f"5090 replay receipt path changed: {candidate_id}")
        if file_sha256(source) != row.get("receipt_sha256"):
            raise RuntimeError(f"5090 replay receipt hash changed: {candidate_id}")
        receipt = _validate_receipt(source)
        if (
            receipt.get("candidate_id") != candidate_id
            or receipt.get("algorithm_fingerprint") != row.get("algorithm_fingerprint")
            or receipt.get("confirmation20_opened") is not False
        ):
            raise RuntimeError(f"5090 replay receipt identity changed: {candidate_id}")
        destination = operations / "terminal_receipts" / f"{candidate_id}.json"
        _atomic_exact_copy(source, destination)
        if file_sha256(destination) != file_sha256(source):
            raise RuntimeError(f"canonical receipt copy changed: {candidate_id}")
        promoted.append({
            "candidate_id": candidate_id,
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "source_receipt_path": str(source),
            "source_receipt_sha256": file_sha256(source),
            "canonical_receipt_path": str(destination),
            "canonical_receipt_sha256": file_sha256(destination),
            "byte_identical": True,
        })
    promotion = {
        "schema": "final-unsb-route1-5090-canonical-receipt-promotion-v1",
        "status": "CANONICAL_5090_RECEIPTS_REGISTERED",
        "source_result_path": str(result_path),
        "source_result_sha256": file_sha256(result_path),
        "promoted_receipts": promoted,
        "checkpoint_transfer": False,
        "formula_changed": False,
        "ranking_changed": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    support.atomic_json(operations / PROMOTION_FILE, promotion)
    return promotion


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("5090 receipt-promotion worktree must be clean")
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
        "source_receipt_suffix": "_5090",
        "destination_overwrite_allowed": False,
        "byte_identical_copy_required": True,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("5090 receipt-promotion contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("5090 receipt-promotion worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("5090 receipt-promotion worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"5090 receipt-promotion source changed: {relative}")
    fixed = {
        "source_receipt_suffix": "_5090",
        "destination_overwrite_allowed": False,
        "byte_identical_copy_required": True,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"5090 receipt-promotion contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 30:
        raise RuntimeError("5090 receipt-promotion polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("5090 receipt-promotion timeout is too short")


class CanonicalReceiptSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "CROSS_RUNTIME_5090_RECEIPT_PROMOTION_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-5090-receipt-promotion-state-v1",
            "updated": support.now(),
            "status": status,
            "pid": os.getpid(),
            "checkpoint_transfer": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def run(self) -> int:
        result_path = self.operations / RESULT_FILE
        while not result_path.is_file():
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for 5090 cross-runtime result")
            self.state("WAITING_FOR_COMPLETE_5090_CROSS_RUNTIME_PORTFOLIO")
            time.sleep(int(self.contract["poll_seconds"]))
        result = promote_canonical_receipts(self.run_root)
        self.state(
            result["status"],
            promotion_sha256=file_sha256(self.operations / PROMOTION_FILE),
            promoted_candidate_ids=[
                row["candidate_id"] for row in result["promoted_receipts"]
            ],
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
            run_root / "operations" / "CROSS_RUNTIME_5090_RECEIPT_PROMOTION.lock"
        ):
            return CanonicalReceiptSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "CROSS_RUNTIME_5090_RECEIPT_PROMOTION_FATAL.json",
            {
                "schema": "final-unsb-route1-5090-receipt-promotion-fatal-v1",
                "updated": support.now(),
                "status": "FAILED",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "pid": os.getpid(),
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
