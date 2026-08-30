"""Durably continue an authorized causal revision through true e200.

The negative-e200 auditor is deliberately not allowed to invent mathematics.
It may therefore stop at ``REVISION_DERIVATION_REQUIRED``.  This successor
closes the execution gap: it waits for a separately frozen, source-bound
revision authorization, validates that the revision is the single permitted
Generation-2 child of the audited parent, runs its durable candidate executor,
materializes a terminal receipt and ranks that receipt with the two original
Generation-1 receipts.

No paired score, fixed window, handoff rule or best checkpoint is available to
this process.  Absence of an authorization is a wait state, never permission
to synthesize a formula.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from operations import local_route1_candidate_executor as support
except ModuleNotFoundError:  # direct execution from operations/
    import local_route1_candidate_executor as support  # type: ignore[no-redef]

from operations.local_route1_cross_version_adjudicate import adjudicate
from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-cross-version-revision-successor-contract-v1"
AUTH_SCHEMA = "final-unsb-route1-cross-version-revision-execution-authorization-v1"
STATE_SCHEMA = "final-unsb-route1-cross-version-revision-successor-state-v1"
FINAL_SCHEMA = "final-unsb-route1-cross-version-final-revision-outcome-v2"
REVISION_REQUIRED = "REVISION_DERIVATION_REQUIRED"
NO_REVISION = "NO_REVISION_APPLICABLE_FINAL_FALLBACK"
EXECUTOR_SCHEMA = "final-unsb-route1-candidate-executor-contract-v1"
ORIGINAL_IDS = (
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
    "G1-02B-PLAYER-CONDITIONAL-RSMG",
)
SUCCESSOR_SOURCES = (
    "operations/local_route1_cross_version_revision_successor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "operations/local_route1_cross_version_adjudicate.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _environment(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.successor_repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("revision successor worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "successor_repo": str(repo),
        "successor_git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "successor_source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SUCCESSOR_SOURCES
        },
        "run_root": str(args.run_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "development_seeds": [2026],
        "deferred_seeds": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("revision successor contract schema mismatch")
    repo = Path(str(contract["successor_repo"])).resolve()
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "successor_git_commit"
    ):
        raise RuntimeError("revision successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("revision successor worktree is dirty")
    for relative, expected in contract.get("successor_source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"revision successor source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("revision successor manifest changed")
    if contract.get("development_seeds") != [2026] or contract.get("deferred_seeds") != [2027, 2028]:
        raise RuntimeError("revision successor seed policy changed")
    if int(contract.get("poll_seconds", 0)) < 15 or int(contract.get("timeout_seconds", 0)) < 3600:
        raise RuntimeError("revision successor polling/timeout contract is unsafe")
    for key in ("paired_metric_scheduling", "paired_controller_access", "confirmation20_opened"):
        if contract.get(key) is not False:
            raise RuntimeError(f"revision successor requires {key}=false")


def _critical_revision_sources(repo: Path, implementation: dict[str, Any]) -> dict[str, str]:
    relatives = {
        "operations/local_route1_candidate_executor.py",
        "operations/local_route1_candidate_terminal_receipt.py",
        "operations/local_route1_cross_version_adjudicate.py",
        "research/local_route1/candidates.py",
        "research/local_route1/candidate_gate.py",
        "research/local_route1/candidate_runner.py",
    }
    for row in implementation.get("source_files", []):
        if isinstance(row, dict) and row.get("path"):
            relatives.add(str(row["path"]))
    return {
        relative: support.file_sha256(repo / relative)
        for relative in sorted(relatives)
    }


def default_authorization(
    *, contract: dict[str, Any], candidate_repo: Path, candidate_id: str,
    executor_contract_path: Path,
) -> dict[str, Any]:
    validate_contract(contract)
    run_root = Path(str(contract["run_root"])).resolve()
    outcome_path = run_root / "operations" / "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json"
    outcome = _read_json(outcome_path)
    if outcome.get("schema") != CROSS_VERSION_FINAL_OUTCOME_SCHEMA or outcome.get("status") != REVISION_REQUIRED:
        raise RuntimeError("revision authorization requires an eligible frozen defect outcome")
    candidate_id = support.safe_candidate_id(candidate_id)
    if candidate_id in ORIGINAL_IDS:
        raise RuntimeError("revision candidate must have a new identity")
    repo = candidate_repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("revision candidate worktree must be clean")
    executor_path = executor_contract_path.resolve()
    executor = _read_json(executor_path)
    card_path = run_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = run_root / "derive" / "implementations" / f"{candidate_id}.json"
    gate_path = run_root / "derive" / "gates" / f"{candidate_id}.json"
    ledger_path = run_root / "derive" / "HYPOTHESIS_LEDGER.json"
    card = _read_json(card_path)
    implementation = _read_json(implementation_path)
    gate = _read_json(gate_path)
    value = {
        "schema": AUTH_SCHEMA,
        "created": support.now(),
        "source_revision_need_sha256": file_sha256(outcome_path),
        "parent_candidate_id": str(outcome["selected_candidate_id"]),
        "candidate_id": candidate_id,
        "generation": 2,
        "revision_count": 1,
        "candidate_repo": str(repo),
        "candidate_git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "candidate_source_sha256": _critical_revision_sources(repo, implementation),
        "executor_contract": str(executor_path),
        "executor_contract_sha256": file_sha256(executor_path),
        "algorithm_fingerprint": executor.get("algorithm_fingerprint"),
        "candidate_fingerprint": executor.get("candidate_fingerprint"),
        "derivation_card_sha256": file_sha256(card_path),
        "implementation_sha256": file_sha256(implementation_path),
        "gate_sha256": file_sha256(gate_path),
        "hypothesis_ledger_sha256": file_sha256(ledger_path),
        "seed": 2026,
        "batch_size": 1,
        "target_data_epochs": 200,
        "best_checkpoint_selection": False,
        "fixed_window_or_handoff": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    validate_authorization(contract, outcome, value)
    return value


def validate_authorization(
    contract: dict[str, Any], outcome: dict[str, Any], authorization: dict[str, Any],
) -> dict[str, Any]:
    if authorization.get("schema") != AUTH_SCHEMA:
        raise RuntimeError("revision authorization schema mismatch")
    run_root = Path(str(contract["run_root"])).resolve()
    outcome_path = run_root / "operations" / "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json"
    if authorization.get("source_revision_need_sha256") != file_sha256(outcome_path):
        raise RuntimeError("revision authorization does not bind the defect outcome")
    if outcome.get("status") != REVISION_REQUIRED:
        raise RuntimeError("revision authorization is not applicable")
    parent = str(outcome.get("selected_candidate_id", ""))
    if authorization.get("parent_candidate_id") != parent:
        raise RuntimeError("revision parent differs from the audited parent")
    candidate_id = support.safe_candidate_id(str(authorization.get("candidate_id", "")))
    if candidate_id in ORIGINAL_IDS or candidate_id == parent:
        raise RuntimeError("revision candidate identity is not new")
    if authorization.get("generation") != 2 or authorization.get("revision_count") != 1:
        raise RuntimeError("only one Generation-2 causal revision is permitted")
    if authorization.get("seed") != 2026 or authorization.get("batch_size") != 1:
        raise RuntimeError("revision must use the emergency batch1 seed2026 protocol")
    if authorization.get("target_data_epochs") != 200 or authorization.get("best_checkpoint_selection") is not False:
        raise RuntimeError("revision must run to fixed e200")
    for key in (
        "fixed_window_or_handoff", "paired_metric_scheduling",
        "paired_controller_access", "confirmation20_opened",
    ):
        if authorization.get(key) is not False:
            raise RuntimeError(f"revision authorization requires {key}=false")
    repo = Path(str(authorization["candidate_repo"])).resolve()
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != authorization.get(
        "candidate_git_commit"
    ):
        raise RuntimeError("revision candidate worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("revision candidate worktree is dirty")
    for relative, expected in authorization.get("candidate_source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"revision candidate source changed: {relative}")
    for relative in (
        "operations/local_route1_candidate_terminal_receipt.py",
        "operations/local_route1_cross_version_adjudicate.py",
    ):
        if authorization["candidate_source_sha256"].get(relative) != contract[
            "successor_source_sha256"
        ].get(relative):
            raise RuntimeError(f"revision verifier semantics differ: {relative}")
    executor_path = Path(str(authorization["executor_contract"])).resolve()
    if file_sha256(executor_path) != authorization.get("executor_contract_sha256"):
        raise RuntimeError("revision executor contract changed")
    executor = _read_json(executor_path)
    required = {
        "schema": EXECUTOR_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_repo": str(repo),
        "candidate_git_commit": authorization["candidate_git_commit"],
        "run_root": str(run_root),
        "manifest_sha256": contract["manifest_sha256"],
        "target_data_epochs": 200,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in required.items():
        if executor.get(key) != expected:
            raise RuntimeError(f"revision executor contract mismatch: {key}")
    for key in ("algorithm_fingerprint", "candidate_fingerprint"):
        if authorization.get(key) != executor.get(key):
            raise RuntimeError(f"revision authorization changed {key}")
    card_path = run_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = run_root / "derive" / "implementations" / f"{candidate_id}.json"
    gate_path = run_root / "derive" / "gates" / f"{candidate_id}.json"
    ledger_path = run_root / "derive" / "HYPOTHESIS_LEDGER.json"
    if file_sha256(card_path) != authorization.get("derivation_card_sha256"):
        raise RuntimeError("revision derivation card changed")
    if file_sha256(implementation_path) != authorization.get("implementation_sha256"):
        raise RuntimeError("revision implementation changed")
    if file_sha256(gate_path) != authorization.get("gate_sha256"):
        raise RuntimeError("revision gate changed")
    if file_sha256(ledger_path) != authorization.get("hypothesis_ledger_sha256"):
        raise RuntimeError("revision hypothesis ledger changed")
    card = _read_json(card_path)
    gate = _read_json(gate_path)
    ledger = _read_json(ledger_path)
    records = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == candidate_id
    ]
    if (
        card.get("parent_candidate_id") != parent
        or card.get("paired_target_available_to_training") is not False
    ):
        raise RuntimeError("revision card lineage or target boundary is invalid")
    if (
        len(records) != 1
        or records[0].get("parent_candidate_id") != parent
        or records[0].get("generation") != 2
        or records[0].get("revision_count") != 1
        or records[0].get("status") != "FROZEN_FOR_GATES"
        or records[0].get("paired_controller_access") is not False
        or records[0].get("confirmation20_opened") is not False
    ):
        raise RuntimeError("revision ledger does not contain one frozen Generation-2 child")
    if (
        gate.get("status") != "PASS_LONG_RUN"
        or gate.get("algorithm_fingerprint") != executor.get("algorithm_fingerprint")
        or gate.get("candidate_fingerprint") != executor.get("candidate_fingerprint")
        or gate.get("paired_metric_used_for_promotion") is not False
        or gate.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("revision executable gate is not valid")
    return executor


class RevisionSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(str(self.contract["run_root"])).resolve()
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "CROSS_VERSION_REVISION_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "CROSS_VERSION_REVISION_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": STATE_SCHEMA, "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(), "paired_metric_scheduling": False,
            "paired_controller_access": False, "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-cross-version-revision-successor-event-v1",
            "time": support.now(), "event": event, "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False, "paired_controller_access": False,
            "confirmation20_opened": False, **fields,
        })

    def wait_json(self, path: Path, status: str) -> dict[str, Any]:
        started = time.time()
        while True:
            if path.is_file():
                return _read_json(path)
            self.state(status, waiting_for=str(path), elapsed_seconds=time.time() - started)
            if time.time() - started > int(self.contract["timeout_seconds"]):
                raise TimeoutError(f"revision successor timed out waiting for {path}")
            time.sleep(int(self.contract["poll_seconds"]))

    def run(self) -> int:
        self.event("REVISION_SUCCESSOR_START", contract=str(self.contract_path))
        outcome_path = self.operations / "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json"
        outcome = self.wait_json(outcome_path, "WAITING_FOR_TARGET_BLIND_REVISION_NEED")
        if outcome.get("schema") != CROSS_VERSION_FINAL_OUTCOME_SCHEMA:
            raise RuntimeError("revision need outcome schema mismatch")
        if outcome.get("status") == NO_REVISION:
            self.state("INAPPLICABLE_NO_SAFE_REVISION", selected_fallback=outcome.get("selected_candidate_id"))
            return 0
        if outcome.get("status") != REVISION_REQUIRED:
            raise RuntimeError("unknown causal revision outcome")
        auth_path = self.operations / "CAUSAL_REVISION_EXECUTION_AUTHORIZATION.json"
        authorization = self.wait_json(
            auth_path, "WAITING_FOR_SOURCE_BOUND_MATHEMATICAL_REVISION",
        )
        executor = validate_authorization(self.contract, outcome, authorization)
        repo = Path(str(authorization["candidate_repo"])).resolve()
        candidate_id = str(authorization["candidate_id"])
        stdout_path = self.operations / f"REVISION_EXECUTOR_{candidate_id}.stdout.log"
        stderr_path = self.operations / f"REVISION_EXECUTOR_{candidate_id}.stderr.log"
        command = [
            self.contract["python"],
            str(repo / "operations" / "local_route1_candidate_executor.py"),
            "--contract", authorization["executor_contract"],
        ]
        with stdout_path.open("a", encoding="utf-8") as out, stderr_path.open("a", encoding="utf-8") as err:
            process = subprocess.Popen(command, cwd=repo, env=_environment(repo), stdout=out, stderr=err)
            while process.poll() is None:
                self.state(
                    "REVISION_E200_EXECUTOR_RUNNING", candidate_id=candidate_id,
                    child_pid=process.pid,
                )
                time.sleep(int(self.contract["poll_seconds"]))
            returncode = int(process.wait())
        if returncode != 0:
            raise RuntimeError(f"revision candidate executor failed with {returncode}")
        trajectory_path = self.run_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        trajectory = _read_json(trajectory_path)
        if not any(int(row.get("epoch", -1)) == 200 for row in trajectory.get("trajectory", [])):
            raise RuntimeError("revision candidate executor returned without complete e200")
        receipt_path = self.operations / "terminal_receipts" / f"{candidate_id}.json"
        receipt_result = subprocess.run(
            [
                self.contract["python"], "-m",
                "operations.local_route1_candidate_terminal_receipt",
                "--output", str(self.run_root), "--candidate-id", candidate_id,
                "--receipt", str(receipt_path),
            ],
            cwd=repo, env=_environment(repo), capture_output=True, text=True,
            check=False,
        )
        if receipt_result.returncode:
            raise RuntimeError(
                "revision terminal receipt failed:\n"
                f"{receipt_result.stdout}\n{receipt_result.stderr}"
            )
        cross = _read_json(self.operations / "CROSS_VERSION_E200_ADJUDICATION.json")
        original_receipts = [
            self.operations / "terminal_receipts" / f"{row['candidate_id']}.json"
            for row in cross.get("ranking", [])
        ]
        if {path.stem for path in original_receipts} != set(ORIGINAL_IDS):
            raise RuntimeError("revision ranking lost an original Generation-1 receipt")
        revision_rank_path = self.operations / "CROSS_VERSION_REVISION_E200_ADJUDICATION.json"
        revision_rank = adjudicate([*original_receipts, receipt_path], revision_rank_path)
        final = {
            "schema": FINAL_SCHEMA,
            "status": "REVISION_E200_COMPLETE_FINAL_ADJUDICATION",
            "source_revision_need_sha256": file_sha256(outcome_path),
            "source_revision_authorization_sha256": file_sha256(auth_path),
            "source_revision_adjudication_sha256": file_sha256(revision_rank_path),
            "parent_candidate_id": outcome["selected_candidate_id"],
            "revision_candidate_id": candidate_id,
            "revision_count": 1,
            "selected_candidate_id": revision_rank["selected_candidate_id"],
            "revision_candidate_trajectory_sha256": file_sha256(trajectory_path),
            "revision_candidate_receipt_sha256": file_sha256(receipt_path),
            "selection_role": revision_rank["selection_role"],
            "seed": 2026,
            "target_data_epochs": 200,
            "best_checkpoint_selection": False,
            "fixed_window_or_handoff": False,
            "paired_metric_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        support.atomic_json(
            self.operations / "CROSS_VERSION_FINAL_CAUSAL_REVISION_RESULT.json", final,
        )
        self.state(
            "REVISION_E200_COMPLETE_FINAL_ADJUDICATION", candidate_id=candidate_id,
            selected_candidate_id=revision_rank["selected_candidate_id"],
        )
        self.event(
            "REVISION_E200_COMPLETE", candidate_id=candidate_id,
            selected_candidate_id=revision_rank["selected_candidate_id"],
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--authorize-revision", action="store_true")
    value.add_argument("--successor-repo", type=Path)
    value.add_argument("--candidate-repo", type=Path)
    value.add_argument("--candidate-id")
    value.add_argument("--executor-contract", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=604800)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = ("successor_repo", "run_root", "manifest", "python")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    contract = _read_json(args.contract)
    if args.authorize_revision:
        required = ("candidate_repo", "candidate_id", "executor_contract")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--authorize-revision missing arguments: {missing}")
        value = default_authorization(
            contract=contract, candidate_repo=args.candidate_repo,
            candidate_id=args.candidate_id,
            executor_contract_path=args.executor_contract,
        )
        path = Path(str(contract["run_root"])) / "operations" / "CAUSAL_REVISION_EXECUTION_AUTHORIZATION.json"
        support.atomic_json(path, value)
        print(json.dumps({"status": "REVISION_AUTHORIZED", **value}, indent=2))
        return 0
    run_root = Path(str(contract["run_root"]))
    try:
        with support.executor_lock(
            run_root / "operations" / "CROSS_VERSION_REVISION_SUCCESSOR.lock"
        ):
            return RevisionSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "CROSS_VERSION_REVISION_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-cross-version-revision-successor-fatal-v1",
                "time": support.now(), "status": "FAILED", "error": repr(error),
                "traceback": traceback.format_exc(), "supervisor_pid": os.getpid(),
                "paired_metric_scheduling": False, "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
