"""Durable winner-only ablation gate, e200 execution and finalization."""

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

from operations.local_route1_candidate_terminal_receipt import materialize_receipt
from operations.local_route1_winner_ablation_adjudicate import adjudicate
from research.local_route1.cross_version_final_delivery import (
    materialize_cross_version_final_delivery,
)
from research.local_route1.single_seed_development import (
    materialize_single_seed_development_freeze,
    validate_single_seed_development_freeze,
)
from research.local_route1.final_selection import resolve_e200_selection_path
from research.local_route1.winner_ablations import (
    materialize_winner_ablation_definitions,
)


SCHEMA = "final-unsb-route1-winner-ablation-successor-contract-v1"
SOURCE_RELATIVES = (
    "operations/local_route1_winner_ablation_successor.py",
    "operations/local_route1_winner_ablation_freeze.py",
    "operations/local_route1_winner_ablation_adjudicate.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/winner_ablations.py",
    "research/local_route1/cross_version_final_delivery.py",
    "research/local_route1/single_seed_development.py",
    "research/local_route1/final_selection.py",
    "research/local_route1/generation1_gates.py",
    "src/models/route1/bvcp_ablation.py",
    "src/models/route1/pcrsmg_ablation.py",
    "src/models/route1/amtnc_ablation.py",
    "src/models/route1/mcrb_ablation.py",
    "src/models/route1_amtnc_ablation_model.py",
    "src/models/route1_mcrb_ablation_model.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("winner ablation successor worktree must be clean")
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
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "baseline_environment_record": str(
            args.baseline_environment_record.resolve()
        ),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "batch_size": 1,
        "target_data_epochs": 200,
        "e200_execution_policy": "SEQUENTIAL_SINGLE_STREAM_BY_MEASURED_WALL_CLOCK",
        "maximum_parallel_e200_executors": 1,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "seed_validation_policy": "DEFER_ADDITIONAL_SEEDS_FOR_ALGORITHM_SEARCH",
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("winner ablation successor contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("winner ablation successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("winner ablation successor worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"winner ablation successor source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("winner ablation successor manifest changed")
    if int(contract.get("batch_size", 0)) != 1:
        raise RuntimeError("winner ablations must retain scientific batch1")
    if int(contract.get("target_data_epochs", 0)) != 200:
        raise RuntimeError("winner ablations must run true e200")
    if contract.get("e200_execution_policy") != (
        "SEQUENTIAL_SINGLE_STREAM_BY_MEASURED_WALL_CLOCK"
    ):
        raise RuntimeError("winner ablation e200 execution policy changed")
    if int(contract.get("maximum_parallel_e200_executors", 0)) != 1:
        raise RuntimeError("winner ablation e200 execution must remain single-stream")
    if contract.get("selection_seeds") != [2026]:
        raise RuntimeError("winner ablations require the frozen seed2026 winner")
    if contract.get("deferred_seed_validation") != [2027, 2028]:
        raise RuntimeError("winner ablation deferred seed set changed")
    if contract.get("seed_validation_policy") != (
        "DEFER_ADDITIONAL_SEEDS_FOR_ALGORITHM_SEARCH"
    ):
        raise RuntimeError("winner ablation seed policy changed")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("winner ablation successor poll interval too short")
    if int(contract.get("timeout_seconds", 0)) < 3600:
        raise RuntimeError("winner ablation successor timeout too short")
    for key in ("paired_metric_scheduling", "paired_controller_access", "confirmation20_opened"):
        if contract.get(key) is not False:
            raise RuntimeError(f"winner ablation successor requires {key}=false")


class WinnerAblationSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "WINNER_ABLATION_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "WINNER_ABLATION_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-winner-ablation-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(), "batch_size": 1,
            "target_data_epochs": 200,
            "e200_execution_policy": self.contract["e200_execution_policy"],
            "maximum_parallel_e200_executors": 1,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-winner-ablation-successor-event-v1",
            "time": support.now(), "event": event,
            "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_for_selection_and_freeze(self) -> dict[str, Any] | None:
        started = time.time()
        while True:
            cross = None
            cross_path = None
            pending_reason = None
            try:
                cross_path = resolve_e200_selection_path(self.run_root)
                cross = _read_json(cross_path)
            except RuntimeError as error:
                pending_reason = str(error)
            winner = None if cross is None else str(cross["selected_candidate_id"])
            freeze_path = self.operations / "SINGLE_SEED_DEVELOPMENT_FREEZE.json"
            freeze = None
            if cross is not None and not freeze_path.is_file():
                freeze = materialize_single_seed_development_freeze(self.run_root)
            elif freeze_path.is_file():
                freeze = validate_single_seed_development_freeze(self.run_root)
            complete = freeze is not None and freeze.get("candidate_id") == winner
            self.state(
                "WAITING_FOR_SINGLE_SEED_DEVELOPMENT_WINNER",
                winner=winner,
                cross_version_ready=cross is not None,
                source_e200_selection=(None if cross_path is None else str(cross_path)),
                pending_reason=pending_reason,
                development_freeze_status=(
                    None if freeze is None else freeze.get("status")
                ),
                cross_seed_stability_claimed=False,
                elapsed_seconds=time.time() - started,
            )
            if complete:
                return cross
            if time.time() - started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("winner ablation successor timed out waiting for selection/freeze")
            time.sleep(int(self.contract["poll_seconds"]))

    def _run_checked(self, command: list[str], *, label: str) -> None:
        result = subprocess.run(
            command, cwd=self.repo, env=_env(self.repo), capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")

    def run_gates(self, candidate_ids: list[str]) -> None:
        processes = []
        for candidate_id in candidate_ids:
            stdout = self.operations / f"WINNER_ABLATION_GATE_{candidate_id}.stdout.log"
            stderr = self.operations / f"WINNER_ABLATION_GATE_{candidate_id}.stderr.log"
            command = [
                self.contract["python"], "-m", "research.local_route1.run",
                "--stage", "candidate", "--candidate-action", "gate",
                "--candidate-id", candidate_id,
                "--output", str(self.run_root),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"], "--gpu", "0",
            ]
            out = stdout.open("a", encoding="utf-8")
            err = stderr.open("a", encoding="utf-8")
            process = subprocess.Popen(
                command, cwd=self.repo, env=_env(self.repo), stdout=out, stderr=err,
            )
            processes.append((candidate_id, process, out, err))
        while any(process.poll() is None for _, process, _, _ in processes):
            self.state(
                "WINNER_ABLATION_GPU_GATES_RUNNING",
                children={candidate_id: process.pid for candidate_id, process, _, _ in processes},
            )
            time.sleep(30)
        failures = []
        for candidate_id, process, out, err in processes:
            out.close(); err.close()
            if int(process.wait()) != 0:
                failures.append(candidate_id)
        if failures:
            raise RuntimeError(f"winner ablation GPU gates failed: {failures}")

    def _init_executor_contract(self, candidate_id: str) -> Path:
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
        if path.is_file():
            return path
        self._run_checked([
            self.contract["python"], "operations/local_route1_candidate_executor.py",
            "--init-contract", "--contract", str(path),
            "--main-repo", str(self.repo), "--candidate-repo", str(self.repo),
            "--candidate-id", candidate_id, "--run-root", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--python", self.contract["python"],
            "--baseline-environment-record", self.contract["baseline_environment_record"],
        ], label=f"initialize {candidate_id} executor")
        return path

    def run_e200(self, candidate_ids: list[str]) -> None:
        completed = []
        for candidate_id in candidate_ids:
            contract = self._init_executor_contract(candidate_id)
            stdout = self.operations / f"WINNER_ABLATION_EXECUTOR_{candidate_id}.stdout.log"
            stderr = self.operations / f"WINNER_ABLATION_EXECUTOR_{candidate_id}.stderr.log"
            out = stdout.open("a", encoding="utf-8")
            err = stderr.open("a", encoding="utf-8")
            process = subprocess.Popen(
                [self.contract["python"], "operations/local_route1_candidate_executor.py",
                 "--contract", str(contract)],
                cwd=self.repo, env=_env(self.repo), stdout=out, stderr=err,
            )
            try:
                while process.poll() is None:
                    self.state(
                        "WINNER_ABLATION_E200_RUNNING_SINGLE_STREAM",
                        active_candidate_id=candidate_id,
                        active_child_pid=process.pid,
                        active_data_epoch=support.current_epoch(
                            self.run_root, candidate_id,
                        ),
                        completed_candidate_ids=list(completed),
                    )
                    time.sleep(30)
                returncode = int(process.wait())
            finally:
                out.close()
                err.close()
            if returncode != 0:
                raise RuntimeError(
                    f"winner ablation e200 executor failed: {candidate_id}"
                )
            completed.append(candidate_id)
            self.event(
                "WINNER_ABLATION_E200_CANDIDATE_COMPLETE",
                candidate_id=candidate_id,
                completed_candidate_ids=list(completed),
            )

    def run(self) -> int:
        self.event("WINNER_ABLATION_SUCCESSOR_START", contract=str(self.contract_path))
        cross = self.wait_for_selection_and_freeze()
        if cross is None:
            return 0
        adjudication_path = self.operations / "WINNER_ABLATION_ADJUDICATION.json"
        if adjudication_path.is_file():
            # Recovery may use a newer adjudicator around already source-bound,
            # complete e200 receipts.  The final-delivery materializer performs
            # the strict adjudication/selection/receipt checks; do not try to
            # regenerate frozen candidate definitions (whose non-training gate
            # hashes may legitimately differ in the recovery worktree).
            delivery = materialize_cross_version_final_delivery(self.run_root)
            winner = str(cross["selected_candidate_id"])
            self.state(
                "WINNER_ABLATIONS_AND_FINAL_DELIVERY_COMPLETE",
                candidate_id=delivery["candidate_id"],
                final_candidate=str(self.run_root / "final" / "CANDIDATE.json"),
                resumed_from_complete_adjudication=True,
            )
            self.event(
                "WINNER_ABLATION_SUCCESSOR_COMPLETE",
                winner=winner,
                resumed_from_complete_adjudication=True,
            )
            return 0
        frozen = materialize_winner_ablation_definitions(self.run_root)
        candidate_ids = [
            frozen["ablation_candidate_ids"][role]
            for role in ("proposal_only", "observable_only")
        ]
        self.event("WINNER_ABLATIONS_SOURCE_FROZEN", candidate_ids=candidate_ids)
        self.run_gates(candidate_ids)
        self.event("WINNER_ABLATION_GATES_PASS", candidate_ids=candidate_ids)
        self.run_e200(candidate_ids)

        receipts = {
            candidate_id: self.operations / "terminal_receipts" / f"{candidate_id}.json"
            for candidate_id in candidate_ids
        }
        for candidate_id, path in receipts.items():
            materialize_receipt(self.run_root, candidate_id, path)
        winner = str(cross["selected_candidate_id"])
        full_receipt = self.operations / "terminal_receipts" / f"{winner}.json"
        result = adjudicate(
            output_root=self.run_root,
            cross_adjudication_path=resolve_e200_selection_path(self.run_root),
            proposal_receipt_path=receipts[frozen["ablation_candidate_ids"]["proposal_only"]],
            observable_receipt_path=receipts[frozen["ablation_candidate_ids"]["observable_only"]],
            full_receipt_path=full_receipt,
            output_path=self.operations / "WINNER_ABLATION_ADJUDICATION.json",
        )
        delivery = materialize_cross_version_final_delivery(self.run_root)
        self.state(
            "WINNER_ABLATIONS_AND_FINAL_DELIVERY_COMPLETE",
            candidate_id=delivery["candidate_id"],
            final_candidate=str(self.run_root / "final" / "CANDIDATE.json"),
        )
        self.event("WINNER_ABLATION_SUCCESSOR_COMPLETE", winner=winner)
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=345600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "run_root", "train_view", "data_root", "manifest", "python",
            "baseline_environment_record",
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
    run_root = Path(contract["run_root"])
    try:
        with support.executor_lock(
            run_root / "operations" / "WINNER_ABLATION_SUCCESSOR.lock"
        ):
            return WinnerAblationSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(run_root / "operations" / "WINNER_ABLATION_SUCCESSOR_FATAL.json", {
            "schema": "final-unsb-route1-winner-ablation-successor-fatal-v1",
            "time": support.now(), "status": "FAILED", "error": repr(error),
            "traceback": traceback.format_exc(), "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
