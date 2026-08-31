"""Replay the complete 5090 PCNR alternate from the matched 4090 common e0."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from operations.local_route1_cross_version_adjudicate import _validate_receipt
from research.local_route1.pcnr_alternate_replay import (
    CANDIDATE_ID,
    RESULT_SCHEMA,
    register_pcnr_alternate,
    select_pcnr_alternate,
)
from research.local_route1.portable_extended_frontier import (
    validate_portable_extended_frontier,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-pcnr-alternate-4090-successor-contract-v1"
RESULT_FILE = "PCNR_ALTERNATE_4090_RESULT.json"
SOURCE_RELATIVES = (
    "operations/local_route1_pcnr_alternate_4090_successor.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/pcnr_alternate_replay.py",
    "research/local_route1/portable_extended_frontier.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git_identity(repo: Path, *, label: str) -> dict[str, str]:
    repo = repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError(f"{label} worktree is dirty: {repo}")
    return {
        "path": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
    }


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    source_repo = args.source_repo.resolve()
    manifest = args.manifest.resolve()
    environment = args.baseline_environment_record.resolve()
    for path, label in ((manifest, "manifest"), (environment, "environment")):
        if not path.is_file():
            raise RuntimeError(f"PCNR alternate 4090 {label} is missing: {path}")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": _git_identity(repo, label="PCNR orchestration"),
        "source_repo": _git_identity(source_repo, label="PCNR algorithm source"),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "run_root": str(args.run_root.resolve()),
        "authority_path": str(args.authority.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": support.file_sha256(manifest),
        "python": str(args.python.resolve()),
        "baseline_environment_record": str(environment),
        "baseline_environment_record_sha256": support.file_sha256(environment),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "candidate_id": CANDIDATE_ID,
        "source_classification": "evidence_backed_alternate",
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "restart_from_destination_common_e0": True,
        "action_priority_is_not_an_exclusivity_rule": True,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_only_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("PCNR alternate 4090 contract schema mismatch")
    for key, label in (("repo", "orchestration"), ("source_repo", "source")):
        identity = contract.get(key) or {}
        repo = Path(str(identity.get("path", "")))
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != identity.get(
            "git_commit"
        ):
            raise RuntimeError(f"PCNR alternate {label} worktree moved")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"PCNR alternate {label} worktree is dirty")
    repo = Path(contract["repo"]["path"])
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"PCNR alternate successor source changed: {relative}")
    for key in ("manifest", "baseline_environment_record"):
        if support.file_sha256(Path(contract[key])) != contract.get(f"{key}_sha256"):
            raise RuntimeError(f"PCNR alternate 4090 {key} changed")
    fixed = {
        "candidate_id": CANDIDATE_ID,
        "source_classification": "evidence_backed_alternate",
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "restart_from_destination_common_e0": True,
        "action_priority_is_not_an_exclusivity_rule": True,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_only_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"PCNR alternate 4090 contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("PCNR alternate polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("PCNR alternate timeout is too short")


class PCNRAlternate4090Successor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"]["path"])
        self.source_repo = Path(self.contract["source_repo"]["path"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "PCNR_ALTERNATE_4090_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-pcnr-alternate-4090-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_id": CANDIDATE_ID,
            "action_priority_is_not_an_exclusivity_rule": True,
            "cross_host_deltas_merged": False,
            "paired_metrics_used_for_formula_or_training_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _run_checked(
        self, command: list[str], *, label: str, timeout: int = 7200,
    ) -> None:
        result = subprocess.run(
            command, cwd=self.source_repo, env=_env(self.source_repo),
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            check=False, timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")

    def wait_authority(self) -> Path:
        path = Path(self.contract["authority_path"])
        while not path.is_file():
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for portable extended 5090 evidence")
            self.state("WAITING_FOR_PORTABLE_EXTENDED_5090_FRONTIER")
            time.sleep(int(self.contract["poll_seconds"]))
        value = validate_portable_extended_frontier(_read_json(path))
        selected = select_pcnr_alternate(value)
        if self.contract["source_repo"]["git_commit"] != selected["receipt"].get(
            "training_git_commit"
        ):
            raise RuntimeError("PCNR authority/source worktree commit mismatch")
        frozen = self.operations / "PCNR_ALTERNATE_5090_AUTHORITY.FROZEN.json"
        if frozen.is_file():
            if frozen.read_bytes() != path.read_bytes():
                raise RuntimeError("PCNR source authority changed after destination freeze")
        else:
            frozen.write_bytes(path.read_bytes())
        return frozen

    def prepare(self, authority: Path) -> Path:
        registration = register_pcnr_alternate(
            self.run_root,
            authority_path=authority,
            source_repo=self.source_repo,
            python=Path(self.contract["python"]),
        )
        self.state(
            registration["status"],
            algorithm_fingerprint=registration["candidate"]["algorithm_fingerprint"],
        )
        self._run_checked([
            self.contract["python"], "-m", "research.local_route1.run",
            "--stage", "candidate", "--candidate-action", "gate",
            "--candidate-id", CANDIDATE_ID,
            "--output", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--gpu", "0",
        ], label="PCNR alternate 4090 gate")
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{CANDIDATE_ID}_4090_ALTERNATE.json"
        if not path.is_file():
            self._run_checked([
                self.contract["python"],
                str(self.source_repo / "operations/local_route1_candidate_executor.py"),
                "--init-contract", "--contract", str(path),
                "--main-repo", str(self.source_repo),
                "--candidate-repo", str(self.source_repo),
                "--candidate-id", CANDIDATE_ID,
                "--run-root", str(self.run_root),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", self.contract["python"],
                "--baseline-environment-record", self.contract[
                    "baseline_environment_record"
                ],
            ], label="PCNR alternate executor freeze")
        return path

    def _existing_result(self) -> bool:
        path = self.operations / RESULT_FILE
        if not path.is_file():
            return False
        value = _read_json(path)
        if value.get("schema") != RESULT_SCHEMA or value.get("candidate_id") != CANDIDATE_ID:
            raise RuntimeError("existing PCNR alternate result has a different identity")
        receipt = Path(str(value.get("receipt_path", ""))).resolve()
        if not receipt.is_file() or file_sha256(receipt) != value.get("receipt_sha256"):
            raise RuntimeError("existing PCNR alternate result receipt changed")
        _validate_receipt(receipt)
        self.state(value["status"], result_sha256=file_sha256(path), resumed=True)
        return True

    def run(self) -> int:
        if self._existing_result():
            return 0
        authority = self.wait_authority()
        executor = self.prepare(authority)
        stdout_path = self.operations / "PCNR_ALTERNATE_4090.stdout.log"
        stderr_path = self.operations / "PCNR_ALTERNATE_4090.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen([
                self.contract["python"],
                str(self.source_repo / "operations/local_route1_candidate_executor.py"),
                "--contract", str(executor),
            ], cwd=self.source_repo, env=_env(self.source_repo), stdout=stdout, stderr=stderr)
            while process.poll() is None:
                if time.time() - self.started > int(self.contract["timeout_seconds"]):
                    process.terminate()
                    raise TimeoutError("PCNR alternate exceeded durable timeout")
                self.state(
                    "PCNR_ALTERNATE_4090_E200_RUNNING",
                    worker_pid=process.pid,
                    data_epoch=support.current_epoch(self.run_root, CANDIDATE_ID),
                )
                time.sleep(int(self.contract["poll_seconds"]))
            if process.wait() != 0:
                raise RuntimeError("PCNR alternate candidate executor failed")

        receipt_path = (
            self.operations / "terminal_receipts" / f"{CANDIDATE_ID}_4090.json"
        )
        self._run_checked([
            self.contract["python"],
            str(self.source_repo / "operations/local_route1_candidate_terminal_receipt.py"),
            "--output", str(self.run_root),
            "--candidate-id", CANDIDATE_ID,
            "--receipt", str(receipt_path),
        ], label="PCNR alternate terminal receipt", timeout=1800)
        receipt = _validate_receipt(receipt_path)
        source = select_pcnr_alternate(_read_json(authority))
        if receipt.get("algorithm_fingerprint") != source["receipt"].get(
            "algorithm_fingerprint"
        ):
            raise RuntimeError("destination PCNR algorithm differs from source e200 receipt")
        result = {
            "schema": RESULT_SCHEMA,
            "status": "PCNR_EVIDENCE_BACKED_ALTERNATE_4090_REPLAY_COMPLETE_E200",
            "candidate_id": CANDIDATE_ID,
            "source_classification": source["ranking"]["classification"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "destination_candidate_fingerprint": receipt["candidate_fingerprint"],
            "trajectory_status": receipt["trajectory_status"],
            "ranking_fields": receipt["ranking_fields"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
            "source_authority_sha256": file_sha256(authority),
            "source_receipt_sha256": source["evidence"]["receipt_sha256"],
            "source_trajectory_sha256": source["evidence"]["trajectory_sha256"],
            "restart_from_destination_common_e0": True,
            "same_host_ranking_pending_global_adjudication": True,
            "action_priority_is_not_an_exclusivity_rule": True,
            "cross_host_deltas_merged": False,
            "paired_metrics_used_only_for_resource_allocation": True,
            "paired_metrics_used_for_formula_or_training_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        path = self.operations / RESULT_FILE
        support.atomic_json(path, result)
        self.state(result["status"], result_sha256=file_sha256(path))
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--source-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--authority", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "source_repo", "run_root", "authority", "train_view",
            "data_root", "manifest", "python", "baseline_environment_record",
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
            run_root / "operations" / "PCNR_ALTERNATE_4090_SUCCESSOR.lock"
        ):
            return PCNRAlternate4090Successor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "PCNR_ALTERNATE_4090_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-pcnr-alternate-4090-successor-fatal-v1",
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
