"""Run two evidence-qualified 4090 algorithms from the matched 5090 common e0."""

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
from research.local_route1.cross_runtime_portfolio import (
    REPLAY_IDS,
    RESULT_SCHEMA,
    register_cross_runtime_candidate,
    validate_portable_cross_runtime_portfolio,
)
from research.local_route1.protocol import file_sha256


SCHEMA = "final-unsb-route1-cross-runtime-5090-successor-contract-v1"
RESULT_FILE = "CROSS_RUNTIME_PORTFOLIO_5090_RESULT.json"
SOURCE_RELATIVES = (
    "operations/local_route1_cross_runtime_5090_successor.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/cross_runtime_portfolio.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git_identity(repo: Path) -> dict[str, str]:
    repo = repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError(f"cross-runtime worktree is dirty: {repo}")
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
    manifest = args.manifest.resolve()
    environment = args.baseline_environment_record.resolve()
    for path, label in ((manifest, "manifest"), (environment, "environment")):
        if not path.is_file():
            raise RuntimeError(f"cross-runtime 5090 {label} is missing: {path}")
    source_repos = {
        REPLAY_IDS[0]: _git_identity(args.pcrsmg_proposal_repo),
        REPLAY_IDS[1]: _git_identity(args.amtnc_repo),
    }
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": _git_identity(repo),
        "source_repos": source_repos,
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
        "replay_candidate_ids": list(REPLAY_IDS),
        "maximum_parallel_replays": 2,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "restart_from_destination_common_e0": True,
        "formula_changed": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("cross-runtime 5090 contract schema mismatch")
    identities = [contract.get("repo", {}), *contract.get("source_repos", {}).values()]
    for identity in identities:
        repo = Path(str(identity.get("path", "")))
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != identity.get(
            "git_commit"
        ):
            raise RuntimeError("cross-runtime 5090 worktree moved")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError("cross-runtime 5090 worktree is dirty")
    if set(contract.get("source_repos", {})) != set(REPLAY_IDS):
        raise RuntimeError("cross-runtime 5090 source repository set changed")
    repo = Path(contract["repo"]["path"])
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"cross-runtime 5090 source changed: {relative}")
    for key in ("manifest", "baseline_environment_record"):
        if support.file_sha256(Path(contract[key])) != contract.get(f"{key}_sha256"):
            raise RuntimeError(f"cross-runtime 5090 {key} changed")
    fixed = {
        "replay_candidate_ids": list(REPLAY_IDS),
        "maximum_parallel_replays": 2,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "restart_from_destination_common_e0": True,
        "formula_changed": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"cross-runtime 5090 contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("cross-runtime 5090 polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("cross-runtime 5090 timeout is too short")


class CrossRuntime5090Successor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"]["path"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "CROSS_RUNTIME_5090_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-cross-runtime-5090-successor-state-v1",
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "replay_candidate_ids": list(REPLAY_IDS),
            "maximum_parallel_replays": 2,
            "cross_host_deltas_merged": False,
            "paired_metrics_used_for_formula_or_training_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _run_checked(
        self, command: list[str], *, cwd: Path, label: str, timeout: int = 7200,
    ) -> None:
        result = subprocess.run(
            command, cwd=cwd, env=_env(cwd), capture_output=True, text=True,
            encoding="utf-8", errors="replace", check=False, timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(f"{label} failed:\n{result.stdout}\n{result.stderr}")

    def wait_authority(self) -> Path:
        path = Path(self.contract["authority_path"])
        while not path.is_file():
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for 4090 portable replay portfolio")
            self.state("WAITING_FOR_PORTABLE_4090_REPLAY_PORTFOLIO")
            time.sleep(int(self.contract["poll_seconds"]))
        authority = validate_portable_cross_runtime_portfolio(_read_json(path))
        evidence = {
            row["candidate_id"]: row for row in authority["candidate_evidence"]
        }
        for candidate_id in REPLAY_IDS:
            if self.contract["source_repos"][candidate_id]["git_commit"] != evidence[
                candidate_id
            ]["receipt"].get("training_git_commit"):
                raise RuntimeError("cross-runtime source repo/receipt commit mismatch")
        frozen = self.operations / "PORTABLE_4090_REPLAY_PORTFOLIO.FROZEN.json"
        if frozen.is_file():
            if frozen.read_bytes() != path.read_bytes():
                raise RuntimeError("4090 replay portfolio changed after 5090 freeze")
        else:
            frozen.write_bytes(path.read_bytes())
        return frozen

    def prepare_candidate(self, authority: Path, candidate_id: str) -> Path:
        source_repo = Path(self.contract["source_repos"][candidate_id]["path"])
        registration = register_cross_runtime_candidate(
            self.run_root,
            authority_path=authority,
            candidate_id=candidate_id,
            source_repo=source_repo,
            python=Path(self.contract["python"]),
        )
        self.state(
            registration["status"], candidate_id=candidate_id,
            algorithm_fingerprint=registration["candidate"]["algorithm_fingerprint"],
        )
        self._run_checked([
            self.contract["python"], "-m", "research.local_route1.run",
            "--stage", "candidate", "--candidate-action", "gate",
            "--candidate-id", candidate_id,
            "--output", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--gpu", "0",
        ], cwd=source_repo, label=f"cross-runtime 5090 gate {candidate_id}")
        path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}_5090_REPLAY.json"
        if not path.is_file():
            self._run_checked([
                self.contract["python"],
                str(source_repo / "operations/local_route1_candidate_executor.py"),
                "--init-contract", "--contract", str(path),
                "--main-repo", str(source_repo),
                "--candidate-repo", str(source_repo),
                "--candidate-id", candidate_id,
                "--run-root", str(self.run_root),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", self.contract["python"],
                "--baseline-environment-record", self.contract[
                    "baseline_environment_record"
                ],
            ], cwd=source_repo, label=f"cross-runtime executor freeze {candidate_id}")
        return path

    def _existing_result(self) -> bool:
        path = self.operations / RESULT_FILE
        if not path.is_file():
            return False
        value = _read_json(path)
        if value.get("schema") != RESULT_SCHEMA:
            raise RuntimeError("existing cross-runtime result schema changed")
        if {row.get("candidate_id") for row in value.get("candidate_results", [])} != set(
            REPLAY_IDS
        ):
            raise RuntimeError("existing cross-runtime result set changed")
        for row in value["candidate_results"]:
            receipt = Path(str(row.get("receipt_path", ""))).resolve()
            if not receipt.is_file() or file_sha256(receipt) != row.get(
                "receipt_sha256"
            ):
                raise RuntimeError("existing cross-runtime receipt changed")
            _validate_receipt(receipt)
        self.state(value["status"], result_sha256=file_sha256(path), resumed=True)
        return True

    def run(self) -> int:
        if self._existing_result():
            return 0
        authority = self.wait_authority()
        contracts = {
            candidate_id: self.prepare_candidate(authority, candidate_id)
            for candidate_id in REPLAY_IDS
        }
        processes = []
        for candidate_id in REPLAY_IDS:
            source_repo = Path(self.contract["source_repos"][candidate_id]["path"])
            stdout = (self.operations / f"CROSS_RUNTIME_5090_{candidate_id}.stdout.log").open(
                "a", encoding="utf-8"
            )
            stderr = (self.operations / f"CROSS_RUNTIME_5090_{candidate_id}.stderr.log").open(
                "a", encoding="utf-8"
            )
            process = subprocess.Popen([
                self.contract["python"],
                str(source_repo / "operations/local_route1_candidate_executor.py"),
                "--contract", str(contracts[candidate_id]),
            ], cwd=source_repo, env=_env(source_repo), stdout=stdout, stderr=stderr)
            processes.append((candidate_id, source_repo, process, stdout, stderr))
        try:
            while any(process.poll() is None for _, _, process, _, _ in processes):
                if time.time() - self.started > int(self.contract["timeout_seconds"]):
                    for _, _, process, _, _ in processes:
                        if process.poll() is None:
                            process.terminate()
                    raise TimeoutError("cross-runtime 5090 portfolio exceeded timeout")
                self.state(
                    "CROSS_RUNTIME_5090_PORTFOLIO_E200_RUNNING",
                    workers={
                        candidate_id: {
                            "pid": process.pid,
                            "data_epoch": support.current_epoch(
                                self.run_root, candidate_id
                            ),
                        }
                        for candidate_id, _, process, _, _ in processes
                    },
                )
                time.sleep(int(self.contract["poll_seconds"]))
        finally:
            for _, _, _, stdout, stderr in processes:
                stdout.close()
                stderr.close()
        failures = [
            candidate_id for candidate_id, _, process, _, _ in processes
            if int(process.wait()) != 0
        ]
        if failures:
            raise RuntimeError(f"cross-runtime 5090 candidates failed: {failures}")
        authority_value = validate_portable_cross_runtime_portfolio(
            _read_json(authority)
        )
        source = {
            row["candidate_id"]: row
            for row in authority_value["candidate_evidence"]
        }
        rows = []
        for candidate_id, source_repo, _, _, _ in processes:
            receipt_path = (
                self.operations / "terminal_receipts" / f"{candidate_id}_5090.json"
            )
            self._run_checked([
                self.contract["python"],
                str(source_repo / "operations/local_route1_candidate_terminal_receipt.py"),
                "--output", str(self.run_root),
                "--candidate-id", candidate_id,
                "--receipt", str(receipt_path),
            ], cwd=source_repo, label=f"cross-runtime receipt {candidate_id}", timeout=1800)
            receipt = _validate_receipt(receipt_path)
            if receipt.get("algorithm_fingerprint") != source[candidate_id][
                "receipt"
            ].get("algorithm_fingerprint"):
                raise RuntimeError("cross-runtime destination algorithm changed")
            rows.append({
                "candidate_id": candidate_id,
                "source_classification": source[candidate_id]["source_classification"],
                "algorithm_fingerprint": receipt["algorithm_fingerprint"],
                "destination_candidate_fingerprint": receipt["candidate_fingerprint"],
                "trajectory_status": receipt["trajectory_status"],
                "ranking_fields": receipt["ranking_fields"],
                "receipt_path": str(receipt_path),
                "receipt_sha256": file_sha256(receipt_path),
            })
        result = {
            "schema": RESULT_SCHEMA,
            "status": "CROSS_RUNTIME_5090_PORTFOLIO_COMPLETE_E200",
            "source_authority_sha256": file_sha256(authority),
            "candidate_results": rows,
            "same_host_5090_adjudication_pending": True,
            "maximum_parallel_replays": 2,
            "restart_from_destination_common_e0": True,
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
    value.add_argument("--pcrsmg-proposal-repo", type=Path)
    value.add_argument("--amtnc-repo", type=Path)
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
            "repo", "pcrsmg_proposal_repo", "amtnc_repo", "run_root",
            "authority", "train_view", "data_root", "manifest", "python",
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
            run_root / "operations" / "CROSS_RUNTIME_5090_SUCCESSOR.lock"
        ):
            return CrossRuntime5090Successor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "CROSS_RUNTIME_5090_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-cross-runtime-5090-successor-fatal-v1",
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
