"""Conditionally replay one strict 5090 frontier winner on the 4090 host."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from operations.local_route1_cross_version_adjudicate import _validate_receipt
from research.local_route1.frontier_adjudication import FRONTIER_IDS


SCHEMA = "final-unsb-route1-frontier-cross-host-successor-contract-v1"
REPLAY_READY = "REPLAY_REQUEST_READY_REQUIRES_4090_SOURCE_BOUND_EXECUTOR"
NO_REPLAY = "NO_4090_REPLAY_FRONTIER_CURRENT_IMPLEMENTATIONS_NEGATIVE"
SOURCE_RELATIVES = (
    "operations/local_route1_frontier_cross_host_successor.py",
    "operations/local_route1_freeze_frontier_expansion.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "research/local_route1/frontier_adjudication.py",
    "research/local_route1/generation1_gates.py",
)


class RemoteReadError(RuntimeError):
    pass


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def _git_identity(repo: Path) -> dict[str, str]:
    repo = repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError(f"frontier cross-host worktree is dirty: {repo}")
    return {
        "path": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
    }


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    candidate_repo = args.candidate_repo.resolve()
    identity_file = args.identity_file.resolve()
    known_hosts_file = args.known_hosts_file.resolve()
    for path, label in (
        (identity_file, "SSH identity"),
        (known_hosts_file, "SSH known-hosts"),
        (args.baseline_environment_record.resolve(), "baseline environment"),
    ):
        if not path.is_file():
            raise RuntimeError(f"{label} file is missing: {path}")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": _git_identity(repo),
        "candidate_repo": _git_identity(candidate_repo),
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
        "baseline_environment_record": str(args.baseline_environment_record.resolve()),
        "baseline_environment_record_sha256": support.file_sha256(
            args.baseline_environment_record.resolve()
        ),
        "remote": {
            "host": str(args.remote_host),
            "port": int(args.remote_port),
            "user": str(args.remote_user),
            "identity_file": str(identity_file),
            "identity_file_sha256": support.file_sha256(identity_file),
            "known_hosts_file": str(known_hosts_file),
            "known_hosts_file_sha256": support.file_sha256(known_hosts_file),
            "run_root": str(args.remote_run_root),
        },
        "candidate_ids": list(FRONTIER_IDS),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "wait_for_local_winner_ablations": True,
        "complete_remote_e200_only": True,
        "maximum_4090_replays": 1,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("frontier cross-host successor contract schema mismatch")
    for key in ("repo", "candidate_repo"):
        identity = contract.get(key, {})
        repo = Path(identity.get("path", ""))
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != identity.get(
            "git_commit"
        ):
            raise RuntimeError(f"{key} worktree moved")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"{key} worktree is dirty")
    repo = Path(contract["repo"]["path"])
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"frontier cross-host source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("frontier cross-host manifest changed")
    if support.file_sha256(Path(contract["baseline_environment_record"])) != contract.get(
        "baseline_environment_record_sha256"
    ):
        raise RuntimeError("frontier cross-host environment record changed")
    remote = contract.get("remote", {})
    for key in ("identity_file", "known_hosts_file"):
        path = Path(remote.get(key, ""))
        if not path.is_file() or support.file_sha256(path) != remote.get(f"{key}_sha256"):
            raise RuntimeError(f"remote {key} identity changed")
    if int(remote.get("port", 0)) <= 0 or not remote.get("host") or not remote.get("user"):
        raise RuntimeError("frontier remote endpoint is incomplete")
    if tuple(contract.get("candidate_ids", [])) != FRONTIER_IDS:
        raise RuntimeError("frontier replay candidate set changed")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("frontier cross-host polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("frontier cross-host timeout is too short")
    fixed = {
        "wait_for_local_winner_ablations": True,
        "complete_remote_e200_only": True,
        "maximum_4090_replays": 1,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"frontier cross-host contract changed: {key}")


def validate_remote_decision(
    decision: dict[str, Any], adjudication: dict[str, Any],
) -> str | None:
    if decision.get("schema") != "final-unsb-route1-frontier-4090-replay-decision-v1":
        raise RuntimeError("remote frontier replay decision schema mismatch")
    if decision.get("status") not in (REPLAY_READY, NO_REPLAY):
        raise RuntimeError("remote frontier replay decision is not terminal")
    if adjudication.get("schema") != "final-unsb-route1-frontier-e200-adjudication-v1":
        raise RuntimeError("remote frontier adjudication schema mismatch")
    if decision.get("complete_e200_only") is not True:
        raise RuntimeError("remote frontier decision did not wait for e200")
    for payload in (decision, adjudication):
        for key in (
            "intermediate_metric_routing", "cross_host_deltas_merged",
            "paired_controller_access", "confirmation20_opened",
        ):
            if payload.get(key) is not False:
                raise RuntimeError(f"remote frontier decision requires {key}=false")
        if payload.get("selection_seeds") != [2026]:
            raise RuntimeError("remote frontier decision changed the selection seed")
    candidate_id = decision.get("recommended_candidate_id")
    if decision["status"] == NO_REPLAY:
        if candidate_id is not None:
            raise RuntimeError("negative frontier decision unexpectedly requests replay")
        return None
    if candidate_id not in FRONTIER_IDS:
        raise RuntimeError("remote frontier replay candidate is not frozen")
    if candidate_id != adjudication.get("recommended_4090_replay_candidate_id"):
        raise RuntimeError("remote replay decision/adjudication candidate mismatch")
    if decision.get("recommended_algorithm_fingerprint") != adjudication.get(
        "recommended_4090_replay_algorithm_fingerprint"
    ):
        raise RuntimeError("remote replay decision/adjudication algorithm mismatch")
    return str(candidate_id)


class FrontierCrossHostSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"]["path"])
        self.candidate_repo = Path(self.contract["candidate_repo"]["path"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "FRONTIER_CROSS_HOST_SUCCESSOR_STATE.json"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-frontier-cross-host-successor-state-v1",
            "updated": support.now(), "status": status,
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

    def _timed_out(self) -> bool:
        return time.time() - self.started > int(self.contract["timeout_seconds"])

    def _ssh(self, command: str, *, timeout: int = 180) -> subprocess.CompletedProcess[str]:
        remote = self.contract["remote"]
        return subprocess.run([
            "ssh", "-i", remote["identity_file"], "-p", str(remote["port"]),
            "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", f"UserKnownHostsFile={remote['known_hosts_file']}",
            "-o", "ConnectTimeout=20", f"{remote['user']}@{remote['host']}",
            command,
        ], capture_output=True, text=True, check=False, timeout=timeout)

    def _remote_json(self, path: str) -> tuple[dict[str, Any], str] | None:
        result = self._ssh(
            f"if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; else exit 44; fi"
        )
        if result.returncode == 44:
            return None
        if result.returncode:
            raise RemoteReadError(result.stderr.strip() or f"remote read exit {result.returncode}")
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RemoteReadError(f"remote JSON decode failed: {path}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"remote JSON is not an object: {path}")
        return value, hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()

    def wait_local_ablations(self) -> None:
        state_path = self.operations / "WINNER_ABLATION_SUCCESSOR_STATE.json"
        fatal_path = self.operations / "WINNER_ABLATION_SUCCESSOR_FATAL.json"
        while not self._timed_out():
            if fatal_path.is_file():
                raise RuntimeError(f"local winner ablation successor failed: {fatal_path}")
            state = _read_json(state_path) if state_path.is_file() else {}
            if state.get("status") == "WINNER_ABLATIONS_AND_FINAL_DELIVERY_COMPLETE":
                if not (self.operations / "WINNER_ABLATION_ADJUDICATION.json").is_file():
                    raise RuntimeError("winner ablation completion lacks adjudication")
                return
            self.state(
                "WAITING_FOR_LOCAL_WINNER_ABLATIONS",
                local_ablation_status=state.get("status"),
                local_ablation_data_epoch=state.get("active_data_epoch"),
            )
            time.sleep(int(self.contract["poll_seconds"]))
        raise TimeoutError("timed out waiting for local winner ablations")

    def wait_remote_decision(self) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
        root = self.contract["remote"]["run_root"]
        decision_path = f"{root}/operations/FRONTIER_4090_REPLAY_DECISION.json"
        adjudication_path = f"{root}/operations/FRONTIER_E200_ADJUDICATION.json"
        last_error = None
        while not self._timed_out():
            try:
                decision_record = self._remote_json(decision_path)
                adjudication_record = self._remote_json(adjudication_path)
                if decision_record is not None and adjudication_record is not None:
                    decision, decision_sha256 = decision_record
                    adjudication, adjudication_sha256 = adjudication_record
                    if decision.get("frontier_adjudication_sha256") != adjudication_sha256:
                        raise RuntimeError("remote frontier adjudication changed after decision")
                    candidate_id = validate_remote_decision(decision, adjudication)
                    evidence = {
                        "schema": "final-unsb-route1-frontier-remote-terminal-envelope-v1",
                        "recorded": support.now(),
                        "decision": decision,
                        "decision_sha256": decision_sha256,
                        "adjudication": adjudication,
                        "adjudication_sha256": adjudication_sha256,
                        "complete_e200_only": True,
                        "cross_host_deltas_merged": False,
                        "paired_controller_access": False,
                        "confirmation20_opened": False,
                    }
                    support.atomic_json(
                        self.operations / "FRONTIER_5090_TERMINAL_ENVELOPE.json", evidence
                    )
                    return candidate_id, decision, adjudication
                last_error = None
            except (OSError, subprocess.SubprocessError, RemoteReadError) as error:
                last_error = str(error)
            self.state(
                "WAITING_FOR_REMOTE_COMPLETE_FRONTIER_DECISION",
                remote_error=last_error,
            )
            time.sleep(int(self.contract["poll_seconds"]))
        raise TimeoutError(f"timed out waiting for remote frontier decision: {last_error}")

    def _run_checked(
        self, argv: list[str], *, cwd: Path, timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            argv, cwd=cwd, env=_env(cwd), capture_output=True, text=True,
            check=False, timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(argv)}\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return result

    def prepare_replay(self, candidate_id: str) -> Path:
        python = self.contract["python"]
        self._run_checked([
            python,
            str(self.candidate_repo / "operations/local_route1_freeze_frontier_expansion.py"),
            "--output", str(self.run_root),
        ], cwd=self.candidate_repo, timeout=900)
        self.state("RUNNING_SELECTED_FRONTIER_4090_GPU_GATE", candidate_id=candidate_id)
        self._run_checked([
            python, "-m", "research.local_route1.run",
            "--stage", "candidate", "--candidate-action", "gate",
            "--candidate-id", candidate_id,
            "--output", str(self.run_root),
            "--train-view", self.contract["train_view"],
            "--data-root", self.contract["data_root"],
            "--manifest", self.contract["manifest"],
            "--gpu", "0",
        ], cwd=self.candidate_repo, timeout=7200)
        contract_path = self.operations / f"CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
        if not contract_path.is_file():
            self._run_checked([
                python,
                str(self.candidate_repo / "operations/local_route1_candidate_executor.py"),
                "--init-contract", "--contract", str(contract_path),
                "--main-repo", str(self.candidate_repo),
                "--candidate-repo", str(self.candidate_repo),
                "--candidate-id", candidate_id,
                "--run-root", str(self.run_root),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", python,
                "--baseline-environment-record", self.contract[
                    "baseline_environment_record"
                ],
            ], cwd=self.candidate_repo, timeout=900)
        return contract_path

    def run_replay(self, candidate_id: str, contract_path: Path) -> Path:
        stdout_path = self.operations / f"FRONTIER_4090_REPLAY_{candidate_id}.stdout.log"
        stderr_path = self.operations / f"FRONTIER_4090_REPLAY_{candidate_id}.stderr.log"
        stdout = stdout_path.open("a", encoding="utf-8")
        stderr = stderr_path.open("a", encoding="utf-8")
        process = subprocess.Popen([
            self.contract["python"],
            str(self.candidate_repo / "operations/local_route1_candidate_executor.py"),
            "--contract", str(contract_path),
        ], cwd=self.candidate_repo, env=_env(self.candidate_repo), stdout=stdout, stderr=stderr)
        try:
            while process.poll() is None:
                self.state(
                    "FRONTIER_4090_REPLAY_RUNNING",
                    candidate_id=candidate_id,
                    child_pid=process.pid,
                    data_epoch=support.current_epoch(self.run_root, candidate_id),
                )
                time.sleep(int(self.contract["poll_seconds"]))
            returncode = int(process.wait())
        finally:
            stdout.close()
            stderr.close()
        if returncode:
            raise RuntimeError(f"frontier 4090 replay failed: {candidate_id}")
        receipt_path = self.operations / "terminal_receipts" / f"{candidate_id}.json"
        self._run_checked([
            self.contract["python"],
            str(self.candidate_repo / "operations/local_route1_candidate_terminal_receipt.py"),
            "--output", str(self.run_root),
            "--candidate-id", candidate_id,
            "--receipt", str(receipt_path),
        ], cwd=self.candidate_repo, timeout=1800)
        _validate_receipt(receipt_path)
        return receipt_path

    def run(self) -> int:
        self.wait_local_ablations()
        candidate_id, decision, adjudication = self.wait_remote_decision()
        if candidate_id is None:
            result = {
                "schema": "final-unsb-route1-frontier-cross-host-result-v1",
                "status": "COMPLETE_REMOTE_FRONTIER_NEGATIVE_4090_REPLAY_SKIPPED",
                "remote_decision_status": decision["status"],
                "remote_selected_frontier_candidate_id": adjudication[
                    "selected_frontier_candidate_id"
                ],
                "maximum_4090_replays": 1,
                "actual_4090_replays": 0,
                "cross_host_deltas_merged": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
            support.atomic_json(self.operations / "FRONTIER_CROSS_HOST_RESULT.json", result)
            self.state(result["status"])
            return 0
        contract_path = self.prepare_replay(candidate_id)
        receipt_path = self.run_replay(candidate_id, contract_path)
        receipt = _validate_receipt(receipt_path)
        result = {
            "schema": "final-unsb-route1-frontier-cross-host-result-v1",
            "status": "COMPLETE_ONE_FRONTIER_4090_REPLAY_ADJUDICATION_REQUIRED",
            "candidate_id": candidate_id,
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "trajectory_status": receipt["trajectory_status"],
            "receipt_path": str(receipt_path),
            "receipt_sha256": support.file_sha256(receipt_path),
            "maximum_4090_replays": 1,
            "actual_4090_replays": 1,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        support.atomic_json(self.operations / "FRONTIER_CROSS_HOST_RESULT.json", result)
        self.state(result["status"], candidate_id=candidate_id)
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--candidate-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--remote-host")
    value.add_argument("--remote-port", type=int)
    value.add_argument("--remote-user")
    value.add_argument("--remote-run-root")
    value.add_argument("--identity-file", type=Path)
    value.add_argument("--known-hosts-file", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=604800)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "candidate_repo", "run_root", "train_view", "data_root",
            "manifest", "python", "baseline_environment_record", "remote_host",
            "remote_port", "remote_user", "remote_run_root", "identity_file",
            "known_hosts_file",
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
            run_root / "operations" / "FRONTIER_CROSS_HOST_SUCCESSOR.lock"
        ):
            return FrontierCrossHostSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "FRONTIER_CROSS_HOST_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-frontier-cross-host-successor-fatal-v1",
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
