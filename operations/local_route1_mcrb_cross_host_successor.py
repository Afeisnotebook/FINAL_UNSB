"""Durably route the 5090 MCRB result into the authoritative 4090 chain.

The supplemental 5090 trajectory is allowed to decide whether an exact,
from-e0 4090 replay is worth paying for only after the complete e200 artifact
exists.  No intermediate paired metric is read.  A positive 5090 trajectory
starts the already frozen MCRB implementation on the 4090; a negative one is
preserved as cross-runtime evidence and does not consume the 4090 replay.

After the local AM-TNC revision and the conditional MCRB replay (if any) are
terminal, this successor creates the all-candidate seed-2026 selection and
starts the winner-only ablation successor.  It contains no credentials: SSH
uses a dedicated key named in the runtime contract.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from research.local_route1.candidates import load_candidate_registration
from research.local_route1.final_selection import (
    materialize_final_e200_selection,
    validate_e200_selection,
)


SCHEMA = "final-unsb-route1-mcrb-cross-host-successor-contract-v1"
STATE_SCHEMA = "final-unsb-route1-mcrb-cross-host-successor-state-v1"
EVENT_SCHEMA = "final-unsb-route1-mcrb-cross-host-successor-event-v1"
CANDIDATE_ID = "G1-03-STATE-FEEDBACK-MISSING"
AMTNC_ID = "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS"
REMOTE_PASS = "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION"
REMOTE_FAIL = "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION"
ALLOWED_REMOTE_STATUSES = {REMOTE_PASS, REMOTE_FAIL}
EXPECTED_UPDATES = 30000
EXPECTED_EPOCHS = 200
SOURCE_RELATIVES = (
    "operations/local_route1_mcrb_cross_host_successor.py",
    "operations/local_route1_freeze_mcrb_generation1.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "operations/local_route1_winner_ablation_successor.py",
    "operations/local_route1_cross_version_adjudicate.py",
    "research/local_route1/final_selection.py",
)


class RemoteReadError(RuntimeError):
    """A retryable transport/read failure, distinct from scientific invalidity."""


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
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError(f"successor worktree is dirty: {repo}")
    return {
        "path": str(repo.resolve()),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
    }


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    mcrb_repo = args.mcrb_repo.resolve()
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
        "mcrb_repo": _git_identity(mcrb_repo),
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
            "repo": str(args.remote_repo),
            "run_root": str(args.remote_run_root),
            "python": str(args.remote_python),
        },
        "candidate_id": CANDIDATE_ID,
        "revision_candidate_id": AMTNC_ID,
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "batch_size": 1,
        "target_data_epochs": EXPECTED_EPOCHS,
        "target_updates": EXPECTED_UPDATES,
        "remote_e200_only_routing": True,
        "cross_host_deltas_merged": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("MCRB cross-host successor contract schema mismatch")
    if contract.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("MCRB candidate identity changed")
    if contract.get("revision_candidate_id") != AMTNC_ID:
        raise RuntimeError("AM-TNC dependency identity changed")
    for name in ("repo", "mcrb_repo"):
        identity = contract.get(name, {})
        repo = Path(identity.get("path", ""))
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != identity.get(
            "git_commit"
        ):
            raise RuntimeError(f"{name} worktree moved")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"{name} worktree is dirty")
    repo = Path(contract["repo"]["path"])
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"successor source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("manifest changed after successor freeze")
    if support.file_sha256(Path(contract["baseline_environment_record"])) != contract.get(
        "baseline_environment_record_sha256"
    ):
        raise RuntimeError("baseline environment record changed")
    remote = contract.get("remote", {})
    for key in ("identity_file", "known_hosts_file"):
        path = Path(remote.get(key, ""))
        if not path.is_file() or support.file_sha256(path) != remote.get(f"{key}_sha256"):
            raise RuntimeError(f"remote {key} identity changed")
    if int(remote.get("port", 0)) <= 0 or not remote.get("host") or not remote.get("user"):
        raise RuntimeError("remote endpoint is incomplete")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("poll interval is too short")
    if int(contract.get("timeout_seconds", 0)) < 3600:
        raise RuntimeError("successor timeout is too short")
    if contract.get("selection_seeds") != [2026]:
        raise RuntimeError("successor may select only seed2026")
    if contract.get("deferred_seed_validation") != [2027, 2028]:
        raise RuntimeError("deferred seed policy changed")
    if int(contract.get("batch_size", 0)) != 1:
        raise RuntimeError("MCRB replay must retain batch1")
    if int(contract.get("target_data_epochs", 0)) != EXPECTED_EPOCHS:
        raise RuntimeError("MCRB replay must reach e200")
    if int(contract.get("target_updates", 0)) != EXPECTED_UPDATES:
        raise RuntimeError("MCRB replay must reach 30000 updates")
    if contract.get("remote_e200_only_routing") is not True:
        raise RuntimeError("intermediate remote metrics may not route the replay")
    for key in (
        "cross_host_deltas_merged", "paired_metric_scheduling",
        "paired_controller_access", "confirmation20_opened",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"successor requires {key}=false")


def validate_remote_trajectory(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "final-unsb-route1-candidate-trajectory-v1":
        raise RuntimeError("remote MCRB trajectory schema mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("remote trajectory is not the frozen MCRB candidate")
    if value.get("status") not in ALLOWED_REMOTE_STATUSES:
        raise RuntimeError("remote MCRB trajectory is not terminal")
    rows = value.get("trajectory", [])
    terminal = [row for row in rows if int(row.get("epoch", -1)) == EXPECTED_EPOCHS]
    if len(terminal) != 1 or int(terminal[0].get("updates", -1)) != EXPECTED_UPDATES:
        raise RuntimeError("remote MCRB trajectory has no unique e200/30000 terminal row")
    if value.get("paired_metrics_used_for_training_or_gate") is not False:
        raise RuntimeError("remote MCRB used paired metrics for training or gating")
    if value.get("confirmation20_opened") is not False:
        raise RuntimeError("remote MCRB opened confirmation20")
    return value


def validate_remote_receipt(value: dict[str, Any], trajectory: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != "final-unsb-route1-candidate-terminal-receipt-v1":
        raise RuntimeError("remote MCRB terminal receipt schema mismatch")
    if value.get("candidate_id") != CANDIDATE_ID:
        raise RuntimeError("remote MCRB terminal receipt candidate mismatch")
    if value.get("trajectory_status") != trajectory.get("status"):
        raise RuntimeError("remote receipt/trajectory status mismatch")
    integrity = value.get("terminal_integrity", {})
    if integrity.get("status") != "ACCEPTED_COMPLETE_E200_ARTIFACT_SET":
        raise RuntimeError("remote receipt does not accept the complete e200 artifact set")
    for bucket in ("milestone_checkpoint_sha256", "metric_sha256"):
        hashes = integrity.get(bucket, {})
        if not isinstance(hashes, dict) or "200" not in hashes:
            raise RuntimeError(f"remote receipt terminal integrity is missing e200 {bucket}")
    if integrity.get("evaluation_crn_matched_to_plain") is not True:
        raise RuntimeError("remote receipt is not matched to same-host plain CRN")
    if integrity.get("paired_metric_used_for_training_or_control") is not False:
        raise RuntimeError("remote terminal integrity used paired metrics for control")
    if integrity.get("confirmation20_opened") is not False:
        raise RuntimeError("remote terminal integrity opened confirmation20")
    if value.get("paired_metrics_used_for_training_or_control") is not False:
        raise RuntimeError("remote receipt used paired metrics for training or control")
    if value.get("paired_metrics_used_only_after_complete_trajectory") is not True:
        raise RuntimeError("remote receipt evaluated paired metrics before trajectory completion")
    if value.get("confirmation20_opened") is not False:
        raise RuntimeError("remote receipt opened confirmation20")
    return value


class MCRBCrossHostSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"]["path"])
        self.mcrb_repo = Path(self.contract["mcrb_repo"]["path"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "MCRB_CROSS_HOST_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "MCRB_CROSS_HOST_SUCCESSOR_EVENTS.jsonl"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": STATE_SCHEMA,
            "updated": support.now(),
            "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_id": CANDIDATE_ID,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "cross_host_deltas_merged": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": EVENT_SCHEMA,
            "time": support.now(),
            "event": event,
            "supervisor_pid": os.getpid(),
            "candidate_id": CANDIDATE_ID,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _timed_out(self) -> bool:
        return time.time() - self.started > int(self.contract["timeout_seconds"])

    def _ssh(self, command: str, *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
        remote = self.contract["remote"]
        return subprocess.run(
            [
                "ssh", "-i", remote["identity_file"], "-p", str(remote["port"]),
                "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                "-o", f"UserKnownHostsFile={remote['known_hosts_file']}",
                "-o", "ConnectTimeout=20",
                f"{remote['user']}@{remote['host']}", command,
            ],
            capture_output=True, text=True, check=False, timeout=timeout,
        )

    def _remote_cat(self, path: str) -> dict[str, Any] | None:
        result = self._ssh(
            f"if test -f {shlex.quote(path)}; then cat {shlex.quote(path)}; else exit 44; fi",
            timeout=120,
        )
        if result.returncode == 44:
            return None
        if result.returncode:
            raise RemoteReadError(
                f"remote read failed ({result.returncode}): {result.stderr.strip()}"
            )
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise RemoteReadError(f"remote JSON decode failed: {path}") from error
        if not isinstance(value, dict):
            raise RuntimeError(f"remote JSON is not an object: {path}")
        return value

    def wait_for_remote_e200(self) -> tuple[dict[str, Any], dict[str, Any]]:
        remote = self.contract["remote"]
        candidate_root = f"{remote['run_root']}/candidates/{CANDIDATE_ID}"
        trajectory_path = f"{candidate_root}/CANDIDATE_TRAJECTORY.json"
        heartbeat_path = f"{candidate_root}/HEARTBEAT.json"
        last_error = None
        while not self._timed_out():
            try:
                trajectory = self._remote_cat(trajectory_path)
                if trajectory is not None:
                    if trajectory.get("status") == "INCOMPLETE_E200":
                        pass
                    else:
                        trajectory = validate_remote_trajectory(trajectory)
                        break
                heartbeat = self._remote_cat(heartbeat_path) or {}
                self.state(
                    "WAITING_FOR_REMOTE_MCRB_E200",
                    remote_data_epoch=heartbeat.get("data_epoch"),
                    remote_updates=heartbeat.get("updates"),
                    elapsed_seconds=time.time() - self.started,
                )
                last_error = None
            except (OSError, subprocess.SubprocessError, RemoteReadError) as error:
                last_error = str(error)
                self.state(
                    "WAITING_FOR_REMOTE_MCRB_RECONNECT",
                    remote_error=last_error,
                    elapsed_seconds=time.time() - self.started,
                )
            time.sleep(int(self.contract["poll_seconds"]))
        else:
            raise TimeoutError(f"timed out waiting for remote MCRB e200: {last_error}")

        receipt_path = f"{remote['run_root']}/operations/terminal_receipts/{CANDIDATE_ID}.json"
        command = " && ".join((
            f"cd {shlex.quote(remote['repo'])}",
            (
                f"PYTHONPATH={shlex.quote(remote['repo'])}:{shlex.quote(remote['repo'] + '/src')} "
                f"{shlex.quote(remote['python'])} -m operations.local_route1_candidate_terminal_receipt "
                f"--output {shlex.quote(remote['run_root'])} --candidate-id {shlex.quote(CANDIDATE_ID)} "
                f"--receipt {shlex.quote(receipt_path)}"
            ),
        ))
        result = self._ssh(command, timeout=1800)
        if result.returncode:
            raise RuntimeError(
                "remote MCRB terminal receipt failed:\n"
                f"{result.stdout}\n{result.stderr}"
            )
        receipt = self._remote_cat(receipt_path)
        if receipt is None:
            raise RuntimeError("remote MCRB receipt command returned without a receipt")
        receipt = validate_remote_receipt(receipt, trajectory)
        envelope = {
            "schema": "final-unsb-route1-cross-host-terminal-evidence-v1",
            "recorded": support.now(),
            "source_role": "supplemental_5090_complete_e200_routing_only",
            "candidate_id": CANDIDATE_ID,
            "trajectory": trajectory,
            "terminal_receipt": receipt,
            "cross_host_deltas_merged": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        path = self.operations / "remote_terminal_receipts" / f"{CANDIDATE_ID}_5090.json"
        support.atomic_json(path, envelope)
        self.event(
            "REMOTE_MCRB_E200_TERMINAL",
            trajectory_status=trajectory["status"],
            evidence=str(path),
            evidence_sha256=support.file_sha256(path),
        )
        return trajectory, receipt

    def wait_for_amtnc_revision(self) -> dict[str, Any]:
        path = self.operations / "CROSS_VERSION_REVISION_E200_ADJUDICATION.json"
        fatal = self.operations / "CROSS_VERSION_REVISION_SUCCESSOR_FATAL.json"
        while not self._timed_out():
            if fatal.is_file():
                raise RuntimeError(f"AM-TNC revision successor failed: {fatal}")
            if path.is_file():
                return validate_e200_selection(path)
            heartbeat = self.run_root / "candidates" / AMTNC_ID / "HEARTBEAT.json"
            value = _read_json(heartbeat) if heartbeat.is_file() else {}
            self.state(
                "WAITING_FOR_AMTNC_REVISION_E200",
                amtnc_data_epoch=value.get("data_epoch"),
                amtnc_updates=value.get("updates"),
                elapsed_seconds=time.time() - self.started,
            )
            time.sleep(int(self.contract["poll_seconds"]))
        raise TimeoutError("timed out waiting for AM-TNC revision e200")

    def _run_checked(
        self, argv: list[str], *, cwd: Path, timeout: int | None = None,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            argv, cwd=cwd, env=_env(cwd), capture_output=True, text=True,
            check=False, timeout=timeout,
        )
        if result.returncode:
            raise RuntimeError(
                f"command failed ({result.returncode}): {argv}\n"
                f"{result.stdout}\n{result.stderr}"
            )
        return result

    def prepare_mcrb_4090(self) -> None:
        python = self.contract["python"]
        self.state("MATERIALIZING_MCRB_4090")
        self._run_checked(
            [
                python, "operations/local_route1_freeze_mcrb_generation1.py",
                "--output", str(self.run_root),
            ],
            cwd=self.mcrb_repo, timeout=600,
        )
        gate_path = self.run_root / "candidates" / CANDIDATE_ID / "CANDIDATE_GATE.json"
        gate_ready = False
        if gate_path.is_file():
            try:
                registration = load_candidate_registration(
                    self.run_root, CANDIDATE_ID, require_gate=True,
                )
                gate_ready = registration.gate is not None and registration.gate.get("status") == "PASS"
            except RuntimeError:
                gate_ready = False
        if not gate_ready:
            self.state("RUNNING_MCRB_4090_GATE")
            self._run_checked(
                [
                    python, "-m", "research.local_route1.run",
                    "--stage", "candidate", "--candidate-action", "gate",
                    "--candidate-id", CANDIDATE_ID,
                    "--output", str(self.run_root),
                    "--train-view", self.contract["train_view"],
                    "--data-root", self.contract["data_root"],
                    "--manifest", self.contract["manifest"],
                    "--gpu", "0",
                ],
                cwd=self.mcrb_repo, timeout=7200,
            )
        registration = load_candidate_registration(
            self.run_root, CANDIDATE_ID, require_gate=True,
        )
        if registration.gate is None or registration.gate.get("status") != "PASS":
            raise RuntimeError("MCRB 4090 gate did not freeze as PASS")
        self.event(
            "MCRB_4090_GATE_PASS",
            algorithm_fingerprint=registration.algorithm_fingerprint,
            candidate_fingerprint=registration.candidate_fingerprint,
        )

    def run_mcrb_4090(self) -> Path:
        python = self.contract["python"]
        contract_path = self.operations / "CANDIDATE_EXECUTOR_CONTRACT_G1-03-MCRB-4090.json"
        if not contract_path.is_file():
            self._run_checked(
                [
                    python, "operations/local_route1_candidate_executor.py",
                    "--init-contract", "--contract", str(contract_path),
                    "--main-repo", str(self.mcrb_repo),
                    "--candidate-repo", str(self.mcrb_repo),
                    "--candidate-id", CANDIDATE_ID,
                    "--run-root", str(self.run_root),
                    "--train-view", self.contract["train_view"],
                    "--data-root", self.contract["data_root"],
                    "--manifest", self.contract["manifest"],
                    "--python", python,
                    "--baseline-environment-record", self.contract["baseline_environment_record"],
                ],
                cwd=self.mcrb_repo, timeout=1200,
            )
        log_dir = self.operations / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = log_dir / "mcrb_4090_candidate_executor.stdout.log"
        stderr_path = log_dir / "mcrb_4090_candidate_executor.stderr.log"
        with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
            "a", encoding="utf-8"
        ) as stderr:
            process = subprocess.Popen(
                [
                    python, "operations/local_route1_candidate_executor.py",
                    "--contract", str(contract_path),
                ],
                cwd=self.mcrb_repo, env=_env(self.mcrb_repo), stdout=stdout,
                stderr=stderr, start_new_session=True,
            )
        self.event("MCRB_4090_EXECUTOR_START", child_pid=process.pid)
        heartbeat = self.run_root / "candidates" / CANDIDATE_ID / "HEARTBEAT.json"
        trajectory_path = self.run_root / "candidates" / CANDIDATE_ID / "CANDIDATE_TRAJECTORY.json"
        while process.poll() is None:
            value = _read_json(heartbeat) if heartbeat.is_file() else {}
            self.state(
                "MCRB_4090_E200_EXECUTOR_RUNNING",
                child_pid=process.pid,
                data_epoch=value.get("data_epoch"),
                updates=value.get("updates"),
                elapsed_seconds=time.time() - self.started,
            )
            if self._timed_out():
                raise TimeoutError("timed out waiting for MCRB 4090 replay")
            time.sleep(int(self.contract["poll_seconds"]))
        if int(process.wait()) != 0:
            raise RuntimeError(f"MCRB 4090 candidate executor failed; see {stderr_path}")
        trajectory = validate_remote_trajectory(_read_json(trajectory_path))
        receipt_path = self.operations / "terminal_receipts" / f"{CANDIDATE_ID}.json"
        self._run_checked(
            [
                python, "-m", "operations.local_route1_candidate_terminal_receipt",
                "--output", str(self.run_root), "--candidate-id", CANDIDATE_ID,
                "--receipt", str(receipt_path),
            ],
            cwd=self.mcrb_repo, timeout=1800,
        )
        self.event(
            "MCRB_4090_E200_TERMINAL",
            trajectory_status=trajectory["status"],
            receipt=str(receipt_path),
            receipt_sha256=support.file_sha256(receipt_path),
        )
        return receipt_path

    def materialize_final_selection(
        self, revision: dict[str, Any], mcrb_receipt: Path | None,
    ) -> dict[str, Any]:
        receipts = [
            self.operations / "terminal_receipts" / f"{row['candidate_id']}.json"
            for row in revision["ranking"]
        ]
        if mcrb_receipt is not None:
            receipts.append(mcrb_receipt)
        if len({path.stem for path in receipts}) != len(receipts):
            raise RuntimeError("final e200 receipt set contains duplicate candidates")
        missing = [str(path) for path in receipts if not path.is_file()]
        if missing:
            raise RuntimeError(f"final e200 receipts are missing: {missing}")
        selection = materialize_final_e200_selection(self.run_root, receipts)
        self.event(
            "ROUTE1_FINAL_E200_SELECTION_FROZEN",
            selected_candidate_id=selection["selected_candidate_id"],
            candidate_ids=[row["candidate_id"] for row in selection["ranking"]],
        )
        return selection

    def start_winner_ablation_successor(self, selection: dict[str, Any]) -> int:
        python = self.contract["python"]
        short = self.contract["repo"]["git_commit"][:12]
        path = self.operations / f"WINNER_ABLATION_SUCCESSOR_CONTRACT_{short}.json"
        if not path.is_file():
            self._run_checked(
                [
                    python, "operations/local_route1_winner_ablation_successor.py",
                    "--init-contract", "--contract", str(path),
                    "--repo", str(self.repo), "--run-root", str(self.run_root),
                    "--train-view", self.contract["train_view"],
                    "--data-root", self.contract["data_root"],
                    "--manifest", self.contract["manifest"],
                    "--python", python,
                    "--baseline-environment-record", self.contract["baseline_environment_record"],
                    "--poll-seconds", str(self.contract["poll_seconds"]),
                    "--timeout-seconds", str(self.contract["timeout_seconds"]),
                ],
                cwd=self.repo, timeout=1200,
            )
        log_dir = self.operations / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        stdout = (log_dir / "winner_ablation_successor_final.stdout.log").open(
            "a", encoding="utf-8"
        )
        stderr = (log_dir / "winner_ablation_successor_final.stderr.log").open(
            "a", encoding="utf-8"
        )
        process = subprocess.Popen(
            [
                python, "operations/local_route1_winner_ablation_successor.py",
                "--contract", str(path),
            ],
            cwd=self.repo, env=_env(self.repo), stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        stdout.close()
        stderr.close()
        self.event(
            "WINNER_ABLATION_SUCCESSOR_STARTED",
            child_pid=process.pid,
            selected_candidate_id=selection["selected_candidate_id"],
            contract=str(path),
        )
        return process.pid

    def run(self) -> int:
        self.event("MCRB_CROSS_HOST_SUCCESSOR_START", contract=str(self.contract_path))
        remote_trajectory, _ = self.wait_for_remote_e200()
        revision = self.wait_for_amtnc_revision()
        mcrb_receipt = None
        if remote_trajectory["status"] == REMOTE_PASS:
            self.state("REMOTE_MCRB_PASS_PREPARING_4090_REPLAY")
            self.prepare_mcrb_4090()
            mcrb_receipt = self.run_mcrb_4090()
        else:
            self.state(
                "REMOTE_MCRB_NEGATIVE_4090_REPLAY_NOT_STARTED",
                remote_trajectory_status=remote_trajectory["status"],
            )
            self.event(
                "MCRB_4090_REPLAY_SKIPPED_COMPLETE_REMOTE_NEGATIVE",
                remote_trajectory_status=remote_trajectory["status"],
            )
        selection = self.materialize_final_selection(revision, mcrb_receipt)
        child_pid = self.start_winner_ablation_successor(selection)
        self.state(
            "FINAL_SELECTION_COMPLETE_WINNER_ABLATION_SUCCESSOR_STARTED",
            selected_candidate_id=selection["selected_candidate_id"],
            winner_ablation_successor_pid=child_pid,
            mcrb_4090_replay_started=mcrb_receipt is not None,
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--mcrb-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--remote-host")
    value.add_argument("--remote-port", type=int)
    value.add_argument("--remote-user")
    value.add_argument("--identity-file", type=Path)
    value.add_argument("--known-hosts-file", type=Path)
    value.add_argument("--remote-repo")
    value.add_argument("--remote-run-root")
    value.add_argument("--remote-python")
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=345600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "mcrb_repo", "run_root", "train_view", "data_root", "manifest",
            "python", "baseline_environment_record", "remote_host", "remote_port",
            "remote_user", "identity_file", "known_hosts_file", "remote_repo",
            "remote_run_root", "remote_python",
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
    state_path = run_root / "operations" / "MCRB_CROSS_HOST_SUCCESSOR_STATE.json"
    try:
        with support.executor_lock(
            run_root / "operations" / "MCRB_CROSS_HOST_SUCCESSOR.lock"
        ):
            return MCRBCrossHostSuccessor(args.contract).run()
    except Exception as error:
        failure = {
            "schema": "final-unsb-route1-mcrb-cross-host-successor-fatal-v1",
            "time": support.now(),
            "status": "FAILED",
            "exception_type": type(error).__name__,
            "exception": str(error),
            "traceback": traceback.format_exc(),
            "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        support.atomic_json(
            run_root / "operations" / "MCRB_CROSS_HOST_SUCCESSOR_FATAL.json", failure
        )
        support.atomic_json(state_path, {**failure, "schema": STATE_SCHEMA})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
