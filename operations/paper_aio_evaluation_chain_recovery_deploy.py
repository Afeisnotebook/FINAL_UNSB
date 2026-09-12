"""Deploy a replacement paper evaluation chain after an upstream run moves.

The training lanes are immutable.  This helper creates new output roots and
durable control supervisors whose only semantic difference from the retired
chain is the source-bound AM-TNC recovery export/release path.  Existing
blocked waiters are retained as historical evidence and are never overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any


SCHEMA = "final-unsb-paper-evaluation-chain-recovery-deployment-v1"
COMMAND_SCHEMA = "final-unsb-paper-control-child-command-v1"
MANIFEST_SHA256 = "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
STCGR_LANE = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != value:
            raise RuntimeError(f"refusing to replace a different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, encoding="utf-8"
    ).strip()


def _argument(command: list[str], name: str) -> str:
    index = command.index(name)
    return command[index + 1]


def _state_path(role: str, command: list[str]) -> Path:
    output = Path(_argument(command, "--output"))
    if role == "unified_evaluation":
        filename = "UNIFIED_EVALUATION_SUCCESSOR_STATE.json"
    elif role == "amtnc_evaluation":
        filename = "ALGORITHM_EVALUATION_SUCCESSOR_amtnc_STATE.json"
    elif role == "stcgr_evaluation":
        filename = f"ALGORITHM_EVALUATION_SUCCESSOR_{STCGR_LANE}_STATE.json"
    elif role == "final_delivery":
        filename = "FINAL_DELIVERY_STATE.json"
    else:  # pragma: no cover - construction is closed over these four roles.
        raise RuntimeError(f"unsupported role: {role}")
    return output / "operations" / filename


def build_child_commands(args: argparse.Namespace, commit: str) -> dict[str, list[str]]:
    python = str(args.python.resolve())
    repo = str(args.repo.resolve())
    manifest = str(args.manifest.resolve())
    data_root = str(args.data_root.resolve())
    train_view = str(args.train_view.resolve())
    gpu_lock = str(args.gpu_lock.resolve())
    import_root = str(args.import_root.resolve())
    common = [
        "--repo",
        repo,
        "--required-control-git-commit",
        commit,
    ]
    unified = [
        python,
        "-u",
        "-m",
        "operations.paper_aio_unified_evaluation_successor",
        *common,
        "--output",
        str(args.first_wave_output.resolve()),
        "--import-root",
        import_root,
        "--manifest",
        manifest,
        "--data-root",
        data_root,
        "--train-view",
        train_view,
        "--lane-source",
        "plain=5090B_MATCHED_PLAIN",
        "--lane-source",
        "proposal=5090C",
        "--lane-source",
        "cut=5090B",
        "--lane-source",
        "cyclegan=5090B",
        "--gpu-release-state",
        str(args.amtnc_release_state.resolve()),
        "--gpu-release-status",
        "COMPLETE_E200",
        "--gpu-lock",
        gpu_lock,
        "--gpu",
        str(args.gpu),
        "--poll-seconds",
        str(args.poll_seconds),
        "--timeout-hours",
        str(args.timeout_hours),
    ]
    amtnc = [
        python,
        "-u",
        "-m",
        "operations.paper_aio_algorithm_evaluation_successor",
        *common,
        "--mode",
        "static_pair",
        "--method-lane",
        "amtnc",
        "--method-source-root",
        str(args.amtnc_export_root.resolve()),
        "--method-source-host",
        args.amtnc_source_host,
        "--plain-source-root",
        str(args.plain_export_root.resolve()),
        "--plain-source-host",
        args.plain_source_host,
        "--output",
        str(args.amtnc_evaluation_output.resolve()),
        "--manifest",
        manifest,
        "--data-root",
        data_root,
        "--train-view",
        train_view,
        "--gpu-lock",
        gpu_lock,
        "--gpu",
        str(args.gpu),
        "--poll-seconds",
        str(args.poll_seconds),
        "--timeout-hours",
        str(args.timeout_hours),
    ]
    stcgr = [
        python,
        "-u",
        "-m",
        "operations.paper_aio_algorithm_evaluation_successor",
        *common,
        "--mode",
        "dynamic_candidate",
        "--method-lane",
        STCGR_LANE,
        "--method-source-root",
        import_root,
        "--method-source-host",
        args.stcgr_source_host,
        "--candidate-authority",
        str(args.stcgr_candidate_authority.resolve()),
        "--candidate-metadata-receipt",
        str(args.stcgr_candidate_metadata_receipt.resolve()),
        "--first-wave-cohort",
        str(
            args.first_wave_output.resolve()
            / "gates"
            / "UNIFIED_EVALUATION_COHORT.json"
        ),
        "--output",
        str(args.first_wave_output.resolve()),
        "--manifest",
        manifest,
        "--data-root",
        data_root,
        "--train-view",
        train_view,
        "--gpu-lock",
        gpu_lock,
        "--gpu",
        str(args.gpu),
        "--poll-seconds",
        str(args.poll_seconds),
        "--timeout-hours",
        str(args.timeout_hours),
    ]
    final_delivery = [
        python,
        "-u",
        "-m",
        "operations.paper_aio_final_delivery_successor",
        *common,
        "--python",
        python,
        "--output",
        str(args.final_delivery_output.resolve()),
        "--first-wave-output",
        str(args.first_wave_output.resolve()),
        "--amtnc-output",
        str(args.amtnc_evaluation_output.resolve()),
        "--import-root",
        import_root,
        "--amtnc-export-root",
        str(args.amtnc_export_root.resolve()),
        "--candidate-authority",
        str(args.stcgr_candidate_authority.resolve()),
        "--lane-source",
        "plain=5090B_MATCHED_PLAIN",
        "--lane-source",
        "proposal=5090C",
        "--lane-source",
        "cut=5090B",
        "--lane-source",
        "cyclegan=5090B",
        "--stcgr-source-host",
        args.stcgr_source_host,
        "--first-wave-state",
        str(_state_path("unified_evaluation", unified)),
        "--amtnc-state",
        str(_state_path("amtnc_evaluation", amtnc)),
        "--stcgr-state",
        str(_state_path("stcgr_evaluation", stcgr)),
        "--manifest",
        manifest,
        "--data-root",
        data_root,
        "--train-view",
        train_view,
        "--gpu-lock",
        gpu_lock,
        "--gpu",
        str(args.gpu),
        "--poll-seconds",
        str(args.poll_seconds),
        "--timeout-hours",
        str(args.timeout_hours),
    ]
    return {
        "unified_evaluation": unified,
        "amtnc_evaluation": amtnc,
        "stcgr_evaluation": stcgr,
        "final_delivery": final_delivery,
    }


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _launch(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return process.pid


def _validate_inputs(args: argparse.Namespace) -> tuple[str, str]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit:
        raise RuntimeError("control checkout moved")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("control checkout is dirty")
    if not args.python.is_file():
        raise RuntimeError("pinned evaluator Python is missing")
    python_sha = _sha256(args.python)
    if python_sha != args.required_python_sha256:
        raise RuntimeError("pinned evaluator Python hash changed")
    if _sha256(args.manifest) != MANIFEST_SHA256:
        raise RuntimeError("full-data manifest hash changed")
    for path in (
        args.data_root,
        args.train_view,
        args.import_root,
        args.plain_export_root,
        args.stcgr_candidate_authority,
        args.stcgr_candidate_metadata_receipt,
    ):
        if not path.exists():
            raise RuntimeError(f"required input is missing: {path}")
    release = _read_json(args.amtnc_release_state)
    if release.get("schema") != "final-unsb-paper-supervisor-v1":
        raise RuntimeError("AM-TNC release state schema changed")
    if release.get("status") not in {"CHILD_RUNNING", "COMPLETE_E200"}:
        raise RuntimeError("AM-TNC recovery is not a live or complete release dependency")
    if release.get("paired_metric_control") is not False:
        raise RuntimeError("AM-TNC release state permits paired control")
    if release.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 is not sealed")
    return commit, python_sha


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    commit, python_sha = _validate_inputs(args)
    control_root = args.control_output.resolve()
    commands = build_child_commands(args, commit)
    selected_roles = list(commands) if not args.role else list(dict.fromkeys(args.role))
    command_paths: dict[str, Path] = {}
    command_hashes: dict[str, str] = {}
    for role in selected_roles:
        command = commands[role]
        payload = {
            "schema": COMMAND_SCHEMA,
            "role": role,
            "command": command,
            "cwd": str(args.repo.resolve()),
            "state_path": str(_state_path(role, command)),
        }
        path = control_root / "commands" / f"{role}.json"
        _write_json_exclusive(path, payload)
        command_paths[role] = path
        command_hashes[role] = _sha256(path)

    supervisor_pids: dict[str, int] = {}
    supervisor_states: dict[str, Path] = {}
    for role in selected_roles:
        output = control_root / "supervisors" / role
        state = output / "CONTROL_SUPERVISOR_STATE.json"
        if state.exists():
            existing = _read_json(state)
            existing_pid = int(existing.get("pid", 0))
            if _pid_alive(existing_pid):
                supervisor_pids[role] = existing_pid
                supervisor_states[role] = state
                continue
            raise RuntimeError(f"stale prior deployment requires audit: {state}")
        supervisor_command = [
            str(args.python.resolve()),
            "-u",
            "-m",
            "operations.paper_aio_control_supervisor",
            "--repo",
            str(args.repo.resolve()),
            "--required-control-git-commit",
            commit,
            "--role",
            role,
            "--child-command",
            str(command_paths[role]),
            "--output",
            str(output),
            "--poll-seconds",
            "30",
            "--restart-delay-seconds",
            "15",
            "--max-restarts",
            "5",
            "--timeout-hours",
            str(args.timeout_hours),
        ]
        supervisor_pids[role] = _launch(
            supervisor_command,
            cwd=args.repo.resolve(),
            log=control_root / "logs" / f"{role}.log",
        )
        supervisor_states[role] = state

    deadline = time.monotonic() + args.startup_timeout_seconds
    observed: dict[str, dict[str, Any]] = {}
    while time.monotonic() < deadline:
        observed.clear()
        all_ready = True
        for role, pid in supervisor_pids.items():
            if not _pid_alive(pid) or not supervisor_states[role].is_file():
                all_ready = False
                continue
            state = _read_json(supervisor_states[role])
            child_path = _state_path(role, commands[role])
            child = _read_json(child_path) if child_path.is_file() else {}
            if str(state.get("status", "")).startswith("BLOCKED"):
                raise RuntimeError(f"{role} supervisor blocked: {state}")
            if any(token in str(child.get("status", "")) for token in ("FAIL", "BLOCKED")):
                raise RuntimeError(f"{role} child blocked: {child}")
            observed[role] = {
                "supervisor_pid": pid,
                "supervisor_status": state.get("status"),
                "child_pid": state.get("child_pid"),
                "child_status": child.get("status"),
                "supervisor_state": str(supervisor_states[role]),
                "child_state": str(child_path),
            }
            if not child:
                all_ready = False
        if all_ready:
            break
        time.sleep(1)
    else:
        raise TimeoutError("replacement evaluation chain did not reach waiting states")

    health_output = control_root / "health"
    health_state = health_output / "HEALTH_WATCH_STATE.json"
    health_command = [
        str(args.python.resolve()),
        "-u",
        "-m",
        "operations.paper_aio_health_watch",
        "--output",
        str(health_output),
        "--host-label",
        "4090A_EVALUATION_CHAIN_RECOVERY",
    ]
    for role, value in observed.items():
        health_command.extend(
            [
                "--watch",
                f"{role}_supervisor|{value['supervisor_pid']}|{value['supervisor_state']}|600|0",
                "--watch",
                f"{role}_child|0|{value['child_state']}|600|0",
            ]
        )
    health_command.extend(
        [
            "--disk-path",
            str(args.disk_path.resolve()),
            "--estimated-remaining-write-gib",
            "20",
            "--minimum-headroom-gib",
            "16",
            "--poll-seconds",
            "60",
            "--timeout-hours",
            str(args.timeout_hours),
        ]
    )
    health_pid = _launch(
        health_command,
        cwd=args.repo.resolve(),
        log=control_root / "logs" / "health.log",
    )

    receipt = {
        "schema": SCHEMA,
        "status": "REPLACEMENT_EVALUATION_CHAIN_RUNNING",
        "captured_unix_time": time.time(),
        "control_repo": str(args.repo.resolve()),
        "control_git_commit": commit,
        "runtime_python": str(args.python.resolve()),
        "runtime_python_sha256": python_sha,
        "manifest_sha256": MANIFEST_SHA256,
        "amtnc_release_state": str(args.amtnc_release_state.resolve()),
        "amtnc_export_root": str(args.amtnc_export_root.resolve()),
        "plain_export_root": str(args.plain_export_root.resolve()),
        "outputs": {
            "first_wave": str(args.first_wave_output.resolve()),
            "amtnc_evaluation": str(args.amtnc_evaluation_output.resolve()),
            "final_delivery": str(args.final_delivery_output.resolve()),
        },
        "selected_roles": selected_roles,
        "command_sha256": command_hashes,
        "roles": observed,
        "health_watcher_pid": health_pid,
        "health_state": str(health_state),
        "retired_chain_mutated": False,
        "training_processes_changed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }
    receipt["deployment_identity_sha256"] = _json_sha256(receipt)
    _write_json_exclusive(control_root / "EVALUATION_CHAIN_RECOVERY_DEPLOYMENT.json", receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--required-python-sha256", required=True)
    value.add_argument("--control-output", type=Path, required=True)
    value.add_argument("--first-wave-output", type=Path, required=True)
    value.add_argument("--amtnc-evaluation-output", type=Path, required=True)
    value.add_argument("--final-delivery-output", type=Path, required=True)
    value.add_argument("--import-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--amtnc-release-state", type=Path, required=True)
    value.add_argument("--amtnc-export-root", type=Path, required=True)
    value.add_argument("--amtnc-source-host", default="4090A_AMTNC_OVERFLOW_RECOVERY")
    value.add_argument("--plain-export-root", type=Path, required=True)
    value.add_argument("--plain-source-host", default="4090A")
    value.add_argument("--stcgr-source-host", default="5090A")
    value.add_argument("--stcgr-candidate-authority", type=Path, required=True)
    value.add_argument("--stcgr-candidate-metadata-receipt", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--disk-path", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    value.add_argument("--startup-timeout-seconds", type=int, default=30)
    value.add_argument(
        "--role",
        action="append",
        choices=(
            "unified_evaluation",
            "amtnc_evaluation",
            "stcgr_evaluation",
            "final_delivery",
        ),
        help="Deploy only the selected role(s); omission deploys the complete chain.",
    )
    return value


def main() -> None:
    args = parser().parse_args()
    print(json.dumps(deploy(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
