"""Keep an existing paper training supervisor recoverable without disturbing it.

The lane's own ``paper_aio_supervisor.py`` remains the only process allowed to
launch training.  This outer guard adopts a live supervisor, watches only
process identity and full-state metadata, and relaunches the exact frozen
supervisor command only after both supervisor and trainer are absent.  It
never reads evaluation outputs or performance values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

try:  # pragma: no cover - deployment is Linux; tests also exercise helpers.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


CONTRACT_SCHEMA = "final-unsb-paper-training-supervisor-guard-contract-v3"
STATE_SCHEMA = "final-unsb-paper-training-supervisor-guard-state-v3"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any] | None:
    values = (args.runtime_root, args.runtime_manifest)
    if not any(value is not None for value in values):
        return None
    if not all(value is not None for value in values):
        raise RuntimeError(
            "runtime-root and runtime-manifest must be provided together"
        )
    python = args.python.resolve(strict=True)
    runtime_root = args.runtime_root.resolve(strict=True)
    runtime_manifest = args.runtime_manifest.resolve(strict=True)
    try:
        python.relative_to(runtime_root)
    except ValueError as error:
        raise RuntimeError("recovery Python must be inside the isolated runtime") from error
    actual_python_sha256 = _sha256(python)
    if (
        args.required_python_sha256 is not None
        and actual_python_sha256 != args.required_python_sha256
    ):
        raise RuntimeError("recovery Python hash does not match the required identity")
    manifest_value = _read_json(runtime_manifest)
    entries = manifest_value.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("runtime recovery manifest has no entries")
    return {
        "runtime_root": str(runtime_root),
        "runtime_manifest": str(runtime_manifest),
        "runtime_manifest_sha256": _sha256(runtime_manifest),
        "runtime_manifest_entry_count": len(entries),
        "python": str(python),
        "python_sha256": actual_python_sha256,
    }


def verify_runtime_identity(
    identity: dict[str, Any], *, verify_all_entries: bool
) -> dict[str, Any]:
    """Fail closed if the isolated recovery runtime has changed.

    The inexpensive interpreter and manifest checks run on every guard poll.
    Full entry hashing is reserved for startup/preflight and the instant before
    a recovery launch, so a healthy training process is not burdened by I/O.
    """
    root = Path(identity["runtime_root"]).resolve(strict=True)
    manifest = Path(identity["runtime_manifest"]).resolve(strict=True)
    python = Path(identity["python"]).resolve(strict=True)
    if (
        _sha256(python) != identity["python_sha256"]
        or _sha256(manifest) != identity["runtime_manifest_sha256"]
    ):
        raise RuntimeError("isolated runtime interpreter or manifest changed")
    result = {
        "python_sha256": identity["python_sha256"],
        "manifest_sha256": identity["runtime_manifest_sha256"],
        "entry_count": int(identity["runtime_manifest_entry_count"]),
        "full_entry_hashes_verified": False,
    }
    if not verify_all_entries:
        return result
    value = _read_json(manifest)
    entries = value.get("entries")
    if not isinstance(entries, list) or len(entries) != result["entry_count"]:
        raise RuntimeError("isolated runtime manifest entry count changed")
    for entry in entries:
        if not isinstance(entry, dict):
            raise RuntimeError("invalid isolated runtime manifest entry")
        relative = entry.get("relative_path")
        expected_sha256 = entry.get("sha256")
        expected_size = entry.get("size")
        if not isinstance(relative, str) or not relative:
            raise RuntimeError("invalid isolated runtime relative path")
        candidate = (root / relative).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise RuntimeError("isolated runtime manifest escapes its root") from error
        if (
            not candidate.is_file()
            or candidate.stat().st_size != int(expected_size)
            or candidate.stat().st_mode & 0o222
            or _sha256(candidate) != expected_sha256
        ):
            raise RuntimeError(f"isolated runtime entry identity failed: {relative}")
    result["full_entry_hashes_verified"] = True
    return result


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _cmdline(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError:
        return []
    return [part.decode("utf-8", "replace") for part in raw.split(b"\0") if part]


def _argument(tokens: list[str], name: str) -> str | None:
    try:
        index = tokens.index(name)
    except ValueError:
        return None
    return tokens[index + 1] if index + 1 < len(tokens) else None


def guard_lane_identity(lane: str, candidate_id: str | None) -> str:
    """Return the checkpoint lane identity for a static or candidate runner."""
    if lane == "candidate":
        value = str(candidate_id or "")
        if not _SAFE_ID.fullmatch(value):
            raise RuntimeError("candidate guard requires a safe candidate identity")
        return value
    if candidate_id is not None:
        raise RuntimeError("candidate-id is only valid when lane is candidate")
    if not _SAFE_ID.fullmatch(lane):
        raise RuntimeError("unsafe lane identity")
    return lane


def command_matches_lane(
    tokens: list[str], *, output: Path, lane: str, candidate_id: str | None
) -> bool:
    """Match the exact runner CLI, including candidate identity when present."""
    if not tokens or _argument(tokens, "--output") != str(output.resolve()):
        return False
    if _argument(tokens, "--lane") != lane:
        return False
    observed_candidate = _argument(tokens, "--candidate-id")
    if lane == "candidate":
        return observed_candidate == candidate_id
    return observed_candidate is None


def matching_lane_processes(
    output: Path, lane: str, candidate_id: str | None = None
) -> dict[str, list[int]]:
    """Return exact-output/lane supervisor and trainer PIDs from Linux procfs."""
    result: dict[str, list[int]] = {"supervisors": [], "trainers": []}
    proc = Path("/proc")
    if not proc.is_dir():
        return result
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        tokens = _cmdline(int(entry.name))
        if not command_matches_lane(
            tokens, output=output, lane=lane, candidate_id=candidate_id
        ):
            continue
        if any(token.endswith("paper_aio_supervisor.py") for token in tokens):
            result["supervisors"].append(int(entry.name))
        if (
            "research.paper_aio.run" in tokens
            and _argument(tokens, "--stage") == "train"
        ):
            result["trainers"].append(int(entry.name))
    for values in result.values():
        values.sort()
    return result


def process_decision(
    *, supervisor_status: str, supervisors: list[int], trainers: list[int]
) -> str:
    if supervisor_status == "COMPLETE_E200":
        return "COMPLETE"
    if len(supervisors) > 1 or len(trainers) > 1:
        return "BLOCK_DUPLICATE"
    if len(supervisors) == 1:
        return "ADOPT"
    if len(trainers) == 1:
        return "WAIT_ORPHAN_TRAINER"
    return "RESTART"


def next_no_progress_count(
    prior: int, *, last_restart_step: int | None, current_step: int
) -> int:
    if last_restart_step is None or current_step > last_restart_step:
        return 0
    return int(prior) + 1


def _checkpoint(args: argparse.Namespace, *, verify_file_hash: bool) -> dict[str, Any]:
    lane_id = guard_lane_identity(args.lane, args.candidate_id)
    sidecar = args.output / "lanes" / lane_id / "full_state_latest.pt.json"
    checkpoint = args.output / "lanes" / lane_id / "full_state_latest.pt"
    if not sidecar.is_file() or not checkpoint.is_file():
        raise RuntimeError("complete full-state checkpoint and sidecar are required")
    value = _read_json(sidecar)
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise RuntimeError("full-state sidecar metadata is missing")
    step = int(value.get("step", -1))
    target_steps = int(value.get("target_steps", -1))
    if (
        value.get("schema") != "final-unsb-paper-aio-full-state-v1"
        or value.get("lane_id") != lane_id
        or step < 0
        or target_steps <= 0
        or step > target_steps
        or metadata.get("git_commit") != args.required_training_git_commit
        or metadata.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
    ):
        raise RuntimeError("full-state sidecar identity is invalid for exact resume")
    expected = str(value.get("full_state_sha256", ""))
    if verify_file_hash and (not expected or _sha256(checkpoint) != expected):
        raise RuntimeError("full-state checkpoint hash does not match its sidecar")
    return {
        "step": step,
        "target_steps": target_steps,
        "sha256": expected,
        "sidecar": str(sidecar),
    }


def _supervisor_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python.resolve()),
        str((args.training_repo / "operations" / "paper_aio_supervisor.py").resolve()),
        "--repo", str(args.training_repo.resolve()),
        "--output", str(args.output.resolve()),
        "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--lane", args.lane,
        "--gpu", str(args.gpu),
        "--maximum-consecutive-failures", "3",
    ]
    if args.candidate_id is not None:
        command.extend(["--candidate-id", args.candidate_id])
    return command


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    control_repo = args.control_repo.resolve()
    training_repo = args.training_repo.resolve()
    lane_id = guard_lane_identity(args.lane, args.candidate_id)
    python = args.python.resolve(strict=True)
    python_sha256 = _sha256(python)
    if (
        args.required_python_sha256 is not None
        and python_sha256 != args.required_python_sha256
    ):
        raise RuntimeError("recovery Python hash does not match the required identity")
    if args.poll_seconds < 10 or args.restart_delay_seconds < 5:
        raise RuntimeError("unsafe polling or restart delay")
    if args.maximum_consecutive_no_progress_restarts not in {1, 2, 3}:
        raise RuntimeError("unsafe no-progress restart budget")
    if args.timeout_hours < 24:
        raise RuntimeError("guard timeout must be at least 24 hours")
    control_source = control_repo / "operations" / "paper_aio_training_supervisor_guard.py"
    training_source = training_repo / "operations" / "paper_aio_supervisor.py"
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN",
        "lane_id": lane_id,
        "runner_lane": args.lane,
        "candidate_id": args.candidate_id,
        "control_repo": str(control_repo),
        "control_git_commit": args.required_control_git_commit,
        "control_source": str(control_source),
        "control_source_sha256": _sha256(control_source),
        "training_repo": str(training_repo),
        "training_git_commit": args.required_training_git_commit,
        "training_supervisor_source": str(training_source),
        "training_supervisor_source_sha256": _sha256(training_source),
        "protocol_fingerprint": args.required_protocol_fingerprint,
        "output": str(args.output.resolve()),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "python": str(python),
        "python_sha256": python_sha256,
        "gpu": int(args.gpu),
        "initial_supervisor_pid": int(args.initial_supervisor_pid),
        "supervisor_command": _supervisor_command(args),
        "poll_seconds": int(args.poll_seconds),
        "restart_delay_seconds": int(args.restart_delay_seconds),
        "maximum_consecutive_no_progress_restarts": int(
            args.maximum_consecutive_no_progress_restarts
        ),
        "timeout_hours": float(args.timeout_hours),
        "performance_values_available_to_guard": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    runtime_identity = _runtime_identity(args)
    if runtime_identity is not None:
        verify_runtime_identity(runtime_identity, verify_all_entries=True)
        contract["runtime_identity"] = runtime_identity
    return contract


def _verify(contract: dict[str, Any], *, verify_runtime_tree: bool = False) -> None:
    control_repo = Path(contract["control_repo"])
    training_repo = Path(contract["training_repo"])
    if (
        _git(control_repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(control_repo, "status", "--porcelain")
        or _sha256(Path(contract["control_source"]))
        != contract["control_source_sha256"]
        or _git(training_repo, "rev-parse", "HEAD")
        != contract["training_git_commit"]
        or _git(training_repo, "status", "--porcelain")
        or _sha256(Path(contract["training_supervisor_source"]))
        != contract["training_supervisor_source_sha256"]
        or _sha256(Path(contract["python"]).resolve(strict=True))
        != contract["python_sha256"]
    ):
        raise RuntimeError("frozen control or training source identity changed")
    protocol = _read_json(Path(contract["output"]) / "PAPER_PROTOCOL.json")
    authorization_name = (
        f"CANDIDATE_AUTHORIZATION_{contract['lane_id']}.json"
        if contract["runner_lane"] == "candidate"
        else f"LANE_AUTHORIZATION_{contract['lane_id']}.json"
    )
    authorization = _read_json(Path(contract["output"]) / "gates" / authorization_name)
    if (
        protocol.get("protocol_fingerprint") != contract["protocol_fingerprint"]
        or authorization.get("status") != "PASS"
        or authorization.get("lane_id") != contract["lane_id"]
        or authorization.get("protocol_fingerprint")
        != contract["protocol_fingerprint"]
        or authorization.get("paired_metric_control") is not False
        or authorization.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("protocol or lane authorization identity changed")
    runtime_identity = contract.get("runtime_identity")
    if runtime_identity is not None:
        if not isinstance(runtime_identity, dict):
            raise RuntimeError("invalid runtime identity contract")
        verify_runtime_identity(
            runtime_identity, verify_all_entries=verify_runtime_tree
        )


def _state(contract: dict[str, Any], status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "lane_id": contract["lane_id"],
        "control_git_commit": contract["control_git_commit"],
        "training_git_commit": contract["training_git_commit"],
        "protocol_fingerprint": contract["protocol_fingerprint"],
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "runtime_identity_bound": "runtime_identity" in contract,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.control_repo = args.control_repo.resolve()
    args.training_repo = args.training_repo.resolve()
    args.output = args.output.resolve()
    guard_output = args.guard_output.resolve()
    guard_output.mkdir(parents=True, exist_ok=True)
    contract_path = guard_output / "TRAINING_SUPERVISOR_GUARD_CONTRACT.json"
    state_path = guard_output / "TRAINING_SUPERVISOR_GUARD_STATE.json"
    lock_path = guard_output / "TRAINING_SUPERVISOR_GUARD.lock"
    log_path = guard_output / "TRAINING_SUPERVISOR_RECOVERY.log"
    contract = _contract(args)
    created = not contract_path.is_file()
    if created:
        _atomic_json(contract_path, contract)
    elif _read_json(contract_path) != contract:
        raise RuntimeError("training supervisor guard contract changed")
    _verify(contract, verify_runtime_tree=True)
    processes = matching_lane_processes(args.output, args.lane, args.candidate_id)
    if created and (
        int(args.initial_supervisor_pid) not in processes["supervisors"]
        or len(processes["supervisors"]) != 1
        or len(processes["trainers"]) > 1
    ):
        raise RuntimeError("initial live process set is not uniquely adoptable")
    if args.verify_only:
        checkpoint = _checkpoint(args, verify_file_hash=True)
        return _state(
            contract,
            "PASS_RUNTIME_AND_CHECKPOINT_RECOVERY_PREFLIGHT",
            checkpoint_step=checkpoint["step"],
            checkpoint_sha256=checkpoint["sha256"],
            runtime_entry_count=contract.get("runtime_identity", {}).get(
                "runtime_manifest_entry_count"
            ),
            full_runtime_entry_hashes_verified=True,
            supervisor_pids=processes["supervisors"],
            trainer_pids=processes["trainers"],
        )

    started = time.time()
    total_restarts = 0
    no_progress_restarts = 0
    initial_checkpoint = _checkpoint(args, verify_file_hash=False)
    last_restart_step: int | None = int(initial_checkpoint["step"])
    with lock_path.open("a+", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("training supervisor guard requires Linux flock")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("training supervisor guard is already running") from error
        while True:
            _verify(contract)
            supervisor_state_path = (
                args.output / "gates" / f"SUPERVISOR_{contract['lane_id']}.json"
            )
            supervisor_state = (
                _read_json(supervisor_state_path)
                if supervisor_state_path.is_file()
                else {}
            )
            processes = matching_lane_processes(
                args.output, args.lane, args.candidate_id
            )
            decision = process_decision(
                supervisor_status=str(supervisor_state.get("status", "")),
                supervisors=processes["supervisors"],
                trainers=processes["trainers"],
            )
            checkpoint = _checkpoint(args, verify_file_hash=False)
            common = {
                "supervisor_pids": processes["supervisors"],
                "trainer_pids": processes["trainers"],
                "supervisor_status": supervisor_state.get("status"),
                "checkpoint_step": checkpoint["step"],
                "checkpoint_sha256": checkpoint["sha256"],
                "total_restarts": total_restarts,
                "consecutive_no_progress_restarts": no_progress_restarts,
            }
            if decision == "COMPLETE":
                result = _state(contract, "COMPLETE_E200", **common)
                _atomic_json(state_path, result)
                return result
            if decision == "BLOCK_DUPLICATE":
                result = _state(contract, "BLOCKED_DUPLICATE_LANE_PROCESS", **common)
                _atomic_json(state_path, result)
                return result
            if time.time() - started >= contract["timeout_hours"] * 3600:
                result = _state(contract, "BLOCKED_GUARD_TIMEOUT", **common)
                _atomic_json(state_path, result)
                return result
            if decision == "ADOPT":
                _atomic_json(state_path, _state(contract, "MONITORING_EXISTING_SUPERVISOR", **common))
                time.sleep(contract["poll_seconds"])
                continue
            if decision == "WAIT_ORPHAN_TRAINER":
                _atomic_json(state_path, _state(contract, "WAITING_FOR_ORPHAN_TRAINER_EXIT", **common))
                time.sleep(contract["poll_seconds"])
                continue

            if int(checkpoint["step"]) >= int(checkpoint["target_steps"]):
                result = _state(
                    contract,
                    "BLOCKED_TERMINAL_CHECKPOINT_WITHOUT_COMPLETE_RUN_STATE",
                    **common,
                )
                _atomic_json(state_path, result)
                return result

            no_progress_restarts = next_no_progress_count(
                no_progress_restarts,
                last_restart_step=last_restart_step,
                current_step=int(checkpoint["step"]),
            )
            if (
                no_progress_restarts
                > contract["maximum_consecutive_no_progress_restarts"]
            ):
                result = _state(
                    contract,
                    "BLOCKED_NO_PROGRESS_RESTART_BUDGET_EXHAUSTED",
                    **{**common, "consecutive_no_progress_restarts": no_progress_restarts},
                )
                _atomic_json(state_path, result)
                return result
            verified = _checkpoint(args, verify_file_hash=True)
            _verify(contract, verify_runtime_tree=True)
            with log_path.open("a", encoding="utf-8") as log:
                log.write(
                    f"\n[{time.time():.3f}] exact supervisor recovery from "
                    f"step {verified['step']}\n"
                )
                log.flush()
                environment = os.environ.copy()
                environment["PYTHONPATH"] = os.pathsep.join(
                    [str(args.training_repo), str(args.training_repo / "src")]
                )
                child = subprocess.Popen(
                    contract["supervisor_command"],
                    cwd=args.training_repo,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=environment,
                    start_new_session=True,
                )
            total_restarts += 1
            last_restart_step = int(verified["step"])
            _atomic_json(
                state_path,
                _state(
                    contract,
                    "RECOVERY_SUPERVISOR_LAUNCHED",
                    recovered_supervisor_pid=child.pid,
                    recovery_checkpoint_step=verified["step"],
                    recovery_checkpoint_sha256=verified["sha256"],
                    total_restarts=total_restarts,
                    consecutive_no_progress_restarts=no_progress_restarts,
                    supervisor_pids=[],
                    trainer_pids=[],
                    supervisor_status=supervisor_state.get("status"),
                ),
            )
            time.sleep(contract["restart_delay_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--control-repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--training-repo", type=Path, required=True)
    value.add_argument("--required-training-git-commit", required=True)
    value.add_argument("--required-protocol-fingerprint", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--lane", required=True)
    value.add_argument("--candidate-id")
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--runtime-root", type=Path)
    value.add_argument("--runtime-manifest", type=Path)
    value.add_argument("--required-python-sha256")
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--initial-supervisor-pid", type=int, required=True)
    value.add_argument("--guard-output", type=Path, required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--restart-delay-seconds", type=int, default=30)
    value.add_argument("--maximum-consecutive-no-progress-restarts", type=int, default=2)
    value.add_argument("--timeout-hours", type=float, default=720.0)
    value.add_argument("--verify-only", action="store_true")
    return value


def main() -> int:
    result = run(parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if not str(result["status"]).startswith("BLOCKED") else 2


if __name__ == "__main__":
    raise SystemExit(main())
