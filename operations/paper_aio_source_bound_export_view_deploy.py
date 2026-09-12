"""Deploy an approved read-only source view for a recovered paper lane.

The paper exporter freezes its contract, state and lock below ``source-output``.
That intentionally prevents two exporter identities from sharing one writable
control root.  A recovered lane can nevertheless need a replacement exporter
after its original control state has terminated.  This helper creates a fresh
control root whose scientific lane and supervisor entries are absolute
read-only symlinks to the original run, then deploys the existing exporter,
its recovery supervisor and a metric-blind health watcher there.  Supported
lane/host pairs are explicit so this cannot become an unrestricted provenance
rewriter.

It never loads a checkpoint, changes training, copies model weights, reads a
performance value, or opens confirmation20.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


SCHEMA = "final-unsb-paper-source-bound-export-view-deployment-v1"
VIEW_SCHEMA = "final-unsb-paper-read-only-source-view-v1"
RECOVERY_VIEW_PROFILES = {
    ("4090A", "amtnc"): {
        "profile": "4090A_AMTNC_OVERFLOW_RECOVERY",
        "health_label": "4090A_AMTNC_NORMALIZED_EXPORT",
    },
    ("5090A", "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"): {
        "profile": "5090A_STCGR_EXACT_RESUME_RECOVERY",
        "health_label": "5090A_STCGR_RECOVERY_SOURCE_EXPORT",
    },
}


def recovery_view_profile(source_host_label: str, lane: str) -> dict[str, str]:
    key = (str(source_host_label), str(lane))
    if key not in RECOVERY_VIEW_PROFILES:
        raise RuntimeError(f"unsupported recovery source-view pair: {key!r}")
    return dict(RECOVERY_VIEW_PROFILES[key])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != value:
            raise RuntimeError(f"refusing to replace a different artifact: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *arguments], text=True, encoding="utf-8"
    ).strip()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _launch(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    handle = log.open("ab", buffering=0)
    child = subprocess.Popen(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    handle.close()
    return child.pid


def build_export_command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python.resolve()),
        str((args.repo.resolve() / "operations" / "paper_aio_export_successor.py")),
        "--repo",
        str(args.repo.resolve()),
        "--source-output",
        str(args.view_output.resolve()),
        "--destination",
        str(args.destination.resolve()),
        "--lane",
        str(args.lane),
        "--source-host-label",
        str(args.source_host_label),
        "--required-training-git-commit",
        str(args.required_training_git_commit),
        "--required-training-protocol-fingerprint",
        str(args.required_training_protocol_fingerprint),
        "--poll-seconds",
        str(int(args.poll_seconds)),
        "--timeout-hours",
        str(float(args.timeout_hours)),
    ]


def build_recovery_command(
    args: argparse.Namespace, *, export_contract: Path, export_state: Path
) -> list[str]:
    return [
        str(args.python.resolve()),
        str(
            args.repo.resolve()
            / "operations"
            / "paper_aio_export_successor_recovery_supervisor.py"
        ),
        "--repo",
        str(args.repo.resolve()),
        "--required-control-git-commit",
        str(args.required_control_git_commit),
        "--python",
        str(args.python.resolve()),
        "--adopt-python",
        str(args.python.resolve()),
        "--export-contract",
        str(export_contract.resolve()),
        "--export-state",
        str(export_state.resolve()),
        "--output",
        str((args.view_output.resolve() / "export_recovery")),
        "--poll-seconds",
        "30",
        "--restart-delay-seconds",
        "15",
        "--max-restarts",
        "5",
        "--timeout-hours",
        str(float(args.timeout_hours)),
    ]


def _validate(
    args: argparse.Namespace,
) -> tuple[str, str, Path, Path, dict[str, Any], dict[str, str]]:
    if os.name == "nt":
        raise RuntimeError("source-view deployment is supported only on Linux hosts")
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(repo, "status", "--porcelain"):
        raise RuntimeError("control checkout is not frozen at the required commit")
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError("pinned exporter runtime is absent")
    python_sha = _sha256(python)
    if python_sha != args.required_python_sha256:
        raise RuntimeError("pinned exporter runtime hash changed")
    profile = recovery_view_profile(args.source_host_label, args.lane)

    source = args.source_output.resolve()
    lane = (source / "lanes" / args.lane).resolve()
    supervisor = (source / "gates" / f"SUPERVISOR_{args.lane}.json").resolve()
    if not lane.is_dir() or not supervisor.is_file():
        raise RuntimeError("recovered source lane or supervisor state is absent")
    supervisor_state = _read_json(supervisor)
    if supervisor_state.get("schema") != "final-unsb-paper-supervisor-v1":
        raise RuntimeError("recovered supervisor schema changed")
    if supervisor_state.get("status") not in {"CHILD_RUNNING", "COMPLETE_E200"}:
        raise RuntimeError("recovered source lane is not live or complete")
    if (
        supervisor_state.get("paired_metric_control") is not False
        or supervisor_state.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("recovered supervisor violates the frozen scientific boundary")

    latest = lane / "full_state_latest.pt.json"
    if not latest.is_file():
        raise RuntimeError("recovered source latest sidecar is absent")
    metadata = _read_json(latest).get("metadata", {})
    if metadata.get("git_commit") != args.required_training_git_commit:
        raise RuntimeError("recovered source training commit differs")
    if (
        metadata.get("protocol_fingerprint")
        != args.required_training_protocol_fingerprint
    ):
        raise RuntimeError("recovered source training protocol differs")
    if metadata.get("lane_id") != args.lane:
        raise RuntimeError("recovered source lane metadata differs")
    if metadata.get("confirmation20_opened") is not False:
        raise RuntimeError("recovered source sidecar opened confirmation20")
    return commit, python_sha, lane, supervisor, supervisor_state, profile


def _make_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        raise RuntimeError(f"source view path already exists: {link}")
    link.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(str(target), str(link), target_is_directory=target.is_dir())
    if link.resolve() != target.resolve():
        raise RuntimeError(f"source view link resolved to an unexpected target: {link}")


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    (
        commit,
        python_sha,
        source_lane,
        source_supervisor,
        supervisor_state,
        profile,
    ) = _validate(args)
    view = args.view_output.resolve()
    if view.exists():
        raise RuntimeError("source view output already exists; refusing ambiguous redeployment")
    view.mkdir(parents=True)
    (view / "operations").mkdir()
    _make_link(view / "lanes" / args.lane, source_lane)
    _make_link(view / "gates" / f"SUPERVISOR_{args.lane}.json", source_supervisor)

    latest = source_lane / "full_state_latest.pt.json"
    view_receipt = {
        "schema": VIEW_SCHEMA,
        "status": "READ_ONLY_SOURCE_VIEW_READY",
        "control_git_commit": commit,
        "source_output": str(args.source_output.resolve()),
        "view_output": str(view),
        "lane_id": args.lane,
        "physical_source_host_label": args.source_host_label,
        "recovery_profile": profile["profile"],
        "source_lane_target": str(source_lane),
        "source_supervisor_target": str(source_supervisor),
        "source_supervisor_status": supervisor_state.get("status"),
        "initial_latest_sidecar_sha256": _sha256(latest),
        "source_files_modified": False,
        "checkpoint_loaded": False,
        "checkpoint_copy_performed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json_exclusive(view / "SOURCE_VIEW.json", view_receipt)

    export_command = build_export_command(args)
    exporter_pid = _launch(
        export_command, cwd=args.repo.resolve(), log=view / "logs" / "exporter.log"
    )
    export_contract = view / "operations" / f"EXPORT_SUCCESSOR_{args.lane}_CONTRACT.json"
    export_state = view / "operations" / f"EXPORT_SUCCESSOR_{args.lane}_STATE.json"
    deadline = time.monotonic() + args.startup_timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_alive(exporter_pid):
            raise RuntimeError("normalized source exporter exited during startup")
        if export_contract.is_file() and export_state.is_file():
            state = _read_json(export_state)
            if state.get("status") == "WAITING_FOR_COMPLETE_E200":
                break
            if str(state.get("status", "")).startswith(("BLOCK", "FAIL", "FATAL")):
                raise RuntimeError(f"normalized source exporter blocked: {state}")
        time.sleep(1)
    else:
        raise TimeoutError("normalized source exporter did not reach its waiting state")

    recovery_command = build_recovery_command(
        args, export_contract=export_contract, export_state=export_state
    )
    recovery_pid = _launch(
        recovery_command,
        cwd=args.repo.resolve(),
        log=view / "logs" / "export_recovery.log",
    )
    recovery_state = view / "export_recovery" / "EXPORT_RECOVERY_STATE.json"
    deadline = time.monotonic() + args.startup_timeout_seconds
    while time.monotonic() < deadline:
        if not _pid_alive(recovery_pid):
            raise RuntimeError("normalized export recovery supervisor exited during startup")
        if recovery_state.is_file():
            state = _read_json(recovery_state)
            if state.get("status") == "MONITORING_EXISTING_EXPORT":
                break
            if str(state.get("status", "")).startswith(("BLOCK", "FAIL", "FATAL")):
                raise RuntimeError(f"normalized export recovery supervisor blocked: {state}")
        time.sleep(1)
    else:
        raise TimeoutError("normalized export recovery supervisor did not adopt the exporter")

    health_command = [
        str(args.python.resolve()),
        str(args.repo.resolve() / "operations" / "paper_aio_health_watch.py"),
        "--output",
        str(view / "health"),
        "--host-label",
        profile["health_label"],
        "--watch",
        f"normalized_export_recovery|{recovery_pid}|{recovery_state}|600|0",
        "--watch",
        f"normalized_export_child|0|{export_state}|600|0",
        "--disk-path",
        str(args.disk_path.resolve()),
        "--estimated-remaining-write-gib",
        "1",
        "--minimum-headroom-gib",
        "16",
        "--poll-seconds",
        "60",
        "--timeout-hours",
        str(float(args.timeout_hours)),
    ]
    health_pid = _launch(
        health_command, cwd=args.repo.resolve(), log=view / "logs" / "health.log"
    )

    result = {
        "schema": SCHEMA,
        "status": "NORMALIZED_SOURCE_BOUND_EXPORT_CHAIN_RUNNING",
        "captured_unix_time": time.time(),
        "control_repo": str(args.repo.resolve()),
        "control_git_commit": commit,
        "runtime_python": str(args.python.resolve()),
        "runtime_python_sha256": python_sha,
        "source_output": str(args.source_output.resolve()),
        "view_output": str(view),
        "destination": str(args.destination.resolve()),
        "lane_id": args.lane,
        "source_host_label": args.source_host_label,
        "recovery_profile": profile["profile"],
        "training_git_commit": args.required_training_git_commit,
        "training_protocol_fingerprint": args.required_training_protocol_fingerprint,
        "exporter_pid": exporter_pid,
        "export_recovery_supervisor_pid": recovery_pid,
        "health_watcher_pid": health_pid,
        "export_contract": str(export_contract),
        "export_contract_sha256": _sha256(export_contract),
        "export_state": str(export_state),
        "export_recovery_state": str(recovery_state),
        "health_state": str(view / "health" / "HEALTH_WATCH_STATE.json"),
        "source_files_modified": False,
        "training_processes_changed": False,
        "checkpoint_loaded": False,
        "checkpoint_copy_performed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write_json_exclusive(view / "SOURCE_VIEW_DEPLOYMENT.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--required-python-sha256", required=True)
    value.add_argument("--source-output", type=Path, required=True)
    value.add_argument("--view-output", type=Path, required=True)
    value.add_argument("--destination", type=Path, required=True)
    value.add_argument("--lane", default="amtnc")
    value.add_argument("--source-host-label", default="4090A")
    value.add_argument("--required-training-git-commit", required=True)
    value.add_argument("--required-training-protocol-fingerprint", required=True)
    value.add_argument("--disk-path", type=Path, required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    value.add_argument("--startup-timeout-seconds", type=int, default=30)
    return value


def main() -> None:
    print(json.dumps(deploy(parser().parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
