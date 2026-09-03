"""Start a fresh runtime-matched plain lane after a metric-blind predecessor.

The successor is intentionally narrow: it can only train the frozen paper
``plain`` lane from e0.  It waits for an external predecessor supervisor to
reach COMPLETE_E200, runs the complete engineering gate chain, and requires a
2000-update exact runtime match to a named peer before long training begins.
An optional metric-blind two-epoch capacity gate can pause the new lane and
wait for a co-resident lane when immediate execution would increase makespan.
It never reads a metric file or continues a checkpoint across hosts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is POSIX; tests import on Windows
    fcntl = None


BLOCKED_PREFIXES = ("BLOCKED", "FAIL")
TERMINAL_SUPERVISOR_STATES = {"COMPLETE_E200"}
CONTRACT_SCHEMA = "final-unsb-paper-cross-host-plain-successor-contract-v2"
STATE_SCHEMA = "final-unsb-paper-cross-host-plain-successor-v2"
CONTROL_SOURCE_RELATIVES = (
    "operations/paper_aio_cross_host_plain_successor.py",
)


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def immutable_json(path: Path, payload: dict) -> None:
    if path.is_file():
        if read_json(path) != payload:
            raise RuntimeError(f"immutable capacity receipt changed: {path}")
        return
    atomic_json(path, payload)


def predecessor_decision(status: str | None, *, timed_out: bool) -> str:
    if status in TERMINAL_SUPERVISOR_STATES:
        return "START"
    if status and status.startswith(BLOCKED_PREFIXES):
        return "BLOCK"
    if timed_out:
        return "TIMEOUT"
    return "WAIT"


def capacity_contract(args: argparse.Namespace) -> dict | None:
    fields = {
        "co_resident_supervisor_state": (
            str(args.co_resident_supervisor_state.resolve())
            if args.co_resident_supervisor_state is not None
            else None
        ),
        "co_resident_heartbeat": (
            str(args.co_resident_heartbeat.resolve())
            if args.co_resident_heartbeat is not None
            else None
        ),
        "co_resident_lane_id": args.co_resident_lane_id,
        "plain_isolated_epoch_seconds": args.plain_isolated_epoch_seconds,
        "co_resident_isolated_epoch_seconds": (args.co_resident_isolated_epoch_seconds),
    }
    supplied = [value is not None for value in fields.values()]
    if not any(supplied) and int(args.capacity_probe_epochs) == 0:
        return None
    if not all(supplied) or int(args.capacity_probe_epochs) < 2:
        raise RuntimeError(
            "co-resident capacity gate requires all inputs and at least two probe epochs"
        )
    numeric = (
        float(args.plain_isolated_epoch_seconds),
        float(args.co_resident_isolated_epoch_seconds),
        float(args.minimum_makespan_saving_seconds),
    )
    if numeric[0] <= 0 or numeric[1] <= 0 or numeric[2] < 0:
        raise RuntimeError("co-resident capacity timing inputs must be positive")
    return {
        **fields,
        "capacity_probe_epochs": int(args.capacity_probe_epochs),
        "minimum_makespan_saving_seconds": numeric[2],
    }


def project_colocation_makespan(
    *,
    target_epochs: float,
    plain_completed_epochs: float,
    plain_colocated_epoch_seconds: float,
    plain_isolated_epoch_seconds: float,
    co_resident_completed_epochs: float,
    co_resident_colocated_epoch_seconds: float,
    co_resident_isolated_epoch_seconds: float,
) -> dict:
    values = (
        target_epochs,
        plain_completed_epochs,
        plain_colocated_epoch_seconds,
        plain_isolated_epoch_seconds,
        co_resident_completed_epochs,
        co_resident_colocated_epoch_seconds,
        co_resident_isolated_epoch_seconds,
    )
    if (
        target_epochs <= 0
        or any(float(value) < 0 for value in values[1:])
        or any(float(value) <= 0 for value in values[2:4] + values[5:])
    ):
        raise RuntimeError("capacity projection inputs are invalid")
    plain_remaining = max(0.0, target_epochs - plain_completed_epochs)
    co_resident_remaining = max(0.0, target_epochs - co_resident_completed_epochs)
    co_resident_colocated_remaining_seconds = (
        co_resident_remaining * co_resident_colocated_epoch_seconds
    )
    plain_colocated_completion_seconds = plain_remaining * plain_colocated_epoch_seconds
    if plain_colocated_completion_seconds <= co_resident_colocated_remaining_seconds:
        continue_now_seconds = plain_colocated_completion_seconds
    else:
        plain_epochs_during_colocation = (
            co_resident_colocated_remaining_seconds / plain_colocated_epoch_seconds
        )
        continue_now_seconds = (
            co_resident_colocated_remaining_seconds
            + max(0.0, plain_remaining - plain_epochs_during_colocation)
            * plain_isolated_epoch_seconds
        )
    wait_for_release_seconds = (
        co_resident_remaining * co_resident_isolated_epoch_seconds
        + plain_remaining * plain_isolated_epoch_seconds
    )
    return {
        "plain_remaining_epochs": plain_remaining,
        "co_resident_remaining_epochs": co_resident_remaining,
        "continue_now_seconds": continue_now_seconds,
        "wait_for_release_seconds": wait_for_release_seconds,
        "continue_now_saving_seconds": (
            wait_for_release_seconds - continue_now_seconds
        ),
    }


def validate_runtime_receipt(
    receipt: dict,
    *,
    host_label: str,
    required_protocol_fingerprint: str,
) -> None:
    if (
        receipt.get("schema") != "final-unsb-paper-runtime-twin-receipt-v1"
        or receipt.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or receipt.get("host_label") != host_label
        or receipt.get("updates") != 2000
        or receipt.get("protocol_fingerprint") != required_protocol_fingerprint
        or receipt.get("exact_runtime_equivalence") is not True
        or receipt.get("differences") != {}
        or receipt.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("cross-host plain runtime twin did not pass exactly")


def gate_commands(args: argparse.Namespace) -> list[list[str]]:
    common = [
        "--output",
        str(args.training_output.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--train-view",
        str(args.train_view.resolve()),
        "--gpu",
        str(args.gpu),
    ]
    python = str(args.python.resolve())
    twin_checkpoint = (
        args.training_output.resolve()
        / "runtime_twin"
        / args.host_label
        / "lanes"
        / "plain"
        / "full_state_latest.pt"
    )
    return [
        [
            python,
            "-m",
            "research.paper_aio.run",
            "--stage",
            "preflight",
            "--host-label",
            args.host_label,
            *common,
        ],
        [
            python,
            "-m",
            "research.paper_aio.run",
            "--stage",
            "resume-gate",
            "--lane",
            "plain",
            *common,
        ],
        [
            python,
            "-m",
            "research.paper_aio.run",
            "--stage",
            "runtime-twin",
            "--host-label",
            args.host_label,
            "--peer-receipt",
            str(args.peer_runtime_receipt.resolve()),
            *common,
        ],
        [
            python,
            "-m",
            "research.paper_aio.run",
            "--stage",
            "evaluation-repeat-gate",
            "--lane",
            "plain",
            "--checkpoint",
            str(twin_checkpoint),
            *common,
        ],
        [
            python,
            "-m",
            "research.paper_aio.run",
            "--stage",
            "authorize",
            "--lane",
            "plain",
            *common,
        ],
    ]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-repo", type=Path, required=True)
    parser.add_argument("--training-output", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--predecessor-state", type=Path, required=True)
    parser.add_argument("--peer-runtime-receipt", type=Path, required=True)
    parser.add_argument("--host-label", required=True)
    parser.add_argument("--source-host-label", required=True)
    parser.add_argument("--required-training-git-commit", required=True)
    parser.add_argument("--required-protocol-fingerprint", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--timeout-hours", type=float, default=720.0)
    parser.add_argument("--co-resident-supervisor-state", type=Path)
    parser.add_argument("--co-resident-heartbeat", type=Path)
    parser.add_argument("--co-resident-lane-id")
    parser.add_argument("--capacity-probe-epochs", type=int, default=0)
    parser.add_argument("--plain-isolated-epoch-seconds", type=float)
    parser.add_argument("--co-resident-isolated-epoch-seconds", type=float)
    parser.add_argument("--minimum-makespan-saving-seconds", type=float, default=3600)
    return parser.parse_args(argv)


def git_identity(repo: Path) -> tuple[str, bool]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        text=True,
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=repo,
            text=True,
        ).strip()
    )
    return head, dirty


def frozen_contract(
    args: argparse.Namespace, *, colocation: dict | None,
) -> dict:
    control_repo = Path(__file__).resolve().parents[1]
    training_repo = args.training_repo.resolve()
    control_head, control_dirty = git_identity(control_repo)
    training_head, training_dirty = git_identity(training_repo)
    if control_dirty:
        raise RuntimeError("cross-host successor control checkout is not frozen")
    if training_head != args.required_training_git_commit or training_dirty:
        raise RuntimeError("frozen plain training checkout identity changed")
    peer_path = args.peer_runtime_receipt.resolve()
    manifest = args.manifest.resolve()
    if not peer_path.is_file() or not manifest.is_file():
        raise RuntimeError("cross-host successor lacks peer receipt or manifest")
    peer = read_json(peer_path)
    manifest_sha256 = file_sha256(manifest)
    if (
        peer.get("schema") != "final-unsb-paper-runtime-twin-receipt-v1"
        or peer.get("status") != "LOCAL_TWIN_COMPLETE"
        or peer.get("updates") != 2000
        or peer.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
        or peer.get("manifest_sha256") != manifest_sha256
        or peer.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("peer runtime receipt is invalid")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(control_repo),
        "control_git_commit": control_head,
        "control_source_sha256": {
            relative: file_sha256(control_repo / relative)
            for relative in CONTROL_SOURCE_RELATIVES
        },
        "training_repo": str(training_repo),
        "training_git_commit": training_head,
        "training_output": str(args.training_output.resolve()),
        "python": str(args.python.resolve()),
        "manifest": str(manifest),
        "manifest_sha256": manifest_sha256,
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "predecessor_state": str(args.predecessor_state.resolve()),
        "peer_runtime_receipt": str(peer_path),
        "peer_runtime_receipt_sha256": file_sha256(peer_path),
        "peer_host_label": str(peer.get("host_label", "")),
        "host_label": str(args.host_label),
        "source_host_label": str(args.source_host_label),
        "required_protocol_fingerprint": str(
            args.required_protocol_fingerprint
        ),
        "gpu": int(args.gpu),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "co_resident_capacity_gate": colocation,
        "fresh_e0_required": True,
        "cross_host_checkpoint_resume": False,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def verify_frozen_contract(contract: dict) -> None:
    control_repo = Path(contract["control_repo"])
    training_repo = Path(contract["training_repo"])
    control_head, control_dirty = git_identity(control_repo)
    training_head, training_dirty = git_identity(training_repo)
    if control_head != contract["control_git_commit"] or control_dirty:
        raise RuntimeError("cross-host successor control checkout moved")
    if training_head != contract["training_git_commit"] or training_dirty:
        raise RuntimeError("cross-host successor training checkout moved")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(control_repo / relative) != expected:
            raise RuntimeError(f"cross-host successor source changed: {relative}")
    if file_sha256(Path(contract["manifest"])) != contract["manifest_sha256"]:
        raise RuntimeError("cross-host successor manifest changed")
    if (
        file_sha256(Path(contract["peer_runtime_receipt"]))
        != contract["peer_runtime_receipt_sha256"]
    ):
        raise RuntimeError("cross-host successor peer runtime receipt changed")


def run_logged(command: list[str], *, cwd: Path, log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{time.time():.3f}] {json.dumps(command)}\n")
        handle.flush()
        return subprocess.run(
            command,
            cwd=cwd,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode


def state_payload(args: argparse.Namespace, *, status: str, **extra) -> dict:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "predecessor_state": str(args.predecessor_state.resolve()),
        "lane_id": "plain",
        "host_label": args.host_label,
        "source_host_label": args.source_host_label,
        "training_git_commit": args.required_training_git_commit,
        "required_protocol_fingerprint": args.required_protocol_fingerprint,
        "fresh_e0_required": True,
        "cross_host_checkpoint_resume": False,
        "co_resident_capacity_gate_configured": (int(args.capacity_probe_epochs) >= 2),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run_capacity_gate(
    args: argparse.Namespace,
    *,
    repo: Path,
    output: Path,
    python: Path,
    protocol: dict,
    log: Path,
    contract: dict,
) -> dict:
    companion_state_path = Path(contract["co_resident_supervisor_state"]).resolve()
    companion_heartbeat_path = Path(contract["co_resident_heartbeat"]).resolve()
    companion_state = (
        read_json(companion_state_path) if companion_state_path.is_file() else {}
    )
    if companion_state.get("status") == "COMPLETE_E200":
        return {
            "status": "BYPASS_CO_RESIDENT_ALREADY_COMPLETE",
            "decision": "CONTINUE_PLAIN_NOW",
            "co_resident_lane_id": contract["co_resident_lane_id"],
            "performance_values_read": False,
        }

    training = protocol.get("training") or {}
    steps_per_epoch = int(training.get("steps_per_data_epoch", -1))
    target_updates = int(training.get("target_updates", -1))
    target_epochs = int(training.get("target_data_epochs", -1))
    if (
        steps_per_epoch <= 0
        or target_updates != steps_per_epoch * target_epochs
        or target_epochs != 200
    ):
        raise RuntimeError("capacity gate requires the frozen 200-epoch protocol")
    probe_updates = steps_per_epoch * int(contract["capacity_probe_epochs"])
    probe_command = [
        str(python),
        "-m",
        "research.paper_aio.run",
        "--stage",
        "train",
        "--lane",
        "plain",
        "--resume",
        "--engineering-stop-after-updates",
        str(probe_updates),
        "--output",
        str(output),
        "--manifest",
        str(args.manifest.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--train-view",
        str(args.train_view.resolve()),
        "--gpu",
        str(args.gpu),
    ]
    code = run_logged(probe_command, cwd=repo, log=log)
    if code:
        raise RuntimeError(f"co-resident capacity probe failed with code {code}")
    plain_heartbeat_path = output / "lanes" / "plain" / "HEARTBEAT.json"
    plain = read_json(plain_heartbeat_path)
    companion = read_json(companion_heartbeat_path)
    if (
        plain.get("lane_id") != "plain"
        or int(plain.get("updates", -1)) != probe_updates
        or float(plain.get("data_epoch", -1)) != contract["capacity_probe_epochs"]
        or float(plain.get("epoch_wall_seconds", -1)) <= 0
        or plain.get("paired_controller_access") is not False
        or plain.get("confirmation20_opened") is not False
        or companion.get("lane_id") != contract["co_resident_lane_id"]
        or float(companion.get("data_epoch", -1)) < 0
        or float(companion.get("epoch_wall_seconds", -1)) <= 0
        or companion.get("paired_controller_access") is not False
        or companion.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("co-resident capacity heartbeat is invalid")
    projection = project_colocation_makespan(
        target_epochs=float(target_epochs),
        plain_completed_epochs=float(plain["data_epoch"]),
        plain_colocated_epoch_seconds=float(plain["epoch_wall_seconds"]),
        plain_isolated_epoch_seconds=float(contract["plain_isolated_epoch_seconds"]),
        co_resident_completed_epochs=float(companion["data_epoch"]),
        co_resident_colocated_epoch_seconds=float(companion["epoch_wall_seconds"]),
        co_resident_isolated_epoch_seconds=float(
            contract["co_resident_isolated_epoch_seconds"]
        ),
    )
    decision = (
        "CONTINUE_PLAIN_NOW"
        if projection["continue_now_saving_seconds"]
        >= contract["minimum_makespan_saving_seconds"]
        else "WAIT_FOR_CO_RESIDENT_RELEASE"
    )
    return {
        "status": "PASS_METRIC_BLIND_CO_RESIDENT_MAKESPAN_GATE",
        "decision": decision,
        "co_resident_lane_id": contract["co_resident_lane_id"],
        "probe_epochs": contract["capacity_probe_epochs"],
        "plain_colocated_epoch_seconds": plain["epoch_wall_seconds"],
        "plain_isolated_epoch_seconds": contract["plain_isolated_epoch_seconds"],
        "co_resident_data_epoch": companion["data_epoch"],
        "co_resident_colocated_epoch_seconds": companion["epoch_wall_seconds"],
        "co_resident_isolated_epoch_seconds": (
            contract["co_resident_isolated_epoch_seconds"]
        ),
        "minimum_makespan_saving_seconds": (
            contract["minimum_makespan_saving_seconds"]
        ),
        "projection": projection,
        "probe_full_state_preserved_for_exact_resume": True,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def wait_for_co_resident_release(
    args: argparse.Namespace,
    *,
    contract: dict,
    state_path: Path,
    capacity_gate: dict,
    started: float,
) -> None:
    companion_state_path = Path(contract["co_resident_supervisor_state"]).resolve()
    while True:
        companion = (
            read_json(companion_state_path) if companion_state_path.is_file() else {}
        )
        status = companion.get("status")
        if status == "COMPLETE_E200" or (
            isinstance(status, str) and status.startswith(BLOCKED_PREFIXES)
        ):
            return
        if time.time() - started >= args.timeout_hours * 3600:
            raise RuntimeError("timed out waiting for co-resident GPU release")
        atomic_json(
            state_path,
            state_payload(
                args,
                status="WAITING_FOR_CO_RESIDENT_RELEASE_AFTER_CAPACITY_GATE",
                co_resident_status=status,
                capacity_gate=capacity_gate,
            ),
        )
        time.sleep(args.poll_seconds)


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    if args.poll_seconds < 30 or args.timeout_hours < 24:
        raise RuntimeError("successor polling/timeout is unsafe")
    colocation = capacity_contract(args)
    repo = args.training_repo.resolve()
    output = args.training_output.resolve()
    python = args.python.resolve()
    if not python.is_file():
        raise RuntimeError(f"frozen Python runtime is missing: {python}")
    operations = output / "operations"
    state_path = operations / "CROSS_HOST_PLAIN_SUCCESSOR_STATE.json"
    contract_path = operations / "CROSS_HOST_PLAIN_SUCCESSOR_CONTRACT.json"
    lock_path = operations / "CROSS_HOST_PLAIN_SUCCESSOR.lock"
    log = output / "logs" / "CROSS_HOST_PLAIN_SUCCESSOR.log"
    operations.mkdir(parents=True, exist_ok=True)
    contract = frozen_contract(args, colocation=colocation)
    immutable_json(contract_path, contract)
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("cross-host plain successor requires POSIX file locking")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.time()
        while True:
            try:
                verify_frozen_contract(contract)
            except Exception as error:
                atomic_json(
                    state_path,
                    state_payload(
                        args,
                        status="BLOCKED_FROZEN_CONTRACT_DRIFT",
                        control_git_commit=contract["control_git_commit"],
                        peer_runtime_receipt_sha256=contract[
                            "peer_runtime_receipt_sha256"
                        ],
                        failure_type=type(error).__name__,
                    ),
                )
                return 2
            predecessor = (
                read_json(args.predecessor_state.resolve())
                if args.predecessor_state.is_file()
                else {}
            )
            decision = predecessor_decision(
                predecessor.get("status"),
                timed_out=(time.time() - started) >= args.timeout_hours * 3600,
            )
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status={
                        "WAIT": "WAITING_FOR_PREDECESSOR_E200",
                        "START": "PREDECESSOR_COMPLETE_STARTING_EXACT_GATES",
                        "BLOCK": "BLOCKED_PREDECESSOR_NOT_RECOVERABLE",
                        "TIMEOUT": "BLOCKED_PREDECESSOR_TIMEOUT",
                    }[decision],
                    predecessor_status=predecessor.get("status"),
                ),
            )
            if decision == "START":
                break
            if decision in {"BLOCK", "TIMEOUT"}:
                return 3
            time.sleep(args.poll_seconds)

        verify_frozen_contract(contract)
        commands = gate_commands(args)
        for index, command in enumerate(commands, start=1):
            verify_frozen_contract(contract)
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status="RUNNING_EXACT_ENGINEERING_GATES",
                    gate_index=index,
                    gate_count=len(commands),
                ),
            )
            code = run_logged(command, cwd=repo, log=log)
            if code:
                atomic_json(
                    state_path,
                    state_payload(
                        args,
                        status="BLOCKED_ENGINEERING_GATE_FAILURE",
                        gate_index=index,
                        child_returncode=code,
                    ),
                )
                return code

        runtime_path = output / "gates" / f"RUNTIME_TWIN_{args.host_label}.json"
        validate_runtime_receipt(
            read_json(runtime_path),
            host_label=args.host_label,
            required_protocol_fingerprint=args.required_protocol_fingerprint,
        )
        protocol = read_json(output / "PAPER_PROTOCOL.json")
        if protocol.get("protocol_fingerprint") != args.required_protocol_fingerprint:
            raise RuntimeError("plain protocol fingerprint changed after gates")

        capacity_gate = None
        if colocation is not None:
            verify_frozen_contract(contract)
            atomic_json(
                state_path,
                state_payload(
                    args, status="RUNNING_METRIC_BLIND_CO_RESIDENT_CAPACITY_GATE"
                ),
            )
            try:
                capacity_gate = run_capacity_gate(
                    args,
                    repo=repo,
                    output=output,
                    python=python,
                    protocol=protocol,
                    log=log,
                    contract=colocation,
                )
                capacity_receipt_path = (
                    operations / "CORESIDENT_MAKESPAN_CAPACITY_GATE.json"
                )
                immutable_json(capacity_receipt_path, capacity_gate)
                capacity_gate = {
                    **capacity_gate,
                    "receipt": str(capacity_receipt_path),
                    "receipt_sha256": file_sha256(capacity_receipt_path),
                }
            except Exception as error:
                atomic_json(
                    state_path,
                    state_payload(
                        args,
                        status="BLOCKED_CO_RESIDENT_CAPACITY_GATE_FAILURE",
                        failure_type=type(error).__name__,
                    ),
                )
                return 4
            if capacity_gate["decision"] == "WAIT_FOR_CO_RESIDENT_RELEASE":
                wait_for_co_resident_release(
                    args,
                    contract=colocation,
                    state_path=state_path,
                    capacity_gate=capacity_gate,
                    started=started,
                )
                verify_frozen_contract(contract)

        export_log = output / "logs" / "EXPORT_SUCCESSOR_plain.log"
        export_handle = export_log.open("a", encoding="utf-8")
        export_command = [
            str(python),
            str(repo / "operations" / "paper_aio_export_successor.py"),
            "--repo",
            str(repo),
            "--source-output",
            str(output),
            "--destination",
            str(output / "exports"),
            "--lane",
            "plain",
            "--source-host-label",
            args.source_host_label,
            "--required-training-git-commit",
            args.required_training_git_commit,
            "--required-training-protocol-fingerprint",
            args.required_protocol_fingerprint,
            "--poll-seconds",
            "60",
            "--timeout-hours",
            str(args.timeout_hours),
        ]
        exporter = subprocess.Popen(
            export_command,
            cwd=repo,
            stdout=export_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        supervisor_command = [
            str(python),
            str(repo / "operations" / "paper_aio_supervisor.py"),
            "--repo",
            str(repo),
            "--output",
            str(output),
            "--manifest",
            str(args.manifest.resolve()),
            "--data-root",
            str(args.data_root.resolve()),
            "--train-view",
            str(args.train_view.resolve()),
            "--lane",
            "plain",
            "--gpu",
            str(args.gpu),
        ]
        supervisor_log = output / "logs" / "SUPERVISOR_plain.log"
        supervisor_handle = supervisor_log.open("a", encoding="utf-8")
        supervisor = subprocess.Popen(
            supervisor_command,
            cwd=repo,
            stdout=supervisor_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while supervisor.poll() is None:
            atomic_json(
                state_path,
                state_payload(
                    args,
                    status="PLAIN_SUPERVISOR_RUNNING",
                    supervisor_pid=supervisor.pid,
                    export_successor_pid=exporter.pid,
                    exact_runtime_equivalence=True,
                    capacity_gate=capacity_gate,
                ),
            )
            time.sleep(args.poll_seconds)
        supervisor_handle.close()
        export_handle.close()
        supervisor_state_path = output / "gates" / "SUPERVISOR_plain.json"
        supervisor_state = (
            read_json(supervisor_state_path) if supervisor_state_path.is_file() else {}
        )
        complete = (
            supervisor.returncode == 0
            and supervisor_state.get("status") == "COMPLETE_E200"
        )
        atomic_json(
            state_path,
            state_payload(
                args,
                status="COMPLETE_PLAIN_E200"
                if complete
                else "BLOCKED_PLAIN_SUPERVISOR_EXIT",
                supervisor_pid=supervisor.pid,
                export_successor_pid=exporter.pid,
                supervisor_returncode=supervisor.returncode,
                supervisor_status=supervisor_state.get("status"),
                exact_runtime_equivalence=True,
                capacity_gate=capacity_gate,
            ),
        )
        return 0 if complete else (supervisor.returncode or 5)


if __name__ == "__main__":
    raise SystemExit(main())
