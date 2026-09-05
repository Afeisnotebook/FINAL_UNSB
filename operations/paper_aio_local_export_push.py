"""Push a completed local source-bound export into the unified paper host.

This is a transport-only successor for a training host that cannot accept an
inbound SSH connection.  It waits for the immutable local export set, checks
every receipt and payload hash, uploads into a fixed remote staging root, and
publishes IMPORT_LANE.json last.  Authentication is read from an environment
variable and is never written to the contract, state, logs, or Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any

from operations import paper_aio_export_relay as relay


CONTRACT_SCHEMA = "final-unsb-paper-local-export-push-contract-v1"
STATE_SCHEMA = "final-unsb-paper-local-export-push-state-v1"
IMPORT_LANE_SCHEMA = relay.IMPORT_LANE_SCHEMA
IMPORT_SET_SCHEMA = relay.IMPORT_SET_SCHEMA
EPOCHS = relay.UNIFIED_EPOCHS
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class LocalExportNotReady(Exception):
    """The local exporter has not published its terminal set yet."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise LocalExportNotReady(str(path)) from error
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for attempt in range(10):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.05 * (attempt + 1))
    finally:
        if temporary.exists():
            temporary.unlink()


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT,
    ).strip()


def inside_local(value: str | Path, root: Path, label: str) -> Path:
    path = Path(value).resolve()
    root = Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes frozen local root: {path}") from error
    return path


def remote_root(value: str) -> str:
    path = PurePosixPath(str(value))
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe remote destination root: {value!r}")
    return str(path)


def _repo_identity(repo: Path, required_commit: str) -> None:
    if git(repo, "rev-parse", "HEAD") != required_commit:
        raise RuntimeError("local push control checkout moved")
    if git(repo, "status", "--porcelain"):
        raise RuntimeError("local push control checkout is dirty")


def local_export_rows(contract: dict[str, Any]) -> list[dict[str, Any]]:
    export_root = Path(contract["export_root"])
    source_run_root = Path(contract["source_run_root"])
    lane_id = contract["lane_id"]
    set_path = export_root / lane_id / "EXPORT_SET.json"
    payload = read_json(set_path)
    if (
        payload.get("schema") != relay.DCLGAN_EXPORT_SET_SCHEMA
        or payload.get("status") != "COMPLETE_SOURCE_BOUND_EXPORT_SET"
        or payload.get("lane_id") != lane_id
        or payload.get("source_host_label") != contract["source_host_label"]
        or payload.get("epochs") != list(EPOCHS)
        or payload.get("performance_values_read") is not False
        or payload.get("checkpoint_copy_performed") is not False
        or payload.get("paired_metric_control") is not False
        or payload.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("invalid local DCLGAN export set")
    source_rows = payload.get("exports")
    if not isinstance(source_rows, list) or len(source_rows) != len(EPOCHS):
        raise RuntimeError("incomplete local DCLGAN export set")
    by_epoch: dict[int, dict[str, Any]] = {}
    for source_row in source_rows:
        epoch = int(source_row.get("epoch", -1))
        if epoch in by_epoch or epoch not in EPOCHS:
            raise RuntimeError("duplicate or unexpected local DCLGAN epoch")
        receipt_path = inside_local(
            source_row.get("receipt", ""), export_root, "export receipt",
        )
        receipt_bytes = receipt_path.read_bytes()
        receipt_hash = bytes_sha256(receipt_bytes)
        if receipt_hash != source_row.get("receipt_sha256"):
            raise RuntimeError(f"local DCLGAN receipt hash differs at e{epoch}")
        receipt = json.loads(receipt_bytes.decode("utf-8"))
        relay.validate_export_receipt(
            receipt, lane_id=lane_id, epoch=epoch,
            source_host_label=contract["source_host_label"],
        )
        if (
            receipt.get("training_git_commit")
            != contract["required_training_git_commit"]
            or receipt.get("training_protocol_fingerprint")
            != contract["required_training_protocol_fingerprint"]
            or receipt.get("manifest_sha256") != contract["required_manifest_sha256"]
        ):
            raise RuntimeError(f"local DCLGAN receipt identity differs at e{epoch}")
        checkpoint = inside_local(
            receipt["source_checkpoint"], source_run_root, "source checkpoint",
        )
        sidecar = inside_local(
            receipt["source_sidecar"], source_run_root, "source sidecar",
        )
        if (
            not checkpoint.is_file()
            or file_sha256(checkpoint) != receipt["checkpoint_sha256"]
            or not sidecar.is_file()
            or file_sha256(sidecar) != receipt["sidecar_sha256"]
        ):
            raise RuntimeError(f"local DCLGAN payload hash differs at e{epoch}")
        by_epoch[epoch] = {
            "epoch": epoch,
            "receipt": receipt,
            "receipt_path": receipt_path,
            "receipt_sha256": receipt_hash,
            "checkpoint": checkpoint,
            "sidecar": sidecar,
        }
    if tuple(sorted(by_epoch)) != EPOCHS:
        raise RuntimeError("local DCLGAN epoch set differs")
    return [by_epoch[epoch] for epoch in EPOCHS]


def _connect(contract: dict[str, Any]):
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError("local export push requires paramiko") from error
    password = os.environ.get(contract["password_env"])
    if not password:
        raise RuntimeError(
            f"missing local push password environment: {contract['password_env']}"
        )
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(
        relay.PinnedHostKeyPolicy(contract["expected_host_key_sha256"])
    )
    try:
        client.connect(
            hostname=contract["destination_host"],
            port=int(contract["destination_port"]),
            username=contract["destination_user"], password=password,
            look_for_keys=False, allow_agent=False, timeout=30,
            banner_timeout=30, auth_timeout=30,
        )
    except (paramiko.AuthenticationException, paramiko.SSHException,
            EOFError, OSError, socket.error) as error:
        client.close()
        raise relay.TransientRelayNetwork(
            "local push SSH connection unavailable"
        ) from error
    return client


def destination_identity(client, contract: dict[str, Any]) -> dict[str, Any]:
    command = (
        "printf '%s\\n' \"$(hostname)\"; "
        "nvidia-smi --query-gpu=uuid --format=csv,noheader"
    )
    _, stdout, stderr = client.exec_command(command, timeout=60)
    lines = [
        line.strip() for line in stdout.read().decode("utf-8", "replace").splitlines()
        if line.strip()
    ]
    error = stderr.read().decode("utf-8", "replace").strip()
    status = stdout.channel.recv_exit_status()
    if status != 0 or error or len(lines) != 2:
        raise relay.TransientRelayNetwork("cannot read destination identity")
    identity = {"hostname": lines[0], "gpu_uuid": lines[1]}
    if (
        identity["hostname"] != contract["required_destination_hostname"]
        or identity["gpu_uuid"] != contract["required_destination_gpu_uuid"]
    ):
        raise RuntimeError(
            "local push destination physical identity differs from frozen contract"
        )
    return identity


def destination_preflight(contract: dict[str, Any]) -> dict[str, Any]:
    client = _connect(contract)
    try:
        identity = destination_identity(client, contract)
        try:
            sftp = client.open_sftp()
            with sftp:
                sftp.stat(contract["destination_root"])
                stat = sftp.statvfs(contract["destination_root"])
        except (EOFError, OSError, socket.error) as error:
            raise relay.TransientRelayNetwork(
                "cannot inspect local push destination"
            ) from error
        return {
            **identity,
            "destination_root": contract["destination_root"],
            "free_gib": int(stat.f_bavail) * int(stat.f_frsize) / 1024 ** 3,
            "status": "PASS_PINNED_DESTINATION_IDENTITY_AND_CAPACITY",
        }
    finally:
        client.close()


def _ensure_remote_directory(sftp, path: str) -> None:
    target = PurePosixPath(path)
    current = PurePosixPath("/")
    for part in target.parts[1:]:
        current = current / part
        try:
            sftp.stat(str(current))
        except OSError as error:
            if getattr(error, "errno", None) != 2:
                raise
            sftp.mkdir(str(current))


def _remote_sha256(client, path: str) -> str:
    command = "sha256sum -- " + shlex.quote(path)
    _, stdout, stderr = client.exec_command(command, timeout=3600)
    result = stdout.read().decode("utf-8", "replace").strip()
    error = stderr.read().decode("utf-8", "replace").strip()
    status = stdout.channel.recv_exit_status()
    if status != 0 or error or not result:
        raise relay.TransientRelayNetwork(f"remote hash failed for {path}")
    return result.split()[0]


def _remove_part(sftp, path: str) -> None:
    try:
        sftp.remove(path)
    except OSError as error:
        if getattr(error, "errno", None) != 2:
            raise


def upload_file(client, sftp, source: Path, destination: str, expected: str) -> None:
    destination = str(PurePosixPath(destination))
    _ensure_remote_directory(sftp, str(PurePosixPath(destination).parent))
    try:
        sftp.stat(destination)
    except OSError as error:
        if getattr(error, "errno", None) != 2:
            raise relay.TransientRelayNetwork(
                f"remote stat failed for {destination}"
            ) from error
    else:
        if _remote_sha256(client, destination) != expected:
            raise RuntimeError(f"existing remote import differs: {destination}")
        return
    stat = sftp.statvfs(str(PurePosixPath(destination).parent))
    available = int(stat.f_bavail) * int(stat.f_frsize)
    required = source.stat().st_size + 2 * 1024 ** 3
    if available < required:
        raise RuntimeError(
            f"insufficient remote capacity for {destination}: "
            f"required {required}, available {available}"
        )
    temporary = destination + f".{os.getpid()}.part"
    _remove_part(sftp, temporary)
    try:
        with source.open("rb") as local, sftp.open(temporary, "wb") as remote:
            while True:
                block = local.read(1024 * 1024)
                if not block:
                    break
                remote.write(block)
        if _remote_sha256(client, temporary) != expected:
            raise RuntimeError(f"uploaded DCLGAN file hash differs: {destination}")
        try:
            sftp.rename(temporary, destination)
        except OSError:
            # A prior recoverable attempt may have won the publication race.
            if _remote_sha256(client, destination) != expected:
                raise
            _remove_part(sftp, temporary)
        if _remote_sha256(client, destination) != expected:
            raise RuntimeError(f"published DCLGAN file hash differs: {destination}")
    finally:
        _remove_part(sftp, temporary)


def upload_bytes(client, sftp, value: bytes, destination: str) -> str:
    digest = bytes_sha256(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="final_unsb_push_", suffix=".tmp",
    )
    os.close(descriptor)
    temporary_local = Path(temporary_name)
    try:
        temporary_local.write_bytes(value)
        upload_file(client, sftp, temporary_local, destination, digest)
    finally:
        if temporary_local.exists():
            temporary_local.unlink()
    return digest


def publish(contract: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    client = _connect(contract)
    try:
        try:
            import paramiko
            destination_identity(client, contract)
            try:
                sftp = client.open_sftp()
            except Exception as error:
                raise relay.TransientRelayNetwork(
                    "local push SFTP channel unavailable"
                ) from error
            with sftp:
                lane_root = (
                    PurePosixPath(contract["destination_root"]) / "sources"
                    / contract["source_host_label"] / contract["lane_id"]
                )
                imported = []
                for row in rows:
                    epoch = row["epoch"]
                    checkpoint_remote = str(lane_root / f"e{epoch:03d}.pt")
                    sidecar_remote = str(lane_root / f"e{epoch:03d}.pt.json")
                    receipt_remote = str(lane_root / f"e{epoch:03d}.export.json")
                    receipt = row["receipt"]
                    upload_file(
                        client, sftp, row["checkpoint"], checkpoint_remote,
                        receipt["checkpoint_sha256"],
                    )
                    upload_file(
                        client, sftp, row["sidecar"], sidecar_remote,
                        receipt["sidecar_sha256"],
                    )
                    receipt_bytes = row["receipt_path"].read_bytes()
                    upload_bytes(client, sftp, receipt_bytes, receipt_remote)
                    imported.append({
                        "epoch": epoch,
                        "export_receipt": receipt_remote,
                        "export_receipt_sha256": row["receipt_sha256"],
                        "checkpoint": checkpoint_remote,
                        "checkpoint_sha256": receipt["checkpoint_sha256"],
                        "sidecar": sidecar_remote,
                        "sidecar_sha256": receipt["sidecar_sha256"],
                        "scientific_state_sha256": receipt["scientific_state_sha256"],
                    })
                lane = {
                    "schema": IMPORT_LANE_SCHEMA,
                    "status": "COMPLETE_VERIFIED_IMPORTED_LANE",
                    "source_host_label": contract["source_host_label"],
                    "lane_id": contract["lane_id"],
                    "epochs": list(EPOCHS),
                    "source_export_set_sha256": file_sha256(
                        Path(contract["export_root"]) / contract["lane_id"]
                        / "EXPORT_SET.json"
                    ),
                    "imports": imported,
                    "checkpoint_copy_performed": True,
                    "source_checkpoint_mutation": False,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                }
                lane_bytes = (
                    json.dumps(lane, ensure_ascii=False, indent=2) + "\n"
                ).encode()
                lane_path = str(lane_root / "IMPORT_LANE.json")
                lane_sha = upload_bytes(client, sftp, lane_bytes, lane_path)
                import_set = {
                    "schema": IMPORT_SET_SCHEMA,
                    "status": "COMPLETE_VERIFIED_IMPORT_SET",
                    "relay_id": contract["relay_id"],
                    "source_host_label": contract["source_host_label"],
                    "lanes": [contract["lane_id"]],
                    "epochs": list(EPOCHS),
                    "lane_imports": {
                        contract["lane_id"]: {
                            "receipt": lane_path,
                            "receipt_sha256": lane_sha,
                        }
                    },
                    "checkpoint_copy_performed": True,
                    "source_checkpoint_mutation": False,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                }
                result_path = str(
                    PurePosixPath(contract["destination_root"]) / "operations"
                    / f"IMPORT_SET_{contract['relay_id']}.json"
                )
                result_bytes = (
                    json.dumps(import_set, ensure_ascii=False, indent=2) + "\n"
                ).encode()
                result_sha = upload_bytes(client, sftp, result_bytes, result_path)
                return {
                    "remote_import_lane": lane_path,
                    "remote_import_lane_sha256": lane_sha,
                    "remote_import_set": result_path,
                    "remote_import_set_sha256": result_sha,
                }
        except relay.TransientRelayNetwork:
            raise
        except (paramiko.SSHException, EOFError, OSError, socket.error) as error:
            raise relay.TransientRelayNetwork(
                "local push transport interrupted"
            ) from error
    finally:
        client.close()


def make_contract(args: argparse.Namespace) -> dict[str, Any]:
    for name in ("repo", "source_run_root", "export_root", "state_root"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if not _SAFE_ID.fullmatch(args.source_host_label):
        raise ValueError("source host label must be a safe identifier")
    if not _SAFE_ID.fullmatch(args.relay_id):
        raise ValueError("relay id must be a safe identifier")
    if args.lane_id != "dclgan":
        raise ValueError("local export push currently accepts only frozen DCLGAN")
    if not args.password_env.startswith("FINAL_UNSB_"):
        raise ValueError("password environment must use FINAL_UNSB_ prefix")
    if not args.expected_host_key_sha256.startswith("SHA256:"):
        raise ValueError("local export push requires a pinned SSH host key")
    if not 30 <= args.poll_seconds <= 600 or args.timeout_hours < 24:
        raise ValueError("unsafe local export push waiting policy")
    inside_local(args.export_root, args.source_run_root, "export root")
    _repo_identity(args.repo, args.required_control_git_commit)
    script = Path(__file__).resolve()
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(args.repo),
        "control_git_commit": args.required_control_git_commit,
        "control_script": str(script),
        "control_script_sha256": file_sha256(script),
        "source_run_root": str(args.source_run_root),
        "export_root": str(args.export_root),
        "state_root": str(args.state_root),
        "source_host_label": args.source_host_label,
        "lane_id": args.lane_id,
        "relay_id": args.relay_id,
        "required_training_git_commit": args.required_training_git_commit,
        "required_training_protocol_fingerprint": (
            args.required_training_protocol_fingerprint
        ),
        "required_manifest_sha256": args.required_manifest_sha256,
        "destination_host": args.destination_host,
        "destination_port": args.destination_port,
        "destination_user": args.destination_user,
        "expected_host_key_sha256": args.expected_host_key_sha256,
        "required_destination_hostname": args.required_destination_hostname,
        "required_destination_gpu_uuid": args.required_destination_gpu_uuid,
        "destination_root": remote_root(args.destination_root),
        "password_env": args.password_env,
        "poll_seconds": args.poll_seconds,
        "timeout_hours": args.timeout_hours,
        "password_persisted": False,
        "performance_values_available_to_scheduler": False,
        "source_checkpoint_mutation": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def verify_contract(contract: dict[str, Any]) -> None:
    _repo_identity(
        Path(contract["control_repo"]), contract["control_git_commit"],
    )
    if file_sha256(Path(contract["control_script"])) != contract["control_script_sha256"]:
        raise RuntimeError("local export push script changed")


def acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    handle.seek(0)
    if handle.read(1) == b"":
        handle.write(b"0")
        handle.flush()
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def state(contract: dict[str, Any], *, status: str, **extra: Any) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "relay_id": contract["relay_id"],
        "source_host_label": contract["source_host_label"],
        "lane_id": contract["lane_id"],
        "password_persisted": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract = make_contract(args)
    root = Path(contract["state_root"])
    contract_path = root / "LOCAL_EXPORT_PUSH_CONTRACT.json"
    state_path = root / "LOCAL_EXPORT_PUSH_STATE.json"
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("local export push contract changed")
    else:
        atomic_json(contract_path, contract)
    lock = acquire_lock(root / "LOCAL_EXPORT_PUSH.lock")
    started = time.time()
    try:
        preflight: dict[str, Any] | None = None
        while preflight is None:
            verify_contract(contract)
            try:
                preflight = destination_preflight(contract)
            except relay.TransientRelayNetwork as error:
                atomic_json(state_path, state(
                    contract,
                    status="WAITING_FOR_PINNED_DESTINATION_CONNECTIVITY",
                    contract=str(contract_path),
                    contract_sha256=file_sha256(contract_path),
                    last_error_type=type(error).__name__,
                    elapsed_seconds=time.time() - started,
                ))
                if time.time() - started > contract["timeout_hours"] * 3600:
                    raise TimeoutError("local export push preflight timed out")
                time.sleep(contract["poll_seconds"])
        while True:
            verify_contract(contract)
            try:
                rows = local_export_rows(contract)
                before = {
                    str(row[key]): file_sha256(row[key])
                    for row in rows for key in ("receipt_path", "checkpoint", "sidecar")
                }
                result = publish(contract, rows)
                after = {
                    str(row[key]): file_sha256(row[key])
                    for row in rows for key in ("receipt_path", "checkpoint", "sidecar")
                }
                if before != after:
                    raise RuntimeError("local DCLGAN source changed during push")
            except (LocalExportNotReady, relay.TransientRelayNetwork) as error:
                atomic_json(state_path, state(
                    contract,
                    status="WAITING_FOR_COMPLETE_LOCAL_EXPORT_OR_TRANSIENT_NETWORK",
                    contract=str(contract_path),
                    contract_sha256=file_sha256(contract_path),
                    last_error_type=type(error).__name__,
                    elapsed_seconds=time.time() - started,
                    destination_preflight=preflight,
                ))
            else:
                final = state(
                    contract, status="COMPLETE_VERIFIED_REMOTE_IMPORT",
                    contract=str(contract_path),
                    contract_sha256=file_sha256(contract_path),
                    destination_preflight=preflight, **result,
                )
                atomic_json(state_path, final)
                return final
            if time.time() - started > contract["timeout_hours"] * 3600:
                raise TimeoutError("local export push exceeded its frozen timeout")
            time.sleep(contract["poll_seconds"])
    except Exception as error:
        atomic_json(state_path, state(
            contract, status="FAIL_CLOSED_REQUIRES_CODEX_AUDIT",
            contract=str(contract_path), contract_sha256=file_sha256(contract_path),
            error_type=type(error).__name__, error_message=str(error),
        ))
        raise
    finally:
        lock.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--source-run-root", type=Path, required=True)
    value.add_argument("--export-root", type=Path, required=True)
    value.add_argument("--state-root", type=Path, required=True)
    value.add_argument("--source-host-label", required=True)
    value.add_argument("--lane-id", default="dclgan")
    value.add_argument("--relay-id", required=True)
    value.add_argument("--required-training-git-commit", required=True)
    value.add_argument("--required-training-protocol-fingerprint", required=True)
    value.add_argument("--required-manifest-sha256", required=True)
    value.add_argument("--destination-host", required=True)
    value.add_argument("--destination-port", type=int, required=True)
    value.add_argument("--destination-user", required=True)
    value.add_argument("--expected-host-key-sha256", required=True)
    value.add_argument("--required-destination-hostname", required=True)
    value.add_argument("--required-destination-gpu-uuid", required=True)
    value.add_argument("--destination-root", required=True)
    value.add_argument("--password-env", required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=480)
    return value


def main() -> int:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
