"""Durably relay the complete repaired portfolio from 5090 to 4090.

Passwords are accepted only through process environment variables and removed
from ``os.environ`` immediately.  The contract, state, logs, and command line
contain no credentials.  The destination is created atomically and is never
overwritten with different bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import paramiko

from operations import local_route1_candidate_executor as support
from research.local_route1.repaired_replay_portfolio import (
    validate_portable_authority,
)


SCHEMA = "final-unsb-route1-repaired-portfolio-relay-contract-v1"
PASSWORD_ENV = {
    "source": "UNSB_5090_PASSWORD",
    "destination": "UNSB_4090_PASSWORD",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "source": {
            "host": str(args.source_host),
            "port": int(args.source_port),
            "user": str(args.source_user),
            "path": str(args.source_path),
        },
        "destination": {
            "host": str(args.destination_host),
            "port": int(args.destination_port),
            "user": str(args.destination_user),
            "path": str(args.destination_path),
        },
        "local_spool": str(args.local_spool.resolve()),
        "state_path": str(args.state.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "maximum_consecutive_failures": 3,
        "complete_source_e200_only": True,
        "destination_overwrite_allowed": False,
        "cross_host_checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("repaired portfolio relay contract schema mismatch")
    for role in ("source", "destination"):
        endpoint = contract.get(role)
        if not isinstance(endpoint, dict):
            raise RuntimeError(f"relay contract lacks {role} endpoint")
        for key in ("host", "user", "path"):
            if not isinstance(endpoint.get(key), str) or not endpoint[key]:
                raise RuntimeError(f"relay {role} endpoint lacks {key}")
        if not 1 <= int(endpoint.get("port", 0)) <= 65535:
            raise RuntimeError(f"relay {role} port is invalid")
        if any("password" in str(key).lower() for key in endpoint):
            raise RuntimeError("relay endpoint must never persist a password")
    fixed = {
        "maximum_consecutive_failures": 3,
        "complete_source_e200_only": True,
        "destination_overwrite_allowed": False,
        "cross_host_checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"relay contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 30:
        raise RuntimeError("relay polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("relay timeout is too short")
    spool = Path(str(contract.get("local_spool", ""))).resolve()
    state = Path(str(contract.get("state_path", ""))).resolve()
    if spool == state:
        raise RuntimeError("relay spool and state paths must differ")


def _client(endpoint: dict[str, Any], password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        endpoint["host"],
        port=int(endpoint["port"]),
        username=endpoint["user"],
        password=password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def _remote_sha256(sftp: paramiko.SFTPClient, path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class RepairedPortfolioRelay:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.state_path = Path(self.contract["state_path"])
        self.spool = Path(self.contract["local_spool"])
        self.started = time.time()
        self.passwords = {}
        for role, variable in PASSWORD_ENV.items():
            value = os.environ.pop(variable, None)
            if not value:
                raise RuntimeError(f"missing relay password environment: {variable}")
            self.passwords[role] = value

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-repaired-portfolio-relay-state-v1",
            "updated": support.now(),
            "status": status,
            "pid": os.getpid(),
            "contract_sha256": support.file_sha256(self.contract_path),
            "credentials_persisted": False,
            "cross_host_checkpoint_transfer": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def source_ready(self) -> bool:
        endpoint = self.contract["source"]
        client = _client(endpoint, self.passwords["source"])
        try:
            with client.open_sftp() as sftp:
                try:
                    sftp.stat(endpoint["path"])
                except FileNotFoundError:
                    return False
                self.spool.parent.mkdir(parents=True, exist_ok=True)
                temporary = self.spool.with_suffix(self.spool.suffix + ".tmp")
                sftp.get(endpoint["path"], str(temporary))
                os.replace(temporary, self.spool)
        finally:
            client.close()
        validate_portable_authority(_read_json(self.spool))
        return True

    def upload_exact(self) -> str:
        local_sha = support.file_sha256(self.spool)
        endpoint = self.contract["destination"]
        client = _client(endpoint, self.passwords["destination"])
        temporary = f"{endpoint['path']}.tmp.{os.getpid()}"
        try:
            with client.open_sftp() as sftp:
                try:
                    existing_sha = _remote_sha256(sftp, endpoint["path"])
                except FileNotFoundError:
                    existing_sha = None
                if existing_sha is not None:
                    if existing_sha != local_sha:
                        raise RuntimeError(
                            "relay destination exists with different complete authority"
                        )
                    return local_sha
                sftp.put(str(self.spool), temporary)
                if _remote_sha256(sftp, temporary) != local_sha:
                    raise RuntimeError("relay temporary upload hash mismatch")
                sftp.rename(temporary, endpoint["path"])
                if _remote_sha256(sftp, endpoint["path"]) != local_sha:
                    raise RuntimeError("relay destination hash mismatch after atomic rename")
        finally:
            client.close()
        return local_sha

    def run(self) -> int:
        failures = 0
        while True:
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("repaired portfolio relay timed out")
            try:
                if not self.source_ready():
                    failures = 0
                    self.state("WAITING_FOR_COMPLETE_5090_PORTABLE_AUTHORITY")
                    time.sleep(int(self.contract["poll_seconds"]))
                    continue
                source_sha = support.file_sha256(self.spool)
                self.state(
                    "VALIDATED_COMPLETE_5090_AUTHORITY",
                    source_sha256=source_sha,
                )
                destination_sha = self.upload_exact()
                self.state(
                    "COMPLETE_IDENTICAL_AUTHORITY_RELAYED_TO_4090",
                    source_sha256=source_sha,
                    destination_sha256=destination_sha,
                    hashes_equal=source_sha == destination_sha,
                )
                return 0
            except Exception as error:
                failures += 1
                self.state(
                    "TRANSIENT_RELAY_FAILURE",
                    consecutive_failures=failures,
                    error=repr(error),
                )
                if failures >= int(self.contract["maximum_consecutive_failures"]):
                    raise
                time.sleep(int(self.contract["poll_seconds"]))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--source-host")
    value.add_argument("--source-port", type=int, default=22)
    value.add_argument("--source-user")
    value.add_argument("--source-path")
    value.add_argument("--destination-host")
    value.add_argument("--destination-port", type=int, default=22)
    value.add_argument("--destination-user")
    value.add_argument("--destination-path")
    value.add_argument("--local-spool", type=Path)
    value.add_argument("--state", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "source_host", "source_user", "source_path", "destination_host",
            "destination_user", "destination_path", "local_spool", "state",
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
    state_path = Path(contract["state_path"])
    try:
        return RepairedPortfolioRelay(args.contract).run()
    except Exception as error:
        support.atomic_json(state_path.with_name(
            "REPAIRED_PORTFOLIO_RELAY_FATAL.json"
        ), {
            "schema": "final-unsb-route1-repaired-portfolio-relay-fatal-v1",
            "updated": support.now(),
            "status": "FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "credentials_persisted": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

