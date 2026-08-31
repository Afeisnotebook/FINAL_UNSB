"""Atomically relay the portable complete 5090 frontier to the 4090 host."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from operations.local_route1_extended_frontier_relay import (
    ExtendedFrontierRelay,
    PASSWORD_ENV,
    _client,
)
from research.local_route1.complete_5090_frontier import (
    validate_portable_complete_5090,
)


SCHEMA = "final-unsb-route1-complete-5090-frontier-relay-contract-v1"


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
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("complete 5090 relay contract schema mismatch")
    for role in ("source", "destination"):
        endpoint = contract.get(role)
        if not isinstance(endpoint, dict):
            raise RuntimeError(f"complete 5090 relay lacks {role} endpoint")
        for key in ("host", "user", "path"):
            if not isinstance(endpoint.get(key), str) or not endpoint[key]:
                raise RuntimeError(f"complete 5090 relay {role} lacks {key}")
        if not 1 <= int(endpoint.get("port", 0)) <= 65535:
            raise RuntimeError(f"complete 5090 relay {role} port is invalid")
        if any("password" in str(key).lower() for key in endpoint):
            raise RuntimeError("complete 5090 relay must not persist a password")
    fixed = {
        "maximum_consecutive_failures": 3,
        "complete_source_e200_only": True,
        "destination_overwrite_allowed": False,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"complete 5090 relay changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 30:
        raise RuntimeError("complete 5090 relay polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("complete 5090 relay timeout is too short")
    if Path(contract["local_spool"]).resolve() == Path(
        contract["state_path"]
    ).resolve():
        raise RuntimeError("complete 5090 relay spool and state must differ")


class Complete5090FrontierRelay(ExtendedFrontierRelay):
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
                raise RuntimeError(f"missing complete 5090 relay password: {variable}")
            self.passwords[role] = value

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-complete-5090-frontier-relay-state-v1",
            "updated": support.now(),
            "status": status,
            "pid": os.getpid(),
            "contract_sha256": support.file_sha256(self.contract_path),
            "credentials_persisted": False,
            "checkpoint_transfer": False,
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
        validate_portable_complete_5090(_read_json(self.spool))
        return True

    def run(self) -> int:
        failures = 0
        while True:
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("complete 5090 frontier relay timed out")
            try:
                if not self.source_ready():
                    failures = 0
                    self.state("WAITING_FOR_PORTABLE_COMPLETE_5090_FRONTIER")
                    time.sleep(int(self.contract["poll_seconds"]))
                    continue
                source_sha = support.file_sha256(self.spool)
                self.state(
                    "VALIDATED_PORTABLE_COMPLETE_5090_FRONTIER",
                    source_sha256=source_sha,
                )
                destination_sha = self.upload_exact()
                self.state(
                    "COMPLETE_5090_FRONTIER_RELAYED_TO_4090",
                    source_sha256=source_sha,
                    destination_sha256=destination_sha,
                    hashes_equal=source_sha == destination_sha,
                )
                return 0
            except Exception as error:
                failures += 1
                self.state(
                    "TRANSIENT_COMPLETE_5090_FRONTIER_RELAY_FAILURE",
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
        return Complete5090FrontierRelay(args.contract).run()
    except Exception as error:
        support.atomic_json(
            state_path.with_name("COMPLETE_5090_FRONTIER_RELAY_FATAL.json"),
            {
                "schema": "final-unsb-route1-complete-5090-frontier-relay-fatal-v1",
                "updated": support.now(),
                "status": "FAILED",
                "error": repr(error),
                "traceback": traceback.format_exc(),
                "credentials_persisted": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
