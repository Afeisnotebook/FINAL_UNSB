"""Durably retrieve the authoritative multi-algorithm route-1 supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path, PurePosixPath
from typing import Any

import paramiko

from operations import local_route1_candidate_executor as support
from research.local_route1.related_multi_algorithm_final_delivery import (
    FINAL_SUBDIR,
    HJPCNR_RECEIPT,
    POINTER,
    POINTER_SCHEMA,
    PUBLISHED_FILES,
    RELATED_4090,
    RELATED_5090,
    RELATED_COMBINED,
)


SCHEMA = "final-unsb-route1-related-multi-algorithm-final-relay-contract-v1"
PASSWORD_ENV = "UNSB_4090_PASSWORD"
EXTRA_FILES = (RELATED_4090, RELATED_5090, RELATED_COMBINED, HJPCNR_RECEIPT)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_pointer(value: dict[str, Any]) -> dict[str, Any]:
    if (
        value.get("schema") != POINTER_SCHEMA
        or value.get("status") != "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_COMPLETE"
        or value.get("action_priority_is_not_scientific_exclusivity") is not True
        or value.get("algorithm_discovery_collapsed_to_single_candidate") is not False
    ):
        raise RuntimeError("related final pointer is not terminal")
    fixed = {
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise RuntimeError(f"related final pointer changed: {key}")
    hashes = value.get("final_file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(PUBLISHED_FILES):
        raise RuntimeError("related final published file set changed")
    return value


def validate_local_delivery(destination: Path) -> dict[str, Any]:
    destination = Path(destination).resolve()
    pointer = validate_pointer(_read_json(destination / POINTER))
    for name, expected in pointer["final_file_sha256"].items():
        if support.file_sha256(destination / name) != expected:
            raise RuntimeError(f"relayed related final file changed: {name}")
    extras = {
        RELATED_4090: pointer["related_4090_host_adjudication_sha256"],
        RELATED_5090: pointer["related_5090_host_adjudication_sha256"],
        RELATED_COMBINED: pointer["related_multi_host_adjudication_sha256"],
        HJPCNR_RECEIPT: pointer["hjpcnr_gain_source_receipt_sha256"],
    }
    for name, expected in extras.items():
        if support.file_sha256(destination / name) != expected:
            raise RuntimeError(f"relayed related input changed: {name}")
    algorithm_set = _read_json(destination / "ALGORITHM_SET.json")
    action = _read_json(destination / "ACTION_PRIORITY.json")
    if (
        algorithm_set.get("action_priority_candidate_id")
        != pointer.get("action_priority_candidate_id")
        or action.get("candidate_id") != pointer.get("action_priority_candidate_id")
        or algorithm_set.get("algorithm_discovery_collapsed_to_single_candidate")
        is not False
        or algorithm_set.get("cross_host_deltas_merged") is not False
        or algorithm_set.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("relayed related algorithm-set identity changed")
    return pointer


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "source": {
            "host": str(args.source_host),
            "port": int(args.source_port),
            "user": str(args.source_user),
            "run_root": str(PurePosixPath(args.source_run_root)),
        },
        "destination": str(args.destination.resolve()),
        "state_path": str(args.state.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "maximum_consecutive_failures": 3,
        "destination_overwrite_allowed": False,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "credentials_persisted": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("related final relay contract schema mismatch")
    source = contract.get("source")
    if not isinstance(source, dict):
        raise RuntimeError("related final relay lacks source")
    for key in ("host", "user", "run_root"):
        if not isinstance(source.get(key), str) or not source[key]:
            raise RuntimeError(f"related final relay source lacks {key}")
    if not 1 <= int(source.get("port", 0)) <= 65535:
        raise RuntimeError("related final relay source port is invalid")
    if any("password" in str(key).lower() for key in source):
        raise RuntimeError("related final relay may not persist a password")
    fixed = {
        "maximum_consecutive_failures": 3,
        "destination_overwrite_allowed": False,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "credentials_persisted": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"related final relay changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 30:
        raise RuntimeError("related final relay polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("related final relay timeout is too short")


def _remote_sha256(sftp: paramiko.SFTPClient, path: str) -> str:
    digest = hashlib.sha256()
    with sftp.open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class RelatedMultiAlgorithmFinalRelay:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.destination = Path(self.contract["destination"])
        self.state_path = Path(self.contract["state_path"])
        self.started = time.time()
        self.password = os.environ.pop(PASSWORD_ENV, None)
        if not self.password:
            raise RuntimeError(f"missing related final relay password: {PASSWORD_ENV}")

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-related-multi-algorithm-final-relay-state-v1",
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

    def _client(self) -> paramiko.SSHClient:
        source = self.contract["source"]
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(
            source["host"], port=int(source["port"]), username=source["user"],
            password=self.password, timeout=20, banner_timeout=20, auth_timeout=20,
            look_for_keys=False, allow_agent=False,
        )
        return client

    def _remote(self, *parts: str) -> str:
        root = PurePosixPath(self.contract["source"]["run_root"])
        return str(root.joinpath(*parts))

    def pointer_ready(self) -> bool:
        client = self._client()
        try:
            with client.open_sftp() as sftp:
                try:
                    sftp.stat(self._remote("operations", POINTER))
                except FileNotFoundError:
                    return False
        finally:
            client.close()
        return True

    def retrieve(self) -> dict[str, Any]:
        staging = self.destination / "staging"
        staging.mkdir(parents=True, exist_ok=True)
        client = self._client()
        try:
            with client.open_sftp() as sftp:
                pointer_path = staging / POINTER
                sftp.get(self._remote("operations", POINTER), str(pointer_path))
                pointer = validate_pointer(_read_json(pointer_path))
                for name, expected in pointer["final_file_sha256"].items():
                    local = staging / name
                    remote = self._remote(*FINAL_SUBDIR.parts, name)
                    sftp.get(remote, str(local))
                    if support.file_sha256(local) != expected or _remote_sha256(
                        sftp, remote,
                    ) != expected:
                        raise RuntimeError(f"related final relay hash mismatch: {name}")
                extras = {
                    RELATED_4090: pointer["related_4090_host_adjudication_sha256"],
                    RELATED_5090: pointer["related_5090_host_adjudication_sha256"],
                    RELATED_COMBINED: pointer["related_multi_host_adjudication_sha256"],
                    HJPCNR_RECEIPT: pointer["hjpcnr_gain_source_receipt_sha256"],
                }
                for name, expected in extras.items():
                    local = staging / name
                    remote = self._remote("operations", name)
                    sftp.get(remote, str(local))
                    if support.file_sha256(local) != expected or _remote_sha256(
                        sftp, remote,
                    ) != expected:
                        raise RuntimeError(f"related final input hash mismatch: {name}")
        finally:
            client.close()
        self.destination.mkdir(parents=True, exist_ok=True)
        names = [POINTER, *PUBLISHED_FILES, *EXTRA_FILES]
        for name in names:
            source = staging / name
            target = self.destination / name
            if target.is_file():
                if target.read_bytes() != source.read_bytes():
                    raise RuntimeError(f"related final relay destination differs: {name}")
                source.unlink()
            else:
                os.replace(source, target)
        pointer = validate_local_delivery(self.destination)
        support.atomic_json(self.destination / "RELAY_MANIFEST.json", {
            "schema": "final-unsb-route1-related-multi-algorithm-final-relay-manifest-v1",
            "status": "COMPLETE_EXACT_RELATED_MULTI_ALGORITHM_DELIVERY_RETRIEVED",
            "action_priority_candidate_id": pointer["action_priority_candidate_id"],
            "algorithm_set_status": pointer["algorithm_set_status"],
            "file_sha256": {
                name: support.file_sha256(self.destination / name) for name in names
            },
            "credentials_persisted": False,
            "checkpoint_transfer": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        return pointer

    def run(self) -> int:
        failures = 0
        while True:
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("related final result relay timed out")
            try:
                if not self.pointer_ready():
                    failures = 0
                    self.state("WAITING_FOR_REMOTE4090_RELATED_FINAL_POINTER")
                    time.sleep(int(self.contract["poll_seconds"]))
                    continue
                pointer = self.retrieve()
                self.state(
                    "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_RETRIEVED",
                    action_priority_candidate_id=pointer[
                        "action_priority_candidate_id"
                    ],
                    algorithm_set_status=pointer["algorithm_set_status"],
                    strict_viable_candidate_count=pointer[
                        "strict_viable_candidate_count"
                    ],
                    destination_manifest_sha256=support.file_sha256(
                        self.destination / "RELAY_MANIFEST.json"
                    ),
                )
                return 0
            except Exception as error:
                failures += 1
                self.state(
                    "TRANSIENT_RELATED_FINAL_RELAY_FAILURE",
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
    value.add_argument("--source-run-root")
    value.add_argument("--destination", type=Path)
    value.add_argument("--state", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "source_host", "source_user", "source_run_root", "destination", "state",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    try:
        return RelatedMultiAlgorithmFinalRelay(args.contract).run()
    except Exception as error:
        contract = _read_json(args.contract)
        support.atomic_json(Path(contract["state_path"]), {
            "schema": "final-unsb-route1-related-multi-algorithm-final-relay-fatal-v1",
            "updated": support.now(),
            "status": "FAILED",
            "error": repr(error),
            "traceback": traceback.format_exc(),
            "pid": os.getpid(),
            "credentials_persisted": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
