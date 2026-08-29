"""Durably wait for and import a verified independent HNEK into a canonical run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from operations.local_route1_import_handoff import (
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
    LANE,
    atomic_json,
    file_sha256,
    import_lane,
    lock,
    paths,
    read_json,
    sidecar_epoch,
    validate_canonical,
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def source_ready(source_root: Path) -> bool:
    contract_path = source_root / "operations" / "INDEPENDENT_PROBE_CONTRACT.json"
    result_path = source_root / "operations" / "INDEPENDENT_PROBE_RESULT.json"
    if not contract_path.is_file() or not result_path.is_file():
        return False
    contract = read_json(contract_path)
    result = read_json(result_path)
    return (
        contract.get("status") == "COMPLETE_MATCHED_BASELINE_VERIFIED"
        and result.get("status") == "COMPLETE_MATCHED_BASELINE_VERIFIED"
        and sidecar_epoch(source_root, LANE) == 200
    )


def canonical_import_state(canonical_root: Path) -> str:
    locations = paths(canonical_root)
    if sidecar_epoch(canonical_root, LANE) == 200:
        if locations["receipt"].is_file() and read_json(locations["receipt"]).get("status") == "IMPORT_VERIFIED":
            return "ALREADY_IMPORTED"
        return "CANONICAL_HNEK_EXISTS_WITHOUT_IMPORT_RECEIPT"
    _, state = validate_canonical(canonical_root)
    if state.get("lane") == LANE:
        return "CANONICAL_HNEK_ALREADY_SCHEDULED"
    return "WAITING"


def record_path(canonical_root: Path) -> Path:
    return canonical_root / "operations" / "INDEPENDENT_HNEK_AUTOIMPORT_WATCH.json"


def write_record(
    canonical_root: Path, source_root: Path, status: str, **fields: Any,
) -> None:
    script = Path(__file__).resolve()
    payload = {
        "schema": "final-unsb-route1-independent-hnek-autoimport-watch-v1",
        "updated": now(),
        "status": status,
        "watcher_pid": os.getpid(),
        "canonical_root": str(canonical_root.resolve()),
        "source_root": str(source_root.resolve()),
        "watcher_script": str(script),
        "watcher_script_sha256": file_sha256(script),
        "training_git_commit": EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "training_update_changed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        **fields,
    }
    atomic_json(record_path(canonical_root), payload)


def wait_and_import(
    canonical_root: Path, source_root: Path, *, poll_seconds: int,
    timeout_seconds: int,
) -> int:
    canonical_root = canonical_root.resolve()
    source_root = source_root.resolve()
    watch_lock = canonical_root / "operations" / "INDEPENDENT_HNEK_AUTOIMPORT_WATCH.lock"
    with lock(watch_lock):
        validate_canonical(canonical_root)
        state = canonical_import_state(canonical_root)
        if state == "ALREADY_IMPORTED":
            write_record(canonical_root, source_root, "ALREADY_IMPORTED")
            return 0
        if state != "WAITING":
            raise RuntimeError(state)
        write_record(
            canonical_root, source_root, "WAITING_FOR_SOURCE_AND_MATCHED_PLAIN_E200",
            poll_seconds=poll_seconds, timeout_seconds=timeout_seconds,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            state = canonical_import_state(canonical_root)
            if state == "ALREADY_IMPORTED":
                write_record(canonical_root, source_root, "ALREADY_IMPORTED")
                return 0
            if state != "WAITING":
                raise RuntimeError(state)
            plain_epoch = sidecar_epoch(canonical_root, "plain")
            ready = source_ready(source_root)
            write_record(
                canonical_root, source_root, "WAITING_FOR_SOURCE_AND_MATCHED_PLAIN_E200",
                canonical_plain_data_epoch=plain_epoch,
                independent_hnek_ready=ready,
                seconds_remaining=max(0, int(deadline - time.monotonic())),
            )
            if plain_epoch == 200 and ready:
                write_record(canonical_root, source_root, "IMPORT_START")
                import_lane(canonical_root, source_root, require_paused=False)
                if canonical_import_state(canonical_root) != "ALREADY_IMPORTED":
                    raise RuntimeError("import returned without a verified canonical receipt")
                receipt = read_json(paths(canonical_root)["receipt"])
                write_record(
                    canonical_root, source_root, "IMPORT_COMPLETE",
                    imported_tree_sha256=receipt["tree_sha256"],
                    imported_files=receipt["file_count"],
                    canonical_plain_data_epoch=200,
                )
                return 0
            time.sleep(poll_seconds)
        raise TimeoutError("independent HNEK auto-import timed out")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--canonical-root", type=Path, required=True)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--poll-seconds", type=int, default=30)
    value.add_argument("--timeout-seconds", type=int, default=43_200)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.poll_seconds < 5 or args.timeout_seconds <= 0:
        raise SystemExit("poll/timeout values are invalid")
    canonical = args.canonical_root.resolve()
    source = args.source_root.resolve()
    try:
        return wait_and_import(
            canonical, source, poll_seconds=int(args.poll_seconds),
            timeout_seconds=int(args.timeout_seconds),
        )
    except Exception as exc:
        write_record(
            canonical, source, "FAILED",
            exception_type=type(exc).__name__, exception=str(exc),
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
