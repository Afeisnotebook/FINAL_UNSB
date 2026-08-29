"""Durably complete an already-armed independent-HNEK handoff.

The existing handoff coordinator owns the scientifically sensitive pause gate:
it observes the final HJ child, stops only the canonical executor parent, and
waits until HJ has written a complete e200 state.  This companion process does
not weaken that gate.  It waits for the coordinator's explicit
``READY_FOR_INDEPENDENT_HNEK_IMPORT`` record, waits for a fully verified source
tree, calls the existing no-overwrite importer, and finally resumes the same
executor PID.  A restart after import but before resume is handled explicitly.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Imported by tests/package users.
    from operations import local_route1_import_handoff as handoff
except ModuleNotFoundError:  # Executed directly beside the coordinator.
    import local_route1_import_handoff as handoff  # type: ignore[no-redef]


WATCH_SCHEMA = "final-unsb-route1-hnek-handoff-autocomplete-v1"
WAITING_RECORD_STATES = {
    None,
    "ARMED",
    "EXECUTOR_PAUSED_FINAL_HJ_CHILD_RUNNING",
}


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def watch_paths(canonical_root: Path) -> tuple[Path, Path]:
    operations = canonical_root / "operations"
    return (
        operations / "INDEPENDENT_HNEK_HANDOFF_AUTOCOMPLETE.json",
        operations / "INDEPENDENT_HNEK_HANDOFF_AUTOCOMPLETE.lock",
    )


def write_watch(
    canonical_root: Path, source_root: Path, status: str, **fields: Any,
) -> dict[str, Any]:
    path, _ = watch_paths(canonical_root)
    previous = handoff.read_json(path) if path.is_file() else {}
    payload = {
        "schema": WATCH_SCHEMA,
        "updated": now(),
        "created": previous.get("created", now()),
        "status": status,
        "watcher_pid": os.getpid(),
        "canonical_root": str(canonical_root.resolve()),
        "source_root": str(source_root.resolve()),
        "training_git_commit": handoff.EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": handoff.EXPECTED_PROTOCOL,
        "manifest_sha256": handoff.EXPECTED_MANIFEST,
        "training_update_changed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        **fields,
    }
    atomic_json(path, payload)
    return payload


def source_ready(source_root: Path) -> bool:
    operations = source_root / "operations"
    contract_path = operations / "INDEPENDENT_PROBE_CONTRACT.json"
    result_path = operations / "INDEPENDENT_PROBE_RESULT.json"
    if not contract_path.is_file() or not result_path.is_file():
        return False
    result = handoff.read_json(result_path)
    status = result.get("status")
    if status == "COMPLETE_MATCHED_BASELINE_VERIFIED":
        return handoff.sidecar_epoch(source_root, handoff.LANE) == 200
    if status in {"FAILED", "ENGINEERING_INVALID"}:
        raise RuntimeError(f"independent HNEK source reached terminal failure: {status}")
    return False


def next_action(
    *, record_status: str | None, receipt_status: str | None, ready: bool,
) -> str:
    """Return the only safe transition for the observed durable records."""
    if record_status == "EXECUTOR_RESUMED_AFTER_VERIFIED_IMPORT":
        return "COMPLETE"
    if receipt_status == "IMPORT_VERIFIED":
        if record_status != "READY_FOR_INDEPENDENT_HNEK_IMPORT":
            raise RuntimeError("verified import exists without a resumable handoff record")
        return "RESUME_ONLY"
    if record_status == "READY_FOR_INDEPENDENT_HNEK_IMPORT":
        return "IMPORT_AND_RESUME" if ready else "WAIT_SOURCE"
    if record_status in WAITING_RECORD_STATES:
        return "WAIT_RECORD"
    raise RuntimeError(f"unexpected handoff status: {record_status}")


def autocomplete(
    canonical_root: Path, source_root: Path, poll_seconds: int, timeout_seconds: int,
) -> int:
    canonical_root = canonical_root.resolve()
    source_root = source_root.resolve()
    record_path = handoff.paths(canonical_root)["record"]
    receipt_path = handoff.paths(canonical_root)["receipt"]
    _, lock_path = watch_paths(canonical_root)
    with handoff.lock(lock_path):
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            handoff.validate_canonical(canonical_root)
            record = handoff.read_json(record_path) if record_path.is_file() else {}
            receipt = handoff.read_json(receipt_path) if receipt_path.is_file() else {}
            recorded_source = record.get("source_root")
            if recorded_source and Path(str(recorded_source)).resolve() != source_root:
                raise RuntimeError("armed handoff source differs from autocomplete source")
            ready = source_ready(source_root)
            action = next_action(
                record_status=record.get("status"),
                receipt_status=receipt.get("status"),
                ready=ready,
            )
            if action == "COMPLETE":
                write_watch(canonical_root, source_root, "COMPLETE_ALREADY_RESUMED")
                return 0
            if action == "RESUME_ONLY":
                handoff.resume(canonical_root)
                write_watch(canonical_root, source_root, "COMPLETE_IMPORTED_AND_RESUMED")
                return 0
            if action == "IMPORT_AND_RESUME":
                handoff.validate_source(source_root)
                handoff.import_lane(canonical_root, source_root, require_paused=True)
                handoff.resume(canonical_root)
                write_watch(canonical_root, source_root, "COMPLETE_IMPORTED_AND_RESUMED")
                return 0
            write_watch(
                canonical_root,
                source_root,
                "WAITING_FOR_HANDOFF_READY" if action == "WAIT_RECORD" else "WAITING_FOR_SOURCE_E200",
                handoff_status=record.get("status"),
                independent_hnek_ready=ready,
                seconds_remaining=max(0, int(deadline - time.monotonic())),
            )
            time.sleep(poll_seconds)
        raise TimeoutError("independent HNEK handoff autocomplete timed out")


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
    try:
        return autocomplete(
            args.canonical_root, args.source_root,
            args.poll_seconds, args.timeout_seconds,
        )
    except Exception as exc:
        canonical = args.canonical_root.resolve()
        source = args.source_root.resolve()
        write_watch(
            canonical, source, "FAILED",
            exception_type=type(exc).__name__,
            exception=str(exc), traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
