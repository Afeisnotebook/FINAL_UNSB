"""Safely admit an independent HNEK e200 tree into a canonical route-1 run.

The scientific anchor is never edited.  On a host where canonical HJ would
finish before the independent HNEK run, ``arm`` stops only the durable executor
parent during HJ's final chunk.  The HJ child finishes and writes e200 normally.
``import`` then copies a fully verified, same-host independent HNEK tree into an
empty canonical lane directory.  ``resume`` releases the executor, which sees
both lanes at e200 and continues to the frozen proxy gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TRAINING_COMMIT = "0da2a37086cca5bc4ad4488bb07c53096a7152ed"
EXPECTED_PROTOCOL = "b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2"
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
LANE = "hnek"
PREDECESSOR = "hj"


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(root.rglob("*")) if path.is_file()
    ]


def tree_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def process_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def process_state(pid: int) -> str:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return "MISSING"
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("State:"):
            return line.split(":", 1)[1].strip()
    return "UNKNOWN"


def process_ppid(pid: int) -> int:
    path = Path(f"/proc/{pid}/status")
    if not path.is_file():
        return -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("PPid:"):
            return int(line.split(":", 1)[1].strip())
    return -1


@contextmanager
def lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            owner = int(read_json(path).get("pid", -1))
        except Exception:
            owner = -1
        if process_exists(owner):
            raise RuntimeError(f"handoff lock owned by live PID {owner}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": now()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if path.exists():
            try:
                owner = int(read_json(path).get("pid", -1))
            except Exception:
                owner = -1
            if owner == os.getpid():
                path.unlink()


def validate_identity(payload: dict[str, Any], *, prefix: str) -> None:
    expected = {
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    }
    if prefix == "contract":
        expected = {
            "training_git_commit": EXPECTED_TRAINING_COMMIT,
            "training_protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise RuntimeError(f"{prefix} identity mismatch for {key}")


def sidecar_epoch(root: Path, lane: str) -> int:
    path = root / "anchors" / lane / "full_state_latest.pt.json"
    if not path.is_file():
        return 0
    sidecar = read_json(path)
    if sidecar.get("schema") != "final-unsb-local-route1-full-state-v1":
        raise RuntimeError(f"{lane} sidecar schema mismatch")
    metadata = sidecar.get("metadata", {})
    validate_identity(metadata, prefix=f"{lane} sidecar")
    if sidecar.get("probe_id") != lane:
        raise RuntimeError(f"{lane} sidecar probe mismatch")
    epoch = int(sidecar.get("physical_epoch_completed", -1))
    if int(sidecar.get("step", -1)) != epoch * 150:
        raise RuntimeError(f"{lane} sidecar update/epoch mismatch")
    return epoch


def validate_checkpoint_file(root: Path, lane: str) -> dict[str, Any]:
    checkpoint = root / "anchors" / lane / "full_state_latest.pt"
    sidecar_path = Path(str(checkpoint) + ".json")
    if not checkpoint.is_file() or not sidecar_path.is_file():
        raise FileNotFoundError(f"{lane} checkpoint/sidecar missing")
    sidecar = read_json(sidecar_path)
    sidecar_epoch(root, lane)
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != sidecar.get("full_state_sha256"):
        raise RuntimeError(f"{lane} checkpoint hash differs from sidecar")
    if not sidecar.get("scientific_state_sha256"):
        raise RuntimeError(f"{lane} scientific state hash missing")
    return {
        "checkpoint_sha256": checkpoint_hash,
        "scientific_state_sha256": sidecar["scientific_state_sha256"],
    }


def paths(canonical_root: Path) -> dict[str, Path]:
    operations = canonical_root / "operations"
    return {
        "operations": operations,
        "contract": operations / "EXECUTOR_CONTRACT.json",
        "state": operations / "EXECUTION_STATE.json",
        "record": operations / "INDEPENDENT_HNEK_HANDOFF.json",
        "events": operations / "INDEPENDENT_HNEK_HANDOFF_EVENTS.jsonl",
        "receipt": operations / "INDEPENDENT_HNEK_IMPORT.json",
        "lock": operations / "INDEPENDENT_HNEK_HANDOFF.lock",
    }


def validate_canonical(canonical_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    locations = paths(canonical_root)
    contract = read_json(locations["contract"])
    state = read_json(locations["state"])
    if contract.get("schema") != "final-unsb-route1-executor-contract-v1":
        raise RuntimeError("canonical executor contract schema mismatch")
    validate_identity(contract, prefix="contract")
    validate_identity(state, prefix="state")
    if Path(str(contract.get("run_root"))).resolve() != canonical_root.resolve():
        raise RuntimeError("canonical executor run_root mismatch")
    return contract, state


def write_record(canonical_root: Path, status: str, **payload: Any) -> dict[str, Any]:
    locations = paths(canonical_root)
    previous = read_json(locations["record"]) if locations["record"].is_file() else {}
    record = {
        "schema": "final-unsb-route1-independent-hnek-handoff-v1",
        "updated": now(),
        "status": status,
        "canonical_root": str(canonical_root.resolve()),
        "training_git_commit": EXPECTED_TRAINING_COMMIT,
        "training_protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
        "created": previous.get("created", now()),
        **{key: value for key, value in previous.items() if key not in {"updated", "status"}},
        **payload,
    }
    atomic_json(locations["record"], record)
    append_jsonl(locations["events"], {"time": now(), "event": status, **payload})
    return record


def arm(canonical_root: Path, source_root: Path, poll_seconds: int, timeout_seconds: int) -> int:
    if os.name == "nt":
        raise RuntimeError("executor SIGSTOP handoff is Linux-only")
    locations = paths(canonical_root)
    script = Path(__file__).resolve()
    with lock(locations["lock"]):
        contract, state = validate_canonical(canonical_root)
        if sidecar_epoch(canonical_root, LANE) > 0:
            raise RuntimeError("canonical HNEK already started; refusing concurrent import handoff")
        executor_pid = int(state.get("executor_pid", -1))
        if not process_exists(executor_pid):
            raise RuntimeError("canonical executor PID is not alive")
        write_record(
            canonical_root, "ARMED",
            source_root=str(source_root.resolve()),
            executor_pid=executor_pid,
            executor_contract_sha256=file_sha256(locations["contract"]),
            coordinator_script=str(script),
            coordinator_script_sha256=file_sha256(script),
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            _, state = validate_canonical(canonical_root)
            if sidecar_epoch(canonical_root, LANE) > 0:
                raise RuntimeError("canonical HNEK started before the executor pause gate")
            ready = (
                state.get("status") == "CHUNK_RUNNING"
                and state.get("lane") == PREDECESSOR
                and int(state.get("start_data_epoch", -1)) == 195
                and int(state.get("target_data_epoch", -1)) == 200
            )
            if not ready:
                time.sleep(poll_seconds)
                continue
            current_executor = int(state.get("executor_pid", -1))
            child_pid = int(state.get("child_pid", -1))
            if current_executor != executor_pid or process_ppid(child_pid) != executor_pid:
                raise RuntimeError("final HJ chunk process lineage mismatch")
            os.kill(executor_pid, signal.SIGSTOP)
            time.sleep(1)
            if not process_state(executor_pid).startswith("T"):
                raise RuntimeError("executor did not enter stopped state")
            write_record(
                canonical_root, "EXECUTOR_PAUSED_FINAL_HJ_CHILD_RUNNING",
                source_root=str(source_root.resolve()), executor_pid=executor_pid,
                child_pid=child_pid, pause_signal="SIGSTOP",
                predecessor=PREDECESSOR, predecessor_target_epoch=200,
            )
            while time.monotonic() < deadline:
                run_state_path = canonical_root / "anchors" / PREDECESSOR / "RUN_STATE.json"
                run_state = read_json(run_state_path) if run_state_path.is_file() else {}
                if (
                    sidecar_epoch(canonical_root, PREDECESSOR) == 200
                    and run_state.get("status") == "COMPLETE_E200"
                    and int(run_state.get("final_data_epoch", -1)) == 200
                ):
                    write_record(
                        canonical_root, "READY_FOR_INDEPENDENT_HNEK_IMPORT",
                        source_root=str(source_root.resolve()), executor_pid=executor_pid,
                        child_pid=child_pid, predecessor=PREDECESSOR,
                        predecessor_epoch=200,
                    )
                    return 0
                time.sleep(poll_seconds)
            raise TimeoutError("HJ child did not finish before handoff timeout")
        raise TimeoutError("HJ final chunk was not observed before handoff timeout")


def validate_source(source_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract_path = source_root / "operations" / "INDEPENDENT_PROBE_CONTRACT.json"
    result_path = source_root / "operations" / "INDEPENDENT_PROBE_RESULT.json"
    if not contract_path.is_file() or not result_path.is_file():
        raise RuntimeError("independent probe has not returned a promotable result")
    contract = read_json(contract_path)
    result = read_json(result_path)
    validate_identity(contract, prefix="contract")
    if contract.get("status") != "COMPLETE_MATCHED_BASELINE_VERIFIED":
        raise RuntimeError("independent probe is not matched-baseline verified")
    if result.get("status") != "COMPLETE_MATCHED_BASELINE_VERIFIED":
        raise RuntimeError("independent probe result remains quarantined")
    if (
        contract.get("lane") != LANE
        or contract.get("batch_size_changed") is not False
        or contract.get("training_update_changed") is not False
    ):
        raise RuntimeError("independent source is not the frozen batch1 HNEK lane")
    if contract.get("cross_host_state_used") is not False:
        raise RuntimeError("cross-host independent source is forbidden")
    if contract.get("confirmation20_opened") is not False:
        raise RuntimeError("independent source opened confirmation20")
    if contract.get("paired_controller_access") is not False:
        raise RuntimeError("independent source used paired controller input")
    if sidecar_epoch(source_root, LANE) != 200:
        raise RuntimeError("independent HNEK source is not e200")
    validate_checkpoint_file(source_root, LANE)
    run_state = read_json(source_root / "anchors" / LANE / "RUN_STATE.json")
    if run_state.get("status") != "COMPLETE_E200" or int(run_state.get("final_data_epoch", -1)) != 200:
        raise RuntimeError("independent HNEK RUN_STATE is incomplete")
    if not (source_root / "anchors" / LANE / "metrics" / "e200.json").is_file():
        raise RuntimeError("independent HNEK e200 metric is missing")
    rows = tree_manifest(source_root / "anchors" / LANE)
    return contract, rows


def import_lane(canonical_root: Path, source_root: Path, require_paused: bool) -> int:
    locations = paths(canonical_root)
    with lock(locations["lock"]):
        contract, state = validate_canonical(canonical_root)
        if sidecar_epoch(canonical_root, LANE) > 0:
            raise RuntimeError("canonical HNEK already exists; import never overwrites")
        source_contract, source_rows = validate_source(source_root)
        if Path(str(source_contract.get("matched_plain_root"))).resolve() != canonical_root.resolve():
            raise RuntimeError("independent HNEK is not matched to this canonical plain root")
        if sidecar_epoch(canonical_root, "plain") != 200:
            raise RuntimeError("canonical matched plain is not e200")
        plain_identity = validate_checkpoint_file(canonical_root, "plain")
        if source_contract.get("matched_plain_checkpoint_sha256") != plain_identity["checkpoint_sha256"]:
            raise RuntimeError("source matched-plain checkpoint hash changed before import")
        if (
            source_contract.get("matched_plain_scientific_state_sha256")
            != plain_identity["scientific_state_sha256"]
        ):
            raise RuntimeError("source matched-plain scientific state changed before import")
        executor_pid = int(state.get("executor_pid", -1))
        if require_paused:
            record = read_json(locations["record"])
            if record.get("status") != "READY_FOR_INDEPENDENT_HNEK_IMPORT":
                raise RuntimeError("executor pause record is not ready for import")
            if int(record.get("executor_pid", -1)) != executor_pid:
                raise RuntimeError("paused executor PID changed")
            if not process_state(executor_pid).startswith("T"):
                raise RuntimeError("canonical executor is not stopped")
        elif state.get("lane") == LANE:
            raise RuntimeError("canonical executor is already scheduling HNEK")

        source_lane = source_root / "anchors" / LANE
        destination_lane = canonical_root / "anchors" / LANE
        if destination_lane.exists():
            raise RuntimeError("canonical HNEK directory exists; refusing overwrite")
        shutil.copytree(source_lane, destination_lane)
        destination_rows = tree_manifest(destination_lane)
        if source_rows != destination_rows:
            raise RuntimeError("copied HNEK tree differs from independent source")
        receipt = {
            "schema": "final-unsb-route1-independent-hnek-import-v1",
            "recorded": now(),
            "status": "IMPORT_VERIFIED",
            "source_root": str(source_root.resolve()),
            "canonical_root": str(canonical_root.resolve()),
            "lane": LANE,
            "data_epoch": 200,
            "file_count": len(source_rows),
            "tree_sha256": tree_sha256(source_rows),
            "source_contract_sha256": file_sha256(
                source_root / "operations" / "INDEPENDENT_PROBE_CONTRACT.json"
            ),
            "executor_paused_during_import": require_paused,
            "training_git_commit": EXPECTED_TRAINING_COMMIT,
            "training_protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        atomic_json(locations["receipt"], receipt)
        append_jsonl(locations["events"], {"time": now(), "event": "IMPORT_VERIFIED", **receipt})
        return 0


def resume(canonical_root: Path) -> int:
    locations = paths(canonical_root)
    with lock(locations["lock"]):
        _, state = validate_canonical(canonical_root)
        record = read_json(locations["record"])
        receipt = read_json(locations["receipt"])
        if record.get("status") != "READY_FOR_INDEPENDENT_HNEK_IMPORT":
            raise RuntimeError("handoff is not waiting for resume")
        if receipt.get("status") != "IMPORT_VERIFIED":
            raise RuntimeError("HNEK import receipt is not verified")
        if sidecar_epoch(canonical_root, LANE) != 200:
            raise RuntimeError("canonical imported HNEK is not e200")
        executor_pid = int(record.get("executor_pid", -1))
        if executor_pid != int(state.get("executor_pid", -1)):
            raise RuntimeError("executor PID changed before resume")
        if not process_state(executor_pid).startswith("T"):
            raise RuntimeError("executor is not stopped before SIGCONT")
        os.kill(executor_pid, signal.SIGCONT)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline and process_state(executor_pid).startswith("T"):
            time.sleep(1)
        if process_state(executor_pid).startswith("T"):
            raise RuntimeError("executor remained stopped after SIGCONT")
        write_record(
            canonical_root, "EXECUTOR_RESUMED_AFTER_VERIFIED_IMPORT",
            executor_pid=executor_pid, resume_signal="SIGCONT",
            imported_hnek_tree_sha256=receipt["tree_sha256"],
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=("arm", "import", "resume"))
    value.add_argument("--canonical-root", type=Path, required=True)
    value.add_argument("--source-root", type=Path)
    value.add_argument("--poll-seconds", type=int, default=15)
    value.add_argument("--timeout-seconds", type=int, default=43_200)
    value.add_argument("--require-paused", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    canonical = args.canonical_root.resolve()
    source = args.source_root.resolve() if args.source_root else None
    if args.action in {"arm", "import"} and source is None:
        raise SystemExit(f"{args.action} requires --source-root")
    if args.poll_seconds < 5 or args.timeout_seconds <= 0:
        raise SystemExit("poll/timeout values are invalid")
    if args.action == "arm":
        return arm(canonical, source, args.poll_seconds, args.timeout_seconds)  # type: ignore[arg-type]
    if args.action == "import":
        return import_lane(canonical, source, bool(args.require_paused))  # type: ignore[arg-type]
    return resume(canonical)


if __name__ == "__main__":
    raise SystemExit(main())
