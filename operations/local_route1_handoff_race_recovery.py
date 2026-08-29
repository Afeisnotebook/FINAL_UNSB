"""Recover a missed HJ->independent-HNEK pause gate without mixing states.

This tool is intentionally narrower than the normal handoff.  It is legal only
after the canonical executor and its accidentally started first HNEK chunk are
both dead, HJ e200 has an accepted independent milestone verification, and the
partial canonical HNEK is still bounded to the first five data epochs.  The
partial tree is moved into an operations quarantine; it is never overwritten.
The verified same-host independent HNEK tree can then be imported, after which
the ordinary frozen executor must be restarted as a new process.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from operations import local_route1_import_handoff as handoff
except ModuleNotFoundError:
    import local_route1_import_handoff as handoff  # type: ignore[no-redef]


SCHEMA = "final-unsb-route1-handoff-race-recovery-v1"
MAX_ACCIDENTAL_EPOCH = 5


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def recovery_paths(canonical_root: Path) -> dict[str, Path]:
    operations = canonical_root.resolve() / "operations"
    return {
        "record": operations / "HNEK_HANDOFF_RACE_RECOVERY.json",
        "events": operations / "HNEK_HANDOFF_RACE_RECOVERY_EVENTS.jsonl",
        "quarantine": operations / "quarantine",
    }


def validate_interrupted_state(state: dict[str, Any]) -> tuple[int, int]:
    if state.get("status") != "CHUNK_RUNNING" or state.get("lane") != "hnek":
        raise RuntimeError("recovery requires the accidentally running canonical HNEK chunk")
    if int(state.get("start_data_epoch", -1)) != 0:
        raise RuntimeError("recovery only accepts an accidental HNEK start from shared e0")
    target = int(state.get("target_data_epoch", -1))
    if not 1 <= target <= MAX_ACCIDENTAL_EPOCH:
        raise RuntimeError("accidental HNEK target is outside the bounded first chunk")
    executor_pid = int(state.get("executor_pid", -1))
    child_pid = int(state.get("child_pid", -1))
    if executor_pid <= 0 or child_pid <= 0:
        raise RuntimeError("interrupted state has invalid process lineage")
    return executor_pid, child_pid


def paused_state(state: dict[str, Any], *, partial_epoch: int, quarantine: Path) -> dict:
    result = dict(state)
    result.pop("lane", None)
    result.pop("child_pid", None)
    result.update({
        "updated": now(),
        "status": "PAUSED_AFTER_HJ_E200_FOR_VERIFIED_HNEK_IMPORT",
        "previous_status": state.get("status"),
        "previous_lane": state.get("lane"),
        "previous_child_pid": state.get("child_pid"),
        "partial_hnek_data_epoch": int(partial_epoch),
        "partial_hnek_quarantine": str(quarantine.resolve()),
        "executor_alive": False,
        "confirmation20_opened": False,
    })
    return result


def _accepted_hj_verification(canonical_root: Path) -> dict[str, Any]:
    path = (
        canonical_root / "operations" / "milestone_verifications"
        / "HJ_E200_VERIFICATION.json"
    )
    if not path.is_file():
        raise RuntimeError("HJ e200 milestone verification is missing")
    payload = handoff.read_json(path)
    identity = payload.get("identity", {})
    integrity = payload.get("integrity", {})
    if payload.get("status") != "ACCEPTED_MILESTONE":
        raise RuntimeError("HJ e200 milestone is not accepted")
    if identity.get("probe_id") != "hj" or int(identity.get("data_epoch", -1)) != 200:
        raise RuntimeError("HJ verification identity mismatch")
    required_true = (
        "checkpoint_file_hash_matches_sidecar",
        "scientific_state_hash_matches_sidecar",
        "metric_protocol_matches",
        "evaluation_bundle_matches_frozen_crn",
    )
    if not all(integrity.get(key) is True for key in required_true):
        raise RuntimeError("HJ verification integrity guard failed")
    if integrity.get("paired_metric_used_for_training_control") is not False:
        raise RuntimeError("HJ verification used paired metric for training control")
    if integrity.get("confirmation20_opened") is not False:
        raise RuntimeError("HJ verification opened confirmation20")
    return payload


def _assert_dead(pid: int, label: str) -> None:
    if handoff.process_exists(int(pid)):
        raise RuntimeError(f"{label} PID {pid} is still alive")


def _write_recovery(canonical_root: Path, payload: dict[str, Any]) -> dict[str, Any]:
    paths = recovery_paths(canonical_root)
    record = {**payload, "schema": SCHEMA, "updated": now()}
    handoff.atomic_json(paths["record"], record)
    handoff.append_jsonl(paths["events"], record)
    return record


def prepare(canonical_root: Path, source_root: Path) -> dict[str, Any]:
    canonical_root = canonical_root.resolve()
    source_root = source_root.resolve()
    locations = handoff.paths(canonical_root)
    _contract, state = handoff.validate_canonical(canonical_root)
    executor_pid, child_pid = validate_interrupted_state(state)
    _assert_dead(executor_pid, "canonical executor")
    _assert_dead(child_pid, "canonical HNEK child")
    hj_verification = _accepted_hj_verification(canonical_root)
    if handoff.sidecar_epoch(canonical_root, "hj") != 200:
        raise RuntimeError("canonical HJ is not e200")
    hj_run = handoff.read_json(canonical_root / "anchors" / "hj" / "RUN_STATE.json")
    if hj_run.get("status") != "COMPLETE_E200":
        raise RuntimeError("canonical HJ RUN_STATE is incomplete")

    partial_epoch = handoff.sidecar_epoch(canonical_root, "hnek")
    if not 1 <= partial_epoch <= MAX_ACCIDENTAL_EPOCH:
        raise RuntimeError("partial canonical HNEK is outside the recoverable first chunk")
    partial_identity = handoff.validate_checkpoint_file(canonical_root, "hnek")
    partial_root = canonical_root / "anchors" / "hnek"
    partial_rows = handoff.tree_manifest(partial_root)
    quarantine_root = recovery_paths(canonical_root)["quarantine"]
    quarantine_root.mkdir(parents=True, exist_ok=True)
    quarantine = quarantine_root / (
        "hnek_missed_pause_" + partial_identity["scientific_state_sha256"][:12]
    )
    if quarantine.exists():
        raise RuntimeError("deterministic HNEK quarantine destination already exists")
    partial_root.rename(quarantine)
    if partial_root.exists() or not quarantine.is_dir():
        raise RuntimeError("partial HNEK quarantine move did not complete")

    handoff.atomic_json(
        locations["state"],
        paused_state(state, partial_epoch=partial_epoch, quarantine=quarantine),
    )
    handoff.write_record(
        canonical_root, "RECOVERY_PREPARED_AFTER_MISSED_PAUSE_GATE",
        source_root=str(source_root), executor_pid=executor_pid,
        child_pid=child_pid, partial_hnek_data_epoch=partial_epoch,
        partial_hnek_quarantine=str(quarantine),
    )
    return _write_recovery(canonical_root, {
        "status": "PREPARED_FOR_VERIFIED_IMPORT",
        "canonical_root": str(canonical_root),
        "source_root": str(source_root),
        "stopped_executor_pid": executor_pid,
        "stopped_child_pid": child_pid,
        "partial_hnek_data_epoch": partial_epoch,
        "partial_hnek_quarantine": str(quarantine),
        "partial_hnek_files": len(partial_rows),
        "partial_hnek_tree_sha256": handoff.tree_sha256(partial_rows),
        "hj_e200_verification_sha256": handoff.file_sha256(
            canonical_root / "operations" / "milestone_verifications"
            / "HJ_E200_VERIFICATION.json"
        ),
        "training_state_mixed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })


def complete(canonical_root: Path, source_root: Path) -> dict[str, Any]:
    canonical_root = canonical_root.resolve()
    source_root = source_root.resolve()
    paths = recovery_paths(canonical_root)
    if not paths["record"].is_file():
        raise RuntimeError("handoff race recovery was not prepared")
    record = handoff.read_json(paths["record"])
    if record.get("status") != "PREPARED_FOR_VERIFIED_IMPORT":
        raise RuntimeError("handoff race recovery is not awaiting import")
    if Path(str(record.get("source_root"))).resolve() != source_root:
        raise RuntimeError("recovery source root changed")
    _assert_dead(int(record["stopped_executor_pid"]), "canonical executor")
    _assert_dead(int(record["stopped_child_pid"]), "canonical HNEK child")
    if (canonical_root / "anchors" / "hnek").exists():
        raise RuntimeError("canonical HNEK unexpectedly reappeared before import")

    _source_contract, source_rows = handoff.validate_source(source_root)
    handoff.import_lane(canonical_root, source_root, require_paused=False)
    destination_rows = handoff.tree_manifest(canonical_root / "anchors" / "hnek")
    if source_rows != destination_rows or handoff.sidecar_epoch(canonical_root, "hnek") != 200:
        raise RuntimeError("verified HNEK import differs from source")
    receipt = handoff.read_json(handoff.paths(canonical_root)["receipt"])
    handoff.write_record(
        canonical_root, "RECOVERED_IMPORT_EXECUTOR_RESTART_REQUIRED",
        source_root=str(source_root), imported_hnek_tree_sha256=receipt["tree_sha256"],
    )
    return _write_recovery(canonical_root, {
        **record,
        "status": "IMPORT_VERIFIED_RESTART_REQUIRED",
        "import_receipt": str(handoff.paths(canonical_root)["receipt"]),
        "imported_hnek_tree_sha256": receipt["tree_sha256"],
        "imported_hnek_files": len(destination_rows),
        "executor_inactive_during_import": True,
        "training_state_mixed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })


def validate_restart_state(record: dict[str, Any], state: dict[str, Any]) -> int:
    if record.get("status") != "IMPORT_VERIFIED_RESTART_REQUIRED":
        raise RuntimeError("verified import is not awaiting executor restart")
    new_pid = int(state.get("executor_pid", -1))
    if new_pid <= 0 or new_pid == int(record.get("stopped_executor_pid", -1)):
        raise RuntimeError("executor was not restarted as a new process")
    if state.get("status") not in {"CHUNK_RUNNING", "ANCHOR_PHASE_COMPLETE"}:
        raise RuntimeError("restarted executor has not resumed the registered anchor flow")
    if state.get("status") == "CHUNK_RUNNING" and state.get("lane") != "dt":
        raise RuntimeError("restarted calibrated executor is not running DT")
    return new_pid


def finalize(canonical_root: Path, source_root: Path) -> dict[str, Any]:
    canonical_root = canonical_root.resolve()
    source_root = source_root.resolve()
    paths = recovery_paths(canonical_root)
    record = handoff.read_json(paths["record"])
    if Path(str(record.get("source_root"))).resolve() != source_root:
        raise RuntimeError("recovery source root changed before finalization")
    _contract, state = handoff.validate_canonical(canonical_root)
    new_pid = validate_restart_state(record, state)
    if not handoff.process_exists(new_pid):
        raise RuntimeError("restarted executor PID is not alive")
    if handoff.sidecar_epoch(canonical_root, "hnek") != 200:
        raise RuntimeError("imported HNEK disappeared after executor restart")
    calibration = handoff.read_json(canonical_root / "evidence" / "PROXY_CALIBRATION.json")
    if calibration.get("status") != "CALIBRATED":
        raise RuntimeError("executor restarted without a calibrated proxy")
    handoff.write_record(
        canonical_root, "RECOVERED_IMPORT_EXECUTOR_RESTARTED",
        source_root=str(source_root), restarted_executor_pid=new_pid,
        resumed_lane=state.get("lane"), proxy_status=calibration.get("status"),
    )
    return _write_recovery(canonical_root, {
        **record,
        "status": "RECOVERY_COMPLETE_EXECUTOR_RESTARTED",
        "restarted_executor_pid": new_pid,
        "post_restart_status": state.get("status"),
        "post_restart_lane": state.get("lane"),
        "post_restart_data_epoch": state.get("current_data_epoch"),
        "proxy_status": calibration.get("status"),
        "training_state_mixed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("action", choices=("prepare", "complete", "finalize"))
    value.add_argument("--canonical-root", type=Path, required=True)
    value.add_argument("--source-root", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.canonical_root, args.source_root)
    elif args.action == "complete":
        result = complete(args.canonical_root, args.source_root)
    else:
        result = finalize(args.canonical_root, args.source_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
