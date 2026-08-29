"""Durably verify route-1 milestones as soon as their artifacts are complete.

This watcher is read-only with respect to checkpoints, metrics, and training
state.  It must never schedule, pause, rank, or otherwise control a run from
paired metrics.  Its only output is compact integrity evidence.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from local_route1_verify_milestone import atomic_json, verify


ALLOWED_LANES = ("plain", "hj", "hnek", "dt")


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def process_exists(pid: int) -> bool:
    if int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes

        handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
            0x1000, False, int(pid)
        )
        if not handle:
            return False
        ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
        return True
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            owner_pid = int(read_json(path).get("pid", -1))
        except Exception:
            owner_pid = -1
        if process_exists(owner_pid):
            raise RuntimeError(f"milestone verifier already owns lock with PID {owner_pid}")
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
                owner = read_json(path)
            except Exception:
                owner = {}
            if int(owner.get("pid", -1)) == os.getpid():
                path.unlink()


def artifact_paths(run_root: Path, lane: str, epoch: int) -> tuple[Path, Path, Path]:
    checkpoint = run_root / "anchors" / lane / "milestones" / f"e{epoch:03d}.pt"
    return (
        checkpoint,
        Path(str(checkpoint) + ".json"),
        run_root / "anchors" / lane / "metrics" / f"e{epoch:03d}.json",
    )


def accepted_evidence(path: Path, *, lane: str, epoch: int) -> bool:
    if not path.is_file():
        return False
    try:
        payload = read_json(path)
    except Exception:
        return False
    identity = payload.get("identity", {})
    integrity = payload.get("integrity", {})
    return bool(
        payload.get("schema") == "final-unsb-route1-milestone-verification-v1"
        and payload.get("status") == "ACCEPTED_MILESTONE"
        and identity.get("probe_id") == lane
        and int(identity.get("data_epoch", -1)) == epoch
        and int(identity.get("updates", -1)) == epoch * 150
        and integrity.get("checkpoint_file_hash_matches_sidecar") is True
        and integrity.get("scientific_state_hash_matches_sidecar") is True
        and integrity.get("metric_protocol_matches") is True
        and integrity.get("evaluation_bundle_matches_frozen_crn") is True
        and integrity.get("paired_metric_used_for_training_control") is False
        and integrity.get("confirmation20_opened") is False
    )


def write_state(path: Path, *, run_root: Path, training_repo: Path, epoch: int,
                status: str, lanes: dict[str, dict[str, Any]], started: str) -> None:
    atomic_json(path, {
        "schema": "final-unsb-route1-auto-milestone-verifier-v1",
        "created": started,
        "updated": now(),
        "status": status,
        "watcher_pid": os.getpid(),
        "run_root": str(run_root),
        "training_repo": str(training_repo),
        "epoch": epoch,
        "lanes": lanes,
        "paired_metric_used_for_training_control": False,
        "confirmation20_opened": False,
    })


def watch(
    *, run_root: Path, training_repo: Path, lanes: tuple[str, ...], epoch: int,
    output_dir: Path, state_path: Path, poll_seconds: float,
    timeout_seconds: float, maximum_failures: int,
    verifier: Callable[..., dict[str, Any]] = verify,
) -> int:
    run_root = run_root.resolve()
    training_repo = training_repo.resolve()
    output_dir = output_dir.resolve()
    state_path = state_path.resolve()
    started = now()
    began = time.monotonic()
    records = {
        lane: {
            "status": "WAITING_FOR_ARTIFACTS",
            "verification": str(output_dir / f"{lane.upper()}_E{epoch}_VERIFICATION.json"),
            "consecutive_failures": 0,
        }
        for lane in lanes
    }
    while True:
        all_verified = True
        for lane in lanes:
            record = records[lane]
            evidence_path = Path(record["verification"])
            if accepted_evidence(evidence_path, lane=lane, epoch=epoch):
                record.update({"status": "VERIFIED", "last_error": None})
                continue
            all_verified = False
            artifacts = artifact_paths(run_root, lane, epoch)
            missing = [str(path) for path in artifacts if not path.is_file()]
            if missing:
                record.update({
                    "status": "WAITING_FOR_ARTIFACTS",
                    "missing_artifact_count": len(missing),
                    "last_error": None,
                })
                continue
            record["status"] = "VERIFYING"
            write_state(
                state_path, run_root=run_root, training_repo=training_repo,
                epoch=epoch, status="VERIFYING", lanes=records, started=started,
            )
            try:
                result = verifier(
                    run_root=run_root, training_repo=training_repo, lane=lane,
                    epoch=epoch, require_lpips=epoch >= 100,
                )
                atomic_json(evidence_path, result)
                if not accepted_evidence(evidence_path, lane=lane, epoch=epoch):
                    raise RuntimeError("verifier output did not pass the acceptance schema")
            except Exception as exc:
                failures = int(record.get("consecutive_failures", 0)) + 1
                record.update({
                    "status": "RETRYABLE_VERIFY_FAILURE" if failures < maximum_failures else "FAILED",
                    "consecutive_failures": failures,
                    "last_error": f"{type(exc).__name__}: {exc}",
                })
                write_state(
                    state_path, run_root=run_root, training_repo=training_repo,
                    epoch=epoch, status=record["status"], lanes=records,
                    started=started,
                )
                if failures >= maximum_failures:
                    return 1
                continue
            record.update({
                "status": "VERIFIED",
                "verified_at": now(),
                "consecutive_failures": 0,
                "missing_artifact_count": 0,
                "last_error": None,
            })

        if all(record["status"] == "VERIFIED" for record in records.values()):
            write_state(
                state_path, run_root=run_root, training_repo=training_repo,
                epoch=epoch, status="COMPLETE", lanes=records, started=started,
            )
            return 0
        if time.monotonic() - began >= timeout_seconds:
            write_state(
                state_path, run_root=run_root, training_repo=training_repo,
                epoch=epoch, status="TIMED_OUT", lanes=records, started=started,
            )
            return 75
        write_state(
            state_path, run_root=run_root, training_repo=training_repo,
            epoch=epoch, status="WAITING", lanes=records, started=started,
        )
        time.sleep(max(0.0, poll_seconds))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--training-repo", type=Path, required=True)
    value.add_argument("--lanes", nargs="+", choices=ALLOWED_LANES, required=True)
    value.add_argument("--epoch", type=int, default=200)
    value.add_argument("--output-dir", type=Path)
    value.add_argument("--state", type=Path)
    value.add_argument("--poll-seconds", type=float, default=15.0)
    value.add_argument("--timeout-seconds", type=float, default=43_200.0)
    value.add_argument("--maximum-failures", type=int, default=3)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir
        else run_root / "operations" / "milestone_verifications"
    )
    state_path = (
        args.state.resolve() if args.state
        else run_root / "operations" / "AUTO_MILESTONE_VERIFIER.json"
    )
    if args.epoch <= 0 or args.epoch > 200:
        raise ValueError("epoch must be in [1, 200]")
    if args.maximum_failures <= 0:
        raise ValueError("maximum failures must be positive")
    lock_path = run_root / "operations" / "AUTO_MILESTONE_VERIFIER.lock"
    with exclusive_lock(lock_path):
        return watch(
            run_root=run_root, training_repo=args.training_repo,
            lanes=tuple(args.lanes), epoch=int(args.epoch), output_dir=output_dir,
            state_path=state_path, poll_seconds=float(args.poll_seconds),
            timeout_seconds=float(args.timeout_seconds),
            maximum_failures=int(args.maximum_failures),
        )


if __name__ == "__main__":
    raise SystemExit(main())
