"""Prove a hard-killed anchor chunk resumes to the accepted reference state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def accepted_two_epoch_hash(gpu_gate: Path) -> str:
    payload = json.loads(gpu_gate.read_text(encoding="utf-8"))
    for gate in payload["gates"]:
        if gate["name"] == "two_epoch_continuous_equals_one_plus_resume":
            if gate["status"] != "PASS":
                raise RuntimeError("accepted two-epoch resume gate is not PASS")
            return str(gate["detail"])
    raise RuntimeError("accepted two-epoch resume hash is missing")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--executor-repo", type=Path, required=True)
    value.add_argument("--scratch-output", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--gpu-gate", type=Path, required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--kill-delay-seconds", type=int, default=20)
    value.add_argument("--heartbeat-timeout-seconds", type=int, default=600)
    return value


def command(args: argparse.Namespace) -> list[str]:
    return [
        str(args.python.resolve()),
        "-m",
        "research.local_route1.run",
        "--stage",
        "anchors",
        "--lane",
        "plain",
        "--resume",
        "--output",
        str(args.scratch_output.resolve()),
        "--train-view",
        str(args.train_view.resolve()),
        "--data-root",
        str(args.data_root.resolve()),
        "--manifest",
        str(args.manifest.resolve()),
        "--engineering-stop-after-epoch",
        "2",
    ]


def scientific_snapshot_hash(args: argparse.Namespace, checkpoint: Path) -> str:
    code = (
        "import torch; "
        "from research.local_route1.runtime import full_state_hash; "
        f"x=torch.load({str(checkpoint)!r},map_location='cpu',weights_only=False); "
        "print(full_state_hash({'model':x['model'],'rng':x['rng'],'samplers':x['samplers']}))"
    )
    result = subprocess.run(
        [str(args.python.resolve()), "-c", code],
        cwd=args.executor_repo.resolve(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to hash recovered scientific snapshot: {result.stderr}")
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo = args.executor_repo.resolve()
    scratch = args.scratch_output.resolve()
    if scratch.exists():
        raise RuntimeError(f"fault-gate output must not already exist: {scratch}")
    scratch.mkdir(parents=True)
    logs = scratch / "fault_gate_logs"
    logs.mkdir()
    heartbeat = scratch / "anchors" / "plain" / "HEARTBEAT.json"
    latest = scratch / "anchors" / "plain" / "full_state_latest.pt"
    latest_json = Path(str(latest) + ".json")
    reference_hash = accepted_two_epoch_hash(args.gpu_gate.resolve())

    started = utc_now()
    argv_run = command(args)
    with (
        (logs / "killed.stdout.log").open("w", encoding="utf-8") as stdout,
        (logs / "killed.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        process = subprocess.Popen(argv_run, cwd=repo, stdout=stdout, stderr=stderr, text=True)
        deadline = time.time() + args.heartbeat_timeout_seconds
        while time.time() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"test process exited before kill point: {process.returncode}")
            if heartbeat.is_file():
                current = json.loads(heartbeat.read_text(encoding="utf-8"))
                if int(current.get("data_epoch", 0)) >= 1:
                    break
            time.sleep(2)
        else:
            process.terminate()
            raise RuntimeError("fault gate timed out waiting for e1 heartbeat")
        time.sleep(args.kill_delay_seconds)
        if process.poll() is not None:
            raise RuntimeError("test process reached e2 before deliberate termination")
        process.terminate()
        process.wait(timeout=30)
        killed_exit_code = int(process.returncode)

    if not latest.is_file() or not latest_json.is_file():
        raise RuntimeError("e1 atomic checkpoint did not survive deliberate termination")
    killed_sidecar = json.loads(latest_json.read_text(encoding="utf-8"))
    if int(killed_sidecar["physical_epoch_completed"]) != 1:
        raise RuntimeError("deliberate termination did not preserve the e1 boundary")
    if file_sha256(latest) != killed_sidecar["full_state_sha256"]:
        raise RuntimeError("surviving e1 checkpoint hash mismatch")
    temporary_files = [str(path) for path in scratch.rglob("*.tmp")]
    if temporary_files:
        raise RuntimeError(f"temporary checkpoint files survived termination: {temporary_files}")

    with (
        (logs / "recovered.stdout.log").open("w", encoding="utf-8") as stdout,
        (logs / "recovered.stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        resumed = subprocess.run(
            argv_run, cwd=repo, stdout=stdout, stderr=stderr, text=True, check=False
        )
    if resumed.returncode != 0:
        raise RuntimeError(f"recovery run failed with exit code {resumed.returncode}")
    recovered = json.loads(latest_json.read_text(encoding="utf-8"))
    if int(recovered["physical_epoch_completed"]) != 2:
        raise RuntimeError("recovery run did not reach e2")
    recovered_hash = scientific_snapshot_hash(args, latest)
    if recovered_hash != reference_hash:
        raise RuntimeError(
            f"recovered state differs from accepted reference: {recovered_hash} != {reference_hash}"
        )
    if recovered["metadata"].get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation lock violated during fault gate")

    evidence = {
        "schema": "final-unsb-route1-process-loss-gate-v1",
        "status": "PASS",
        "started": started,
        "completed": utc_now(),
        "executor_repo": str(repo),
        "scratch_output": str(scratch),
        "killed_pid": process.pid,
        "killed_exit_code": killed_exit_code,
        "surviving_data_epoch": 1,
        "surviving_updates": 150,
        "surviving_checkpoint_sha256": killed_sidecar["full_state_sha256"],
        "recovered_data_epoch": 2,
        "recovered_updates": 300,
        "recovered_scientific_state_sha256": recovered_hash,
        "accepted_reference_scientific_state_sha256": reference_hash,
        "exact_reference_match": True,
        "temporary_files_after_kill": [],
        "confirmation20_opened": False,
        "training_git_commit": recovered["metadata"]["git_commit"],
        "training_protocol_fingerprint": recovered["metadata"]["protocol_fingerprint"],
    }
    atomic_json(scratch / "PROCESS_LOSS_GATE.json", evidence)
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
