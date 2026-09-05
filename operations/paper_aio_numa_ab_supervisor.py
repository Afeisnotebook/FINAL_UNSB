"""Non-destructive, metric-blind NUMA scheduling A/B supervisor.

The supervisor waits for a new complete data-epoch checkpoint, freezes (but
does not terminate) the live trainer, and runs two isolated 1000-update
branches from that checkpoint.  It resumes the live in-memory trajectory in
all exit paths.  A new CPU affinity is applied only when every transition-
defining component is bitwise identical and the net remaining-time saving,
including the A/B pause, is at least the preregistered threshold.

This is an operational scheduling gate.  It never reads evaluation metrics,
changes an algorithm/configuration, or opens confirmation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-paper-numa-ab-supervisor-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def parse_cpu_list(value: str) -> set[int]:
    result: set[int] = set()
    for raw in str(value).split(","):
        item = raw.strip()
        if not item:
            raise ValueError("empty CPU-list item")
        if "-" in item:
            left, right = item.split("-", 1)
            start, stop = int(left), int(right)
            if start < 0 or stop < start:
                raise ValueError(f"invalid CPU range: {item}")
            result.update(range(start, stop + 1))
        else:
            cpu = int(item)
            if cpu < 0:
                raise ValueError(f"invalid CPU: {item}")
            result.add(cpu)
    if not result:
        raise ValueError("CPU list is empty")
    return result


def format_cpu_list(values: set[int]) -> str:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("CPU set is empty")
    groups: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:]:
        if cpu == previous + 1:
            previous = cpu
            continue
        groups.append(str(start) if start == previous else f"{start}-{previous}")
        start = previous = cpu
    groups.append(str(start) if start == previous else f"{start}-{previous}")
    return ",".join(groups)


def net_remaining_saving(
    *, original_seconds: float, bound_seconds: float, updates: int,
    remaining_updates: int,
) -> dict[str, float]:
    if min(original_seconds, bound_seconds, updates, remaining_updates) <= 0:
        raise ValueError("timing and update counts must be positive")
    original_remaining = remaining_updates * original_seconds / updates
    bound_remaining = remaining_updates * bound_seconds / updates
    test_cost = original_seconds + bound_seconds
    net_bound = test_cost + bound_remaining
    return {
        "original_remaining_seconds": original_remaining,
        "bound_remaining_seconds_excluding_test": bound_remaining,
        "ab_test_cost_seconds": test_cost,
        "net_bound_remaining_seconds": net_bound,
        "net_saving_seconds": original_remaining - net_bound,
        "net_saving_fraction": 1.0 - net_bound / original_remaining,
    }


def _process_command(pid: int) -> str:
    path = Path(f"/proc/{pid}/cmdline")
    if not path.is_file():
        raise RuntimeError(f"trainer PID is not alive: {pid}")
    return path.read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")


def _process_state(pid: int) -> str:
    for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("State:"):
            return line.split()[1]
    raise RuntimeError(f"cannot read process state: {pid}")


def process_task_affinities(pid: int) -> dict[int, set[int]]:
    task_root = Path(f"/proc/{pid}/task")
    if not task_root.is_dir():
        raise RuntimeError(f"trainer task directory is unavailable: {pid}")
    result = {
        int(path.name): set(os.sched_getaffinity(int(path.name)))
        for path in task_root.iterdir() if path.name.isdigit()
    }
    if not result:
        raise RuntimeError(f"trainer has no visible tasks: {pid}")
    return result


def set_process_task_affinity(pid: int, cpus: set[int]) -> dict[int, set[int]]:
    # The trainer is SIGSTOP'ed before this function is called, so its task
    # set is stable.  Linux sched_setaffinity(pid, ...) changes only one TID;
    # every existing thread must be changed explicitly.
    before = process_task_affinities(pid)
    for tid in before:
        os.sched_setaffinity(tid, cpus)
    after = process_task_affinities(pid)
    if set(after) != set(before) or any(value != cpus for value in after.values()):
        raise RuntimeError("whole-process task affinity application was incomplete")
    return after


def restore_process_task_affinities(
    pid: int, affinities: dict[int, set[int]],
) -> None:
    current = process_task_affinities(pid)
    if set(current) != set(affinities):
        raise RuntimeError("trainer task set changed while stopped")
    for tid, cpus in affinities.items():
        os.sched_setaffinity(tid, cpus)
    restored = process_task_affinities(pid)
    if restored != affinities:
        raise RuntimeError("original per-thread affinities were not restored")


def _wait_stopped(pid: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _process_state(pid) in {"T", "t"}:
            return
        time.sleep(0.05)
    raise RuntimeError(f"trainer did not enter stopped state: {pid}")


def _snapshot_matches(heartbeat: dict[str, Any], sidecar: dict[str, Any]) -> bool:
    return (
        float(heartbeat.get("data_epoch", -1))
        == float(sidecar.get("physical_epoch_completed", -2))
        and int(heartbeat.get("updates", -1)) == int(sidecar.get("step", -2))
        and heartbeat.get("scientific_state_sha256")
        == sidecar.get("scientific_state_sha256")
        and heartbeat.get("paired_controller_access") is False
        and heartbeat.get("confirmation20_opened") is False
    )


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare_branch(
    *, source_output: Path, destination: Path, lane_id: str,
    candidate_id: str | None,
) -> None:
    if destination.exists():
        raise RuntimeError(f"isolated branch already exists: {destination}")
    destination.mkdir(parents=True)
    for relative in (
        Path("shared_e0/unsb_common/e0.pt"),
        Path("shared_e0/unsb_common/e0.pt.json"),
        Path("lanes") / lane_id / "full_state_latest.pt",
        Path("lanes") / lane_id / "full_state_latest.pt.json",
    ):
        _copy_file(source_output / relative, destination / relative)
    protocol = source_output / "PAPER_PROTOCOL.json"
    if protocol.is_file():
        _copy_file(protocol, destination / protocol.name)
    if candidate_id is None:
        relative = Path("gates") / f"LANE_AUTHORIZATION_{lane_id}.json"
        _copy_file(source_output / relative, destination / relative)
    else:
        lock = (
            Path("candidate_locks") / candidate_id / "CANDIDATE_LOCK.json"
        )
        authority = (
            Path("gates") / f"CANDIDATE_AUTHORIZATION_{candidate_id}.json"
        )
        _copy_file(source_output / lock, destination / lock)
        _copy_file(source_output / authority, destination / authority)


def component_hashes(checkpoint: Path, repo: Path) -> dict[str, str]:
    sys.path.insert(0, str(Path(repo).resolve()))
    try:
        import torch
        from research.local_route1.runtime import full_state_hash
        from research.paper_aio.gates import scientific_core

        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = payload["model"]
        hashes = {
            "networks": full_state_hash(model["networks"]),
            "optimizers": full_state_hash(model["optimizers"]),
            "schedulers": full_state_hash(model["schedulers"]),
            "model": full_state_hash(model),
            "samplers": full_state_hash(payload["samplers"]),
            "rng": full_state_hash(payload["rng"]),
            "scientific_core": full_state_hash(scientific_core(payload)),
        }
        return hashes
    finally:
        if sys.path and sys.path[0] == str(Path(repo).resolve()):
            sys.path.pop(0)


def _branch_command(
    *, python: Path, repo: Path, output: Path, lane_id: str,
    candidate_id: str | None, manifest: Path, data_root: Path,
    train_view: Path, gpu: int, stop: int, cpus: set[int],
) -> list[str]:
    lane = "candidate" if candidate_id is not None else lane_id
    command = [
        "taskset", "--cpu-list", format_cpu_list(cpus), str(python),
        "-m", "research.paper_aio.run", "--stage", "train", "--lane", lane,
        "--resume", "--engineering-stop-after-updates", str(stop),
        "--output", str(output), "--manifest", str(manifest),
        "--data-root", str(data_root), "--train-view", str(train_view),
        "--gpu", str(gpu),
    ]
    if candidate_id is not None:
        command.extend(["--candidate-id", candidate_id])
    return command


def _proc_sample(pid: int) -> dict[str, int]:
    result = {"cpu_ticks": 0, "read_bytes": 0, "write_bytes": 0,
              "voluntary_context_switches": 0,
              "nonvoluntary_context_switches": 0}
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        fields = stat.read_text(encoding="utf-8").split()
        result["cpu_ticks"] = int(fields[13]) + int(fields[14])
    io = Path(f"/proc/{pid}/io")
    if io.is_file():
        values = {
            key.rstrip(":"): int(value)
            for key, value in (
                line.split() for line in io.read_text(encoding="utf-8").splitlines()
            )
        }
        result["read_bytes"] = values.get("read_bytes", 0)
        result["write_bytes"] = values.get("write_bytes", 0)
    status = Path(f"/proc/{pid}/status")
    if status.is_file():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("voluntary_ctxt_switches:"):
                result["voluntary_context_switches"] = int(line.split()[1])
            elif line.startswith("nonvoluntary_ctxt_switches:"):
                result["nonvoluntary_context_switches"] = int(line.split()[1])
    return result


def _gpu_sample(gpu: int) -> dict[str, int] | None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi", f"--id={gpu}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True, timeout=10,
        ).strip()
        utilization, memory = (int(item.strip()) for item in raw.split(","))
        return {"gpu_utilization_percent": utilization, "gpu_memory_mib": memory}
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def _run_branch(
    command: list[str], *, repo: Path, log: Path, gpu: int,
    sample_seconds: float = 5.0,
) -> dict[str, Any]:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("wb") as handle:
        process = subprocess.Popen(
            command, cwd=repo, stdout=handle, stderr=subprocess.STDOUT,
        )
        proc_start = _proc_sample(process.pid)
        proc_last = proc_start
        gpu_samples: list[dict[str, int]] = []
        while process.poll() is None:
            proc_last = _proc_sample(process.pid)
            sample = _gpu_sample(gpu)
            if sample is not None:
                gpu_samples.append(sample)
            time.sleep(sample_seconds)
        returncode = process.wait()
        proc_end = proc_last
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, command)
    telemetry = {
        "wall_seconds": time.monotonic() - started,
        "gpu_samples": len(gpu_samples),
        "gpu_utilization_mean_percent": (
            sum(item["gpu_utilization_percent"] for item in gpu_samples) / len(gpu_samples)
            if gpu_samples else None
        ),
        "gpu_utilization_peak_percent": (
            max(item["gpu_utilization_percent"] for item in gpu_samples)
            if gpu_samples else None
        ),
        "gpu_memory_peak_mib": (
            max(item["gpu_memory_mib"] for item in gpu_samples)
            if gpu_samples else None
        ),
        "process_start": proc_start,
        "process_end": proc_end,
        "load_average_end": list(os.getloadavg()),
    }
    for key in proc_start:
        if key in proc_end:
            telemetry[f"process_{key}_delta"] = proc_end[key] - proc_start[key]
    return telemetry


def run(args: argparse.Namespace) -> dict[str, Any]:
    source = args.source_output.resolve()
    work = args.work_root.resolve()
    if work.exists():
        raise RuntimeError(f"work root already exists: {work}")
    work.mkdir(parents=True)
    state_path = work / "NUMA_AB_STATE.json"
    source_lane = source / "lanes" / args.lane_id
    heartbeat_path = source_lane / "HEARTBEAT.json"
    sidecar_path = source_lane / "full_state_latest.pt.json"
    checkpoint_path = source_lane / "full_state_latest.pt"
    initial_heartbeat = _read_json(heartbeat_path)
    initial_epoch = int(float(initial_heartbeat["data_epoch"]))
    original_task_affinities = process_task_affinities(args.trainer_pid)
    original_affinity = set(os.sched_getaffinity(args.trainer_pid))
    if any(value != original_affinity for value in original_task_affinities.values()):
        raise RuntimeError("trainer starts with heterogeneous per-thread affinities")
    local_affinity = parse_cpu_list(args.local_cpus)
    if not local_affinity.issubset(original_affinity):
        raise RuntimeError("local NUMA CPU set is outside trainer affinity")
    command = _process_command(args.trainer_pid)
    required_fragment = args.candidate_id or args.lane_id
    if "research.paper_aio.run" not in command or required_fragment not in command:
        raise RuntimeError("trainer PID command identity differs")
    contract = {
        "schema": SCHEMA,
        "status": "WAITING_FOR_NEXT_COMPLETE_EPOCH",
        "control_source_sha256": _sha256(Path(__file__)),
        "trainer_pid": args.trainer_pid,
        "lane_id": args.lane_id,
        "candidate_id": args.candidate_id,
        "initial_epoch": initial_epoch,
        "updates_per_branch": args.updates,
        "original_cpus": format_cpu_list(original_affinity),
        "original_thread_count": len(original_task_affinities),
        "local_cpus": format_cpu_list(local_affinity),
        "minimum_net_saving_fraction": args.minimum_net_saving_fraction,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _atomic_json(work / "NUMA_AB_CONTRACT.json", contract)
    _atomic_json(state_path, contract)
    deadline = time.monotonic() + args.timeout_hours * 3600
    while time.monotonic() < deadline:
        _process_command(args.trainer_pid)
        heartbeat = _read_json(heartbeat_path)
        sidecar = _read_json(sidecar_path)
        if int(float(heartbeat.get("data_epoch", -1))) > initial_epoch:
            if not _snapshot_matches(heartbeat, sidecar):
                raise RuntimeError("boundary heartbeat and sidecar differ")
            break
        time.sleep(args.poll_seconds)
    else:
        raise TimeoutError("next complete epoch did not arrive")

    adopted = False
    paused = False
    guard: subprocess.Popen[bytes] | None = None
    final_affinity = original_affinity
    try:
        os.kill(args.trainer_pid, signal.SIGSTOP)
        _wait_stopped(args.trainer_pid)
        paused = True
        guard_code = (
            "import os,signal,time;time.sleep(" + str(args.failsafe_seconds)
            + ");os.kill(" + str(args.trainer_pid) + ",signal.SIGCONT)"
        )
        guard = subprocess.Popen(
            [sys.executable, "-c", guard_code],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        heartbeat = _read_json(heartbeat_path)
        sidecar = _read_json(sidecar_path)
        if not _snapshot_matches(heartbeat, sidecar):
            raise RuntimeError("paused source heartbeat and sidecar differ")
        source_step = int(sidecar["step"])
        target_step = int(sidecar["target_steps"])
        stop = source_step + args.updates
        if stop >= target_step:
            raise RuntimeError("A/B branch reaches or exceeds final target")
        source_checkpoint_sha = _sha256(checkpoint_path)
        if source_checkpoint_sha != sidecar.get("full_state_sha256"):
            raise RuntimeError("source checkpoint hash differs from sidecar")
        original_root = work / "original"
        local_root = work / "local_numa"
        prepare_branch(
            source_output=source, destination=original_root,
            lane_id=args.lane_id, candidate_id=args.candidate_id,
        )
        prepare_branch(
            source_output=source, destination=local_root,
            lane_id=args.lane_id, candidate_id=args.candidate_id,
        )
        if (
            _sha256(original_root / "lanes" / args.lane_id / "full_state_latest.pt")
            != source_checkpoint_sha
            or _sha256(local_root / "lanes" / args.lane_id / "full_state_latest.pt")
            != source_checkpoint_sha
        ):
            raise RuntimeError("isolated branch source copy differs")
        _atomic_json(state_path, {**contract, "status": "RUNNING_ORIGINAL_BRANCH", "source_step": source_step})
        original_telemetry = _run_branch(
            _branch_command(
                python=args.python.resolve(), repo=args.repo.resolve(),
                output=original_root, lane_id=args.lane_id,
                candidate_id=args.candidate_id, manifest=args.manifest.resolve(),
                data_root=args.data_root.resolve(), train_view=args.train_view.resolve(),
                gpu=args.gpu, stop=stop, cpus=original_affinity,
            ), repo=args.repo.resolve(), log=work / "original.log", gpu=args.gpu,
        )
        _atomic_json(state_path, {**contract, "status": "RUNNING_LOCAL_NUMA_BRANCH", "source_step": source_step})
        local_telemetry = _run_branch(
            _branch_command(
                python=args.python.resolve(), repo=args.repo.resolve(),
                output=local_root, lane_id=args.lane_id,
                candidate_id=args.candidate_id, manifest=args.manifest.resolve(),
                data_root=args.data_root.resolve(), train_view=args.train_view.resolve(),
                gpu=args.gpu, stop=stop, cpus=local_affinity,
            ), repo=args.repo.resolve(), log=work / "local_numa.log", gpu=args.gpu,
        )
        original_seconds = float(original_telemetry["wall_seconds"])
        local_seconds = float(local_telemetry["wall_seconds"])
        original_checkpoint = original_root / "lanes" / args.lane_id / "full_state_latest.pt"
        local_checkpoint = local_root / "lanes" / args.lane_id / "full_state_latest.pt"
        original_hashes = component_hashes(original_checkpoint, args.repo)
        local_hashes = component_hashes(local_checkpoint, args.repo)
        differences = {
            key: {"original": original_hashes[key], "local_numa": local_hashes[key]}
            for key in original_hashes if original_hashes[key] != local_hashes[key]
        }
        timing = net_remaining_saving(
            original_seconds=original_seconds, bound_seconds=local_seconds,
            updates=args.updates, remaining_updates=target_step - source_step,
        )
        source_unchanged = _sha256(checkpoint_path) == source_checkpoint_sha
        adopted = (
            not differences and source_unchanged
            and timing["net_saving_fraction"] >= args.minimum_net_saving_fraction
        )
        final_affinity = local_affinity if adopted else original_affinity
        applied_task_affinities = set_process_task_affinity(
            args.trainer_pid, final_affinity,
        )
        result = {
            **contract,
            "status": "PASS_ADOPT_LOCAL_NUMA" if adopted else "PASS_KEEP_ORIGINAL",
            "boundary_epoch": int(sidecar["physical_epoch_completed"]),
            "source_step": source_step,
            "stop_step": stop,
            "target_step": target_step,
            "source_checkpoint_sha256": source_checkpoint_sha,
            "source_checkpoint_unchanged": source_unchanged,
            "component_hashes_original": original_hashes,
            "component_hashes_local_numa": local_hashes,
            "component_differences": differences,
            "bitwise_transition_equivalent": not differences,
            "original_seconds": original_seconds,
            "local_numa_seconds": local_seconds,
            "original_telemetry": original_telemetry,
            "local_numa_telemetry": local_telemetry,
            "original_updates_per_second": args.updates / original_seconds,
            "local_numa_updates_per_second": args.updates / local_seconds,
            "timing": timing,
            "adopted_affinity": format_cpu_list(final_affinity),
            "adopted_thread_count": len(applied_task_affinities),
            "all_trainer_threads_verified": True,
            "runtime_cohort": (
                "SCIENTIFIC_TRANSITION_EQUIVALENT_OPERATIONAL_AFFINITY_ANNOTATED"
                if adopted else "UNCHANGED"
            ),
            "main_checkpoint_loaded_by_branch_only": True,
            "main_in_memory_state_mutated": False,
            "main_progress_rolled_back": False,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        }
        _atomic_json(work / "NUMA_AB_RECEIPT.json", result)
        _atomic_json(state_path, result)
        return result
    finally:
        if paused:
            try:
                try:
                    if adopted:
                        set_process_task_affinity(args.trainer_pid, final_affinity)
                    else:
                        restore_process_task_affinities(
                            args.trainer_pid, original_task_affinities,
                        )
                finally:
                    os.kill(args.trainer_pid, signal.SIGCONT)
            finally:
                if guard is not None and guard.poll() is None:
                    guard.terminate()
                    try:
                        guard.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        guard.kill()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--source-output", type=Path, required=True)
    value.add_argument("--work-root", type=Path, required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--lane-id", required=True)
    value.add_argument("--candidate-id")
    value.add_argument("--trainer-pid", type=int, required=True)
    value.add_argument("--local-cpus", required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--updates", type=int, default=1000)
    value.add_argument("--minimum-net-saving-fraction", type=float, default=0.10)
    value.add_argument("--poll-seconds", type=float, default=1.0)
    value.add_argument("--timeout-hours", type=float, default=6.0)
    value.add_argument("--failsafe-seconds", type=int, default=7200)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.updates <= 0:
        raise SystemExit("--updates must be positive")
    if not 0 <= args.minimum_net_saving_fraction < 1:
        raise SystemExit("--minimum-net-saving-fraction must be in [0,1)")
    try:
        result = run(args)
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "status": "FAIL_CLOSED_MAIN_RESUMED",
            "error_type": type(error).__name__,
            "error": str(error),
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        }
        try:
            _atomic_json(args.work_root.resolve() / "NUMA_AB_FAILURE.json", failure)
        finally:
            print(json.dumps(failure, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
