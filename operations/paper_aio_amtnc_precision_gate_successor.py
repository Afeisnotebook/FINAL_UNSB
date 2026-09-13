"""Durably close the AM-TNC e195 precision gate after hash-bound capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


SCHEMA = "final-unsb-paper-amtnc-precision-gate-successor-v1"
CAPTURE_SCHEMA = "final-unsb-paper-transient-checkpoint-capture-v1"
PROBE_SCHEMA = "final-unsb-paper-amtnc-precision-resume-probe-v1"
GATE_SCHEMA = "final-unsb-paper-amtnc-precision-checkpoint-gate-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "status": "FROZEN",
        "capture_state": str(args.capture_state.resolve()),
        "capture_receipt": str(args.capture_receipt.resolve()),
        "e195_checkpoint": str(args.e195_checkpoint.resolve()),
        "resume_probe_script": str(args.resume_probe_script.resolve()),
        "resume_probe_script_sha256": args.required_resume_probe_script_sha256,
        "checkpoint_gate_script": str(args.checkpoint_gate_script.resolve()),
        "checkpoint_gate_script_sha256": (
            args.required_checkpoint_gate_script_sha256
        ),
        "source_output": str(args.source_output.resolve()),
        "training_repo": str(args.training_repo.resolve()),
        "python": str(args.python.resolve()),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "probe_output": str(args.probe_output.resolve()),
        "required_step": int(args.required_step),
        "required_epoch": int(args.required_epoch),
        "required_git_commit": args.required_git_commit,
        "required_protocol_fingerprint": args.required_protocol_fingerprint,
        "gpu": int(args.gpu),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _freeze(path: Path, contract: dict[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != contract:
            raise RuntimeError("precision successor contract changed")
        return
    _atomic_json(path, contract)


def _validate_scripts(args: argparse.Namespace) -> None:
    checks = (
        (args.resume_probe_script, args.required_resume_probe_script_sha256),
        (args.checkpoint_gate_script, args.required_checkpoint_gate_script_sha256),
    )
    for path, expected in checks:
        if not path.resolve().is_file() or _sha256(path.resolve()) != expected:
            raise RuntimeError(f"deployed gate script identity mismatch: {path}")


def _capture_ready(args: argparse.Namespace) -> dict[str, Any] | None:
    if not args.capture_state.is_file() or not args.capture_receipt.is_file():
        return None
    state = _read_json(args.capture_state)
    receipt = _read_json(args.capture_receipt)
    if state != receipt:
        raise RuntimeError("capture state and publish-last receipt differ")
    checkpoint = args.e195_checkpoint.resolve()
    sidecar = Path(str(checkpoint) + ".json")
    if (
        receipt.get("schema") != CAPTURE_SCHEMA
        or receipt.get("status") != "COMPLETE_HASH_BOUND_CAPTURE"
        or int(receipt.get("epoch", -1)) != args.required_epoch
        or int(receipt.get("step", -1)) != args.required_step
        or Path(receipt.get("destination_checkpoint", "")).resolve() != checkpoint
        or not checkpoint.is_file()
        or not sidecar.is_file()
        or receipt.get("checkpoint_sha256") != _sha256(checkpoint)
        or receipt.get("sidecar_sha256") != _sha256(sidecar)
        or receipt.get("source_unchanged_during_capture") is not True
        or receipt.get("performance_values_read") is not False
        or receipt.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("hash-bound e195 capture receipt is invalid")
    return receipt


def _run(command: list[str], *, cwd: Path, log: Path) -> None:
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    log.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        raise RuntimeError(f"precision successor command failed; see {log}")


def run(args: argparse.Namespace) -> int:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contract = _contract(args)
    contract_path = output / "PRECISION_GATE_SUCCESSOR_CONTRACT.json"
    state_path = output / "PRECISION_GATE_SUCCESSOR_STATE.json"
    receipt_path = output / "PRECISION_GATE_SUCCESSOR_RECEIPT.json"
    probe_receipt_path = output / "PRECISION_RESUME_PROBE_RECEIPT.json"
    gate_receipt_path = output / "PRECISION_CHECKPOINT_GATE_RECEIPT.json"
    _freeze(contract_path, contract)
    _validate_scripts(args)
    started = time.monotonic()
    try:
        capture = None
        while time.monotonic() - started < args.timeout_hours * 3600:
            capture = _capture_ready(args)
            if capture is not None:
                break
            _atomic_json(state_path, {
                **contract, "status": "WAITING_FOR_HASH_BOUND_E195",
                "pid": os.getpid(),
            })
            time.sleep(args.poll_seconds)
        if capture is None:
            raise RuntimeError("precision successor timed out waiting for e195")

        if not probe_receipt_path.is_file():
            _atomic_json(state_path, {
                **contract, "status": "RUNNING_ISOLATED_RESUME_PROBE",
                "pid": os.getpid(), "capture": capture,
            })
            command = [
                str(args.python.resolve()), str(args.resume_probe_script.resolve()),
                "--source-output", str(args.source_output.resolve()),
                "--source-checkpoint", str(args.e195_checkpoint.resolve()),
                "--source-sidecar", str(Path(str(args.e195_checkpoint.resolve()) + ".json")),
                "--training-repo", str(args.training_repo.resolve()),
                "--python", str(args.python.resolve()),
                "--manifest", str(args.manifest.resolve()),
                "--data-root", str(args.data_root.resolve()),
                "--train-view", str(args.train_view.resolve()),
                "--probe-output", str(args.probe_output.resolve()),
                "--output", str(probe_receipt_path),
                "--required-step", str(args.required_step),
                "--required-epoch", str(args.required_epoch),
                "--required-git-commit", args.required_git_commit,
                "--required-protocol-fingerprint",
                args.required_protocol_fingerprint,
                "--gpu", str(args.gpu),
            ]
            _run(command, cwd=args.training_repo.resolve(), log=output / "probe.log")
        probe = _read_json(probe_receipt_path)
        if (
            probe.get("schema") != PROBE_SCHEMA
            or probe.get("status") != "PASS_ISOLATED_E195_PLUS_ONE_UPDATE"
            or probe.get("source_preserved") is not True
            or int(probe.get("probe_step", -1)) != args.required_step + 1
        ):
            raise RuntimeError("isolated precision resume receipt is invalid")

        if not gate_receipt_path.is_file():
            _atomic_json(state_path, {
                **contract, "status": "RUNNING_PRECISION_CHECKPOINT_GATE",
                "pid": os.getpid(), "probe": probe,
            })
            command = [
                str(args.python.resolve()),
                str(args.checkpoint_gate_script.resolve()),
                "--e195-checkpoint", str(args.e195_checkpoint.resolve()),
                "--e195-sidecar", str(Path(str(args.e195_checkpoint.resolve()) + ".json")),
                "--resume-probe-checkpoint", probe["probe_checkpoint"],
                "--resume-probe-sidecar", probe["probe_sidecar"],
                "--expected-git-commit", args.required_git_commit,
                "--expected-protocol-fingerprint",
                args.required_protocol_fingerprint,
                "--output", str(gate_receipt_path),
            ]
            _run(command, cwd=args.training_repo.resolve(), log=output / "gate.log")
        gate = _read_json(gate_receipt_path)
        if (
            gate.get("schema") != GATE_SCHEMA
            or gate.get("status")
            != "PASS_E195_FINITE_PROMOTION_AND_EXACT_RESUME"
            or gate.get("precision_promotion_observed") is not True
            or gate.get("full_state_reload_preserves_promoted_dtype") is not True
        ):
            raise RuntimeError("AM-TNC precision checkpoint gate did not pass")
        result = {
            **contract,
            "status": "COMPLETE_E195_PRECISION_RECOVERY_GATE",
            "captured_unix_time": time.time(),
            "capture_receipt_sha256": _sha256(args.capture_receipt.resolve()),
            "probe_receipt": str(probe_receipt_path),
            "probe_receipt_sha256": _sha256(probe_receipt_path),
            "checkpoint_gate_receipt": str(gate_receipt_path),
            "checkpoint_gate_receipt_sha256": _sha256(gate_receipt_path),
            "live_training_signalled": False,
            "live_checkpoint_preserved": True,
        }
        _atomic_json(receipt_path, result)
        _atomic_json(state_path, result)
        return 0
    except Exception as error:
        _atomic_json(state_path, {
            **contract, "status": "FAILED_CLOSED", "pid": os.getpid(),
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in (
        "capture_state", "capture_receipt", "e195_checkpoint",
        "resume_probe_script", "checkpoint_gate_script", "source_output",
        "training_repo", "python", "manifest", "data_root", "train_view",
        "probe_output", "output",
    ):
        value.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    value.add_argument("--required-resume-probe-script-sha256", required=True)
    value.add_argument("--required-checkpoint-gate-script-sha256", required=True)
    value.add_argument("--required-step", type=int, required=True)
    value.add_argument("--required-epoch", type=int, required=True)
    value.add_argument("--required-git-commit", required=True)
    value.add_argument("--required-protocol-fingerprint", required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=float, default=2.0)
    value.add_argument("--timeout-hours", type=float, default=24.0)
    return value


if __name__ == "__main__":
    raise SystemExit(run(parser().parse_args()))
