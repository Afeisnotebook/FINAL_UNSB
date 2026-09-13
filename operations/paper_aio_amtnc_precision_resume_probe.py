"""Run the isolated one-update resume probe for the AM-TNC precision gate.

The live e195 trajectory is read-only.  A fresh output root receives only the
protocol, authorization, e0, and captured e195 full state required to execute
one native training update.  The receipt is published only after the source is
proved unchanged and the probe checkpoint is hash-bound to the repaired code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any


SCHEMA = "final-unsb-paper-amtnc-precision-resume-probe-v1"
FULL_STATE_SCHEMA = "final-unsb-paper-aio-full-state-v1"


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


def _atomic_exclusive(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_json(path) != value:
            raise RuntimeError(f"refusing to replace a different receipt: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True,
    ).strip()


def _validate_source(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.source_checkpoint.resolve()
    sidecar_path = args.source_sidecar.resolve()
    sidecar = _read_json(sidecar_path)
    metadata = sidecar.get("metadata") or {}
    checkpoint_sha = _sha256(checkpoint)
    if (
        sidecar.get("schema") != FULL_STATE_SCHEMA
        or sidecar.get("lane_id") != "amtnc"
        or int(sidecar.get("step", -1)) != args.required_step
        or int(sidecar.get("physical_epoch_completed", -1))
        != args.required_epoch
        or sidecar.get("full_state_sha256") != checkpoint_sha
        or metadata.get("git_commit") != args.required_git_commit
        or metadata.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
        or metadata.get("paired_controller_access") is not False
        or metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("captured AM-TNC source identity mismatch")
    return {
        "checkpoint_sha256": checkpoint_sha,
        "sidecar_sha256": _sha256(sidecar_path),
        "scientific_state_sha256": sidecar.get("scientific_state_sha256"),
    }


def _copy_inputs(args: argparse.Namespace, probe: Path) -> None:
    source = args.source_output.resolve()
    copies = (
        (source / "PAPER_PROTOCOL.json", probe / "PAPER_PROTOCOL.json"),
        (
            source / "gates" / "LANE_AUTHORIZATION_amtnc.json",
            probe / "gates" / "LANE_AUTHORIZATION_amtnc.json",
        ),
        (
            source / "shared_e0" / "unsb_common" / "e0.pt",
            probe / "shared_e0" / "unsb_common" / "e0.pt",
        ),
        (
            source / "shared_e0" / "unsb_common" / "e0.pt.json",
            probe / "shared_e0" / "unsb_common" / "e0.pt.json",
        ),
        (
            args.source_checkpoint.resolve(),
            probe / "lanes" / "amtnc" / "full_state_latest.pt",
        ),
        (
            args.source_sidecar.resolve(),
            probe / "lanes" / "amtnc" / "full_state_latest.pt.json",
        ),
    )
    for origin, destination in copies:
        if not origin.is_file():
            raise RuntimeError(f"required probe input is missing: {origin}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.training_repo.resolve()
    if (
        _git(repo, "rev-parse", "HEAD") != args.required_git_commit
        or _git(repo, "status", "--porcelain")
    ):
        raise RuntimeError("repaired AM-TNC checkout identity changed")
    source = _validate_source(args)
    probe = args.probe_output.resolve()
    if probe.exists():
        raise RuntimeError(f"fresh isolated probe output required: {probe}")
    _copy_inputs(args, probe)
    stop_step = args.required_step + 1
    command = [
        str(args.python.resolve()), "-m", "research.paper_aio.run",
        "--stage", "train", "--lane", "amtnc", "--resume",
        "--engineering-stop-after-updates", str(stop_step),
        "--output", str(probe), "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--train-view", str(args.train_view.resolve()),
        "--gpu", str(args.gpu),
    ]
    environment = dict(os.environ)
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    completed = subprocess.run(
        command, cwd=repo, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, check=False,
    )
    log = probe / "AMTNC_PRECISION_RESUME_PROBE.log"
    log.write_text(completed.stdout, encoding="utf-8", newline="\n")
    if completed.returncode != 0:
        raise RuntimeError(f"AM-TNC precision resume probe failed; see {log}")

    result_checkpoint = probe / "lanes" / "amtnc" / "full_state_latest.pt"
    result_sidecar_path = result_checkpoint.with_suffix(
        result_checkpoint.suffix + ".json"
    )
    result_sidecar = _read_json(result_sidecar_path)
    result_metadata = result_sidecar.get("metadata") or {}
    if (
        result_sidecar.get("schema") != FULL_STATE_SCHEMA
        or result_sidecar.get("lane_id") != "amtnc"
        or int(result_sidecar.get("step", -1)) != stop_step
        or int(result_sidecar.get("physical_epoch_completed", -1))
        != args.required_epoch
        or result_sidecar.get("full_state_sha256") != _sha256(result_checkpoint)
        or result_metadata.get("git_commit") != args.required_git_commit
        or result_metadata.get("protocol_fingerprint")
        != args.required_protocol_fingerprint
        or result_metadata.get("paired_controller_access") is not False
        or result_metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("isolated AM-TNC resume output identity mismatch")
    source_after = _validate_source(args)
    if source_after != source:
        raise RuntimeError("live e195 source changed during isolated resume probe")
    receipt = {
        "schema": SCHEMA,
        "status": "PASS_ISOLATED_E195_PLUS_ONE_UPDATE",
        "captured_unix_time": time.time(),
        "source": source,
        "source_step": args.required_step,
        "source_epoch": args.required_epoch,
        "source_preserved": True,
        "probe_step": stop_step,
        "probe_epoch": int(result_sidecar["physical_epoch_completed"]),
        "probe_checkpoint": str(result_checkpoint),
        "probe_checkpoint_sha256": result_sidecar["full_state_sha256"],
        "probe_sidecar": str(result_sidecar_path),
        "probe_sidecar_sha256": _sha256(result_sidecar_path),
        "probe_scientific_state_sha256": result_sidecar.get(
            "scientific_state_sha256"
        ),
        "log": str(log),
        "log_sha256": _sha256(log),
        "training_git_commit": args.required_git_commit,
        "protocol_fingerprint": args.required_protocol_fingerprint,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _atomic_exclusive(args.output.resolve(), receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--source-output", type=Path, required=True)
    value.add_argument("--source-checkpoint", type=Path, required=True)
    value.add_argument("--source-sidecar", type=Path, required=True)
    value.add_argument("--training-repo", type=Path, required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--probe-output", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--required-step", type=int, required=True)
    value.add_argument("--required-epoch", type=int, required=True)
    value.add_argument("--required-git-commit", required=True)
    value.add_argument("--required-protocol-fingerprint", required=True)
    value.add_argument("--gpu", type=int, default=0)
    return value


if __name__ == "__main__":
    print(json.dumps(run_probe(parser().parse_args()), ensure_ascii=False, indent=2))
