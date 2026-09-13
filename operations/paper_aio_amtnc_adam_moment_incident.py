"""Close a repeated AM-TNC Adam second-moment overflow without metrics.

The operation is read-only with respect to the training run.  It verifies that
the last complete checkpoint is finite and hash-bound, then counts identical
failures in the supervisor log and writes a compact incident receipt outside
the source lane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from research.local_route1.runtime import (
    assert_finite,
    file_sha256,
    full_state_hash,
    write_json,
)


EXCEPTION_PATH = "state.optimizers[0].state.2.exp_avg_sq"
EXCEPTION_LINE = f"RuntimeError: non-finite tensor in {EXCEPTION_PATH}"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def repeated_failure_count(log_text: str) -> int:
    return sum(line.strip() == EXCEPTION_LINE for line in log_text.splitlines())


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--supervisor-log", type=Path, required=True)
    parser.add_argument("--supervisor-state", type=Path, required=True)
    parser.add_argument("--guard-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--required-checkpoint-sha256", required=True)
    parser.add_argument("--required-scientific-state-sha256", required=True)
    parser.add_argument("--required-git-commit", required=True)
    parser.add_argument("--required-protocol-fingerprint", required=True)
    parser.add_argument("--required-epoch", type=int, required=True)
    parser.add_argument("--minimum-repeat-count", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    checkpoint = args.checkpoint.resolve()
    sidecar_path = Path(str(checkpoint) + ".json")
    log_path = args.supervisor_log.resolve()
    supervisor_state_path = args.supervisor_state.resolve()
    guard_state_path = args.guard_state.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"fresh incident output required: {output}")

    checkpoint_before = file_sha256(checkpoint)
    log_before = file_sha256(log_path)
    if checkpoint_before != args.required_checkpoint_sha256:
        raise RuntimeError("last finite checkpoint hash changed")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    sidecar = _read_json(sidecar_path)
    if (
        int(payload.get("physical_epoch_completed", -1)) != args.required_epoch
        or full_state_hash(payload) != args.required_scientific_state_sha256
        or sidecar.get("full_state_sha256") != checkpoint_before
        or sidecar.get("scientific_state_sha256")
        != args.required_scientific_state_sha256
        or sidecar.get("metadata") != payload.get("metadata")
        or payload.get("metadata", {}).get("git_commit")
        != args.required_git_commit
        or payload.get("metadata", {}).get("protocol_fingerprint")
        != args.required_protocol_fingerprint
    ):
        raise RuntimeError("last finite checkpoint identity changed")
    assert_finite(payload["model"])

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    repeat_count = repeated_failure_count(log_text)
    if repeat_count < args.minimum_repeat_count:
        raise RuntimeError("Adam second-moment failure has not repeated")
    supervisor_state = _read_json(supervisor_state_path)
    guard_state = _read_json(guard_state_path)
    checkpoint_after = file_sha256(checkpoint)
    log_after = file_sha256(log_path)
    if checkpoint_after != checkpoint_before or log_after != log_before:
        raise RuntimeError("source changed during incident capture; retry read-only audit")

    receipt = {
        "schema": "final-unsb-paper-amtnc-adam-second-moment-incident-v1",
        "status": "REPEATED_ADAM_SECOND_MOMENT_FLOAT32_OVERFLOW",
        "repeat_count": repeat_count,
        "exception_path": EXCEPTION_PATH,
        "last_finite_epoch": args.required_epoch,
        "last_finite_step": int(payload["step"]),
        "last_finite_checkpoint_sha256": checkpoint_before,
        "last_finite_scientific_state_sha256": (
            args.required_scientific_state_sha256
        ),
        "last_finite_state_all_finite": True,
        "training_git_commit": args.required_git_commit,
        "protocol_fingerprint": args.required_protocol_fingerprint,
        "supervisor_status": supervisor_state.get("status"),
        "supervisor_consecutive_failures": supervisor_state.get(
            "consecutive_failures"
        ),
        "guard_status": guard_state.get("status"),
        "guard_total_restarts": guard_state.get("total_restarts"),
        "guard_consecutive_no_progress_restarts": guard_state.get(
            "consecutive_no_progress_restarts"
        ),
        "supervisor_log_sha256": log_before,
        "source_checkpoint_unchanged": True,
        "source_log_unchanged_during_capture": True,
        "classification": "IMPLEMENTATION_NUMERICAL_REPRESENTATION_FAILURE_NOT_PERFORMANCE_ADJUDICATION",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "scientific_lane_modified": False,
    }
    output.mkdir(parents=True)
    write_json(output / "AMTNC_ADAM_SECOND_MOMENT_INCIDENT.json", receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
