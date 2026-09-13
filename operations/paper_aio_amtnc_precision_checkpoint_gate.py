"""Close the target-blind AM-TNC e195 precision-recovery checkpoint gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch

from research.local_route1.runtime import assert_finite, full_state_hash


SCHEMA = "final-unsb-paper-amtnc-precision-checkpoint-gate-v1"
FULL_STATE_SCHEMA = "final-unsb-paper-aio-full-state-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
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


def _optimizer_precision(model: dict[str, Any]) -> dict[str, Any]:
    float64 = 0
    float32 = 0
    unsupported: list[str] = []
    for optimizer_index, optimizer in enumerate(model.get("optimizers", [])):
        for state_id, state in optimizer.get("state", {}).items():
            value = state.get("exp_avg_sq")
            if value is None:
                continue
            if not torch.is_tensor(value) or not torch.is_floating_point(value):
                unsupported.append(f"{optimizer_index}:{state_id}:non_floating")
            elif value.dtype == torch.float64:
                float64 += 1
            elif value.dtype == torch.float32:
                float32 += 1
            else:
                unsupported.append(
                    f"{optimizer_index}:{state_id}:{str(value.dtype)}"
                )
    return {
        "float64_exp_avg_sq_tensors": float64,
        "float32_exp_avg_sq_tensors": float32,
        "unsupported_exp_avg_sq": unsupported,
    }


def _validate_checkpoint(
    checkpoint: Path, sidecar_path: Path, *, expected_step: int,
    expected_epoch: int, expected_git_commit: str,
    expected_protocol_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sidecar = _read_json(sidecar_path)
    checkpoint_sha = _sha256(checkpoint)
    if (
        sidecar.get("schema") != FULL_STATE_SCHEMA
        or sidecar.get("lane_id") != "amtnc"
        or int(sidecar.get("step", -1)) != expected_step
        or int(sidecar.get("physical_epoch_completed", -1)) != expected_epoch
        or sidecar.get("full_state_sha256") != checkpoint_sha
    ):
        raise RuntimeError("AM-TNC checkpoint sidecar identity mismatch")
    metadata = sidecar.get("metadata") or {}
    if (
        metadata.get("lane_id") != "amtnc"
        or metadata.get("git_commit") != expected_git_commit
        or metadata.get("protocol_fingerprint")
        != expected_protocol_fingerprint
        or metadata.get("paired_controller_access") is not False
        or metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("AM-TNC checkpoint metadata differs")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != FULL_STATE_SCHEMA
        or int(payload.get("step", -1)) != expected_step
        or int(payload.get("physical_epoch_completed", -1)) != expected_epoch
        or payload.get("metadata") != metadata
    ):
        raise RuntimeError("AM-TNC checkpoint payload identity mismatch")
    if full_state_hash(payload) != sidecar.get("scientific_state_sha256"):
        raise RuntimeError("AM-TNC scientific-state hash differs")
    assert_finite(payload.get("model"), "state.model")
    method = ((payload.get("model") or {}).get("method") or {}).get("amtnc")
    if not isinstance(method, dict):
        raise RuntimeError("AM-TNC method state is absent")
    promotions = int(method.get("second_moment_precision_promotions", 0))
    first = method.get("first_second_moment_precision_promotion")
    if (
        promotions <= 0
        or not isinstance(first, dict)
        or first.get("player") not in {"GF", "DE"}
        or int(first.get("parameter_count", 0)) <= 0
        or not 1_659_282 <= int(first.get("zero_based_update", -1)) < 1_667_835
    ):
        raise RuntimeError("AM-TNC precision promotion was not recorded in e195")
    precision = _optimizer_precision(payload["model"])
    if (
        precision["float64_exp_avg_sq_tensors"] <= 0
        or precision["unsupported_exp_avg_sq"]
    ):
        raise RuntimeError("AM-TNC promoted optimizer state is absent or unsupported")
    return payload, {
        "step": expected_step,
        "physical_epoch_completed": expected_epoch,
        "checkpoint_sha256": checkpoint_sha,
        "sidecar_sha256": _sha256(sidecar_path),
        "scientific_state_sha256": sidecar["scientific_state_sha256"],
        "method_precision_promotions": promotions,
        "first_precision_promotion": first,
        **precision,
        "all_model_and_optimizer_tensors_finite": True,
    }


def close_gate(args: argparse.Namespace) -> dict[str, Any]:
    e195, e195_summary = _validate_checkpoint(
        args.e195_checkpoint.resolve(), args.e195_sidecar.resolve(),
        expected_step=1_667_835, expected_epoch=195,
        expected_git_commit=args.expected_git_commit,
        expected_protocol_fingerprint=args.expected_protocol_fingerprint,
    )
    resume, resume_summary = _validate_checkpoint(
        args.resume_probe_checkpoint.resolve(),
        args.resume_probe_sidecar.resolve(),
        expected_step=1_667_836, expected_epoch=195,
        expected_git_commit=args.expected_git_commit,
        expected_protocol_fingerprint=args.expected_protocol_fingerprint,
    )
    e195_method = e195["model"]["method"]["amtnc"]
    resume_method = resume["model"]["method"]["amtnc"]
    if (
        resume_method["first_second_moment_precision_promotion"]
        != e195_method["first_second_moment_precision_promotion"]
        or int(resume_method["second_moment_precision_promotions"])
        < int(e195_method["second_moment_precision_promotions"])
        or resume_summary["float64_exp_avg_sq_tensors"]
        < e195_summary["float64_exp_avg_sq_tensors"]
    ):
        raise RuntimeError("AM-TNC promoted precision did not survive exact resume")
    result = {
        "schema": SCHEMA,
        "status": "PASS_E195_FINITE_PROMOTION_AND_EXACT_RESUME",
        "e195": e195_summary,
        "resume_probe": resume_summary,
        "precision_promotion_observed": True,
        "promoted_state_dtype": "float64",
        "full_state_reload_preserves_promoted_dtype": True,
        "gradient_clipped": False,
        "optimizer_update_skipped": False,
        "hyperparameter_changed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _atomic_exclusive(args.output.resolve(), result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--e195-checkpoint", type=Path, required=True)
    value.add_argument("--e195-sidecar", type=Path, required=True)
    value.add_argument("--resume-probe-checkpoint", type=Path, required=True)
    value.add_argument("--resume-probe-sidecar", type=Path, required=True)
    value.add_argument("--expected-git-commit", required=True)
    value.add_argument("--expected-protocol-fingerprint", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(close_gate(parser().parse_args()), ensure_ascii=False, indent=2))
