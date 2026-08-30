"""Rank complete candidate receipts without reinterpreting their code identities."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
)
from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_NEGATIVE_STATUS,
)
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.runtime import write_json

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)


SCHEMA = "final-unsb-route1-cross-version-e200-adjudication-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _number(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _rank_key(receipt: dict[str, Any]) -> tuple[Any, ...]:
    values = receipt["ranking_fields"]
    wall = _number(receipt.get("median_epoch_wall_seconds"), default=math.inf)
    return (
        -_number(values.get("late_three_mean_macro_psnr_delta"), default=-math.inf),
        -_number(values.get("e200_macro_psnr_delta"), default=-math.inf),
        -int(values.get("late_points_with_four_of_six_positive_domains", -1)),
        -_number(values.get("late_average_worst_domain_delta"), default=-math.inf),
        _number(
            values.get("candidate_best_to_terminal_three_point_rolling_drawdown"),
            default=math.inf,
        ),
        -_number(values.get("late_mean_macro_ssim_delta"), default=-math.inf),
        _number(values.get("late_mean_macro_lpips_delta"), default=math.inf),
        wall,
        str(receipt["candidate_id"]),
    )


def _validate_receipt(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    sidecar_path = Path(str(path) + ".sha256.json")
    if not path.is_file() or not sidecar_path.is_file():
        raise RuntimeError(f"candidate receipt or sidecar missing: {path}")
    receipt = _read_json(path)
    sidecar = _read_json(sidecar_path)
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("status") != "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT"
        or sidecar.get("schema") != SIDECAR_SCHEMA
        or sidecar.get("candidate_id") != receipt.get("candidate_id")
        or sidecar.get("receipt_sha256") != file_sha256(path)
    ):
        raise RuntimeError(f"candidate terminal receipt integrity failed: {path}")
    expected_receipt_source = file_sha256(
        ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
    )
    if receipt.get("receipt_source_sha256") != expected_receipt_source:
        raise RuntimeError("candidate receipt was not produced by the frozen receipt verifier")
    if receipt.get("trajectory_status") not in (POSITIVE_STATUS, NEGATIVE_STATUS):
        raise RuntimeError("candidate receipt has no complete scientific status")
    if receipt.get("terminal_integrity", {}).get("status") != (
        "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"
    ):
        raise RuntimeError("candidate receipt has no accepted terminal artifact set")
    for key in (
        "evaluation_crn_matched_to_same_host_plain",
        "paired_metrics_used_only_after_complete_trajectory",
    ):
        if receipt.get(key) is not True:
            raise RuntimeError(f"candidate receipt requires {key}=true")
    for key in ("paired_metrics_used_for_training_or_control", "confirmation20_opened"):
        if receipt.get(key) is not False:
            raise RuntimeError(f"candidate receipt requires {key}=false")
    return receipt


def adjudicate(receipt_paths: Iterable[Path], output_path: Path) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in receipt_paths]
    if len(paths) < 1:
        raise ValueError("at least one terminal receipt is required")
    receipts = [_validate_receipt(path) for path in paths]
    ids = [str(value["candidate_id"]) for value in receipts]
    if len(set(ids)) != len(ids):
        raise RuntimeError("cross-version adjudication received duplicate candidate ids")
    common_fields = (
        "base_e0_scientific_state_sha256",
        "base_protocol_fingerprint",
        "manifest_sha256",
        "plain_e200_verification_sha256",
    )
    for field in common_fields:
        if len({str(value.get(field)) for value in receipts}) != 1:
            raise RuntimeError(f"candidate receipts differ on authoritative baseline: {field}")

    ranked = sorted(receipts, key=_rank_key)
    eligible = [value for value in ranked if value["trajectory_status"] == POSITIVE_STATUS]
    winner = eligible[0] if eligible else ranked[0]
    result = {
        "schema": SCHEMA,
        "status": (
            "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"
            if eligible else
            CROSS_VERSION_NEGATIVE_STATUS
        ),
        "ranking": [
            {
                "rank": index,
                "candidate_id": value["candidate_id"],
                "trajectory_status": value["trajectory_status"],
                "algorithm_fingerprint": value["algorithm_fingerprint"],
                "candidate_fingerprint": value["candidate_fingerprint"],
                "training_git_commit": value["training_git_commit"],
                "candidate_training_core_fingerprint": value[
                    "candidate_training_core_fingerprint"
                ],
                "trajectory_sha256": value["trajectory_sha256"],
                "ranking_fields": value["ranking_fields"],
                "median_epoch_wall_seconds": value["median_epoch_wall_seconds"],
                "terminal_integrity": value["terminal_integrity"],
            }
            for index, value in enumerate(ranked, start=1)
        ],
        "selected_candidate_id": winner["candidate_id"],
        "selected_training_git_commit": winner["training_git_commit"],
        "selected_algorithm_fingerprint": winner["algorithm_fingerprint"],
        "selected_candidate_fingerprint": winner["candidate_fingerprint"],
        "selection_role": "seed2026_numeric_winner" if eligible else "current_best_fallback",
        "winner_not_loaded_under_a_different_training_core": True,
        "seed_freeze_performed": False,
        "paired_metrics_used_only_after_complete_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_path).resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        adjudicate(args.receipt, args.output), ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
