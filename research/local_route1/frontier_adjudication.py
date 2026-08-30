"""Adjudicate the complete same-host route-1 frontier without cross-host deltas."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from operations.local_route1_cross_version_adjudicate import (
    _rank_key,
    _validate_receipt,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-frontier-e200-adjudication-v1"
REPLAY_STATUS = "FRONTIER_SAME_HOST_4090_REPLAY_RECOMMENDED"
NO_REPLAY_STATUS = "FRONTIER_COMPLETE_NO_4090_REPLAY"
FRONTIER_IDS = (
    "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING",
    "F1-02-ADAM-METRIC-MOVING-COVARIANCE-BARRIER",
)


def adjudicate_frontier(
    receipt_paths: Iterable[Path], output_path: Path,
) -> dict[str, Any]:
    paths = [Path(path).resolve() for path in receipt_paths]
    if len(paths) != len(FRONTIER_IDS):
        raise RuntimeError("frontier adjudication requires exactly two receipts")
    receipts = [_validate_receipt(path) for path in paths]
    ids = [str(receipt["candidate_id"]) for receipt in receipts]
    if set(ids) != set(FRONTIER_IDS) or len(set(ids)) != len(FRONTIER_IDS):
        raise RuntimeError("frontier receipt identities differ from the frozen frontier")

    common_fields = (
        "base_e0_scientific_state_sha256",
        "base_protocol_fingerprint",
        "manifest_sha256",
        "plain_e200_verification_sha256",
    )
    authority: dict[str, str] = {}
    for field in common_fields:
        values = {str(receipt.get(field, "")) for receipt in receipts}
        if len(values) != 1 or not next(iter(values)):
            raise RuntimeError(f"frontier receipts differ on same-host authority: {field}")
        authority[field] = next(iter(values))

    ranked = sorted(receipts, key=_rank_key)
    strict_passes = [
        receipt for receipt in ranked
        if receipt.get("trajectory_status") == POSITIVE_STATUS
    ]
    selected = strict_passes[0] if strict_passes else ranked[0]
    replay = strict_passes[0] if strict_passes else None
    result = {
        "schema": SCHEMA,
        "status": REPLAY_STATUS if replay is not None else NO_REPLAY_STATUS,
        "selection_role": (
            "same_host_5090_strict_gate_winner_pending_4090_replay"
            if replay is not None else "same_host_5090_current_best_fallback"
        ),
        "same_host_authority": authority,
        "ranking": [
            {
                "rank": index,
                "candidate_id": receipt["candidate_id"],
                "trajectory_status": receipt["trajectory_status"],
                "algorithm_fingerprint": receipt["algorithm_fingerprint"],
                "candidate_fingerprint": receipt["candidate_fingerprint"],
                "training_git_commit": receipt["training_git_commit"],
                "candidate_training_core_fingerprint": receipt[
                    "candidate_training_core_fingerprint"
                ],
                "trajectory_sha256": receipt["trajectory_sha256"],
                "receipt_path": str(path),
                "receipt_sha256": file_sha256(path),
                "ranking_fields": receipt["ranking_fields"],
                "median_epoch_wall_seconds": receipt["median_epoch_wall_seconds"],
            }
            for index, (receipt, path) in enumerate(
                sorted(zip(receipts, paths), key=lambda pair: _rank_key(pair[0])),
                start=1,
            )
        ],
        "selected_frontier_candidate_id": selected["candidate_id"],
        "selected_frontier_algorithm_fingerprint": selected["algorithm_fingerprint"],
        "selected_frontier_candidate_fingerprint": selected["candidate_fingerprint"],
        "strict_gate_pass_candidate_ids": [
            receipt["candidate_id"] for receipt in strict_passes
        ],
        "recommended_4090_replay_candidate_id": (
            None if replay is None else replay["candidate_id"]
        ),
        "recommended_4090_replay_algorithm_fingerprint": (
            None if replay is None else replay["algorithm_fingerprint"]
        ),
        "only_complete_strict_gate_can_trigger_replay": True,
        "intermediate_metrics_used_for_routing": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_path).resolve(), result)
    return result

