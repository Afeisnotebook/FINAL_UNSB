from __future__ import annotations

import copy

import pytest

from research.local_route1.frontier_advancement import NEAR, STRICT
from research.local_route1.repaired_replay_portfolio import (
    MAXIMUM_REPLAYS,
    REPAIRED_IDS,
    SCHEMA,
    STATUS,
    _json_file_equivalent_sha256,
    validate_portable_authority,
)


def _row(candidate_id: str, classification: str = STRICT) -> dict:
    card = {
        "schema": "final-unsb-route1-derivation-card-v1",
        "candidate_id": candidate_id,
    }
    implementation = {
        "schema": "final-unsb-route1-candidate-implementation-v1",
        "candidate_id": candidate_id,
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
    }
    return {
        "source_rank": 1,
        "candidate_id": candidate_id,
        "classification": classification,
        "algorithm_fingerprint": "a" * 64,
        "remote_candidate_fingerprint": "b" * 64,
        "remote_training_git_commit": "c" * 40,
        "remote_receipt_sha256": "d" * 64,
        "remote_trajectory_sha256": "e" * 64,
        "derivation_card_sha256": _json_file_equivalent_sha256(card),
        "implementation_sha256": _json_file_equivalent_sha256(implementation),
        "derivation_card": card,
        "implementation": implementation,
        "source_ranking_fields": {},
    }


def _authority(rows: list[dict] | None = None) -> dict:
    return {
        "schema": SCHEMA,
        "status": STATUS,
        "source_adjudication_sha256": "f" * 64,
        "source_same_host_authority": {},
        "source_action_priority_candidate_id": REPAIRED_IDS[0],
        "replay_candidates": [_row(REPAIRED_IDS[0])] if rows is None else rows,
        "complete_source_e200_only": True,
        "maximum_4090_replays": MAXIMUM_REPLAYS,
        "selection_seeds": [2026],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_only_after_complete_e200_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_portable_replay_preserves_two_independent_repaired_algorithms() -> None:
    value = _authority([
        _row(REPAIRED_IDS[0], STRICT),
        _row(REPAIRED_IDS[1], NEAR),
    ])
    assert validate_portable_authority(value) is value
    assert [row["candidate_id"] for row in value["replay_candidates"]] == list(
        REPAIRED_IDS
    )


def test_portable_replay_rejects_single_winner_policy_expansion() -> None:
    rows = [_row(REPAIRED_IDS[index % 2]) for index in range(3)]
    with pytest.raises(RuntimeError, match="two-candidate cap"):
        validate_portable_authority(_authority(rows))


def test_portable_replay_rejects_paired_control_and_unqualified_rows() -> None:
    value = _authority()
    value["paired_controller_access"] = True
    with pytest.raises(RuntimeError, match="paired_controller_access"):
        validate_portable_authority(value)
    value = _authority([_row(REPAIRED_IDS[0], "evidence_backed_alternate")])
    with pytest.raises(RuntimeError, match="strict or near"):
        validate_portable_authority(value)


def test_portable_replay_artifacts_are_content_bound() -> None:
    value = _authority()
    changed = copy.deepcopy(value)
    changed["replay_candidates"][0]["implementation"]["model"] = "changed"
    with pytest.raises(RuntimeError, match="implementation changed"):
        validate_portable_authority(changed)

