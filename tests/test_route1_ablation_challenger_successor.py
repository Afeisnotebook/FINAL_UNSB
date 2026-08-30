import copy

import pytest

from operations.local_route1_ablation_challenger_successor import (
    CHALLENGER_STATUS,
    challenger_from_adjudication,
    materialize_challenger_seed_workspace,
)
from operations.local_route1_winner_ablation_adjudicate import SCHEMA


def _record(status=CHALLENGER_STATUS):
    return {
        "schema": SCHEMA,
        "status": status,
        "roles": {
            "proposal_only": {"candidate_id": "G1-PROPOSAL"},
            "observable_only": {"candidate_id": "G1-OBSERVABLE"},
            "projected_or_full": {"candidate_id": "G1-FULL"},
        },
        "proposal_only_out_ranks_full": status == CHALLENGER_STATUS,
        "selection_change_blocked_pending_seed_validation": (
            status == CHALLENGER_STATUS
        ),
        "selection_changed": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_challenger_successor_routes_only_fail_closed_proposal():
    assert challenger_from_adjudication(_record()) == "G1-PROPOSAL"
    assert challenger_from_adjudication(_record("COMPLETE_NO_SELECTION_CHANGE")) is None


@pytest.mark.parametrize(
    "key,value,match",
    [
        ("selection_changed", True, "fail-closed"),
        ("proposal_only_out_ranks_full", False, "fail-closed"),
        ("paired_controller_access", True, "paired_controller_access"),
        ("confirmation20_opened", True, "confirmation20_opened"),
    ],
)
def test_challenger_successor_rejects_unsafe_or_contradictory_record(
    key, value, match,
):
    record = copy.deepcopy(_record())
    record[key] = value
    with pytest.raises(RuntimeError, match=match):
        challenger_from_adjudication(record)


def test_challenger_seed_workspace_is_hash_bound_and_separate(tmp_path):
    source = tmp_path / "source"
    candidate_id = "G1-PROPOSAL"
    relatives = [
        "audit/LONG_CAUSAL_MATRIX.json",
        "audit/LONG_REVERSAL_ATLAS.jsonl",
        "derive/HYPOTHESIS_LEDGER.json",
        f"derive/cards/{candidate_id}.json",
        f"derive/implementations/{candidate_id}.json",
        f"derive/gates/{candidate_id}.json",
        "shared_e0/e0.pt",
        "shared_e0/e0.pt.json",
        f"candidates/{candidate_id}/CANDIDATE_TRAJECTORY.json",
    ]
    for index, relative in enumerate(relatives):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"authority-{index}".encode())
    workspace = tmp_path / "workspace"
    first = materialize_challenger_seed_workspace(source, workspace, candidate_id)
    second = materialize_challenger_seed_workspace(source, workspace, candidate_id)
    assert first == second
    assert not (workspace / "seed_validation").exists()

    (source / relatives[0]).write_text("changed")
    with pytest.raises(RuntimeError, match="identity changed"):
        materialize_challenger_seed_workspace(source, workspace, candidate_id)
