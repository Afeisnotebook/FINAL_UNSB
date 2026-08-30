from __future__ import annotations

import json
from pathlib import Path

import operations.local_route1_frontier_winner_ablation_successor as module
from operations.local_route1_frontier_winner_ablation_successor import (
    FrontierWinnerAblationSuccessor,
    RESULT,
)
from research.local_route1.frontier_final_delivery import FINAL_SELECTION


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _successor(tmp_path: Path) -> FrontierWinnerAblationSuccessor:
    value = object.__new__(FrontierWinnerAblationSuccessor)
    value.run_root = tmp_path
    value.operations = tmp_path / "operations"
    value.operations.mkdir(parents=True)
    value.contract = {"poll_seconds": 60, "python": "/python"}
    value.repo = tmp_path
    value.contract_path = tmp_path / "contract.json"
    value.event = lambda *_args, **_fields: None
    value.state = lambda *_args, **_fields: None
    value.wait_cross_result = lambda: {
        "status": "COMPLETE_REMOTE_FRONTIER_NEGATIVE_4090_REPLAY_SKIPPED",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    return value


def test_retained_base_reuses_only_its_own_complete_ablations(
    tmp_path: Path, monkeypatch,
) -> None:
    successor = _successor(tmp_path)
    base_id = "BASE"
    successor._base_delivery = lambda: (
        {"candidate_id": base_id},
        {"selected_candidate_id": base_id, "winner_ablation_results": {"full": "base"}},
    )
    receipt_path = successor.operations / "terminal_receipts" / f"{base_id}.json"
    _write(receipt_path, {"candidate_id": base_id})
    selection_path = successor.operations / FINAL_SELECTION
    selection = {
        "selected_candidate_id": base_id,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(selection_path, selection)
    ablation_path = successor.operations / "WINNER_ABLATION_ADJUDICATION.json"
    _write(ablation_path, {"candidate_id": base_id, "confirmation20_opened": False})
    monkeypatch.setattr(
        module, "_same_host_selection",
        lambda *_args: (selection, receipt_path, {"candidate_id": base_id}),
    )
    assert successor.run() == 0
    result = json.loads((successor.operations / RESULT).read_text())
    assert result["status"] == "REUSED_PRE_FRONTIER_SELECTED_WINNER_ABLATIONS"
    assert result["selected_candidate_id"] == base_id
    assert result["new_frontier_ablation_e200_executors"] == 0
    assert result["winner_ablation_evidence"] == {"full": "base"}


def test_new_frontier_winner_runs_two_e200_ablations_before_final_selection(
    tmp_path: Path, monkeypatch,
) -> None:
    successor = _successor(tmp_path)
    base_id = "BASE"
    full_id = "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING"
    proposal_id = "ABL-F1-01-PCNR-PROPOSAL-ONLY"
    observable_id = "ABL-F1-01-PCNR-OBSERVABLE-ONLY"
    successor._base_delivery = lambda: (
        {"candidate_id": base_id},
        {"selected_candidate_id": base_id, "winner_ablation_results": {"full": "base"}},
    )
    paths = {}
    for candidate_id in (base_id, full_id, proposal_id, observable_id):
        path = successor.operations / "terminal_receipts" / f"{candidate_id}.json"
        _write(path, {"candidate_id": candidate_id})
        paths[candidate_id] = path
    frontier_selection = {
        "selected_candidate_id": full_id,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    frontier_selection_path = successor.operations / FINAL_SELECTION
    _write(frontier_selection_path, frontier_selection)
    monkeypatch.setattr(
        module, "_same_host_selection",
        lambda *_args: (
            frontier_selection, paths[full_id], {"candidate_id": full_id},
        ),
    )
    monkeypatch.setattr(
        module, "materialize_winner_ablation_definitions",
        lambda *_args, **_kwargs: {
            "ablation_candidate_ids": {
                "proposal_only": proposal_id,
                "observable_only": observable_id,
            }
        },
    )
    calls = []
    successor.run_gates = lambda ids: calls.append(("gates", list(ids)))
    successor.run_e200 = lambda ids: calls.append(("e200", list(ids)))
    monkeypatch.setattr(module, "materialize_receipt", lambda *_args: None)

    def fake_ablation(**kwargs):
        value = {
            "status": "COMPLETE_NO_SELECTION_CHANGE",
            "roles": {"projected_or_full": {"candidate_id": full_id}},
            "paired_metrics_used_for_training_or_control": False,
            "confirmation20_opened": False,
        }
        _write(kwargs["output_path"], value)
        return value

    monkeypatch.setattr(module, "adjudicate_ablations", fake_ablation)

    def fake_rank(_paths, output_path):
        value = {
            "selected_candidate_id": proposal_id,
            "paired_metrics_used_for_training_or_control": False,
            "confirmation20_opened": False,
        }
        _write(output_path, value)
        return value

    monkeypatch.setattr(module, "rank_receipts", fake_rank)
    monkeypatch.setattr(
        module, "_validate_receipt",
        lambda path: {
            "candidate_id": Path(path).stem,
            "algorithm_fingerprint": f"algorithm-{Path(path).stem}",
        },
    )
    assert successor.run() == 0
    assert calls == [
        ("gates", [proposal_id, observable_id]),
        ("e200", [proposal_id, observable_id]),
    ]
    result = json.loads((successor.operations / RESULT).read_text())
    assert result["status"] == "FRONTIER_SELECTED_ALGORITHM_ABLATIONS_COMPLETE"
    assert result["frontier_full_candidate_id"] == full_id
    assert result["selected_candidate_id"] == proposal_id
    assert result["new_frontier_ablation_e200_executors"] == 2
