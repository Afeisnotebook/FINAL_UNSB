from __future__ import annotations

import json
from pathlib import Path

from research.local_route1 import extended_repaired_frontier as extended
from research.local_route1.frontier_advancement import ALTERNATE, STRICT
from research.local_route1.protocol import file_sha256
from research.local_route1.repaired_frontier_adjudication import RANKABLE_IDS
from research.local_route1.repaired_frontier_followups import REPAIRED_IDS
from research.local_route1.winner_ablations import WINNER_FAMILIES


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_extended_frontier_ranks_both_parent_proposals_and_excludes_observers(
    monkeypatch, tmp_path,
):
    operations = tmp_path / "operations"
    receipts: dict[str, dict] = {}

    def receipt(candidate_id: str, score: float) -> Path:
        path = operations / "terminal_receipts" / f"{candidate_id}.json"
        _write(path, {"candidate_id": candidate_id})
        receipts[str(path.resolve())] = {
            "candidate_id": candidate_id,
            "algorithm_fingerprint": f"algorithm-{candidate_id}",
            "candidate_fingerprint": f"candidate-{candidate_id}",
            "training_git_commit": "a" * 40,
            "trajectory_status": "LONG_HORIZON_POSITIVE_CANDIDATE",
            "trajectory_path": str(tmp_path / f"{candidate_id}.trajectory.json"),
            "trajectory_sha256": "trajectory",
            "ranking_fields": {"score": score},
            "score": score,
            "base_e0_scientific_state_sha256": "e0",
            "base_protocol_fingerprint": "protocol",
            "manifest_sha256": "manifest",
            "plain_e200_verification_sha256": "plain",
        }
        return path

    full_paths = {
        candidate_id: receipt(candidate_id, score)
        for candidate_id, score in zip(RANKABLE_IDS, (0.2, 0.3, 0.1))
    }
    streams = []
    parent_results = []
    proposal_ids = []
    observer_ids = []
    for parent_id, proposal_score in zip(REPAIRED_IDS, (0.9, 0.8)):
        ids = WINNER_FAMILIES[parent_id]["ids"]
        proposal_id = ids["proposal_only"]
        observable_id = ids["observable_only"]
        proposal_ids.append(proposal_id)
        observer_ids.append(observable_id)
        proposal_path = receipt(proposal_id, proposal_score)
        observable_path = receipt(observable_id, -100.0)
        streams.append({
            "parent_candidate_id": parent_id,
            "parent_classification": ALTERNATE,
            "ablation_candidate_ids": ids,
        })
        parent_results.append({
            "parent_candidate_id": parent_id,
            "status": "PARENT_ABLATION_STREAM_COMPLETE_E200",
            "receipts": [
                {
                    "candidate_id": proposal_id,
                    "receipt_path": str(proposal_path.resolve()),
                    "receipt_sha256": file_sha256(proposal_path),
                },
                {
                    "candidate_id": observable_id,
                    "receipt_path": str(observable_path.resolve()),
                    "receipt_sha256": file_sha256(observable_path),
                },
            ],
        })

    repaired_path = operations / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
    _write(repaired_path, {
        "schema": "final-unsb-route1-repaired-frontier-adjudication-v1",
        "ranking": [
            {
                "candidate_id": candidate_id,
                "receipt_path": str(path.resolve()),
                "receipt_sha256": file_sha256(path),
            }
            for candidate_id, path in full_paths.items()
        ],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    followup_path = operations / "REPAIRED_FRONTIER_FOLLOWUPS.json"
    _write(followup_path, {
        "schema": "final-unsb-route1-repaired-frontier-followups-v1",
        "source_adjudication_sha256": file_sha256(repaired_path),
        "eligible_parent_streams": streams,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    _write(operations / "REPAIRED_FOLLOWUP_EXECUTION_RESULT.json", {
        "schema": "final-unsb-route1-repaired-followup-execution-v1",
        "source_plan_sha256": file_sha256(followup_path),
        "parent_results": parent_results,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })

    monkeypatch.setattr(
        extended, "_validate_receipt",
        lambda path: receipts[str(Path(path).resolve())],
    )
    monkeypatch.setattr(extended, "_rank_key", lambda value: (-value["score"],))
    monkeypatch.setattr(
        extended, "_receipt_row",
        lambda path: (
            receipts[str(Path(path).resolve())],
            {"candidate_id": receipts[str(Path(path).resolve())]["candidate_id"]},
            {
                "classification": (
                    STRICT
                    if receipts[str(Path(path).resolve())]["score"] >= 0.5
                    else ALTERNATE
                ),
                "checks": {},
            },
        ),
    )
    monkeypatch.setattr(
        extended, "_observable_identity",
        lambda output_root, candidate_id: {
            "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
            "candidate_id": candidate_id,
        },
    )

    result = extended.materialize_extended_repaired_frontier(tmp_path)
    assert result["action_priority_candidate_id"] == proposal_ids[0]
    assert result["strict_candidate_ids"] == proposal_ids
    assert result["rankable_complete_e200_candidate_count"] == 5
    assert result["observable_only_candidate_ids_excluded_from_ranking"] == observer_ids
    assert all(
        candidate_id not in {row["candidate_id"] for row in result["ranking"]}
        for candidate_id in observer_ids
    )
    assert result["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert len(result["parent_ablation_results"]) == 2
