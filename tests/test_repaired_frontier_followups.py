from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from research.local_route1.frontier_advancement import NEAR, STRICT
from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.repaired_frontier_adjudication import (
    ACTIONABLE_STATUS,
    RANKABLE_IDS,
    SCHEMA as ADJUDICATION_SCHEMA,
)
from research.local_route1.repaired_frontier_followups import (
    REPAIRED_IDS,
    materialize_repaired_frontier_followups,
)
from research.local_route1.winner_ablations import WINNER_FAMILIES


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _parent(tmp_path: Path, candidate_id: str, fingerprint: str) -> tuple[Path, dict]:
    source = (
        ROOT / "research" / "local_route1" / "derivation_cards"
        / f"{candidate_id}.json"
    )
    card = json.loads(source.read_text(encoding="utf-8"))
    card_path = tmp_path / "derive" / "cards" / source.name
    _write(card_path, card)
    receipt_path = (
        tmp_path / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    )
    receipt = {
        "candidate_id": candidate_id,
        "algorithm_fingerprint": fingerprint,
        "derivation_card_sha256": file_sha256(card_path),
    }
    _write(receipt_path, receipt)
    return receipt_path, receipt


def test_repaired_followups_preserve_both_eligible_repairs(monkeypatch, tmp_path):
    _write(tmp_path / "derive" / "HYPOTHESIS_LEDGER.json", {
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [],
    })
    rows = []
    for rank, candidate_id in enumerate(RANKABLE_IDS, start=1):
        fingerprint = f"algorithm-{rank}"
        receipt_path, _ = _parent(tmp_path, candidate_id, fingerprint)
        rows.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "classification": (
                STRICT if candidate_id == REPAIRED_IDS[0]
                else NEAR if candidate_id == REPAIRED_IDS[1]
                else "evidence_backed_alternate"
            ),
            "trajectory_status": "LONG_HORIZON_POSITIVE_CANDIDATE",
            "algorithm_fingerprint": fingerprint,
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
        })
    adjudication_path = (
        tmp_path / "operations" / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
    )
    _write(adjudication_path, {
        "schema": ADJUDICATION_SCHEMA,
        "status": ACTIONABLE_STATUS,
        "ranking": rows,
        "action_priority_candidate_id": REPAIRED_IDS[0],
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "canonical_candidate_is_action_priority_only": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })

    def fake_freeze(output_root: Path, candidate_id: str):
        ledger_path = Path(output_root) / "derive" / "HYPOTHESIS_LEDGER.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        next(row for row in ledger["records"] if row["candidate_id"] == candidate_id)[
            "status"
        ] = "FROZEN_FOR_GATES"
        _write(ledger_path, ledger)
        return SimpleNamespace(to_dict=lambda: {"candidate_id": candidate_id})

    monkeypatch.setattr(
        "research.local_route1.winner_ablations.freeze_candidate_derivation",
        fake_freeze,
    )
    result = materialize_repaired_frontier_followups(tmp_path)
    assert result["eligible_parent_count"] == 2
    assert result["action_priority_is_not_an_exclusivity_rule"] is True
    assert result["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert [row["parent_candidate_id"] for row in result["eligible_parent_streams"]] == list(
        REPAIRED_IDS
    )
    for row in result["eligible_parent_streams"]:
        expected = WINNER_FAMILIES[row["parent_candidate_id"]]["ids"]
        assert row["ablation_candidate_ids"] == expected
        assert row["execution_order_within_stream"] == [
            expected["proposal_only"], expected["observable_only"],
        ]


def test_repaired_followups_do_not_promote_closed_repairs(monkeypatch, tmp_path):
    _write(tmp_path / "derive" / "HYPOTHESIS_LEDGER.json", {
        "schema": "final-unsb-route1-hypothesis-ledger-v1", "records": [],
    })
    rows = []
    for rank, candidate_id in enumerate(RANKABLE_IDS, start=1):
        receipt_path, _ = _parent(tmp_path, candidate_id, f"algorithm-{rank}")
        rows.append({
            "rank": rank,
            "candidate_id": candidate_id,
            "classification": "closed_current_operator",
            "trajectory_status": "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION",
            "algorithm_fingerprint": f"algorithm-{rank}",
            "receipt_path": str(receipt_path),
            "receipt_sha256": file_sha256(receipt_path),
        })
    adjudication_path = (
        tmp_path / "operations" / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
    )
    _write(adjudication_path, {
        "schema": ADJUDICATION_SCHEMA,
        "status": "REPAIRED_FRONTIER_COMPLETE_FALLBACK_ONLY",
        "ranking": rows,
        "action_priority_candidate_id": RANKABLE_IDS[0],
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "canonical_candidate_is_action_priority_only": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    result = materialize_repaired_frontier_followups(tmp_path)
    assert result["eligible_parent_count"] == 0
    assert result["status"] == "NO_REPAIRED_PARENT_ELIGIBLE_FOR_ABLATION_LONG_RUN"
