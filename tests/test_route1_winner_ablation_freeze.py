from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.local_route1.protocol import ROOT, file_sha256
from research.local_route1.winner_ablations import (
    WINNER_FAMILIES,
    materialize_winner_ablation_definitions,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_winner_ablation_freeze_materializes_only_selected_family(monkeypatch, tmp_path):
    parent_id = "G1-01-ROLLOUT-DISTRIBUTION-SPEED"
    parent_card = json.loads((
        ROOT / "research" / "local_route1" / "derivation_cards" / f"{parent_id}.json"
    ).read_text(encoding="utf-8"))
    parent_card_path = tmp_path / "derive" / "cards" / f"{parent_id}.json"
    _write(parent_card_path, parent_card)
    receipt_path = tmp_path / "operations" / "terminal_receipts" / f"{parent_id}.json"
    _write(receipt_path, {
        "candidate_id": parent_id,
        "algorithm_fingerprint": "algorithm-parent",
        "derivation_card_sha256": file_sha256(parent_card_path),
    })
    _write(tmp_path / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json", {
        "schema": "final-unsb-route1-cross-version-e200-adjudication-v1",
        "status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE",
        "ranking": [{
            "candidate_id": parent_id,
            "algorithm_fingerprint": "algorithm-parent",
            "candidate_fingerprint": "candidate-parent",
            "training_git_commit": "a" * 40,
        }],
        "selected_candidate_id": parent_id,
        "selected_algorithm_fingerprint": "algorithm-parent",
        "selected_candidate_fingerprint": "candidate-parent",
        "selected_training_git_commit": "a" * 40,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "derive" / "HYPOTHESIS_LEDGER.json", {
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [],
    })

    def fake_freeze(output_root: Path, candidate_id: str):
        ledger_path = Path(output_root) / "derive" / "HYPOTHESIS_LEDGER.json"
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        record = next(row for row in ledger["records"] if row["candidate_id"] == candidate_id)
        record["status"] = "FROZEN_FOR_GATES"
        _write(ledger_path, ledger)
        return SimpleNamespace(to_dict=lambda: {"candidate_id": candidate_id})

    monkeypatch.setattr(
        "research.local_route1.winner_ablations.freeze_candidate_derivation",
        fake_freeze,
    )
    result = materialize_winner_ablation_definitions(tmp_path)
    expected = WINNER_FAMILIES[parent_id]["ids"]
    assert result["ablation_candidate_ids"] == expected
    assert result["long_horizon_started"] is False
    for role, candidate_id in expected.items():
        card_path = tmp_path / "derive" / "cards" / f"{candidate_id}.json"
        implementation_path = (
            tmp_path / "derive" / "implementations" / f"{candidate_id}.json"
        )
        card = json.loads(card_path.read_text(encoding="utf-8"))
        implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
        assert card["parent_candidate_id"] == parent_id
        assert card["parent_terminal_receipt_sha256"] == file_sha256(receipt_path)
        assert card["ablation_role"] == role
        assert implementation["method"]["route1_ablation_enable"] is True
        assert implementation["gate_hook"]["callable"] == "run_winner_ablation_gate"
        assert all(
            file_sha256(ROOT / row["path"]) == row["sha256"]
            for row in implementation["source_files"]
        )
    assert not (
        tmp_path / "derive" / "cards" / "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY.json"
    ).exists()


@pytest.mark.parametrize("parent_id,selection_name", [
    (
        "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS",
        "CROSS_VERSION_REVISION_E200_ADJUDICATION.json",
    ),
    ("G1-03-STATE-FEEDBACK-MISSING", "ROUTE1_FINAL_E200_SELECTION.json"),
])
def test_extended_winner_families_freeze_from_terminal_selection(
    monkeypatch, tmp_path, parent_id, selection_name,
):
    parent_card = json.loads((
        ROOT / "research" / "local_route1" / "derivation_cards" / f"{parent_id}.json"
    ).read_text(encoding="utf-8"))
    parent_card_path = tmp_path / "derive" / "cards" / f"{parent_id}.json"
    _write(parent_card_path, parent_card)
    receipt_path = tmp_path / "operations" / "terminal_receipts" / f"{parent_id}.json"
    _write(receipt_path, {
        "candidate_id": parent_id,
        "algorithm_fingerprint": "algorithm-parent",
        "derivation_card_sha256": file_sha256(parent_card_path),
    })
    _write(tmp_path / "operations" / selection_name, {
        "schema": "final-unsb-route1-cross-version-e200-adjudication-v1",
        "status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE",
        "ranking": [{
            "candidate_id": parent_id,
            "algorithm_fingerprint": "algorithm-parent",
            "candidate_fingerprint": "candidate-parent",
            "training_git_commit": "b" * 40,
        }],
        "selected_candidate_id": parent_id,
        "selected_algorithm_fingerprint": "algorithm-parent",
        "selected_candidate_fingerprint": "candidate-parent",
        "selected_training_git_commit": "b" * 40,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    })
    _write(tmp_path / "derive" / "HYPOTHESIS_LEDGER.json", {
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "records": [],
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
    result = materialize_winner_ablation_definitions(tmp_path)
    family = WINNER_FAMILIES[parent_id]["family"]
    for role, candidate_id in result["ablation_candidate_ids"].items():
        implementation = json.loads((
            tmp_path / "derive" / "implementations" / f"{candidate_id}.json"
        ).read_text(encoding="utf-8"))
        assert implementation["model"] == f"route1_{family}_ablation"
        assert implementation["method"][f"{family}_ablation_role"] == role
