from __future__ import annotations

import pytest

from research.local_route1.frontier_advancement import ALTERNATE
from research.local_route1.pcnr_alternate_replay import (
    CANDIDATE_ID,
    select_pcnr_alternate,
)


def _authority() -> dict:
    card = {"candidate_id": CANDIDATE_ID}
    implementation = {
        "candidate_id": CANDIDATE_ID,
        "model": "route1_pcnr",
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
    }
    receipt = {
        "schema": "final-unsb-route1-candidate-terminal-receipt-v1",
        "candidate_id": CANDIDATE_ID,
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "candidate",
        "trajectory_sha256": "trajectory",
        "derivation_card_sha256": "card",
        "implementation_sha256": "implementation",
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    return {
        "extended_adjudication": {
            "action_priority_candidate_id": CANDIDATE_ID,
            "evidence_preserved_candidate_ids": [CANDIDATE_ID],
            "ranking": [{
                "candidate_id": CANDIDATE_ID,
                "classification": ALTERNATE,
                "algorithm_fingerprint": "algorithm",
                "candidate_fingerprint": "candidate",
                "trajectory_sha256": "trajectory",
                "receipt_sha256": "receipt",
            }],
        },
        "candidate_evidence": [{
            "candidate_id": CANDIDATE_ID,
            "receipt": receipt,
            "receipt_sha256": "receipt",
            "trajectory_sha256": "trajectory",
            "derivation_card": card,
            "derivation_card_sha256": "card",
            "implementation": implementation,
            "implementation_sha256": "implementation",
        }],
    }


def test_selects_action_priority_evidence_backed_alternate(monkeypatch) -> None:
    monkeypatch.setattr(
        "research.local_route1.pcnr_alternate_replay."
        "validate_portable_extended_frontier",
        lambda value: value,
    )
    monkeypatch.setattr(
        "research.local_route1.pcnr_alternate_replay._canonical_sha256",
        lambda value: "card" if value.get("model") is None else "implementation",
    )
    selected = select_pcnr_alternate(_authority())
    assert selected["receipt"]["candidate_id"] == CANDIDATE_ID
    assert selected["ranking"]["classification"] == ALTERNATE


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda value: value["extended_adjudication"].update(
                action_priority_candidate_id="another"
            ),
            "action priority",
        ),
        (
            lambda value: value["extended_adjudication"].update(
                evidence_preserved_candidate_ids=[]
            ),
            "evidence-preserved",
        ),
        (
            lambda value: value["extended_adjudication"]["ranking"][0].update(
                classification="closed_current_operator"
            ),
            "evidence-backed alternate",
        ),
    ],
)
def test_rejects_unqualified_pcnr_authority(monkeypatch, mutation, message) -> None:
    monkeypatch.setattr(
        "research.local_route1.pcnr_alternate_replay."
        "validate_portable_extended_frontier",
        lambda value: value,
    )
    value = _authority()
    mutation(value)
    with pytest.raises(RuntimeError, match=message):
        select_pcnr_alternate(value)
