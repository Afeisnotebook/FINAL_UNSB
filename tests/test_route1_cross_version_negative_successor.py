from pathlib import Path

from operations import local_route1_cross_version_negative_successor as successor
from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_NEGATIVE_STATUS,
)


def _bare_successor(monkeypatch, cross):
    value = object.__new__(successor.CrossVersionNegativeSuccessor)
    events = []
    states = []
    audits = []
    value.contract_path = "contract.json"
    value.run_root = Path("run-root")
    value.event = lambda event, **fields: events.append((event, fields))
    value.state = lambda status, **fields: states.append((status, fields))
    value.wait_for_cross = lambda: cross
    value.run_audits = lambda: audits.append(True)
    return value, events, states, audits


def test_negative_successor_accepts_canonical_cross_version_negative_status(
    monkeypatch,
):
    value, events, states, audits = _bare_successor(
        monkeypatch, {"status": CROSS_VERSION_NEGATIVE_STATUS},
    )
    monkeypatch.setattr(
        successor, "adjudicate_cross_version_revision_need",
        lambda *_: {
            "status": "REVISION_DERIVATION_REQUIRED",
            "selected_candidate_id": successor.EXPECTED_IDS[1],
            "revision_applicable_candidate_ids": [successor.EXPECTED_IDS[1]],
        },
    )
    assert value.run() == 0
    assert audits == [True]
    assert states[-1][0] == "MATHEMATICAL_REVISION_DERIVATION_REQUIRED"
    assert states[-1][1]["selected_parent_candidate_id"] == successor.EXPECTED_IDS[1]


def test_negative_successor_does_not_audit_a_positive_cross_version_winner(
    monkeypatch,
):
    value, events, states, audits = _bare_successor(
        monkeypatch,
        {"status": "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"},
    )
    assert value.run() == 0
    assert audits == []
    assert states[-1][0] == "INAPPLICABLE_POSITIVE_CROSS_VERSION_WINNER"
    assert states[-1][1]["audits_started"] is False
