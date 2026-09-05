from __future__ import annotations

import json
from pathlib import Path

from operations import paper_aio_final_delivery_successor as final
from operations import paper_aio_relation_registry_review as review
from research.paper_aio import runtime_relation


PLAIN_PROTOCOL = "e5704e445a51dd9c5c12369c94df01cf9532364a71c806b9914ef3963994b07b"
STCGR_PROTOCOL = "2fbdd6f58971657134c305224ab14ae0e0ba53cf421c64f9935c68a4c0873e20"
MANIFEST = "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
PLAIN_HOST = "5090B_MATCHED_PLAIN"


def _metric(host: str, protocol: str) -> dict:
    return {
        "source_host_label": host,
        "training_protocol_fingerprint": protocol,
        "manifest_sha256": MANIFEST,
        "confirmation20_opened": False,
    }


def _late_entry(relation: dict) -> dict:
    return {
        "late_trajectory": [
            {"epoch": epoch, "crn_exact": True, "runtime_relation": relation}
            for epoch in (150, 175, 200)
        ]
    }


def test_future_5090b_control_registry_reaches_final_delivery(tmp_path: Path) -> None:
    base = review.validate_registry(runtime_relation.RELATIONS_PATH)
    proposal_relations = runtime_relation.relation_candidates(base, "proposal")
    assert len(proposal_relations) == 2
    old_proposal = next(
        row for row in proposal_relations
        if row["plain_source_host_label"] == "5090A"
    )
    future_proposal = next(
        row for row in proposal_relations
        if row["plain_source_host_label"] == PLAIN_HOST
    )
    stcgr = runtime_relation.relation_candidates(base, review.STCGR_ID)[0]
    proposed = review.propose_registry(base, [future_proposal, stcgr])
    assert proposed == base

    proposal_relations = runtime_relation.relation_candidates(proposed, "proposal")
    assert len(proposal_relations) == 2
    assert old_proposal in proposal_relations
    assert future_proposal in proposal_relations
    assert runtime_relation.relation_candidates(proposed, review.STCGR_ID) == [stcgr]

    registry = tmp_path / "PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json"
    registry.write_text(json.dumps(proposed), encoding="utf-8")

    proposal_status = runtime_relation.runtime_pair_status(
        method=_metric("5090C", PLAIN_PROTOCOL),
        plain=_metric(PLAIN_HOST, PLAIN_PROTOCOL),
        lane_id="proposal",
        relations_path=registry,
    )
    stcgr_status = runtime_relation.runtime_pair_status(
        method=_metric("5090A", STCGR_PROTOCOL),
        plain=_metric(PLAIN_HOST, PLAIN_PROTOCOL),
        lane_id=review.STCGR_ID,
        candidate_cross_code_gate=True,
        relations_path=registry,
    )

    assert proposal_status["status"] == "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"
    assert stcgr_status["status"] == (
        "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION"
    )
    assert final._validated_control_relation(
        _late_entry(proposal_status),
        lane_id="proposal",
        method_source_host="5090C",
        plain_source_host=PLAIN_HOST,
    )["plain_source_host_label"] == PLAIN_HOST
    assert final._validated_control_relation(
        _late_entry(stcgr_status),
        lane_id=review.STCGR_ID,
        method_source_host="5090A",
        plain_source_host=PLAIN_HOST,
        candidate_cross_code=True,
    )["plain_source_host_label"] == PLAIN_HOST
