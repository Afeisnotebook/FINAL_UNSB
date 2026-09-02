from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import paper_aio_relation_registry_review as review
from research.paper_aio.protocol import file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _base() -> dict:
    return {
        "schema": review.REGISTRY_SCHEMA,
        "status": "ACTIVE_METRIC_BLIND_RELATIONS",
        "relations": {},
    }


def _relation(lane: str, method: str, plain: str) -> dict:
    common = {
        "method_lane": lane,
        "method_source_host_label": method,
        "plain_source_host_label": plain,
        "updates": 2000,
        "manifest_sha256": "m" * 64,
        "e0_core_sha256": "e" * 64,
        "step_core_sha256": "s" * 64,
        "differences": {},
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    if lane == review.STCGR_ID:
        return {
            **common,
            "status": review.CANDIDATE_STATUS,
            "candidate_id": lane,
            "candidate_protocol_fingerprint": "c" * 64,
            "plain_training_protocol_fingerprint": "p" * 64,
            "proof_chain": {
                "candidate_to_parent": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
                "parent_to_plain": "PASS_EXACT_RUNTIME_COHORT",
            },
            **{
                key: key[0] * 64
                for key in (
                    "candidate_runtime_gate_sha256",
                    "candidate_authorization_sha256",
                    "candidate_metadata_import_sha256",
                    "candidate_authority_sha256",
                    "candidate_parent_runtime_receipt_sha256",
                    "plain_runtime_receipt_sha256",
                )
            },
        }
    return {
        **common,
        "status": review.STANDARD_STATUS,
        "training_protocol_fingerprint": "p" * 64,
        "method_runtime_receipt_sha256": "r" * 64,
        "plain_runtime_receipt_sha256": "q" * 64,
        "method_authorization_receipt_sha256": "a" * 64,
        "normalized_environment": {"gpu": "NVIDIA GeForce RTX 5090"},
    }


def _state(path: Path, relation_path: Path, relation_sha: str, lane: str) -> Path:
    cross_code = lane == review.STCGR_ID
    relation = json.loads(relation_path.read_text(encoding="utf-8"))
    return _write(
        path,
        {
            "schema": (
                review.CANDIDATE_STATE_SCHEMA
                if cross_code
                else review.STANDARD_STATE_SCHEMA
            ),
            "status": (
                "COMPLETE_REVIEW_ONLY_CANDIDATE_CONTROL_RELATION"
                if cross_code
                else "COMPLETE_REVIEW_ONLY_RUNTIME_RELATION_CANDIDATE"
            ),
            "candidate_id" if cross_code else "lane_id": lane,
            "method_source_host_label": "5090A" if cross_code else "5090C",
            "plain_source_host_label": "5090B_MATCHED_PLAIN",
            "relation_candidate": str(relation_path.resolve()),
            "relation_candidate_sha256": relation_sha,
            "plain_runtime_receipt_sha256": relation["plain_runtime_receipt_sha256"],
            **({"proof_chain": relation["proof_chain"]} if cross_code else {}),
            "exact_runtime_equivalence": True,
            "registry_edited": False,
            "comparison_authorized": False,
            "performance_values_read": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        },
    )


def test_candidate_validation_is_type_and_host_bound(tmp_path: Path) -> None:
    path = _write(tmp_path / "proposal.json", _relation("proposal", "5090C", "P"))
    value = review.validate_relation_candidate(
        path,
        expected_sha256=file_sha256(path),
        lane_id="proposal",
        method_source_host="5090C",
        plain_source_host="P",
    )
    assert value["status"] == review.STANDARD_STATUS
    with pytest.raises(RuntimeError, match="invalid review-only"):
        review.validate_relation_candidate(
            path,
            expected_sha256=file_sha256(path),
            lane_id="proposal",
            method_source_host="5090A",
            plain_source_host="P",
        )


def test_stcgr_requires_cross_code_candidate_proof(tmp_path: Path) -> None:
    value = _relation(review.STCGR_ID, "5090A", "P")
    path = _write(tmp_path / "stcgr.json", value)
    review.validate_relation_candidate(
        path,
        expected_sha256=file_sha256(path),
        lane_id=review.STCGR_ID,
        method_source_host="5090A",
        plain_source_host="P",
    )
    value["status"] = review.STANDARD_STATUS
    _write(path, value)
    with pytest.raises(RuntimeError, match="invalid review-only"):
        review.validate_relation_candidate(
            path,
            expected_sha256=file_sha256(path),
            lane_id=review.STCGR_ID,
            method_source_host="5090A",
            plain_source_host="P",
        )


def test_proposal_is_idempotent_and_rejects_conflict() -> None:
    first = _relation("proposal", "5090C", "P0")
    second = _relation("proposal", "5090C", "P1")
    value = review.propose_registry(_base(), [first, second])
    assert value["relations"]["proposal"] == [first, second]
    assert review.propose_registry(value, [first]) == value
    conflict = {**first, "step_core_sha256": "x" * 64}
    with pytest.raises(RuntimeError, match="conflicts"):
        review.propose_registry(value, [conflict])


def test_successor_state_binds_candidate_path_hash_and_hosts(tmp_path: Path) -> None:
    candidate = _write(
        tmp_path / "proposal.json",
        _relation("proposal", "5090C", "5090B_MATCHED_PLAIN"),
    )
    state = _state(
        tmp_path / "state.json",
        candidate,
        file_sha256(candidate),
        "proposal",
    )
    value = review.validate_candidate_state(
        state,
        expected_sha256=file_sha256(state),
        candidate_path=candidate,
        candidate_sha256=file_sha256(candidate),
        candidate=_relation("proposal", "5090C", "5090B_MATCHED_PLAIN"),
        lane_id="proposal",
        method_source_host="5090C",
        plain_source_host="5090B_MATCHED_PLAIN",
    )
    assert value["exact_runtime_equivalence"] is True
    value["relation_candidate_sha256"] = "x" * 64
    _write(state, value)
    with pytest.raises(RuntimeError, match="invalid relation successor"):
        review.validate_candidate_state(
            state,
            expected_sha256=file_sha256(state),
            candidate_path=candidate,
            candidate_sha256=file_sha256(candidate),
            candidate=_relation("proposal", "5090C", "5090B_MATCHED_PLAIN"),
            lane_id="proposal",
            method_source_host="5090C",
            plain_source_host="5090B_MATCHED_PLAIN",
        )


def test_review_outputs_proposal_without_editing_registry(tmp_path: Path) -> None:
    registry = _write(tmp_path / "registry.json", _base())
    proposal = _write(
        tmp_path / "proposal.json",
        _relation("proposal", "5090C", "5090B_MATCHED_PLAIN"),
    )
    stcgr = _write(
        tmp_path / "stcgr.json",
        _relation(review.STCGR_ID, "5090A", "5090B_MATCHED_PLAIN"),
    )
    proposal_state = _state(
        tmp_path / "proposal_state.json",
        proposal,
        file_sha256(proposal),
        "proposal",
    )
    stcgr_state = _state(
        tmp_path / "stcgr_state.json",
        stcgr,
        file_sha256(stcgr),
        review.STCGR_ID,
    )
    args = SimpleNamespace(
        registry=registry,
        candidate=[proposal, stcgr],
        expected_candidate_sha256=[file_sha256(proposal), file_sha256(stcgr)],
        candidate_state=[proposal_state, stcgr_state],
        expected_candidate_state_sha256=[
            file_sha256(proposal_state),
            file_sha256(stcgr_state),
        ],
        required_lane=["proposal", review.STCGR_ID],
        method_host=["proposal=5090C", f"{review.STCGR_ID}=5090A"],
        plain_source_host="5090B_MATCHED_PLAIN",
        output=tmp_path / "review",
    )
    receipt = review.review(args)
    assert receipt["status"].startswith("PASS_REVIEW_PROPOSAL")
    assert receipt["registry_edited"] is False
    assert receipt["comparison_authorized"] is False
    assert read_json(registry) == _base()
    proposed = read_json(
        tmp_path / "review" / "PROPOSED_RUNTIME_RELATION_REGISTRY.json"
    )
    assert set(proposed["relations"]) == {"proposal", review.STCGR_ID}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
