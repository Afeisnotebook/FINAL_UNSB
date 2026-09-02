from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import paper_aio_candidate_control_relation_successor as successor
from research.paper_aio.protocol import file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _source_contract() -> dict:
    return {
        "schema": successor.SOURCE_CONTRACT_SCHEMA,
        "plain_source_host_label": "5090B_MATCHED_PLAIN",
        "protocol_fingerprint": "parent-fp",
        "manifest_sha256": "m" * 64,
        "registry_edited": False,
        "comparison_authorized": False,
        "performance_values_read": False,
        "confirmation20_opened": False,
    }


def test_contract_is_review_only_and_pins_all_primary_evidence(
    tmp_path: Path, monkeypatch,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    script = tmp_path / "successor.py"
    script.write_text("frozen", encoding="utf-8")
    monkeypatch.setattr(successor, "__file__", str(script))
    monkeypatch.setattr(successor, "git_identity", lambda _: ("c" * 40, False))
    evidence = {}
    for name in ("gate", "authorization", "metadata", "authority"):
        evidence[name] = _write(tmp_path / f"{name}.json", {"name": name})
    source_contract = _write(tmp_path / "source_contract.json", _source_contract())
    args = SimpleNamespace(
        repo=repo,
        required_control_git_commit="c" * 40,
        candidate_id="candidate",
        method_source_host_label="5090A",
        plain_source_host_label="5090B_MATCHED_PLAIN",
        plain_protocol_fingerprint="parent-fp",
        manifest_sha256="m" * 64,
        candidate_runtime_gate=evidence["gate"],
        required_candidate_runtime_gate_sha256=file_sha256(evidence["gate"]),
        candidate_authorization=evidence["authorization"],
        required_candidate_authorization_sha256=file_sha256(evidence["authorization"]),
        candidate_metadata_import=evidence["metadata"],
        required_candidate_metadata_import_sha256=file_sha256(evidence["metadata"]),
        candidate_authority=evidence["authority"],
        required_candidate_authority_sha256=file_sha256(evidence["authority"]),
        source_successor_contract=source_contract,
        required_source_successor_contract_sha256=file_sha256(source_contract),
        source_successor_state=tmp_path / "source_state.json",
        plain_runtime_receipt=tmp_path / "plain.json",
        destination_output=tmp_path / "out",
        poll_seconds=60,
        timeout_hours=720.0,
    )
    contract = successor.proposed_contract(args)
    assert contract["status"] == "FROZEN_REVIEW_ONLY_SUCCESSOR"
    assert contract["registry_edited"] is False
    assert contract["comparison_authorized"] is False
    assert contract["training_authorized_or_scheduled"] is False
    assert len(contract["candidate_evidence"]) == 4


def test_verified_source_requires_completed_hash_bound_receipt(tmp_path: Path) -> None:
    plain = _write(tmp_path / "plain.json", {"exact": True})
    state = tmp_path / "source_state.json"
    contract = {
        "source_successor_state": str(state),
        "plain_runtime_receipt": str(plain),
        "plain_source_host_label": "5090B_MATCHED_PLAIN",
    }
    with pytest.raises(successor.SourceRelationNotReady):
        successor.require_verified_source(contract)
    _write(state, {
        "schema": successor.SOURCE_STATE_SCHEMA,
        "status": "COMPLETE_REVIEW_ONLY_RUNTIME_RELATION_CANDIDATE",
        "plain_source_host_label": "5090B_MATCHED_PLAIN",
        "plain_runtime_receipt": str(plain.resolve()),
        "plain_runtime_receipt_sha256": file_sha256(plain),
        "exact_runtime_equivalence": True,
        "registry_edited": False,
        "comparison_authorized": False,
        "performance_values_read": False,
        "confirmation20_opened": False,
    })
    assert successor.require_verified_source(contract) == plain.resolve()
    plain.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not bind"):
        successor.require_verified_source(contract)
