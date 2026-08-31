from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from operations import local_route1_portable_extended_frontier_export_successor as successor
from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
    SIDECAR_SCHEMA,
)
from research.local_route1.extended_repaired_frontier import SCHEMA as EXTENDED_SCHEMA
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.portable_extended_frontier import (
    STATUS,
    export_portable_extended_frontier,
    validate_portable_extended_frontier,
)
from research.local_route1.protocol import ROOT, file_sha256


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    return path


def _candidate(root: Path, candidate_id: str) -> Path:
    card = _write(root / "derive" / "cards" / f"{candidate_id}.json", {
        "candidate_id": candidate_id, "name": candidate_id,
    })
    implementation = _write(
        root / "derive" / "implementations" / f"{candidate_id}.json",
        {"candidate_id": candidate_id, "model": "fake"},
    )
    trajectory = _write(
        root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json",
        {"candidate_id": candidate_id, "status": POSITIVE_STATUS},
    )
    receipt = _write(
        root / "operations" / "terminal_receipts" / f"{candidate_id}.json",
        {
            "schema": RECEIPT_SCHEMA,
            "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
            "candidate_id": candidate_id,
            "trajectory_status": POSITIVE_STATUS,
            "algorithm_fingerprint": f"algorithm-{candidate_id}",
            "candidate_fingerprint": f"candidate-{candidate_id}",
            "candidate_training_core_fingerprint": f"core-{candidate_id}",
            "base_e0_scientific_state_sha256": "e0",
            "base_protocol_fingerprint": "protocol",
            "manifest_sha256": "manifest",
            "plain_e200_verification_sha256": "plain",
            "training_git_commit": "a" * 40,
            "receipt_source_sha256": file_sha256(
                ROOT / "operations" / "local_route1_candidate_terminal_receipt.py"
            ),
            "trajectory_path": str(trajectory.resolve()),
            "trajectory_sha256": file_sha256(trajectory),
            "derivation_card_sha256": file_sha256(card),
            "implementation_sha256": file_sha256(implementation),
            "ranking_fields": {
                "late_three_mean_macro_psnr_delta": 0.1,
                "e200_macro_psnr_delta": 0.1,
            },
            "terminal_integrity": {"status": "ACCEPTED_COMPLETE_E200_ARTIFACT_SET"},
            "evaluation_crn_matched_to_same_host_plain": True,
            "paired_metrics_used_only_after_complete_trajectory": True,
            "paired_metrics_used_for_training_or_control": False,
            "confirmation20_opened": False,
        },
    )
    _write(Path(str(receipt) + ".sha256.json"), {
        "schema": SIDECAR_SCHEMA,
        "candidate_id": candidate_id,
        "receipt_sha256": file_sha256(receipt),
    })
    return receipt


def test_portable_extended_frontier_embeds_ranked_and_observable_evidence(tmp_path: Path):
    full = _candidate(tmp_path, "FULL")
    proposal = _candidate(tmp_path, "PROPOSAL")
    observable = _candidate(tmp_path, "OBSERVABLE")
    adjudication = _write(
        tmp_path / "operations" / "EXTENDED_REPAIRED_FRONTIER_E200_ADJUDICATION.json",
        {
            "schema": EXTENDED_SCHEMA,
            "status": "EXTENDED_REPAIRED_FRONTIER_STRICT_ACTION_PRIORITY_AVAILABLE",
            "ranking": [
                {"candidate_id": "PROPOSAL", "receipt_path": str(proposal.resolve()), "receipt_sha256": file_sha256(proposal)},
                {"candidate_id": "FULL", "receipt_path": str(full.resolve()), "receipt_sha256": file_sha256(full)},
            ],
            "parent_ablation_results": [{
                "parent_candidate_id": "FULL",
                "roles": {
                    "proposal_only": {"candidate_id": "PROPOSAL", "receipt_path": str(proposal.resolve()), "receipt_sha256": file_sha256(proposal)},
                    "observable_only": {"candidate_id": "OBSERVABLE", "receipt_path": str(observable.resolve()), "receipt_sha256": file_sha256(observable)},
                    "projected_or_full": {"candidate_id": "FULL", "receipt_path": str(full.resolve()), "receipt_sha256": file_sha256(full)},
                },
            }],
            "canonical_candidate_is_action_priority_only": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "cross_host_deltas_merged": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    )
    result = export_portable_extended_frontier(tmp_path)
    assert result["status"] == STATUS
    assert result["source_adjudication_sha256"] == file_sha256(adjudication)
    assert {row["candidate_id"] for row in result["candidate_evidence"]} == {
        "FULL", "PROPOSAL", "OBSERVABLE",
    }
    validate_portable_extended_frontier(result)
    changed = json.loads(json.dumps(result))
    changed["candidate_evidence"][0]["trajectory"]["status"] = "changed"
    with pytest.raises(RuntimeError, match="embedded artifact changed"):
        validate_portable_extended_frontier(changed)


def test_portable_extended_export_successor_contract_is_evidence_only(
    monkeypatch, tmp_path,
):
    def fake_run_text(command, *, cwd):
        if command[:2] == ["git", "status"]:
            return ""
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        raise AssertionError(command)

    monkeypatch.setattr(successor.support, "run_text", fake_run_text)
    contract = successor.default_contract(Namespace(
        repo=ROOT, run_root=tmp_path, poll_seconds=60, timeout_seconds=1209600,
    ))
    successor.validate_contract(contract)
    assert contract["complete_source_e200_only"] is True
    assert contract["checkpoint_transfer"] is False
    assert contract["cross_host_deltas_merged"] is False


def test_extended_frontier_relay_contract_never_persists_credentials(tmp_path: Path):
    from operations import local_route1_extended_frontier_relay as relay

    contract = relay.default_contract(Namespace(
        source_host="source",
        source_port=12770,
        source_user="root",
        source_path="/source/portable.json",
        destination_host="destination",
        destination_port=22,
        destination_user="yc",
        destination_path="/destination/portable.json",
        local_spool=tmp_path / "spool.json",
        state=tmp_path / "state.json",
        poll_seconds=60,
        timeout_seconds=1209600,
    ))
    relay.validate_contract(contract)
    assert contract["checkpoint_transfer"] is False
    assert contract["destination_overwrite_allowed"] is False
    assert "password" not in json.dumps(contract).lower()
