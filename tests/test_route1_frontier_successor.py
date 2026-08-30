from __future__ import annotations

from pathlib import Path

import pytest

from operations.local_route1_frontier_successor import (
    CANDIDATE_IDS,
    SCHEMA,
    validate_contract,
)


def _contract(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"manifest")
    return {
        "schema": SCHEMA,
        "repo": str(repo),
        "git_commit": "commit",
        "candidate_repo": str(repo),
        "candidate_git_commit": "commit",
        "source_sha256": {},
        "candidate_ids": list(CANDIDATE_IDS),
        "manifest": str(manifest),
        "manifest_sha256": "manifest",
        "gate_poll_seconds": 30,
        "gate_timeout_seconds": 7200,
        "training_poll_seconds": 30,
        "training_timeout_seconds": 172800,
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parallel_e200_executors": 2,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_frontier_contract_rejects_scientific_shortcuts(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        "operations.local_route1_frontier_successor.support.run_text",
        lambda argv, cwd: "commit" if "rev-parse" in argv else "",
    )
    monkeypatch.setattr(
        "operations.local_route1_frontier_successor.support.file_sha256",
        lambda path: "manifest",
    )
    validate_contract(contract)
    for key in (
        "paired_metric_scheduling", "paired_controller_access", "confirmation20_opened",
    ):
        changed = dict(contract, **{key: True})
        with pytest.raises(RuntimeError, match=key):
            validate_contract(changed)


def test_frontier_contract_freezes_two_stream_seed2026_e200(monkeypatch, tmp_path):
    contract = _contract(tmp_path)
    monkeypatch.setattr(
        "operations.local_route1_frontier_successor.support.run_text",
        lambda argv, cwd: "commit" if "rev-parse" in argv else "",
    )
    monkeypatch.setattr(
        "operations.local_route1_frontier_successor.support.file_sha256",
        lambda path: "manifest",
    )
    changes = (
        ("candidate_ids", [CANDIDATE_IDS[0]]),
        ("batch_size", 2),
        ("target_data_epochs", 199),
        ("maximum_parallel_e200_executors", 1),
        ("selection_seeds", [2026, 2027]),
        ("deferred_seed_validation", []),
    )
    for key, value in changes:
        changed = dict(contract, **{key: value})
        with pytest.raises(RuntimeError):
            validate_contract(changed)
