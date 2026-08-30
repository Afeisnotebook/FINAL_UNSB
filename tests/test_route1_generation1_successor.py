from pathlib import Path

import pytest

from operations.local_route1_generation1_successor import (
    DEFAULT_IDS,
    SCHEMA,
    validate_contract,
)


def test_successor_contract_rejects_scientific_shortcuts(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"manifest")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "operations.local_route1_generation1_successor.support.run_text",
        lambda argv, cwd: "commit" if "rev-parse" in argv else "",
    )
    monkeypatch.setattr(
        "operations.local_route1_generation1_successor.support.file_sha256",
        lambda path: (
            "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
            if Path(path) == manifest else "source"
        ),
    )
    contract = {
        "schema": SCHEMA,
        "successor_repo": str(repo),
        "successor_git_commit": "commit",
        "candidate_ids": list(DEFAULT_IDS),
        "manifest": str(manifest),
        "manifest_sha256": "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b",
        "source_sha256": {"source.py": "source"},
        "candidate_wait_poll_seconds": 60,
        "candidate_wait_timeout_seconds": 86400,
        "seed_order": [2027, 2028],
        "seed2028_requires_seed2027_sign_inconsistency": True,
        "freeze_only_numeric_gate_winner": True,
        "algorithm_revision_path": False,
        "handoff_or_window_path": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    validate_contract(contract)
    for key in (
        "algorithm_revision_path", "handoff_or_window_path",
        "paired_metric_scheduling", "paired_controller_access",
        "confirmation20_opened",
    ):
        changed = dict(contract)
        changed[key] = True
        with pytest.raises(RuntimeError, match=key):
            validate_contract(changed)


def test_successor_contract_requires_frozen_candidates_and_seed_order(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"manifest")
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(
        "operations.local_route1_generation1_successor.support.run_text",
        lambda argv, cwd: "commit" if "rev-parse" in argv else "",
    )
    monkeypatch.setattr(
        "operations.local_route1_generation1_successor.support.file_sha256",
        lambda path: (
            "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
            if Path(path) == manifest else "source"
        ),
    )
    base = {
        "schema": SCHEMA, "successor_repo": str(repo),
        "successor_git_commit": "commit", "candidate_ids": list(DEFAULT_IDS),
        "manifest": str(manifest),
        "manifest_sha256": "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b",
        "source_sha256": {}, "candidate_wait_poll_seconds": 60,
        "candidate_wait_timeout_seconds": 86400, "seed_order": [2027, 2028],
        "seed2028_requires_seed2027_sign_inconsistency": True,
        "freeze_only_numeric_gate_winner": True, "algorithm_revision_path": False,
        "handoff_or_window_path": False, "paired_metric_scheduling": False,
        "paired_controller_access": False, "confirmation20_opened": False,
    }
    changed = dict(base, candidate_ids=[DEFAULT_IDS[0]])
    with pytest.raises(RuntimeError, match="candidate set"):
        validate_contract(changed)
    changed = dict(base, seed_order=[2028, 2027])
    with pytest.raises(RuntimeError, match="order"):
        validate_contract(changed)
