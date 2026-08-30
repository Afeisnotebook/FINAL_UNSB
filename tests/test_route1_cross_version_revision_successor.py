import json
from pathlib import Path

import pytest

from operations import local_route1_cross_version_revision_successor as successor
from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
)
from research.local_route1.protocol import file_sha256


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, (dict, list)):
        path.write_text(json.dumps(value), encoding="utf-8")
    else:
        path.write_text(str(value), encoding="utf-8")


def _fixture(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    run_root = tmp_path / "run"
    manifest = tmp_path / "manifest.csv"
    python = tmp_path / "python"
    _write(manifest, "domain,stem\n")
    _write(python, "")

    candidate_id = "G2-01-TARGET-BLIND-REVISION"
    parent_id = "G1-02B-PLAYER-CONDITIONAL-RSMG"
    for relative in successor.SUCCESSOR_SOURCES:
        _write(repo / relative, relative)
    for relative in (
        "operations/local_route1_candidate_executor.py",
        "research/local_route1/candidates.py",
        "research/local_route1/candidate_gate.py",
        "research/local_route1/candidate_runner.py",
        "src/models/route1/revision.py",
    ):
        _write(repo / relative, relative)

    def fake_run_text(argv, *, cwd):
        assert Path(cwd) == repo
        if argv[1:3] == ["status", "--porcelain"]:
            return ""
        if argv[1:3] == ["rev-parse", "HEAD"]:
            return "revision-commit"
        raise AssertionError(argv)

    monkeypatch.setattr(successor.support, "run_text", fake_run_text)

    outcome_path = (
        run_root / "operations" / "CROSS_VERSION_FINAL_CAUSAL_REVISION_OUTCOME.json"
    )
    outcome = {
        "schema": CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
        "status": successor.REVISION_REQUIRED,
        "selected_candidate_id": parent_id,
        "revision_applicable_candidate_ids": [parent_id],
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    _write(outcome_path, outcome)
    card_path = run_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = (
        run_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    gate_path = run_root / "derive" / "gates" / f"{candidate_id}.json"
    _write(card_path, {
        "candidate_id": candidate_id,
        "parent_candidate_id": parent_id,
        "paired_target_available_to_training": False,
    })
    _write(implementation_path, {
        "candidate_id": candidate_id,
        "source_files": [{"path": "src/models/route1/revision.py"}],
    })
    _write(gate_path, {
        "status": "PASS_LONG_RUN",
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "candidate",
        "paired_metric_used_for_promotion": False,
        "confirmation20_opened": False,
    })
    _write(run_root / "derive" / "HYPOTHESIS_LEDGER.json", {
        "records": [{
            "candidate_id": candidate_id,
            "parent_candidate_id": parent_id,
            "generation": 2,
            "revision_count": 1,
            "status": "FROZEN_FOR_GATES",
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }],
    })
    executor_path = run_root / "operations" / "REVISION_EXECUTOR_CONTRACT.json"
    _write(executor_path, {
        "schema": successor.EXECUTOR_SCHEMA,
        "candidate_id": candidate_id,
        "candidate_repo": str(repo.resolve()),
        "candidate_git_commit": "revision-commit",
        "run_root": str(run_root.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "target_data_epochs": 200,
        "paired_metric_early_stop": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        "algorithm_fingerprint": "algorithm",
        "candidate_fingerprint": "candidate",
    })
    contract = {
        "schema": successor.SCHEMA,
        "successor_repo": str(repo.resolve()),
        "successor_git_commit": "revision-commit",
        "successor_source_sha256": {
            relative: file_sha256(repo / relative)
            for relative in successor.SUCCESSOR_SOURCES
        },
        "run_root": str(run_root.resolve()),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": file_sha256(manifest),
        "python": str(python.resolve()),
        "poll_seconds": 60,
        "timeout_seconds": 3600,
        "development_seeds": [2026],
        "deferred_seeds": [2027, 2028],
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    return contract, outcome, repo, candidate_id, executor_path


def test_revision_authorization_binds_single_revision_and_e200(tmp_path, monkeypatch):
    contract, outcome, repo, candidate_id, executor_path = _fixture(
        tmp_path, monkeypatch,
    )
    authorization = successor.default_authorization(
        contract=contract, candidate_repo=repo, candidate_id=candidate_id,
        executor_contract_path=executor_path,
    )
    executor = successor.validate_authorization(contract, outcome, authorization)
    assert executor["candidate_id"] == candidate_id
    assert authorization["generation"] == 2
    assert authorization["revision_count"] == 1
    assert authorization["seed"] == 2026
    assert authorization["batch_size"] == 1
    assert authorization["target_data_epochs"] == 200
    assert authorization["best_checkpoint_selection"] is False
    assert authorization["paired_controller_access"] is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("batch_size", 2, "batch1 seed2026"),
        ("seed", 2027, "batch1 seed2026"),
        ("revision_count", 2, "one Generation-2"),
        ("fixed_window_or_handoff", True, "fixed_window_or_handoff"),
        ("paired_controller_access", True, "paired_controller_access"),
        ("best_checkpoint_selection", True, "fixed e200"),
    ],
)
def test_revision_authorization_rejects_protocol_drift(
    tmp_path, monkeypatch, field, value, message,
):
    contract, outcome, repo, candidate_id, executor_path = _fixture(
        tmp_path, monkeypatch,
    )
    authorization = successor.default_authorization(
        contract=contract, candidate_repo=repo, candidate_id=candidate_id,
        executor_contract_path=executor_path,
    )
    authorization[field] = value
    with pytest.raises(RuntimeError, match=message):
        successor.validate_authorization(contract, outcome, authorization)


def test_revision_authorization_rejects_changed_defect_outcome(tmp_path, monkeypatch):
    contract, outcome, repo, candidate_id, executor_path = _fixture(
        tmp_path, monkeypatch,
    )
    authorization = successor.default_authorization(
        contract=contract, candidate_repo=repo, candidate_id=candidate_id,
        executor_contract_path=executor_path,
    )
    authorization["source_revision_need_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="bind the defect outcome"):
        successor.validate_authorization(contract, outcome, authorization)
