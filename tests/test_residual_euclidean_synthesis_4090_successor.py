from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations import (
    local_route1_residual_euclidean_synthesis_4090_successor as successor,
)


def _contract(tmp_path: Path) -> dict:
    files = {}
    for name in ("sampling.json", "manifest.csv", "environment.json"):
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
        files[name] = path
    repo_root = Path(successor.__file__).resolve().parents[1]
    return {
        "schema": successor.SCHEMA,
        "repo": {"path": str(tmp_path), "git_commit": "frozen"},
        "source_sha256": {
            relative: successor.support.file_sha256(repo_root / relative)
            for relative in successor.SOURCE_RELATIVES
        },
        "run_root": str(tmp_path / "run"),
        "portfolio_result": str(tmp_path / "portfolio.json"),
        "sampling_receipt": str(files["sampling.json"]),
        "sampling_receipt_sha256": successor.support.file_sha256(
            files["sampling.json"]
        ),
        "train_view": str(tmp_path / "view"),
        "data_root": str(tmp_path / "data"),
        "manifest": str(files["manifest.csv"]),
        "manifest_sha256": successor.support.file_sha256(files["manifest.csv"]),
        "python": str(tmp_path / "python"),
        "baseline_environment_record": str(files["environment.json"]),
        "baseline_environment_record_sha256": successor.support.file_sha256(
            files["environment.json"]
        ),
        "poll_seconds": 60,
        "timeout_seconds": 1209600,
        "candidate_id": successor.CANDIDATE_ID,
        "required_sampling_parent_id": successor.PCRSMG_PROPOSAL_ID,
        "required_barrier_parent_id": successor.RFMCRB_ID,
        "required_prior_ledger_boundary_candidate_id": successor.ADAM_SYNTHESIS_ID,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _patch_contract_identity(tmp_path, monkeypatch):
    repo_root = Path(successor.__file__).resolve().parents[1]
    monkeypatch.setattr(
        successor.support,
        "run_text",
        lambda command, cwd: (
            "frozen" if command[1:] == ["rev-parse", "HEAD"] else ""
        ),
    )
    original_hash = successor.support.file_sha256

    def source_aware_hash(path):
        path = Path(path)
        try:
            relative = path.relative_to(tmp_path).as_posix()
        except ValueError:
            return original_hash(path)
        if relative in successor.SOURCE_RELATIVES:
            return original_hash(repo_root / relative)
        return original_hash(path)

    monkeypatch.setattr(successor.support, "file_sha256", source_aware_hash)


def test_euclidean_successor_contract_freezes_route1_scope(tmp_path, monkeypatch):
    contract = _contract(tmp_path)
    _patch_contract_identity(tmp_path, monkeypatch)
    successor.validate_contract(contract)
    changed = json.loads(json.dumps(contract))
    changed["required_barrier_parent_id"] = "wrong"
    with pytest.raises(RuntimeError, match="required_barrier_parent_id"):
        successor.validate_contract(changed)


def test_euclidean_successor_without_rfmcrb_exits_without_candidate(
    tmp_path, monkeypatch,
):
    contract = _contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(successor, "validate_contract", lambda value: None)
    worker = successor.ResidualEuclideanSynthesis4090Successor(contract_path)
    monkeypatch.setattr(worker, "wait_portfolio", lambda: {
        "schema": successor.PORTFOLIO_SCHEMA,
        "candidate_results": [],
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    assert worker.run() == 0
    result = json.loads((
        Path(contract["run_root"]) / "operations"
        / "RESIDUAL_EUCLIDEAN_SYNTHESIS_4090_RESULT.json"
    ).read_text(encoding="utf-8"))
    assert result["status"] == "SYNTHESIS_INAPPLICABLE_RFMCRB_NOT_REPLAYED"
    assert result["old_fixed_margin_operator_run"] is False


def test_euclidean_successor_waits_for_adam_ledger_boundary(tmp_path, monkeypatch):
    contract = _contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(successor, "validate_contract", lambda value: None)
    worker = successor.ResidualEuclideanSynthesis4090Successor(contract_path)
    worker.operations.mkdir(parents=True)
    (worker.operations / "RESIDUAL_SYNTHESIS_4090_RESULT.json").write_text(
        json.dumps({
            "candidate_id": None,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }),
        encoding="utf-8",
    )
    worker.wait_prior_ledger_boundary()

