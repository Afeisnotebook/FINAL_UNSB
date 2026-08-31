from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations import local_route1_residual_synthesis_4090_successor as successor


def _contract(tmp_path: Path) -> dict:
    files = {}
    for name in ("sampling.json", "manifest.csv", "environment.json"):
        path = tmp_path / name
        path.write_text(name + "\n", encoding="utf-8")
        files[name] = path
    source_sha = {}
    for relative in successor.SOURCE_RELATIVES:
        source_sha[relative] = successor.support.file_sha256(
            successor.Path(successor.__file__).resolve().parents[1] / relative
        )
    return {
        "schema": successor.SCHEMA,
        "repo": {"path": str(tmp_path), "git_commit": "frozen"},
        "source_sha256": source_sha,
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
        "required_barrier_parent_id": successor.RFAMMCRB_ID,
        "batch_size": 1,
        "target_data_epochs": 200,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_successor_contract_freezes_single_seed_target_blind_scope(
    tmp_path, monkeypatch,
):
    contract = _contract(tmp_path)
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
    successor.validate_contract(contract)
    changed = json.loads(json.dumps(contract))
    changed["target_data_epochs"] = 50
    with pytest.raises(RuntimeError, match="target_data_epochs"):
        successor.validate_contract(changed)


def test_successor_without_replayed_rf_parent_exits_without_candidate(
    tmp_path, monkeypatch,
):
    contract = _contract(tmp_path)
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(successor, "validate_contract", lambda value: None)
    worker = successor.ResidualSynthesis4090Successor(contract_path)
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
        / "RESIDUAL_SYNTHESIS_4090_RESULT.json"
    ).read_text(encoding="utf-8"))
    assert result["status"] == "SYNTHESIS_INAPPLICABLE_RFAMMCRB_NOT_REPLAYED"
    assert result["old_g3_run"] is False

