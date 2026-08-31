from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from operations import local_route1_single_candidate_receipt_successor as successor


CANDIDATE_ID = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
COMMIT = "7" * 40


def _contract(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    candidate_repo = tmp_path / "candidate_repo"
    run_root = tmp_path / "run"
    repo.mkdir()
    candidate_repo.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    value = {
        "schema": successor.SCHEMA,
        "repo": {"path": str(repo), "git_commit": COMMIT},
        "candidate_repo": {"path": str(candidate_repo), "git_commit": COMMIT},
        "source_sha256": {},
        "candidate_receipt_source_sha256": {},
        "python": str(python),
        "run_root": str(run_root),
        "candidate_id": CANDIDATE_ID,
        "trajectory_path": str(
            run_root / "candidates" / CANDIDATE_ID / "CANDIDATE_TRAJECTORY.json"
        ),
        "canonical_receipt_path": str(
            run_root / "operations" / "terminal_receipts" / f"{CANDIDATE_ID}.json"
        ),
        "poll_seconds": 60,
        "timeout_seconds": 43200,
        "scheduling_bridge_only": True,
        "requires_complete_e200_trajectory": True,
        "checkpoint_transfer": False,
        "formula_changed": False,
        "ranking_changed": False,
        "paired_metrics_used_for_scheduling": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _receipt() -> dict:
    return {
        "candidate_id": CANDIDATE_ID,
        "training_git_commit": COMMIT,
        "verification_git_commit": COMMIT,
        "trajectory_status": "COMPLETE_E200_POSITIVE",
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }


def _patch_validation(monkeypatch):
    monkeypatch.setattr(successor, "validate_contract", lambda value: None)
    monkeypatch.setattr(
        successor, "_validate_receipt",
        lambda path: json.loads(path.read_text(encoding="utf-8")),
    )


def test_completed_candidate_publishes_canonical_receipt_without_ranking(
    monkeypatch, tmp_path: Path,
):
    _patch_validation(monkeypatch)
    contract_path = _contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    trajectory = Path(contract["trajectory_path"])
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    receipt_path = Path(contract["canonical_receipt_path"])

    def fake_run(command, **kwargs):
        assert "--receipt" in command
        assert command[command.index("--receipt") + 1] == str(receipt_path)
        assert "ranking" not in " ".join(command).lower()
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(successor.subprocess, "run", fake_run)
    runner = successor.SingleCandidateReceiptSuccessor(contract_path)
    assert runner.run() == 0
    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert state["status"] == "CANONICAL_SOURCE_BOUND_RECEIPT_AVAILABLE"
    assert state["paired_metrics_used_for_scheduling"] is False
    assert state["confirmation20_opened"] is False


def test_existing_valid_receipt_is_idempotent(monkeypatch, tmp_path: Path):
    _patch_validation(monkeypatch)
    contract_path = _contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    receipt_path = Path(contract["canonical_receipt_path"])
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(_receipt()), encoding="utf-8")
    monkeypatch.setattr(
        successor.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    runner = successor.SingleCandidateReceiptSuccessor(contract_path)
    assert runner.run() == 0
    state = json.loads(runner.state_path.read_text(encoding="utf-8"))
    assert state["resumed"] is True


def test_contract_rejects_paired_metric_scheduling(monkeypatch, tmp_path: Path):
    contract_path = _contract(tmp_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["paired_metrics_used_for_scheduling"] = True
    monkeypatch.setattr(
        successor.support,
        "run_text",
        lambda argv, **kwargs: COMMIT if "rev-parse" in argv else "",
    )
    try:
        successor.validate_contract(contract)
    except RuntimeError as error:
        assert "paired_metrics_used_for_scheduling" in str(error)
    else:
        raise AssertionError("paired scheduling must fail closed")
