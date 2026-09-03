import json
import subprocess
from pathlib import Path

import pytest

from research.paper_aio import freeze
from research.paper_aio.run import parser


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _portfolio(path: Path) -> Path:
    result = lambda lane: {"lane_id": lane}
    return _write(path, {
        "schema": "final-unsb-paper-full-data-algorithm-portfolio-v2",
        "status": "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_WITH_DCLGAN_AWAITING_CONFIRMATION_DECISION",
        "primary_epoch": 200,
        "plain_control": result("plain"),
        "methods": {
            "proposal": {"result": result("proposal")},
            "stcgr": {"result": result("G4-01-STRATIFIED-TIME-CONDITIONAL-GF")},
        },
        "external_baselines": {
            "input": result("input"), "cut": result("cut"),
            "cyclegan": result("cyclegan"), "dclgan": result("dclgan"),
        },
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })


def test_freeze_draft_cannot_self_approve(tmp_path: Path) -> None:
    portfolio = _portfolio(tmp_path / "portfolio.json")
    draft = freeze.create_review_draft(
        portfolio=portfolio,
        claims=["Proposal is compared only with its reviewed matched plain."],
        destination=tmp_path / "draft.json",
    )
    assert draft["status"] == freeze.DRAFT_STATUS
    assert draft["human_approval_recorded"] is False
    assert draft["confirmation_authorized"] is False
    assert set(draft["distribution_lanes"]) == {
        "input", "plain", "proposal", "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
        "cut", "cyclegan", "dclgan",
    }


def test_freeze_materialization_requires_committed_explicit_review(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    portfolio = _portfolio(tmp_path / "portfolio.json")
    _, lanes = freeze.validate_portfolio(portfolio)
    claims = ["fixed e200 claim"]
    review = _write(root / "review.json", {
        "schema": freeze.REVIEW_SCHEMA,
        "status": freeze.REVIEW_STATUS,
        "source_portfolio_path": str(portfolio.resolve()),
        "source_portfolio_sha256": freeze.file_sha256(portfolio),
        "distribution_lanes": lanes,
        "paper_claims": claims,
        "paper_claims_sha256": freeze.object_sha256(claims),
        "human_approval_recorded": True,
        "codex_scientific_review_recorded": True,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    monkeypatch.setattr(freeze, "ROOT", root)
    monkeypatch.setattr(
        freeze, "_committed_json",
        lambda path: (json.loads(review.read_text()), "d" * 40, "review.json"),
    )
    receipt = freeze.materialize_freeze_receipt(
        portfolio=portfolio, review_decision=review,
        destination=root / "freeze.json",
    )
    assert receipt["paper_claims_frozen"] is True
    assert receipt["confirmation_authorized"] is False
    assert receipt["review_decision_git_commit"] == "d" * 40


def test_freeze_cli_requires_explicit_stages() -> None:
    draft = parser().parse_args([
        "--stage", "freeze-draft", "--portfolio", "portfolio.json",
        "--receipt-output", "draft.json", "--paper-claim", "claim",
    ])
    materialize = parser().parse_args([
        "--stage", "freeze-materialize", "--portfolio", "portfolio.json",
        "--review-decision", "review.json", "--receipt-output", "freeze.json",
    ])
    assert draft.stage == "freeze-draft"
    assert materialize.stage == "freeze-materialize"


def test_freeze_draft_rejects_empty_claim_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        freeze.create_review_draft(
            portfolio=_portfolio(tmp_path / "portfolio.json"),
            claims=[], destination=tmp_path / "draft.json",
        )


def test_committed_review_and_freeze_form_a_real_git_chain(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    portfolio = _portfolio(tmp_path / "portfolio.json")
    _, lanes = freeze.validate_portfolio(portfolio)
    claims = ["Every reported comparison uses its frozen e200 protocol."]
    review = _write(root / "review.json", {
        "schema": freeze.REVIEW_SCHEMA,
        "status": freeze.REVIEW_STATUS,
        "source_portfolio_path": str(portfolio.resolve()),
        "source_portfolio_sha256": freeze.file_sha256(portfolio),
        "distribution_lanes": lanes,
        "paper_claims": claims,
        "paper_claims_sha256": freeze.object_sha256(claims),
        "human_approval_recorded": True,
        "codex_scientific_review_recorded": True,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    subprocess.run(["git", "add", "review.json"], cwd=root, check=True)
    subprocess.run([
        "git", "-c", "user.name=Freeze Test", "-c",
        "user.email=freeze@example.invalid", "commit", "-q", "-m", "review",
    ], cwd=root, check=True)
    monkeypatch.setattr(freeze, "ROOT", root)
    receipt_path = root / "freeze.json"
    freeze.materialize_freeze_receipt(
        portfolio=portfolio, review_decision=review, destination=receipt_path,
    )
    subprocess.run(["git", "add", "freeze.json"], cwd=root, check=True)
    subprocess.run([
        "git", "-c", "user.name=Freeze Test", "-c",
        "user.email=freeze@example.invalid", "commit", "-q", "-m", "freeze",
    ], cwd=root, check=True)
    from research.paper_aio import distribution

    monkeypatch.setattr(distribution, "ROOT", root)
    receipt, commit = distribution.committed_freeze_identity(
        receipt_path, lane_id="plain",
    )
    assert len(commit) == 40
    assert receipt["source_portfolio_sha256"] == freeze.file_sha256(portfolio)
    assert receipt["confirmation_authorized"] is False
