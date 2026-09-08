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
            "proposal": {
                "algorithm_id": "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
                "result": result("proposal"),
            },
            "stcgr": {
                "algorithm_id": "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
                "result": result("G4-01-STRATIFIED-TIME-CONDITIONAL-GF"),
            },
            "amtnc": {
                "algorithm_id": "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS",
                "result": result("amtnc"),
            },
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


def _reference_ledger(root: Path) -> Path:
    bib = root / "research" / "paper_aio" / "references.bib"
    bib.parent.mkdir(parents=True, exist_ok=True)
    bib.write_bytes(
        b"@article{test2026,\n"
        b"  title={Test},\n  author={Author, A.},\n  year={2026},\n"
        b"  url={https://example.invalid/test}\n}\n",
    )
    return _write(root / "configs" / "PAPER_REFERENCE_LEDGER.json", {
        "schema": "final-unsb-paper-reference-ledger-v1",
        "status": "PRE_RESULT_PRIMARY_METADATA_LOCK_CORE_STABLE_VOLATILE_REFRESH_REQUIRED",
        "bib": {
            "path": "research/paper_aio/references.bib",
            "sha256": freeze.file_sha256(bib),
            "entry_count": 1,
        },
        "required_groups": {"test_group": ["test2026"]},
        "entries": [{
            "citation_key": "test2026",
            "roles": ["test"],
            "metadata_status": "verified_primary",
            "year": 2026,
            "primary_url": "https://example.invalid/test",
        }],
        "submission_day_refresh_required": [{
            "working_key": "volatile_test",
            "status": "volatile_not_in_core_bib",
            "primary_url": "https://example.invalid/volatile",
        }],
        "authority": {
            "metadata_lock_is_novelty_proof": False,
            "volatile_entries_require_fresh_primary_source_review_before_submission": True,
        },
        "scientific_boundaries": {
            "performance_values_read": False,
            "training_or_queue_changed": False,
            "algorithm_selected": False,
            "confirmation20_opened": False,
            "empirical_claim_frozen": False,
        },
    })


def _theory_bundle(root: Path) -> Path:
    _reference_ledger(root)
    artifacts = {
        "map.md": "map",
        "proposal-card.json": json.dumps({"candidate_id": "proposal"}),
        "proposal-family.md": "family",
        "proposal-audit.json": json.dumps({"status": "PASS"}),
        "stcgr-card.json": json.dumps({"candidate_id": "stcgr"}),
        "stcgr-audit.json": json.dumps({"status": "PASS"}),
        "stcgr-semantic.json": json.dumps({"status": "PASS"}),
        "amtnc-card.json": json.dumps({"candidate_id": "amtnc"}),
        "amtnc-audit.json": json.dumps({"status": "PASS"}),
    }
    for relative, text in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def ref(role: str, relative: str) -> dict:
        return {
            "role": role, "path": relative,
            "sha256": freeze.file_sha256(root / relative),
        }

    path = root / "configs" / "PAPER_ALGORITHM_THEORY_BUNDLE.json"
    return _write(path, {
        "schema": "final-unsb-paper-algorithm-theory-bundle-v1",
        "status": "PRE_RESULT_THREE_OPERATOR_THEORY_BUNDLE_FROZEN",
        "canonical_map": {
            "path": "map.md", "sha256": freeze.file_sha256(root / "map.md"),
        },
        "methods": {
            "proposal": {
                "algorithm_id": "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
                "paper_role": "proposal", "pre_adam_property": "mean",
                "artifacts": [
                    ref("derivation_card", "proposal-card.json"),
                    ref("family_derivation", "proposal-family.md"),
                    ref("formula_implementation_audit", "proposal-audit.json"),
                ],
            },
            "stcgr": {
                "algorithm_id": "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
                "paper_role": "stcgr", "pre_adam_property": "mean",
                "artifacts": [
                    ref("derivation_card", "stcgr-card.json"),
                    ref("formula_implementation_audit", "stcgr-audit.json"),
                    ref("independent_operator_semantic_audit", "stcgr-semantic.json"),
                ],
            },
            "amtnc": {
                "algorithm_id": "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS",
                "paper_role": "amtnc", "pre_adam_property": "mean",
                "artifacts": [
                    ref("derivation_card", "amtnc-card.json"),
                    ref("formula_implementation_audit", "amtnc-audit.json"),
                ],
            },
        },
        "claim_boundaries": {
            "pre_adam_conditional_mean_only": True,
            "expected_adam_displacement_unbiased_claimed": False,
            "full_markov_kernel_unbiased_claimed": False,
            "equal_flop_superiority_claimed": False,
            "terminal_singular_drift_repair_claimed": False,
            "full_data_benefit_claimed_before_e200": False,
            "unique_winner_predeclared": False,
        },
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })


def _commit_theory(root: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run([
        "git", "-c", "user.name=Theory Test", "-c",
        "user.email=theory@example.invalid", "commit", "-q", "-m", "theory",
    ], cwd=root, check=True)


def test_freeze_draft_cannot_self_approve(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bundle = _theory_bundle(root)
    _commit_theory(root)
    monkeypatch.setattr(freeze, "ROOT", root)
    portfolio = _portfolio(tmp_path / "portfolio.json")
    draft = freeze.create_review_draft(
        portfolio=portfolio,
        claims=["Proposal is compared only with its reviewed matched plain."],
        destination=tmp_path / "draft.json",
        theory_bundle=bundle,
    )
    assert draft["status"] == freeze.DRAFT_STATUS
    assert draft["paper_reference_ledger"]["entry_count"] == 1
    assert draft["human_approval_recorded"] is False
    assert draft["confirmation_authorized"] is False
    assert set(draft["distribution_lanes"]) == {
        "input", "plain", "proposal", "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
        "amtnc", "cut", "cyclegan", "dclgan",
    }


def test_freeze_materialization_requires_committed_explicit_review(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bundle = _theory_bundle(root)
    _commit_theory(root)
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
        "algorithm_theory_bundle": freeze.theory_bundle_reference(
            bundle, root=root,
        ),
        "paper_reference_ledger": freeze.reference_ledger_reference(
            root=root,
        ),
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
    assert receipt["algorithm_theory_bundle"]["path"] == (
        "configs/PAPER_ALGORITHM_THEORY_BUNDLE.json"
    )


def test_freeze_cli_requires_explicit_stages() -> None:
    draft = parser().parse_args([
        "--stage", "freeze-draft", "--portfolio", "portfolio.json",
        "--receipt-output", "draft.json", "--paper-claim", "claim",
        "--theory-bundle", "theory.json",
    ])
    materialize = parser().parse_args([
        "--stage", "freeze-materialize", "--portfolio", "portfolio.json",
        "--review-decision", "review.json", "--receipt-output", "freeze.json",
    ])
    assert draft.stage == "freeze-draft"
    assert draft.theory_bundle.name == "theory.json"
    assert materialize.stage == "freeze-materialize"


def test_freeze_draft_rejects_empty_claim_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="nonempty"):
        freeze.create_review_draft(
            portfolio=_portfolio(tmp_path / "portfolio.json"),
            claims=[], destination=tmp_path / "draft.json",
        )


def test_freeze_rejects_portfolio_algorithm_identity_drift(
    tmp_path: Path,
) -> None:
    portfolio = _portfolio(tmp_path / "portfolio.json")
    value = json.loads(portfolio.read_text(encoding="utf-8"))
    value["methods"]["proposal"]["algorithm_id"] = "DIFFERENT-OPERATOR"
    portfolio.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity differs from theory"):
        freeze.validate_portfolio(portfolio)


def test_theory_bundle_is_hash_bound_and_rejects_artifact_drift(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bundle = _theory_bundle(root)
    reference = freeze.theory_bundle_reference(bundle, root=root)
    assert reference["algorithm_ids"] == [
        "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS",
        "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
        "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
    ]
    (root / "proposal-family.md").write_text("drift", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        freeze.theory_bundle_reference(bundle, root=root)


def test_theory_bundle_must_match_committed_git_bytes(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bundle = _theory_bundle(root)
    _commit_theory(root)
    reference = freeze.committed_theory_bundle_reference(bundle, root=root)
    assert reference["status"] == "PRE_RESULT_THREE_OPERATOR_THEORY_BUNDLE_FROZEN"

    family = root / "proposal-family.md"
    family.write_text("uncommitted replacement", encoding="utf-8")
    value = json.loads(bundle.read_text(encoding="utf-8"))
    proposal = value["methods"]["proposal"]["artifacts"]
    next(row for row in proposal if row["role"] == "family_derivation")[
        "sha256"
    ] = freeze.file_sha256(family)
    bundle.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="uncommitted changes"):
        freeze.committed_theory_bundle_reference(bundle, root=root)


def test_committed_review_and_freeze_form_a_real_git_chain(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    bundle = _theory_bundle(root)
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
        "algorithm_theory_bundle": freeze.theory_bundle_reference(
            bundle, root=root,
        ),
        "paper_reference_ledger": freeze.reference_ledger_reference(
            root=root,
        ),
        "human_approval_recorded": True,
        "codex_scientific_review_recorded": True,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
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
