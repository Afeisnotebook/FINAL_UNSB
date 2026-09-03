from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from research.paper_aio import distribution
from research.paper_aio.run import parser


def _freeze(tmp_path, **updates):
    portfolio = tmp_path / "PAPER_ALGORITHM_PORTFOLIO.json"
    portfolio.write_text("{}", encoding="utf-8")
    review = tmp_path / "FREEZE_REVIEW.json"
    review.write_text("{}", encoding="utf-8")
    claims = ["frozen paper claim"]
    value = {
        "schema": distribution.FREEZE_SCHEMA,
        "status": distribution.FREEZE_STATUS,
        "algorithm_configuration_frozen": True,
        "baseline_configuration_frozen": True,
        "paper_claims_frozen": True,
        "e200_results_frozen": True,
        "primary_epoch": 200,
        "manifest_sha256": distribution.EXPECTED_MANIFEST_SHA256,
        "evaluation_bundle_fingerprint": (
            distribution.FROZEN_EVALUATION_BUNDLE_FINGERPRINT
        ),
        "source_portfolio_sha256": "a" * 64,
        "source_portfolio_path": str(portfolio.resolve()),
        "review_decision": "FREEZE_REVIEW.json",
        "review_decision_sha256": distribution.file_sha256(review),
        "review_decision_git_commit": "c" * 40,
        "paper_claims": claims,
        "paper_claims_sha256": distribution.object_sha256(claims),
        "distribution_lanes": ["input", "plain", "proposal"],
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    value.update(updates)
    path = tmp_path / "FREEZE.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_distribution_requires_explicit_complete_freeze(tmp_path):
    path = _freeze(tmp_path)
    assert distribution.validate_freeze_receipt(path, lane_id="plain")[
        "paper_claims_frozen"
    ] is True

    _freeze(tmp_path, paper_claims_frozen=False)
    with pytest.raises(RuntimeError, match="freeze receipt is invalid"):
        distribution.validate_freeze_receipt(path, lane_id="plain")

    _freeze(tmp_path, confirmation20_opened=True)
    with pytest.raises(RuntimeError, match="freeze receipt is invalid"):
        distribution.validate_freeze_receipt(path, lane_id="plain")

    _freeze(tmp_path, distribution_lanes=["input"])
    with pytest.raises(RuntimeError, match="freeze receipt is invalid"):
        distribution.validate_freeze_receipt(path, lane_id="plain")


def test_distribution_requires_the_exact_freeze_receipt_in_git(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    path = _freeze(root)
    monkeypatch.setattr(distribution, "ROOT", root)
    value = json.loads(path.read_text(encoding="utf-8"))
    portfolio = root / "PAPER_ALGORITHM_PORTFOLIO.json"
    value["source_portfolio_sha256"] = distribution.file_sha256(portfolio)
    review = root / "FREEZE_REVIEW.json"
    review_value = {
        "status": "APPROVE_FULL_DATA_ALGORITHM_BASELINE_AND_CLAIM_FREEZE",
        "human_approval_recorded": True,
        "codex_scientific_review_recorded": True,
        "source_portfolio_sha256": value["source_portfolio_sha256"],
        "distribution_lanes": value["distribution_lanes"],
        "paper_claims_sha256": value["paper_claims_sha256"],
    }
    review.write_text(json.dumps(review_value), encoding="utf-8")
    value["review_decision_sha256"] = distribution.file_sha256(review)
    path.write_text(json.dumps(value), encoding="utf-8")

    def clean_git(command, **kwargs):
        if command[1:3] == ["status", "--porcelain"]:
            return ""
        if command[1:3] == ["log", "-1"]:
            return ("c" if command[-1] == "FREEZE_REVIEW.json" else "b") * 40 + "\n"
        if command[1] == "show":
            return (
                review.read_text(encoding="utf-8")
                if command[-1].endswith(":FREEZE_REVIEW.json")
                else path.read_text(encoding="utf-8")
            )
        raise AssertionError(command)

    monkeypatch.setattr(distribution.subprocess, "check_output", clean_git)
    value, commit = distribution.committed_freeze_identity(path, lane_id="plain")
    assert value["paper_claims_frozen"] is True
    assert commit == "b" * 40

    def dirty_git(command, **kwargs):
        if command[1:3] == ["status", "--porcelain"]:
            return " M FREEZE.json\n"
        raise AssertionError(command)

    monkeypatch.setattr(distribution.subprocess, "check_output", dirty_git)
    with pytest.raises(RuntimeError, match="uncommitted"):
        distribution.committed_freeze_identity(path, lane_id="plain")


class _FakeFID:
    @staticmethod
    def kernel_distance(left, right, *, num_subsets, max_subset_size):
        assert num_subsets == distribution.KID_NUM_SUBSETS
        assert max_subset_size == distribution.KID_MAX_SUBSET_SIZE
        return float(np.random.random() + np.mean(left) - np.mean(right))

    @staticmethod
    def fid_from_feats(left, right):
        return float(np.square(np.mean(left, axis=0) - np.mean(right, axis=0)).sum())


def test_distribution_feature_summary_is_deterministic_and_rng_isolated():
    target = {
        "a": np.zeros((3, 2), dtype=np.float64),
        "b": np.ones((3, 2), dtype=np.float64),
    }
    predictions = [
        {"a": target["a"] + 1, "b": target["b"] + 1},
        {"a": target["a"] + 2, "b": target["b"] + 2},
    ]
    np.random.seed(912)
    state = np.random.get_state()
    first = distribution.summarize_feature_sets(
        lane_id="plain", target=target, predictions=predictions,
        fid_module=_FakeFID,
    )
    after = np.random.random()
    np.random.set_state(state)
    expected_after = np.random.random()
    second = distribution.summarize_feature_sets(
        lane_id="plain", target=target, predictions=predictions,
        fid_module=_FakeFID,
    )

    assert first == second
    assert after == expected_after
    assert first["summary"]["replicate_count"] == 2
    assert first["target_counts"] == {"a": 3, "b": 3}
    assert first["pooled_target_count"] == 6
    assert first["summary"]["pooled_fid_mean"] == pytest.approx(5.0)


def test_distribution_feature_summary_rejects_domain_or_count_mismatch():
    target = {"a": np.zeros((3, 2)), "b": np.zeros((3, 2))}
    with pytest.raises(RuntimeError, match="domains are incomplete"):
        distribution.summarize_feature_sets(
            lane_id="plain", target=target,
            predictions=[{"a": np.zeros((3, 2))}], fid_module=_FakeFID,
        )
    with pytest.raises(RuntimeError, match="feature count differs"):
        distribution.summarize_feature_sets(
            lane_id="plain", target=target,
            predictions=[{"a": np.zeros((2, 2)), "b": np.zeros((3, 2))}],
            fid_module=_FakeFID,
        )


def test_feature_model_hash_handles_scalar_state():
    layer = torch.nn.BatchNorm1d(2)
    first = distribution._model_state_sha256(layer)
    second = distribution._model_state_sha256(layer)
    assert first == second
    assert len(first) == 64


def test_distribution_cli_is_explicitly_post_freeze():
    args = parser().parse_args([
        "--stage", "distribution", "--lane", "input",
        "--freeze-receipt", "FREEZE.json",
        "--receipt-output", "distribution.json",
    ])
    assert args.stage == "distribution"
    assert str(args.freeze_receipt).endswith("FREEZE.json")


def test_distribution_lock_cli_requires_explicit_receipt_set():
    args = parser().parse_args([
        "--stage", "distribution-lock", "--freeze-receipt", "FREEZE.json",
        "--distribution-receipt", "input.json",
        "--distribution-receipt", "plain.json",
        "--receipt-output", "DISTRIBUTION_COHORT.json",
    ])
    assert args.stage == "distribution-lock"
    assert [path.name for path in args.distribution_receipt] == [
        "input.json", "plain.json",
    ]


def test_distribution_cohort_requires_exact_frozen_lanes_and_one_runtime(
    tmp_path, monkeypatch,
):
    freeze_path = tmp_path / "FREEZE.json"
    freeze_path.write_text("{}", encoding="utf-8")
    freeze = {"distribution_lanes": ["input", "plain"]}
    monkeypatch.setattr(
        distribution, "committed_freeze_identity",
        lambda _path, lane_id: (freeze, "f" * 40),
    )
    monkeypatch.setattr(distribution, "protocol_fingerprint", lambda: "protocol")
    environment = {"torch": "fixed", "gpu": "one"}
    clean_fid = {"version": "fixed", "feature_model_state_sha256": "m" * 64}

    def receipt(lane):
        value = {
            "schema": distribution.SCHEMA,
            "status": "PASS_POST_FREEZE_DISCOVERY80_DISTRIBUTION_EVALUATION",
            "lane_id": lane,
            "primary_epoch": 200,
            "count_per_domain": 80,
            "domain_count": 6,
            "protocol_fingerprint": "protocol",
            "evaluation_bundle_fingerprint": (
                distribution.FROZEN_EVALUATION_BUNDLE_FINGERPRINT
            ),
            "evaluation_input_sha256": "i" * 64,
            "freeze_receipt": str(freeze_path.resolve()),
            "freeze_receipt_sha256": distribution.file_sha256(freeze_path),
            "freeze_receipt_object_sha256": distribution.object_sha256(freeze),
            "freeze_git_commit": "f" * 40,
            "checkpoint_unchanged": True,
            "generated_images_retained": False,
            "target_path_read_for_post_freeze_evaluation": True,
            "metric_values_used_for_training_or_scheduling": False,
            "best_checkpoint_selection": False,
            "confirmation_authorized": False,
            "confirmation20_opened": False,
            "metrics": {"summary": {}},
            "environment": environment,
            "clean_fid": clean_fid,
        }
        path = tmp_path / f"{lane}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    input_path = receipt("input")
    plain_path = receipt("plain")
    result = distribution.lock_distribution_cohort(
        freeze_receipt=freeze_path, receipts=[input_path, plain_path],
        destination=tmp_path / "COHORT.json",
    )
    assert result["lanes"] == ["input", "plain"]
    assert result["all_lanes_one_runtime"] is True
    assert result["confirmation_authorized"] is False

    changed = json.loads(plain_path.read_text())
    changed["environment"] = {"gpu": "different"}
    plain_path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="one runtime"):
        distribution.lock_distribution_cohort(
            freeze_receipt=freeze_path, receipts=[input_path, plain_path],
            destination=tmp_path / "OTHER.json",
        )


def test_distribution_cohort_rejects_empty_receipt_set(tmp_path, monkeypatch):
    freeze_path = tmp_path / "FREEZE.json"
    freeze_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        distribution, "committed_freeze_identity",
        lambda _path, lane_id: ({"distribution_lanes": ["input", "plain"]}, "f" * 40),
    )
    with pytest.raises(RuntimeError, match="unique receipt paths"):
        distribution.lock_distribution_cohort(
            freeze_receipt=freeze_path, receipts=[],
            destination=tmp_path / "COHORT.json",
        )
