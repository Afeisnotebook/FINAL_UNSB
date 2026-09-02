from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from research.paper_aio import distribution
from research.paper_aio.run import parser


def _freeze(tmp_path, **updates):
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

    def clean_git(command, **kwargs):
        if command[1:3] == ["status", "--porcelain"]:
            return ""
        if command[1:3] == ["log", "-1"]:
            return "b" * 40 + "\n"
        if command[1] == "show":
            return path.read_text(encoding="utf-8")
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
