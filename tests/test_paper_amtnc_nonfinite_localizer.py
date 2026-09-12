from __future__ import annotations

import pytest
import torch

from operations.paper_aio_amtnc_nonfinite_localizer import (
    replay_expected_metadata,
    summarize_replica_geometry,
)


def test_localizer_distinguishes_source_gradient_nonfinite():
    value = summarize_replica_geometry(
        (torch.tensor([float("inf")]),),
        (torch.tensor([1.0]),),
        (torch.tensor([1.0]),),
        parameter_names=("D.weight",),
    )
    assert value["first_issue"]["parameter_name"] == "D.weight"
    assert value["first_issue"]["category"] == "SOURCE_GRADIENT_NONFINITE"


def test_localizer_distinguishes_metric_multiplication_overflow():
    value = summarize_replica_geometry(
        (torch.tensor([4.0e30], dtype=torch.float32),),
        (torch.tensor([4.0e30], dtype=torch.float32),),
        (torch.tensor([1.0e8], dtype=torch.float32),),
        parameter_names=("G.weight",),
    )
    issue = value["first_issue"]
    assert issue["category"] == "ADAM_METRIC_MULTIPLICATION_OVERFLOW"
    assert issue["summaries"]["replica_mean"]["all_finite"] is True
    assert issue["summaries"]["scaled_mean"]["all_finite"] is False


def test_localizer_distinguishes_float32_geometry_product_overflow():
    value = summarize_replica_geometry(
        (torch.tensor([2.0e11], dtype=torch.float32),),
        (torch.tensor([2.0e11], dtype=torch.float32),),
        (torch.tensor([1.0e8], dtype=torch.float32),),
        parameter_names=("E.weight",),
    )
    issue = value["first_issue"]
    assert issue["category"] == "FLOAT32_GEOMETRY_PRODUCT_OVERFLOW"
    assert issue["summaries"]["scaled_mean"]["all_finite"] is True
    assert issue["summaries"]["scaled_mean_square"]["all_finite"] is False


def test_localizer_reports_finite_contribution_without_issue():
    value = summarize_replica_geometry(
        (torch.tensor([1.0]),),
        (torch.tensor([3.0]),),
        (torch.tensor([2.0]),),
        parameter_names=("F.weight",),
    )
    assert value["first_issue"] is None
    assert value["all_issues"] == []
    assert value["category_counts"] == {"FINITE_TENSOR_CONTRIBUTION": 1}


def _metadata(*, commit="new", fingerprint="new-fingerprint"):
    return {
        "project_id": "paper",
        "lane_id": "amtnc",
        "lane_config_sha256": "lane",
        "seed": 2026,
        "manifest_sha256": "manifest",
        "protocol_fingerprint": fingerprint,
        "e0_scientific_state_sha256": "e0",
        "git_commit": commit,
        "steps_per_data_epoch": 8553,
        "target_updates": 1710600,
        "batch_size": 1,
        "sampling_measure": "official_image_proportional_unpaired",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_cross_version_replay_requires_both_parent_identities():
    payload = {"metadata": _metadata(commit="old", fingerprint="old-fingerprint")}
    current = _metadata()
    with pytest.raises(RuntimeError, match="supplied together"):
        replay_expected_metadata(
            payload, current, required_parent_git_commit="old",
        )


def test_cross_version_replay_admits_only_named_parent_identity():
    parent = _metadata(commit="old", fingerprint="old-fingerprint")
    payload = {"metadata": parent}
    current = _metadata()
    observed = replay_expected_metadata(
        payload,
        current,
        required_parent_git_commit="old",
        required_parent_protocol_fingerprint="old-fingerprint",
    )
    assert observed == parent


def test_cross_version_replay_rejects_other_scientific_identity_change():
    parent = _metadata(commit="old", fingerprint="old-fingerprint")
    payload = {"metadata": parent}
    current = _metadata()
    current["seed"] = 2027
    with pytest.raises(RuntimeError, match="identity mismatch for seed"):
        replay_expected_metadata(
            payload,
            current,
            required_parent_git_commit="old",
            required_parent_protocol_fingerprint="old-fingerprint",
        )
