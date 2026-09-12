from __future__ import annotations

import torch

from operations.paper_aio_amtnc_nonfinite_localizer import summarize_replica_geometry


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
