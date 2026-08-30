from __future__ import annotations

import torch

from models.route1.mcrb import log_direction_covariance, project_actual_displacement


def test_numerical_floor_preserves_clean_initial_covariance_scale():
    covariance = torch.tensor([0.0, 1e-14, 2e-14], dtype=torch.float32)
    logged = log_direction_covariance(covariance, configured_floor=1e-30)
    assert torch.isfinite(logged).all()
    assert logged[1] != logged[2]
    assert logged[0] < logged[1]


def test_safe_displacement_is_exact_identity():
    native = [torch.tensor([-2.0, 3.0], dtype=torch.float32)]
    tangent = [torch.tensor([1.0, 0.0], dtype=torch.float32)]
    projected, diagnostics = project_actual_displacement(native, tangent)
    assert projected is native
    assert torch.equal(projected[0], native[0])
    assert diagnostics.unsafe is False
    assert diagnostics.correction_l2 == 0.0


def test_unsafe_displacement_is_closest_halfspace_projection():
    native = [torch.tensor([2.0, 3.0], dtype=torch.float64)]
    tangent = [torch.tensor([1.0, 0.0], dtype=torch.float64)]
    projected, diagnostics = project_actual_displacement(native, tangent)
    assert diagnostics.unsafe is True
    assert torch.equal(projected[0], torch.tensor([0.0, 3.0], dtype=torch.float64))
    assert diagnostics.native_defect_directional_derivative == 2.0
    assert diagnostics.projected_defect_directional_derivative == 0.0
    # Any feasible candidate has first coordinate <=0; zero is the unique
    # minimum-distance choice and the unconstrained orthogonal coordinate stays.
    alternative = torch.tensor([-1.0, 3.0], dtype=torch.float64)
    assert torch.linalg.vector_norm(projected[0] - native[0]) < torch.linalg.vector_norm(
        alternative - native[0]
    )


def test_zero_or_unused_tangent_is_identity():
    native = [torch.tensor([1.0]), torch.tensor([2.0])]
    projected, diagnostics = project_actual_displacement(
        native, [torch.zeros(1), None]
    )
    assert projected is native
    assert diagnostics.unsafe is False


def test_projection_handles_multiple_parameter_blocks():
    native = [torch.tensor([1.0, 2.0]), torch.tensor([3.0])]
    tangent = [torch.tensor([1.0, 0.0]), torch.tensor([1.0])]
    projected, diagnostics = project_actual_displacement(native, tangent)
    derivative = sum(
        float((block * direction).sum().item())
        for block, direction in zip(projected, tangent)
    )
    assert diagnostics.unsafe is True
    assert derivative <= 0.0
    assert abs(derivative) < 1e-5
    assert projected[0][1].item() == 2.0
