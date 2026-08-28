import torch

from production import metrics


class Endpoint(torch.nn.Module):
    def forward(self, x, time_index, z):
        return x + z[:, :1, None, None] * 0.0 + time_index[:, None, None, None] * 0.0


def test_crn_bundle_is_repeatable_and_lane_blind():
    kwargs = dict(
        protocol_hash="abc", domain="domain", stem="stem", replicate=2,
        latent_dim=256, height=8, width=8,
    )
    first = metrics.build_rollout_bundle(**kwargs)
    second = metrics.build_rollout_bundle(**kwargs)
    assert metrics.bundle_hash(first) == metrics.bundle_hash(second)
    assert metrics.bundle_hash(first) != metrics.bundle_hash(
        metrics.build_rollout_bundle(**{**kwargs, "replicate": 3})
    )


def test_metrics_identity_values():
    value = torch.full((1, 3, 8, 8), 0.4)
    assert metrics.psnr_unit(value, value) == 120.0
    assert abs(metrics.ssim_unit(value, value) - 1.0) < 1e-6
