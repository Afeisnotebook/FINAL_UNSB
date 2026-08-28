from types import SimpleNamespace

import pytest
import torch

from models.hj.model import SBModelHJPatchNCE
from models.sb_model import SBModel


class FakeNetF:
    def __call__(self, features, num_patches, sample_ids):
        del num_patches
        ids = [torch.arange(features[0].shape[-1], device=features[0].device)]
        return features, ids if sample_ids is None else sample_ids


class FakeCriterion:
    def __call__(self, query, key):
        return (query - key).square()


class FakeInactiveModel:
    def __init__(self, device):
        self.real_A = torch.zeros(1, 3, 2, 2, device=device)
        self.time_idx = torch.zeros(1, dtype=torch.long, device=device)
        self.nce_layers = [0]
        self.hj_layers = [0]
        self.netF = FakeNetF()
        self.criterionNCE = [FakeCriterion()]
        self.opt = SimpleNamespace(
            ngf=1, flip_equivariance=False, num_patches=4, lambda_NCE=1.0,
        )

    def netG(self, image, time_index, latent, layers, encode_only=False):
        del time_index, layers, encode_only
        return [image.flatten(2) + latent[:, :1, None]]

    def _hj_active(self):
        return False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG invariant")
def test_inactive_hj_consumes_the_exact_plain_rng_stream():
    device = torch.device("cuda:0")
    source = torch.zeros(1, 3, 2, 2, device=device)
    target = torch.ones_like(source)

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    plain = FakeInactiveModel(device)
    plain_loss = SBModel.calculate_NCE_loss(plain, source, target)
    plain_cpu_rng = torch.get_rng_state().clone()
    plain_cuda_rng = torch.cuda.get_rng_state_all()

    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    hj = FakeInactiveModel(device)
    hj_loss = SBModelHJPatchNCE.calculate_NCE_loss(hj, source, target)
    assert torch.equal(plain_loss, hj_loss)
    assert torch.equal(torch.get_rng_state(), plain_cpu_rng)
    assert all(
        torch.equal(left, right)
        for left, right in zip(torch.cuda.get_rng_state_all(), plain_cuda_rng)
    )
