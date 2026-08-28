import torch
import torch.nn.functional as F

from models.det_pad import DeterministicReflectionPad2d


def _cases():
    for shape in [(1, 1, 5, 6), (2, 3, 8, 9)]:
        for pad in (1, 2, 3):
            yield shape, pad


def test_forward_bitwise_matches_reflect():
    for shape, pad in _cases():
        x = torch.randn(shape, dtype=torch.float64)
        got = DeterministicReflectionPad2d(pad)(x)
        ref = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        assert torch.equal(got, ref)


def test_forward_non_symmetric_tuple():
    x = torch.randn(1, 2, 6, 7, dtype=torch.float64)
    got = DeterministicReflectionPad2d((1, 2, 2, 1))(x)
    ref = F.pad(x, (1, 2, 2, 1), mode="reflect")
    assert torch.equal(got, ref)


def test_backward_matches_reference():
    for shape, pad in _cases():
        x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
        got = DeterministicReflectionPad2d(pad)(x).sum()
        got.backward()
        got_grad = x.grad.clone()
        x.grad = None

        ref = F.pad(x, (pad, pad, pad, pad), mode="reflect").sum()
        ref.backward()
        ref_grad = x.grad.clone()
        assert torch.equal(got_grad, ref_grad)


def test_backward_is_reproducible():
    for shape, pad in _cases():
        grads = []
        for _ in range(2):
            x = torch.randn(shape, dtype=torch.float64, requires_grad=True)
            DeterministicReflectionPad2d(pad)(x).sum().backward()
            grads.append(x.grad.clone())
        assert torch.equal(grads[0], grads[1])
