"""Deterministic reflection-pad2d (bitwise forward-compatible, deterministic backward)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DeterministicReflectionPad2d(nn.Module):
    """Reflection padding built only from slice/index-select + cat.

    ``forward`` is bitwise identical to ``F.pad(x, pad, mode="reflect")`` for the
    supported pad sizes, and ``backward`` does not rely on PyTorch's
    ``reflection_pad2d_backward_cuda`` (which has no deterministic implementation).
    """

    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, (tuple, list)):
            if len(padding) == 1:
                p = int(padding[0])
                padding = (p, p, p, p)
            elif len(padding) == 2:
                lr, tb = int(padding[0]), int(padding[1])
                padding = (lr, lr, tb, tb)
            elif len(padding) == 4:
                padding = tuple(int(v) for v in padding)
            else:
                raise ValueError("reflection pad must have 1, 2 or 4 entries")
        else:
            raise TypeError("padding must be int or tuple/list")

        left, right, top, bottom = padding
        if min(left, right, top, bottom) < 0:
            raise ValueError("reflection padding must be non-negative")
        self.padding = (left, right, top, bottom)

    @staticmethod
    def _reflect_axis(x: torch.Tensor, pad_before: int, pad_after: int, dim: int) -> torch.Tensor:
        if pad_before == 0 and pad_after == 0:
            return x
        pieces = []
        if pad_before > 0:
            idx = list(range(pad_before, 0, -1))
            pieces.append(x.index_select(dim, torch.tensor(idx, device=x.device)))
        pieces.append(x)
        if pad_after > 0:
            size = int(x.shape[dim])
            idx = list(range(size - 2, size - 2 - pad_after, -1))
            pieces.append(x.index_select(dim, torch.tensor(idx, device=x.device)))
        return torch.cat(pieces, dim=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.padding
        x = self._reflect_axis(x, left, right, 3)   # width
        x = self._reflect_axis(x, top, bottom, 2)   # height
        return x

    def extra_repr(self) -> str:
        return f"padding={self.padding}"
