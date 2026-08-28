"""Read-only epoch-level diagnostic logging for intervention mechanism evidence."""

from __future__ import annotations

import json
from pathlib import Path


class EpochDiagnostics:
    """Append one JSON line per epoch (read-only, does not affect training)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(fields, ensure_ascii=False) + "\n")


def parameter_grad_l2(model, params) -> float:
    """L2 norm of gradients over an iterable of parameters."""
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += float((p.grad.detach() ** 2).sum().item())
    return total ** 0.5
