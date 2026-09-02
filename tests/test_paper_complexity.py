from types import SimpleNamespace

import pytest
import torch

from research.paper_aio.complexity import parameter_inventory, summarize_milliseconds


def test_summarize_milliseconds_is_stable_and_explicit() -> None:
    result = summarize_milliseconds([4.0, 1.0, 3.0, 2.0])
    assert result["repeats"] == 4
    assert result["median_ms"] == 2.5
    assert result["mean_ms"] == 2.5
    assert result["min_ms"] == 1.0
    assert result["max_ms"] == 4.0


def test_summarize_milliseconds_rejects_invalid_samples() -> None:
    with pytest.raises(ValueError):
        summarize_milliseconds([])
    with pytest.raises(ValueError):
        summarize_milliseconds([1.0, -1.0])


def test_parameter_inventory_counts_each_network() -> None:
    model = SimpleNamespace(
        model_names=["G", "D"],
        netG=torch.nn.Linear(3, 4),
        netD=torch.nn.Linear(4, 1),
    )
    model.optimizers = [
        torch.optim.Adam(
            list(model.netG.parameters()) + list(model.netD.parameters()), lr=1e-3,
        )
    ]
    result = parameter_inventory(model)
    assert result["networks"]["G"]["parameters"] == 16
    assert result["networks"]["D"]["parameters"] == 5
    assert result["unique_parameters"] == 21
    assert result["unique_optimizer_owned_parameters"] == 21
