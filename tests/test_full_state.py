from pathlib import Path

import torch

from production import common, full_state


class DummyModel:
    def __init__(self):
        self.model_names = ["G"]
        self.netG = torch.nn.Linear(3, 2)
        self.optimizers = [torch.optim.Adam(self.netG.parameters(), lr=1e-3)]
        self.schedulers = [torch.optim.lr_scheduler.StepLR(self.optimizers[0], 1)]
        self.extra = {"counter": 7}

    def get_extra_training_state(self):
        return dict(self.extra)

    def load_extra_training_state(self, state):
        self.extra = dict(state)


def test_full_state_restores_network_optimizer_scheduler_rng_and_method(tmp_path: Path):
    common.apply_determinism(2026)
    model = DummyModel()
    loss = model.netG(torch.ones(1, 3)).sum()
    loss.backward()
    model.optimizers[0].step()
    model.schedulers[0].step()
    expected_hash = common.state_tensor_sha256({"G": model.netG})
    path = tmp_path / "state.pt"
    full_state.save(path, model, metadata={"lane_id": "dummy", "epoch_completed": 1})
    expected_next = torch.rand(4)
    with torch.no_grad():
        model.netG.weight.zero_()
    model.extra = {"counter": -1}
    restored = full_state.load(path, model, expected={"lane_id": "dummy"})
    assert restored["epoch_completed"] == 1
    assert common.state_tensor_sha256({"G": model.netG}) == expected_hash
    assert model.extra == {"counter": 7}
    assert torch.equal(torch.rand(4), expected_next)
