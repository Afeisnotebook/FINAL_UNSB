import json
from pathlib import Path

import pytest
import torch

from operations.paper_aio_amtnc_precision_checkpoint_gate import close_gate
from research.local_route1.runtime import full_state_hash
from research.paper_aio.protocol import file_sha256


def _checkpoint(root: Path, *, step: int, promoted: bool = True) -> tuple[Path, Path]:
    epoch = step // 8553
    state = {
        "step": torch.tensor(1.0),
        "exp_avg": torch.tensor([1.0]),
        "exp_avg_sq": torch.tensor(
            [1e40 if promoted else 1.0],
            dtype=torch.float64 if promoted else torch.float32,
        ),
    }
    metadata = {
        "lane_id": "amtnc",
        "git_commit": "g" * 40,
        "protocol_fingerprint": "p" * 64,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    payload = {
        "schema": "final-unsb-paper-aio-full-state-v1",
        "step": step,
        "physical_epoch_completed": epoch,
        "metadata": metadata,
        "model": {
            "networks": {"G": {"w": torch.tensor([1.0])}},
            "optimizers": [{"state": {0: state}, "param_groups": []}],
            "schedulers": [],
            "method": {"amtnc": {
                "second_moment_precision_promotions": 1 if promoted else 0,
                "first_second_moment_precision_promotion": (
                    {"player": "GF", "zero_based_update": 1_665_000,
                     "parameter_count": 1} if promoted else None
                ),
            }},
        },
        "rng": {},
        "samplers": {},
    }
    checkpoint = root / f"{step}.pt"
    torch.save(payload, checkpoint)
    sidecar = Path(str(checkpoint) + ".json")
    sidecar.write_text(json.dumps({
        "schema": "final-unsb-paper-aio-full-state-v1",
        "lane_id": "amtnc",
        "step": step,
        "physical_epoch_completed": epoch,
        "full_state_sha256": file_sha256(checkpoint),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": metadata,
    }), encoding="utf-8")
    return checkpoint, sidecar


def _args(tmp_path: Path, first: tuple[Path, Path], second: tuple[Path, Path]):
    from argparse import Namespace
    return Namespace(
        e195_checkpoint=first[0], e195_sidecar=first[1],
        resume_probe_checkpoint=second[0], resume_probe_sidecar=second[1],
        expected_git_commit="g" * 40,
        expected_protocol_fingerprint="p" * 64,
        output=tmp_path / "gate.json",
    )


def test_precision_checkpoint_gate_requires_finite_promotion_and_resume(tmp_path):
    first = _checkpoint(tmp_path, step=1_667_835)
    second = _checkpoint(tmp_path, step=1_667_836)
    result = close_gate(_args(tmp_path, first, second))
    assert result["status"] == "PASS_E195_FINITE_PROMOTION_AND_EXACT_RESUME"
    assert result["e195"]["float64_exp_avg_sq_tensors"] == 1
    assert result["resume_probe"]["float64_exp_avg_sq_tensors"] == 1


def test_precision_checkpoint_gate_rejects_missing_promotion(tmp_path):
    first = _checkpoint(tmp_path, step=1_667_835, promoted=False)
    second = _checkpoint(tmp_path, step=1_667_836, promoted=False)
    with pytest.raises(RuntimeError, match="promotion"):
        close_gate(_args(tmp_path, first, second))
