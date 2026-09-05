from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from operations import paper_aio_high_resolution_inference as highres
from research.paper_aio.protocol import LaneSpec, _fingerprinted_files


class _IdentityGenerator(torch.nn.Module):
    def forward(self, value):
        return value


class _ExternalModel:
    def __init__(self):
        self.netG_A = _IdentityGenerator()
        self.model_names = ["G_A"]
        self.device = torch.device("cpu")
        self.opt = SimpleNamespace(ngf=64)

    def eval(self):
        self.netG_A.eval()


class _FakeLPIPS(torch.nn.Module):
    def forward(self, prediction, target):
        return (prediction - target).abs().mean().reshape(1)


def _rows(tmp_path):
    rows = []
    for index in range(6):
        source = tmp_path / f"source_{index}.png"
        target = tmp_path / f"target_{index}.png"
        image = Image.new("RGB", (11 + index, 9 + index), color=(20 + index, 30, 40))
        image.save(source)
        image.save(target)
        rows.append({
            "domain": f"domain_{index}", "stem": f"sample_{index}",
            "order": "0", "split": "discovery",
            "input_relpath": source.name, "target_relpath": target.name,
            "input_bytes": str(source.stat().st_size),
            "target_bytes": str(target.stat().st_size),
            "input_sha256": highres.file_sha256(source),
            "target_sha256": highres.file_sha256(target),
        })
    return rows


def _freeze(tmp_path):
    path = tmp_path / "FREEZE.json"
    path.write_text("{}", encoding="utf-8")
    return path


def _patch_runtime(monkeypatch, tmp_path, rows):
    freeze = {"distribution_lanes": ["input", "cyclegan"]}
    monkeypatch.setattr(highres, "COUNT_PER_DOMAIN", 1)
    monkeypatch.setattr(highres, "select_discovery", lambda _rows, _count: list(rows))
    monkeypatch.setattr(
        highres, "committed_freeze_identity",
        lambda _path, lane_id: (freeze, "f" * 40),
    )
    monkeypatch.setattr(highres, "protocol_fingerprint", lambda: "protocol")
    monkeypatch.setattr(highres, "environment_record", lambda: {"runtime": "one"})
    monkeypatch.setattr(highres, "_lpips", lambda _device: _FakeLPIPS())
    return freeze


def test_high_resolution_code_does_not_change_training_fingerprint_scope():
    assert highres.SCRIPT_PATH not in _fingerprinted_files()
    assert highres.SPATIAL_POLICY["inference_resolution"] == 256
    assert highres.SPATIAL_POLICY["retraining"] is False
    assert highres.SPATIAL_POLICY["checkpoint_selection"] is False


def test_high_resolution_fails_before_touching_discovery_without_committed_freeze(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        highres, "committed_freeze_identity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("freeze absent")),
    )
    monkeypatch.setattr(
        highres, "select_discovery",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("split touched")),
    )
    with pytest.raises(RuntimeError, match="freeze absent"):
        highres.evaluate_high_resolution(
            model=None, spec=None, rows=[], data_root=tmp_path,
            destination=tmp_path / "result.json",
            freeze_receipt=tmp_path / "freeze.json", checkpoint=None,
            checkpoint_step=None, checkpoint_metadata=None, gpu=-1,
        )


def test_fixed_256_evaluation_is_rng_and_checkpoint_read_only(tmp_path, monkeypatch):
    rows = _rows(tmp_path)
    _patch_runtime(monkeypatch, tmp_path, rows)
    freeze = _freeze(tmp_path)
    checkpoint = tmp_path / "e200.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    model = _ExternalModel()
    spec = LaneSpec(
        id="cyclegan", backend="internal", family="external_translation",
        model="cycle_gan", role="baseline", method={},
    )
    torch.manual_seed(912)
    state = torch.get_rng_state().clone()
    result = highres.evaluate_high_resolution(
        model=model, spec=spec, rows=rows, data_root=tmp_path,
        destination=tmp_path / "cyclegan.json", freeze_receipt=freeze,
        checkpoint=checkpoint, checkpoint_step=1_710_600,
        checkpoint_metadata={
            "paired_controller_access": False, "confirmation20_opened": False,
        },
        gpu=-1,
    )
    assert torch.equal(torch.get_rng_state(), state)
    assert checkpoint.read_bytes() == b"fixed checkpoint"
    assert result["status"] == highres.STATUS
    assert result["macro_psnr"] == pytest.approx(120.0)
    assert result["macro_lpips"] == pytest.approx(0.0)
    assert result["replicate_count"] == 1
    assert result["main_table"] is False
    assert result["confirmation20_opened"] is False
    highres.validate_receipt(result, expected_lane="cyclegan")


def test_high_resolution_cohort_requires_all_frozen_lanes_in_one_runtime(
    tmp_path, monkeypatch,
):
    rows = _rows(tmp_path)
    _patch_runtime(monkeypatch, tmp_path, rows)
    freeze = _freeze(tmp_path)
    input_result = highres.evaluate_high_resolution(
        model=None, spec=None, rows=rows, data_root=tmp_path,
        destination=tmp_path / "input.json", freeze_receipt=freeze,
        checkpoint=None, checkpoint_step=None, checkpoint_metadata=None, gpu=-1,
    )
    checkpoint = tmp_path / "e200.pt"
    checkpoint.write_bytes(b"fixed checkpoint")
    model_result = highres.evaluate_high_resolution(
        model=_ExternalModel(),
        spec=LaneSpec(
            id="cyclegan", backend="internal", family="external_translation",
            model="cycle_gan", role="baseline", method={},
        ),
        rows=rows, data_root=tmp_path,
        destination=tmp_path / "cyclegan.json", freeze_receipt=freeze,
        checkpoint=checkpoint, checkpoint_step=1_710_600,
        checkpoint_metadata={
            "paired_controller_access": False, "confirmation20_opened": False,
        },
        gpu=-1,
    )
    assert input_result["evaluation_input_sha256"] == model_result["evaluation_input_sha256"]
    cohort = highres.lock_high_resolution_cohort(
        freeze_receipt=freeze,
        receipts=[tmp_path / "input.json", tmp_path / "cyclegan.json"],
        destination=tmp_path / "cohort.json",
    )
    assert cohort["lanes"] == ["cyclegan", "input"]
    assert cohort["supplementary_only"] is True
    broken = json.loads((tmp_path / "cyclegan.json").read_text(encoding="utf-8"))
    broken["environment"] = {"runtime": "another"}
    (tmp_path / "cyclegan.json").write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(RuntimeError, match="one evaluator"):
        highres.lock_high_resolution_cohort(
            freeze_receipt=freeze,
            receipts=[tmp_path / "input.json", tmp_path / "cyclegan.json"],
            destination=tmp_path / "another_cohort.json",
        )


def test_high_resolution_cli_is_explicitly_post_freeze():
    args = highres.parser().parse_args([
        "--mode", "evaluate", "--lane", "input",
        "--data-root", "data", "--freeze-receipt", "FREEZE.json",
        "--receipt-output", "input_256.json",
    ])
    assert args.mode == "evaluate"
    assert args.lane == "input"
    lock = highres.parser().parse_args([
        "--mode", "lock", "--data-root", "data",
        "--freeze-receipt", "FREEZE.json",
        "--receipt-output", "HIGH_RESOLUTION_COHORT.json",
        "--high-resolution-receipt", "input.json",
        "--high-resolution-receipt", "plain.json",
    ])
    assert [path.name for path in lock.high_resolution_receipt] == [
        "input.json", "plain.json",
    ]
