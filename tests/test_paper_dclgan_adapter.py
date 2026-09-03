from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from operations import paper_aio_dclgan_adapter as adapter
from operations import paper_aio_dclgan_gate_supervisor as gate_supervisor
from research.local_route1.runtime import full_state_hash
from research.local_route1.protocol import portable_source_sha256
from research.paper_aio.evaluate import _prediction


ROOT = Path(__file__).resolve().parents[1]


def _rows() -> list[dict]:
    rows = []
    for domain in ("a", "b", "c", "d", "e", "f"):
        rows.extend({
            "domain": domain,
            "split": "discovery",
            "stem": f"{index:04d}",
            "order": str(index),
        } for index in range(80))
        rows.extend({
            "domain": domain,
            "split": "confirmation",
            "stem": f"c{index:04d}",
            "order": str(index),
        } for index in range(20))
    return rows


def test_dclgan_adapter_fingerprint_excludes_installation_path() -> None:
    receipt = {
        "repository": "official",
        "authority": "author_official",
        "commit": "c" * 40,
        "tracked_source_clean": True,
        "hashes": {"model": "m" * 64},
        "upstream_root": "/host/one/DCLGAN",
    }
    manifest = ROOT / "manifests" / "FULL_DATA_MANIFEST.csv"
    first = adapter.adapter_fingerprint(
        upstream_receipt=receipt, manifest_path=manifest,
    )
    receipt["upstream_root"] = "/different/host/DCLGAN"
    second = adapter.adapter_fingerprint(
        upstream_receipt=receipt, manifest_path=manifest,
    )
    assert first == second
    changed = copy.deepcopy(receipt)
    changed["commit"] = "d" * 40
    assert adapter.adapter_fingerprint(
        upstream_receipt=changed, manifest_path=manifest,
    ) != first


def test_dclgan_source_hash_is_cross_platform_line_ending_stable(
    tmp_path: Path,
) -> None:
    windows = tmp_path / "windows.txt"
    linux = tmp_path / "linux.txt"
    windows.write_bytes(b"first\r\nsecond\r\n")
    linux.write_bytes(b"first\nsecond\n")
    assert portable_source_sha256(windows) == portable_source_sha256(linux)


def test_dclgan_confirmation_interface_fails_before_row_access(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="confirmation20 access blocked"):
        adapter.select_dclgan_evaluation_rows(
            _rows(), split="confirmation", count_per_domain=20,
        )
    receipt = adapter.confirmation_lock_gate(output_root=tmp_path)
    assert receipt["status"] == "PASS_CONFIRMATION20_UNADDRESSABLE"
    assert receipt["attempt_rejected_before_manifest_or_image_access"] is True
    assert receipt["confirmation20_opened"] is False


def test_dclgan_discovery_selector_is_frozen() -> None:
    selected = adapter.select_dclgan_evaluation_rows(
        _rows(), split="discovery", count_per_domain=70,
    )
    assert len(selected) == 6 * 70
    assert {row["split"] for row in selected} == {"discovery"}


def test_dclgan_paper_prediction_uses_degraded_to_clean_generator() -> None:
    source = torch.tensor([1.0])
    model = SimpleNamespace(netG_A=lambda value: value + 2.0)
    prediction = _prediction(
        model, adapter.dclgan_lane_spec(), source, bundle={}, nfe=1,
    )
    assert torch.equal(prediction, torch.tensor([3.0]))


@pytest.mark.parametrize("padding", [0, 1, (2, 3), (1, 2, 3, 2)])
def test_deterministic_reflection_pad_is_forward_exact(padding) -> None:
    value = torch.randn(2, 3, 8, 9)
    actual = adapter.DeterministicReflectionPad2d(padding)(value)
    if isinstance(padding, int):
        normalized = (padding, padding, padding, padding)
    elif len(padding) == 2:
        normalized = (padding[0], padding[1], padding[0], padding[1])
    else:
        normalized = padding
    expected = torch.nn.functional.pad(value, normalized, mode="reflect")
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA gate")
def test_deterministic_reflection_pad_has_repeatable_cuda_backward() -> None:
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        source = torch.randn(1, 3, 8, 9, device="cuda")
        gradients = []
        for _ in range(2):
            value = source.detach().clone().requires_grad_(True)
            output = adapter.DeterministicReflectionPad2d((2, 3, 1, 2))(value)
            output.square().sum().backward()
            gradients.append(value.grad.detach().cpu())
        assert torch.equal(gradients[0], gradients[1])
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def test_dclgan_full_model_optimizer_scheduler_state_roundtrip() -> None:
    class Model:
        def __init__(self):
            self.model_names = ["G_A", "F1"]
            self.netG_A = torch.nn.Linear(2, 2)
            self.netF1 = torch.nn.Linear(2, 1)
            self.device = torch.device("cpu")
            self.optimizers = [torch.optim.Adam(
                [*self.netG_A.parameters(), *self.netF1.parameters()], lr=0.01,
            )]
            self.schedulers = [torch.optim.lr_scheduler.StepLR(
                self.optimizers[0], step_size=1,
            )]

    model = Model()
    loss = model.netF1(model.netG_A(torch.ones(1, 2))).sum()
    loss.backward()
    model.optimizers[0].step()
    model.schedulers[0].step()
    expected = adapter.capture_model_state(model)
    expected_hash = full_state_hash(expected)

    with torch.no_grad():
        for parameter in model.netG_A.parameters():
            parameter.add_(10.0)
    model.optimizers[0].param_groups[0]["lr"] = 0.5
    model.schedulers[0].step()
    adapter.load_captured_model_state(model, expected)

    assert full_state_hash(adapter.capture_model_state(model)) == expected_hash


def test_dclgan_cli_exposes_only_frozen_engineering_stages() -> None:
    choices = adapter.build_parser()._option_string_actions["--stage"].choices
    assert set(choices) == {
        "preflight", "train", "exact-resume-gate", "evaluate",
        "evaluation-repeat-gate", "confirmation-lock-gate", "authorize",
    }
    assert adapter.ENGINEERING_MAX_UPDATES == 1_000


def test_dclgan_long_authorization_requires_complete_1000_update_gpu_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "c" * 40
    monkeypatch.setattr(adapter, "verify_upstream", lambda _root: {
        "commit": "u" * 40, "upstream_root": "/source",
    })
    monkeypatch.setattr(adapter, "adapter_fingerprint", lambda **_kwargs: "fp")
    monkeypatch.setattr(adapter, "git_commit", lambda: commit)
    runtime_host = {"hostname": "test", "gpu_index": 0, "gpu_uuid": "gpu"}
    monkeypatch.setattr(adapter, "runtime_host_identity", lambda _gpu: runtime_host)
    monkeypatch.setattr(adapter, "file_sha256", lambda _path: "h" * 64)
    monkeypatch.setattr(
        shutil, "disk_usage",
        lambda _path: shutil._ntuple_diskusage(10**12, 0, 10**12),
    )

    paths = {
        "preflight": tmp_path / "gates" / "DCLGAN_PREFLIGHT.json",
        "resume": (
            tmp_path / "gates" / "DCLGAN_EXACT_RESUME"
            / "EXACT_RESUME_RECEIPT.json"
        ),
        "evaluation": tmp_path / "gates" / "DCLGAN_EVALUATION_REPEAT.json",
        "confirmation": tmp_path / "gates" / "DCLGAN_CONFIRMATION_LOCK.json",
        "run": tmp_path / "lanes" / "dclgan" / "RUN_STATE.json",
    }
    payloads = {
        "preflight": {
            "status": "PASS_SOURCE_AND_CONTROLLED_DATA_PREFLIGHT",
            "adapter_fingerprint": "fp",
            "data": {"content_hashes_verified": True, "content_hash_files": 17_106},
        },
        "resume": {
            "status": "PASS_EXACT_RESUME", "exact": True,
            "total_updates": 1_000, "split_updates": 500,
            "adapter_fingerprint": "fp",
        },
        "evaluation": {
            "status": "PASS_EVALUATION_REPEAT_EXACT", "exact": True,
            "updates": 1_000, "adapter_fingerprint": "fp",
        },
        "confirmation": {
            "status": "PASS_CONFIRMATION20_UNADDRESSABLE",
            "adapter_git_commit": commit,
            "adapter_source_sha256": "h" * 64,
            "confirmation20_opened": False,
        },
        "run": {
            "status": "ENGINEERING_PAUSE", "final_updates": 1_000,
            "updates_per_second": 2.0,
            "metadata": {
                "adapter_fingerprint": "fp", "runtime_host": runtime_host,
            },
            "runtime": {
                "cuda_available": True, "peak_allocated_bytes": 1024,
            },
        },
    }
    for label, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payloads[label]), encoding="utf-8")
    checkpoint = tmp_path / "lanes" / "dclgan" / "full_state_latest.pt"
    checkpoint.write_bytes(b"checkpoint")

    result = adapter.authorize_long_training(
        upstream_root=tmp_path, manifest_path=tmp_path / "manifest.csv",
        output_root=tmp_path,
    )
    assert result["status"] == "PASS_LONG_TRAINING_AUTHORIZED"
    assert result["disk"]["fixed_200_gib_threshold_used"] is False

    payloads["resume"]["total_updates"] = 4
    paths["resume"].write_text(json.dumps(payloads["resume"]), encoding="utf-8")
    with pytest.raises(RuntimeError, match="1000-vs-500"):
        adapter.authorize_long_training(
            upstream_root=tmp_path, manifest_path=tmp_path / "manifest.csv",
            output_root=tmp_path,
        )


def test_dclgan_supervisor_has_fixed_metric_blind_recoverable_stages(
    tmp_path: Path,
) -> None:
    args = SimpleNamespace(
        repo=tmp_path / "repo", upstream_root=tmp_path / "source",
        manifest=tmp_path / "manifest.csv", train_view=tmp_path / "view",
        data_root=tmp_path / "data", output=tmp_path / "output", gpu=0,
    )
    stages = gate_supervisor._stage_commands(args, Path("python"))
    assert [stage for stage, _command in stages] == [
        "preflight", "confirmation_lock", "exact_resume_1000_500",
        "capacity_train_1000", "evaluation_repeat", "authorize",
    ]
    train = dict(stages)["capacity_train_1000"]
    assert "--resume" in train
    exact = dict(stages)["exact_resume_1000_500"]
    assert exact[exact.index("--gate-total-updates") + 1] == "1000"
    assert exact[exact.index("--gate-split-updates") + 1] == "500"


def test_dclgan_gate_terminal_status_is_not_local_host_specific() -> None:
    source = Path(gate_supervisor.__file__).read_text(encoding="utf-8")
    assert "COMPLETE_HOST_BOUND_GPU_GATE_AUTHORIZED" in source
    assert "COMPLETE_LOCAL_GPU_GATE_AUTHORIZED" not in source
