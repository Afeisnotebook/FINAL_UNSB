from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from operations import paper_aio_dclgan_adapter as adapter
from operations import paper_aio_dclgan_export_successor as exporter
from operations import paper_aio_dclgan_long_supervisor as long_supervisor
from operations import paper_aio_dclgan_target_successor as target_successor
from research.local_route1.runtime import full_state_hash


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _checkpoint(
    path: Path, *, step: int, commit: str = "c" * 40,
    fingerprint: str = "f" * 64,
) -> dict:
    metadata = {
        "adapter_git_commit": commit,
        "adapter_fingerprint": fingerprint,
        "manifest_sha256": adapter.EXPECTED_MANIFEST_SHA256,
        "upstream_commit": "u" * 40,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    payload = {
        "schema": adapter.FULL_STATE_SCHEMA,
        "lane_id": adapter.LANE_ID,
        "step": step,
        "metadata": metadata,
        "model": {"weight": torch.tensor([1.0])},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    sidecar = {
        "schema": adapter.FULL_STATE_SCHEMA,
        "lane_id": adapter.LANE_ID,
        "step": step,
        "physical_epoch_completed": step // 8553,
        "full_state_sha256": adapter.file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": metadata,
    }
    _write_json(Path(str(path) + ".json"), sidecar)
    return payload


def test_dclgan_long_supervisor_command_is_fixed_e200_resume(tmp_path: Path) -> None:
    args = SimpleNamespace(
        repo=tmp_path / "repo",
        upstream_root=tmp_path / "source",
        manifest=tmp_path / "manifest.csv",
        train_view=tmp_path / "view",
        output=tmp_path / "output",
        gpu=0,
    )
    command = long_supervisor.training_command(args)
    assert command[command.index("--stop-after-updates") + 1] == "1710600"
    assert "--resume" in command
    assert not any("psnr" in value.lower() for value in command)


def test_dclgan_terminal_decision_verifies_latest_and_e200(tmp_path: Path) -> None:
    lane = tmp_path / "lanes" / adapter.LANE_ID
    latest = lane / "full_state_latest.pt"
    e200 = lane / "milestones" / "e200.pt"
    _checkpoint(latest, step=long_supervisor.TERMINAL_UPDATES)
    _checkpoint(e200, step=long_supervisor.TERMINAL_UPDATES)
    _write_json(lane / "RUN_STATE.json", {
        "status": "COMPLETE_E200",
        "final_updates": long_supervisor.TERMINAL_UPDATES,
        "final_data_epoch": 200.0,
    })
    decision = long_supervisor.terminal_decision(tmp_path)
    assert decision["complete"] is True
    assert decision["final_updates"] == 1_710_600


def test_dclgan_target_successor_wait_start_and_block() -> None:
    assert target_successor.predecessor_decision({}, "COMPLETE") == "WAIT"
    assert target_successor.predecessor_decision(
        {"status": "COMPLETE"}, "COMPLETE",
    ) == "START"
    assert target_successor.predecessor_decision(
        {"status": "BLOCKED_ENGINEERING"}, "COMPLETE",
    ) == "BLOCK"
    assert target_successor.predecessor_decision(
        {"status": "COMPLETE", "paired_metric_control": True}, "COMPLETE",
    ) == "BLOCK"
    assert target_successor.predecessor_decision(
        {"status": "COMPLETE", "performance_values_read": True}, "COMPLETE",
    ) == "BLOCK"


def test_dclgan_target_accepts_source_bound_matched_plain_v2_terminal_state() -> None:
    predecessor = {
        "schema": "final-unsb-paper-cross-host-plain-successor-v2",
        "status": "COMPLETE_PLAIN_E200",
        "fresh_e0_required": True,
        "cross_host_checkpoint_resume": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert target_successor.predecessor_decision(
        predecessor, "COMPLETE_PLAIN_E200",
    ) == "START"


def test_dclgan_target_commands_preserve_host_bound_gate(tmp_path: Path) -> None:
    args = SimpleNamespace(
        repo=tmp_path / "repo",
        upstream_root=tmp_path / "source",
        manifest=tmp_path / "manifest.csv",
        train_view=tmp_path / "view",
        data_root=tmp_path / "data",
        output=tmp_path / "output",
        gpu=0,
        required_git_commit="c" * 40,
        required_adapter_fingerprint="f" * 64,
    )
    gate = target_successor.gate_command(args)
    long = target_successor.long_command(args)
    assert gate[gate.index("--required-git-commit") + 1] == "c" * 40
    assert long[long.index("--required-adapter-fingerprint") + 1] == "f" * 64
    assert "paper_aio_dclgan_gate_supervisor.py" in gate[1]
    assert "paper_aio_dclgan_long_supervisor.py" in long[1]


def test_dclgan_exporter_accepts_only_source_bound_state(tmp_path: Path) -> None:
    commit = "c" * 40
    fingerprint = "f" * 64
    checkpoint = tmp_path / "e100.pt"
    _checkpoint(
        checkpoint,
        step=100 * 8553,
        commit=commit,
        fingerprint=fingerprint,
    )
    destination = tmp_path / "e100.export.json"
    receipt = exporter.export_one(
        checkpoint=checkpoint,
        epoch=100,
        source_host_label="host",
        required_git_commit=commit,
        required_adapter_fingerprint=fingerprint,
        destination=destination,
    )
    assert receipt["status"] == "ACCEPTED_SOURCE_BOUND_DCLGAN_CHECKPOINT"
    assert receipt["training_git_commit"] == commit
    assert receipt["training_protocol_fingerprint"] == fingerprint
    assert receipt["performance_values_read"] is False
    assert receipt["checkpoint_copy_performed"] is False


def test_dclgan_export_epoch_set_is_frozen() -> None:
    assert exporter.EPOCHS == (100, 125, 150, 175, 200)
