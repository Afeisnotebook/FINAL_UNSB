from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from operations.paper_aio_checkpoint_capture import capture_once


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path: Path, *, epoch: int = 195):
    source = tmp_path / "source.pt"
    source.write_bytes(b"finite checkpoint bytes")
    metadata = {
        "git_commit": "a" * 40,
        "protocol_fingerprint": "b" * 64,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    Path(str(source) + ".json").write_text(json.dumps({
        "schema": "final-unsb-paper-aio-full-state-v1",
        "lane_id": "amtnc",
        "step": epoch * 8553,
        "physical_epoch_completed": epoch,
        "full_state_sha256": _sha(source),
        "scientific_state_sha256": "c" * 64,
        "metadata": metadata,
    }), encoding="utf-8")
    return argparse.Namespace(
        source_checkpoint=source,
        destination_checkpoint=tmp_path / "captured" / "e195.pt",
        output=tmp_path / "control",
        lane_id="amtnc",
        required_epoch=195,
        required_step=195 * 8553,
        required_git_commit="a" * 40,
        required_protocol_fingerprint="b" * 64,
        poll_seconds=0.01,
        timeout_hours=1.0,
    )


def test_capture_copies_exact_checkpoint_and_sidecar(tmp_path):
    args = _args(tmp_path)
    result = capture_once(args)
    assert result["status"] == "COMPLETE_HASH_BOUND_CAPTURE"
    assert args.destination_checkpoint.read_bytes() == b"finite checkpoint bytes"
    assert result["checkpoint_sha256"] == _sha(args.destination_checkpoint)
    assert result["signals_training"] is False
    assert result["performance_values_read"] is False
    assert result["confirmation20_opened"] is False


def test_capture_waits_before_epoch_and_fails_after_overwrite(tmp_path):
    args = _args(tmp_path, epoch=194)
    assert capture_once(args) is None
    sidecar = Path(str(args.source_checkpoint) + ".json")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    payload["physical_epoch_completed"] = 196
    payload["step"] = 196 * 8553
    sidecar.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="overwritten"):
        capture_once(args)


def test_capture_rejects_wrong_lineage(tmp_path):
    args = _args(tmp_path)
    args.required_git_commit = "d" * 40
    with pytest.raises(RuntimeError, match="identity mismatch"):
        capture_once(args)
