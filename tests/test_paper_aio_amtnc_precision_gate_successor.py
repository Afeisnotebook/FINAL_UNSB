import argparse
import hashlib
import json
from pathlib import Path

import pytest

from operations.paper_aio_amtnc_precision_gate_successor import (
    _capture_ready,
    _contract,
    _validate_scripts,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _args(tmp_path: Path) -> argparse.Namespace:
    checkpoint = tmp_path / "e195.pt"
    checkpoint.write_bytes(b"e195")
    sidecar = Path(str(checkpoint) + ".json")
    sidecar.write_text("{}", encoding="utf-8")
    capture = {
        "schema": "final-unsb-paper-transient-checkpoint-capture-v1",
        "status": "COMPLETE_HASH_BOUND_CAPTURE",
        "epoch": 195,
        "step": 1_667_835,
        "destination_checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "sidecar_sha256": _sha256(sidecar),
        "source_unchanged_during_capture": True,
        "performance_values_read": False,
        "confirmation20_opened": False,
    }
    state = tmp_path / "capture-state.json"
    receipt = tmp_path / "capture-receipt.json"
    state.write_text(json.dumps(capture), encoding="utf-8")
    receipt.write_text(json.dumps(capture), encoding="utf-8")
    probe_script = tmp_path / "probe.py"
    gate_script = tmp_path / "gate.py"
    probe_script.write_text("pass\n", encoding="utf-8")
    gate_script.write_text("pass\n", encoding="utf-8")
    return argparse.Namespace(
        capture_state=state, capture_receipt=receipt,
        e195_checkpoint=checkpoint, resume_probe_script=probe_script,
        checkpoint_gate_script=gate_script, source_output=tmp_path / "source",
        training_repo=tmp_path / "repo", python=tmp_path / "python",
        manifest=tmp_path / "manifest", data_root=tmp_path / "data",
        train_view=tmp_path / "view", probe_output=tmp_path / "probe-output",
        required_resume_probe_script_sha256=_sha256(probe_script),
        required_checkpoint_gate_script_sha256=_sha256(gate_script),
        required_step=1_667_835, required_epoch=195,
        required_git_commit="5" * 40,
        required_protocol_fingerprint="8" * 64, gpu=0,
    )


def test_capture_ready_and_scripts_are_hash_bound(tmp_path):
    args = _args(tmp_path)
    assert _capture_ready(args)["status"] == "COMPLETE_HASH_BOUND_CAPTURE"
    _validate_scripts(args)


def test_capture_ready_waits_for_publish_last_receipt(tmp_path):
    args = _args(tmp_path)
    args.capture_receipt.unlink()
    assert _capture_ready(args) is None


def test_script_hash_mismatch_fails_closed(tmp_path):
    args = _args(tmp_path)
    args.resume_probe_script.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="script identity mismatch"):
        _validate_scripts(args)


def test_contract_records_metric_blind_firewalls(tmp_path):
    args = _args(tmp_path)
    contract = _contract(args)
    assert contract["performance_values_read"] is False
    assert contract["paired_metric_control"] is False
    assert contract["confirmation20_opened"] is False
