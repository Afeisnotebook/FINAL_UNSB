import argparse
import hashlib
import json
from pathlib import Path

import pytest

from operations.paper_aio_amtnc_precision_resume_probe import (
    _copy_inputs,
    _validate_source,
)


COMMIT = "5" * 40
FINGERPRINT = "8" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> argparse.Namespace:
    source = tmp_path / "source"
    checkpoint = source / "captured" / "e195.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"captured-e195")
    sidecar = checkpoint.with_suffix(".pt.json")
    sidecar.write_text(json.dumps({
        "schema": "final-unsb-paper-aio-full-state-v1",
        "lane_id": "amtnc",
        "step": 1_667_835,
        "physical_epoch_completed": 195,
        "full_state_sha256": _sha256(checkpoint),
        "scientific_state_sha256": "a" * 64,
        "metadata": {
            "git_commit": COMMIT,
            "protocol_fingerprint": FINGERPRINT,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    }), encoding="utf-8")
    (source / "gates").mkdir()
    (source / "shared_e0" / "unsb_common").mkdir(parents=True)
    for path in (
        source / "PAPER_PROTOCOL.json",
        source / "gates" / "LANE_AUTHORIZATION_amtnc.json",
        source / "shared_e0" / "unsb_common" / "e0.pt",
        source / "shared_e0" / "unsb_common" / "e0.pt.json",
    ):
        path.write_bytes(path.name.encode())
    return argparse.Namespace(
        source_output=source,
        source_checkpoint=checkpoint,
        source_sidecar=sidecar,
        required_step=1_667_835,
        required_epoch=195,
        required_git_commit=COMMIT,
        required_protocol_fingerprint=FINGERPRINT,
    )


def test_validate_and_copy_captured_source(tmp_path):
    args = _fixture(tmp_path)
    identity = _validate_source(args)
    assert identity["checkpoint_sha256"] == _sha256(args.source_checkpoint)
    probe = tmp_path / "probe"
    _copy_inputs(args, probe)
    assert (probe / "lanes" / "amtnc" / "full_state_latest.pt").read_bytes() == b"captured-e195"
    assert (probe / "PAPER_PROTOCOL.json").is_file()


def test_validate_rejects_open_confirmation(tmp_path):
    args = _fixture(tmp_path)
    value = json.loads(args.source_sidecar.read_text(encoding="utf-8"))
    value["metadata"]["confirmation20_opened"] = True
    args.source_sidecar.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source identity mismatch"):
        _validate_source(args)


def test_copy_requires_fresh_complete_inputs(tmp_path):
    args = _fixture(tmp_path)
    (args.source_output / "PAPER_PROTOCOL.json").unlink()
    with pytest.raises(RuntimeError, match="required probe input is missing"):
        _copy_inputs(args, tmp_path / "probe")
