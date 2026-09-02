from __future__ import annotations

import argparse

import pytest

from operations import paper_aio_cross_host_plain_successor as successor


def _args(tmp_path):
    return argparse.Namespace(
        training_output=tmp_path / "output",
        manifest=tmp_path / "manifest.csv",
        data_root=tmp_path / "data",
        train_view=tmp_path / "view",
        gpu=0,
        python=tmp_path / "python",
        host_label="5090B_MATCHED",
        peer_runtime_receipt=tmp_path / "peer.json",
    )


def test_predecessor_decision_is_metric_blind_and_fail_closed():
    assert successor.predecessor_decision(None, timed_out=False) == "WAIT"
    assert successor.predecessor_decision("CHILD_RUNNING", timed_out=False) == "WAIT"
    assert successor.predecessor_decision("COMPLETE_E200", timed_out=False) == "START"
    assert successor.predecessor_decision("BLOCKED_IO", timed_out=False) == "BLOCK"
    assert successor.predecessor_decision(None, timed_out=True) == "TIMEOUT"


def test_gate_chain_requires_exact_runtime_twin_before_authorization(tmp_path):
    rendered = [" ".join(row) for row in successor.gate_commands(_args(tmp_path))]
    assert len(rendered) == 5
    assert "--stage preflight" in rendered[0]
    assert "--stage resume-gate --lane plain" in rendered[1]
    assert "--stage runtime-twin" in rendered[2]
    assert "--peer-receipt" in rendered[2]
    assert "--stage evaluation-repeat-gate --lane plain" in rendered[3]
    assert "--stage authorize --lane plain" in rendered[4]


def test_runtime_receipt_must_be_exact_and_confirmation_closed():
    receipt = {
        "schema": "final-unsb-paper-runtime-twin-receipt-v1",
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": "5090B_MATCHED",
        "updates": 2000,
        "protocol_fingerprint": "frozen",
        "exact_runtime_equivalence": True,
        "differences": {},
        "confirmation20_opened": False,
    }
    successor.validate_runtime_receipt(
        receipt, host_label="5090B_MATCHED",
        required_protocol_fingerprint="frozen",
    )
    for key, value in (
        ("status", "FAIL_HOST_SEPARATED"),
        ("updates", 1999),
        ("exact_runtime_equivalence", False),
        ("confirmation20_opened", True),
    ):
        broken = {**receipt, key: value}
        with pytest.raises(RuntimeError, match="did not pass exactly"):
            successor.validate_runtime_receipt(
                broken, host_label="5090B_MATCHED",
                required_protocol_fingerprint="frozen",
            )
