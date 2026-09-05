from __future__ import annotations

import json
from pathlib import Path

import pytest

from operations import paper_aio_repeat_gate_fingerprint_repair as repair


TRAINING = "a" * 64
EVALUATION = "b" * 64


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, dict]:
    output = tmp_path / "run"
    state = output / "operations" / "CROSS_HOST_PLAIN_SUCCESSOR_STATE.json"
    evaluation = output / "gates" / "EVALUATION_REPEAT_plain.json"
    authorization = output / "gates" / "LANE_AUTHORIZATION_plain.json"
    runtime = output / "gates" / "RUNTIME_TWIN_5090B_MATCHED_PLAIN.json"
    _write(state, {
        "status": "BLOCKED_ENGINEERING_GATE_FAILURE", "gate_index": 5,
        "child_returncode": 1, "host_label": "5090B_MATCHED_PLAIN",
        "required_protocol_fingerprint": TRAINING,
    })
    _write(evaluation, {
        "schema": "final-unsb-paper-evaluation-repeat-gate-v1",
        "status": "PASS", "lane_id": "plain",
        "first_result_sha256": "same", "second_result_sha256": "same",
        "protocol_fingerprint": EVALUATION, "split": "discovery",
        "confirmation20_opened": False,
    })
    evaluation_sha = repair.file_sha256(evaluation)
    _write(authorization, {
        "status": "FAIL", "lane_id": "plain",
        "failures": ["lane-specific repeated evaluation receipt is stale"],
        "evaluation_repeat_gate_sha256": evaluation_sha,
        "paired_metric_control": False, "confirmation20_opened": False,
    })
    _write(runtime, {
        "schema": "final-unsb-paper-runtime-twin-receipt-v1",
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": "5090B_MATCHED_PLAIN", "updates": 2000,
        "protocol_fingerprint": TRAINING, "exact_runtime_equivalence": True,
        "differences": {}, "confirmation20_opened": False,
    })
    _write(output / "PAPER_PROTOCOL.json", {
        "protocol_fingerprint": TRAINING,
        "evaluation": {"bundle_seed_fingerprint": EVALUATION},
        "confirmation20_opened": False,
    })
    kwargs = {
        "output": output, "repair_id": "gate5-legacy",
        "host_label": "5090B_MATCHED_PLAIN",
        "required_training_protocol_fingerprint": TRAINING,
        "required_evaluation_bundle_fingerprint": EVALUATION,
        "required_failed_successor_state_sha256": repair.file_sha256(state),
        "required_legacy_evaluation_receipt_sha256": evaluation_sha,
        "required_failed_authorization_sha256": repair.file_sha256(authorization),
        "required_runtime_receipt_sha256": repair.file_sha256(runtime),
    }
    return output, kwargs


def test_known_legacy_fingerprint_mixup_is_repaired_and_archived(tmp_path):
    output, kwargs = _fixture(tmp_path)
    value = repair.repair(**kwargs)
    assert value["status"] == "PASS_KNOWN_LEGACY_METADATA_REPAIR"
    assert value["performance_values_read"] is False
    corrected = repair.read_json(
        output / "gates" / "EVALUATION_REPEAT_plain.json"
    )
    assert corrected["protocol_fingerprint"] == TRAINING
    assert corrected["evaluation_bundle_fingerprint"] == EVALUATION
    archived = output / "operations" / "gate_fingerprint_repair" / (
        "gate5-legacy/original/EVALUATION_REPEAT_plain.json"
    )
    assert repair.file_sha256(archived) == (
        kwargs["required_legacy_evaluation_receipt_sha256"]
    )


def test_repair_refuses_anything_beyond_the_exact_known_failure(tmp_path):
    output, kwargs = _fixture(tmp_path)
    path = output / "gates" / "EVALUATION_REPEAT_plain.json"
    payload = repair.read_json(path)
    payload["first_result_sha256"] = "different"
    _write(path, payload)
    kwargs["required_legacy_evaluation_receipt_sha256"] = repair.file_sha256(path)
    authorization = output / "gates" / "LANE_AUTHORIZATION_plain.json"
    auth = repair.read_json(authorization)
    auth["evaluation_repeat_gate_sha256"] = repair.file_sha256(path)
    _write(authorization, auth)
    kwargs["required_failed_authorization_sha256"] = repair.file_sha256(authorization)
    with pytest.raises(RuntimeError, match="known legacy"):
        repair.repair(**kwargs)
