import json

import pytest

from operations.paper_aio_local_terminal_audit_successor import (
    AUDIT_SCHEMA,
    PROBES,
    _audit_result,
)


def test_full_data_terminal_probe_set_is_fixed_and_multi_algorithm():
    assert {(row["host_label"], row["import_lane"]) for row in PROBES} == {
        ("4090A", "plain"),
        ("4090A", "amtnc"),
        ("5090C", "proposal"),
        ("5090A", "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"),
    }


def test_terminal_result_gate_requires_unchanged_state_rng_and_no_labels(tmp_path):
    path = tmp_path / "TERMINAL_AUDIT.jsonl"
    value = {
        "schema": AUDIT_SCHEMA,
        "status": "TARGET_BLIND_AUDIT_COMPLETE",
        "parent_state_sha256_before": "state",
        "parent_state_sha256_after": "state",
        "parent_rng_sha256_before": "rng",
        "parent_rng_sha256_after": "rng",
        "paired_labels_attached": False,
        "terminal_pathology_confirmed": False,
        "confirmation20_opened": False,
    }
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    assert _audit_result(path) == value
    value["paired_labels_attached"] = True
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boundary"):
        _audit_result(path)
