import json

from operations.paper_aio_terminal_pathology_successor import (
    AUDIT_EPOCHS,
    PROBES,
    audit_release,
)


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _complete_state():
    return {
        "status": "COMPLETE_FIXED_FULL_DATA_TARGET_BLIND_TERMINAL_AUDITS",
        "completed": [
            f"{probe_id}:e{epoch}" for probe_id in PROBES for epoch in AUDIT_EPOCHS
        ],
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_terminal_pathology_release_requires_exact_complete_audit_set(tmp_path):
    state = tmp_path / "state.json"
    assert audit_release(state) == "WAIT"
    value = _complete_state()
    _write(state, value)
    assert audit_release(state) == "READY"
    value["completed"].pop()
    _write(state, value)
    assert audit_release(state) == "BLOCKED"


def test_terminal_pathology_release_rejects_information_boundary_violation(tmp_path):
    state = tmp_path / "state.json"
    value = _complete_state()
    value["performance_values_read"] = True
    _write(state, value)
    assert audit_release(state) == "BLOCKED"
