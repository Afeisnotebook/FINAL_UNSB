import json
import subprocess
import sys
from pathlib import Path

from operations.paper_aio_terminal_pathology_successor import (
    AUDIT_EPOCHS,
    PROBES,
    _automatic_metrics,
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


def test_terminal_pathology_successor_supports_direct_script_entrypoint():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(root / "operations" / "paper_aio_terminal_pathology_successor.py"),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--audit-successor-state" in result.stdout


def test_automatic_metrics_materializes_all_fixed_cells(monkeypatch, tmp_path):
    completed = set()
    calls = []

    def fake_ready(_contract, probe):
        return {
            epoch: {
                "export_receipt": str(tmp_path / f"export-{epoch}.json"),
                "checkpoint": str(tmp_path / f"checkpoint-{epoch}.pt"),
            }
            for epoch in AUDIT_EPOCHS
        }

    def fake_completed(_paths, *, probe, epoch):
        return (probe["lane_id"], epoch) in completed

    def fake_evaluate(**kwargs):
        lane = (
            kwargs["candidate_id"]
            if kwargs["candidate_id"] is not None
            else Path(kwargs["export_receipt"]).stem.split("-")[0]
        )
        calls.append(kwargs)
        # The production function learns lane identity from the source receipt;
        # this test uses the output call order to recover the fixed probe lane.
        probe = list(PROBES.values())[(len(calls) - 1) // len(AUDIT_EPOCHS)]
        epoch = AUDIT_EPOCHS[(len(calls) - 1) % len(AUDIT_EPOCHS)]
        assert lane or probe["lane_id"]
        completed.add((probe["lane_id"], epoch))

    monkeypatch.setattr(
        "operations.paper_aio_terminal_pathology_successor._ready_rows", fake_ready
    )
    monkeypatch.setattr(
        "operations.paper_aio_terminal_pathology_successor._completed_metric",
        fake_completed,
    )
    monkeypatch.setattr(
        "operations.paper_aio_terminal_pathology_successor.evaluate_imported_checkpoint",
        fake_evaluate,
    )
    monkeypatch.setattr(
        "operations.paper_aio_terminal_pathology_successor._acquire_lock",
        lambda _handle: True,
    )
    contract = {
        "import_root": str(tmp_path / "imports"),
        "evaluation_root": str(tmp_path / "evaluation"),
        "gpu_lock": str(tmp_path / "gpu.lock"),
        "train_view": str(tmp_path / "view"),
        "data_root": str(tmp_path / "data"),
        "manifest": str(tmp_path / "manifest.csv"),
        "gpu": 0,
        "candidate_authority": str(tmp_path / "authority.json"),
        "metric_bindings": str(tmp_path / "METRIC_BINDINGS.json"),
        "poll_seconds": 60,
    }
    path = _automatic_metrics(contract, tmp_path / "STATE.json")
    assert path.is_file()
    assert len(calls) == len(PROBES) * len(AUDIT_EPOCHS)
    assert sum(call["candidate_id"] is not None for call in calls) == len(AUDIT_EPOCHS)
    binding = json.loads(path.read_text(encoding="utf-8"))
    assert set(binding["probes"]) == set(PROBES)
