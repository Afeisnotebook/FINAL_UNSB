from __future__ import annotations

import json
from pathlib import Path

from operations.paper_aio_export_successor import source_state_decision


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_export_successor_waits_for_exact_complete_e200(tmp_path: Path) -> None:
    assert source_state_decision(tmp_path, "plain")["status"] == (
        "WAITING_FOR_COMPLETE_E200"
    )
    _write(tmp_path / "lanes" / "plain" / "RUN_STATE.json", {
        "status": "ENGINEERING_PAUSE",
        "final_updates": 8553,
        "final_data_epoch": 1,
        "confirmation20_opened": False,
    })
    assert source_state_decision(tmp_path, "plain")["status"] == (
        "WAITING_FOR_COMPLETE_E200"
    )


def test_export_successor_accepts_only_sealed_complete_e200(tmp_path: Path) -> None:
    state = tmp_path / "lanes" / "cut" / "RUN_STATE.json"
    _write(state, {
        "status": "COMPLETE_E200",
        "final_updates": 1_710_600,
        "final_data_epoch": 200.0,
        "confirmation20_opened": False,
    })
    assert source_state_decision(tmp_path, "cut")["status"] == "READY_COMPLETE_E200"
    value = json.loads(state.read_text(encoding="utf-8"))
    value["confirmation20_opened"] = True
    _write(state, value)
    assert source_state_decision(tmp_path, "cut")["status"] == (
        "WAITING_FOR_COMPLETE_E200"
    )


def test_export_successor_propagates_supervisor_engineering_block(tmp_path: Path) -> None:
    _write(tmp_path / "gates" / "SUPERVISOR_proposal.json", {
        "status": "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE",
    })
    result = source_state_decision(tmp_path, "proposal")
    assert result["status"] == "BLOCKED_SOURCE_LANE_ENGINEERING_FAILURE"

