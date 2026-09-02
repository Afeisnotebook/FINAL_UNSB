from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from operations import paper_aio_relation_registry_review_successor as successor
from operations.paper_aio_relation_registry_review import (
    CANDIDATE_STATE_SCHEMA,
    STANDARD_STATE_SCHEMA,
    STCGR_ID,
)


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _state(*, candidate: bool = False) -> dict:
    return {
        "schema": CANDIDATE_STATE_SCHEMA if candidate else STANDARD_STATE_SCHEMA,
        "status": successor.STCGR_COMPLETE
        if candidate
        else successor.PROPOSAL_COMPLETE,
        "candidate_id" if candidate else "lane_id": STCGR_ID
        if candidate
        else "proposal",
        "method_source_host_label": "5090A" if candidate else "5090C",
        "plain_source_host_label": "5090B_MATCHED_PLAIN",
        "relation_candidate": str(Path.cwd() / "candidate.json"),
        "relation_candidate_sha256": "a" * 64,
        "exact_runtime_equivalence": True,
        "registry_edited": False,
        "comparison_authorized": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _release(path: Path, *, candidate: bool = False) -> str:
    return successor.relation_state_release(
        path,
        expected_schema=CANDIDATE_STATE_SCHEMA if candidate else STANDARD_STATE_SCHEMA,
        expected_status=successor.STCGR_COMPLETE
        if candidate
        else successor.PROPOSAL_COMPLETE,
        lane_field="candidate_id" if candidate else "lane_id",
        expected_lane=STCGR_ID if candidate else "proposal",
        method_host="5090A" if candidate else "5090C",
        plain_host="5090B_MATCHED_PLAIN",
    )


def test_relation_state_release_waits_then_accepts_exact_terminal_state(tmp_path):
    path = tmp_path / "state.json"
    assert _release(path) == "WAIT"
    _write(path, _state())
    assert _release(path) == "READY"
    _write(path, _state(candidate=True))
    assert _release(path, candidate=True) == "READY"


def test_relation_state_release_fails_closed_on_boundary_or_host_change(tmp_path):
    path = tmp_path / "state.json"
    value = _state()
    value["performance_values_read"] = True
    _write(path, value)
    assert _release(path) == "BLOCKED"
    value = _state()
    value["plain_source_host_label"] = "wrong"
    _write(path, value)
    assert _release(path) == "BLOCKED"


def test_registry_review_successor_supports_direct_script_entrypoint():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(
                root / "operations" / "paper_aio_relation_registry_review_successor.py"
            ),
            "--help",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--proposal-state" in result.stdout
