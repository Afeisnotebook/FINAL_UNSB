from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEY = "dclgan_v7_delivery_rebind_20260913"
CONTROL_COMMIT = "da41652b7ff73bdd9e66db72fc72169bd08374f1"
REFERENCE = "/home/yc/runs/FINAL_UNSB_PAPER_UNIFIED_EVAL_V7_DA41652"
BASE_DELIVERY = "/home/yc/runs/FINAL_UNSB_PAPER_FINAL_DELIVERY_V7_DA41652"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dclgan_v7_chain_is_registered_in_current_authorities() -> None:
    project = _read(ROOT / "PROJECT_STATE.json")["paper_aio_20260902"][KEY]
    portfolio = _read(ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json")[KEY]
    matrix = _read(ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json")[KEY]

    for record in (project, portfolio, matrix):
        assert record["control_git_commit"] == CONTROL_COMMIT
        assert record["reference_output"] == REFERENCE
        assert record["base_delivery_output"] == BASE_DELIVERY
        assert record["evaluation_supervisor_pid"] == 135947
        assert record["evaluation_child_pid"] == 135956
        assert record["addendum_supervisor_pid"] == 136368
        assert record["addendum_child_pid"] == 136377
        assert record["health_watcher_pid"] == 136765
        assert set(record["retired_wait_only_pids"]).isdisjoint(
            {
                record["evaluation_supervisor_pid"],
                record["evaluation_child_pid"],
                record["addendum_supervisor_pid"],
                record["addendum_child_pid"],
                record["health_watcher_pid"],
            }
        )
        assert record["old_outputs_deleted"] is False
        assert record["training_processes_changed"] is False
        assert record["performance_values_read"] is False
        assert record["confirmation20_opened"] is False


def test_active_dclgan_delivery_fields_no_longer_reference_failed_dynamic_v1() -> None:
    project = _read(ROOT / "PROJECT_STATE.json")["paper_aio_20260902"]
    project_dclgan = project["external_dclgan_negcut_source_gate"]
    portfolio_dclgan = _read(
        ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json"
    )["external_second_priority_source_gate"]
    matrix_dclgan = next(
        row
        for row in _read(
            ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json"
        )["nonblocking_extensions"]
        if row["id"] == "dclgan"
    )

    assert project_dclgan["dclgan_unified_reference_output"] == REFERENCE
    assert portfolio_dclgan["unified_reference_output"] == REFERENCE
    assert matrix_dclgan["delivery_gate"].endswith(
        "current V7 first-wave cohort"
    )
    assert project_dclgan["dclgan_unified_evaluation_successor_pid"] == 135956
    assert portfolio_dclgan["unified_evaluation_successor_pid"] == 135956
    assert matrix_dclgan["evaluation_child_pid"] == 135956


def test_dclgan_v7_compact_evidence_preserves_scientific_boundaries() -> None:
    evidence = _read(
        ROOT
        / "evidence"
        / "paper_aio"
        / "PAPER_AIO_DCLGAN_V7_DELIVERY_REBIND_20260913T185629.json"
    )
    assert evidence["replacement"]["reference_output"] == REFERENCE
    assert evidence["replacement"]["base_delivery_output"] == BASE_DELIVERY
    assert evidence["replacement"]["health_status_after_two_polls"] == (
        "HEALTHY_ZERO_ALERTS"
    )
    assert evidence["retirement"]["all_listed_old_pids_absent_after_replacement"]
    assert evidence["retirement"]["old_outputs_deleted"] is False
    assert evidence["continuity"]["local_dclgan_training_changed"] is False
    assert evidence["continuity"]["amtnc_training_changed"] is False
    assert all(value is False for value in evidence["scientific_boundaries"].values())
