from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KEY = "amtnc_v7_post_training_chain_rebind_20260913"
CURRENT_RELEASE = (
    "/home/yc/runs/FINAL_UNSB_PAPER_AMTNC_RAW_SQUARE_RECOVERY_5676c91/"
    "gates/SUPERVISOR_amtnc.json"
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v7_chain_is_registered_in_all_current_paper_authorities() -> None:
    project = _read(ROOT / "PROJECT_STATE.json")["paper_aio_20260902"][KEY]
    portfolio = _read(ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json")[KEY]
    matrix = _read(ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json")[KEY]

    for record in (project, portfolio, matrix):
        assert record["current_amtnc_release_state"] == CURRENT_RELEASE
        assert record["control_git_commit"] == (
            "da41652b7ff73bdd9e66db72fc72169bd08374f1"
        )
        assert record["supervisor_pids"] == [125400, 125401, 125402, 125403]
        assert record["child_pids"] == [125436, 125440, 125437, 125444]
        assert record["health_watcher_pid"] == 125502
        assert set(record["supervisor_pids"]).isdisjoint(
            record["retired_wait_only_pids"]
        )
        assert set(record["child_pids"]).isdisjoint(
            record["retired_wait_only_pids"]
        )
        assert record["old_outputs_deleted"] is False
        assert record["training_processes_changed"] is False
        assert record["performance_values_read"] is False
        assert record["confirmation20_opened"] is False


def test_v7_compact_evidence_records_stale_and_current_release_boundaries() -> None:
    evidence = _read(
        ROOT
        / "evidence"
        / "paper_aio"
        / "PAPER_AIO_AMTNC_V7_POST_TRAINING_CHAIN_REBIND_20260913T183727.json"
    )
    assert evidence["replacement"]["current_release_state"] == CURRENT_RELEASE
    assert "478211c" in evidence["incident"]["stale_release_state"]
    assert evidence["replacement"]["health_status_after_two_polls"] == (
        "HEALTHY_ZERO_ALERTS"
    )
    assert evidence["retirement"]["all_absent_after_sigterm"] is True
    assert evidence["retirement"]["old_outputs_deleted"] is False
    assert evidence["protected_live_training"]["all_alive_after_retirement"] is True
    assert all(value is False for value in evidence["scientific_boundary"].values())
