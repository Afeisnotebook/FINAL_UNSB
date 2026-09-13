import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_terminal_audit_live_authority_is_consistent_and_fail_closed() -> None:
    project = _load("PROJECT_STATE.json")["paper_aio_20260902"]
    portfolio = _load("configs/FULL_DATA_METHOD_PORTFOLIO.json")
    matrix = _load("configs/PAPER_DELIVERY_COMPLETION_MATRIX.json")
    matrix_extensions = {row["id"]: row for row in matrix["nonblocking_extensions"]}

    project_node = project["local_evaluation_node"]
    portfolio_node = portfolio["local_evaluation_node"]
    matrix_node = matrix_extensions["terminal_causal_audit"]
    snapshots = [
        project["terminal_audit_read_recovery_refresh_20260913"],
        portfolio["terminal_audit_read_recovery_refresh_20260913"],
        matrix["terminal_audit_read_recovery_refresh_20260913"],
    ]

    assert project_node["terminal_audit_control_supervisor_pid"] == 5180
    assert portfolio_node["terminal_audit_control_supervisor_pid"] == 5180
    assert matrix_node["terminal_audit_supervisor_pid"] == 5180
    assert project_node["terminal_audit_successor_pid"] == 16424
    assert portfolio_node["terminal_audit_successor_pid"] == 16424
    assert matrix_node["terminal_audit_child_pid"] == 16424

    assert project_node["terminal_audit_control_restart_count"] == 9
    assert portfolio_node["terminal_audit_control_restart_count"] == 9
    assert matrix_node["terminal_audit_launch_count"] == 9
    assert project_node["terminal_audit_total_permission_error_tracebacks"] == 8
    assert portfolio_node["terminal_audit_total_permission_error_tracebacks"] == 8
    assert matrix_node["terminal_audit_failed_child_count"] == 8

    for snapshot in snapshots:
        assert snapshot["current_child_pid"] == 16424
        assert snapshot["launch_count"] == 9
        assert snapshot["failed_child_count"] == 8
        assert snapshot["remaining_launch_budget"] == 11
        assert snapshot["healthy_child_replaced"] is False
        assert snapshot["training_processes_changed"] is False
        assert snapshot["performance_values_read"] is False
        assert snapshot["confirmation20_opened"] is False
