from __future__ import annotations

from pathlib import Path

from operations.paper_aio_export_successor_recovery_supervisor import (
    EXPORT_STATE_SCHEMA,
    command_matches_contract,
    export_state_decision,
    render_export_command,
)


def _contract(tmp_path: Path) -> dict:
    repo = tmp_path / "repo"
    return {
        "control_repo": str(repo),
        "source_output": str(tmp_path / "run"),
        "destination": str(tmp_path / "run" / "exports"),
        "lane_id": "cyclegan",
        "source_host_label": "5090B",
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "poll_seconds": 60,
        "timeout_hours": 480.0,
    }


def _state(status: str) -> dict:
    return {
        "schema": EXPORT_STATE_SCHEMA,
        "status": status,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_export_recovery_state_decision_is_fail_closed_and_metric_blind() -> None:
    assert export_state_decision({}) == "WAIT"
    assert export_state_decision(_state("WAITING_FOR_COMPLETE_E200")) == "WAIT"
    assert export_state_decision(_state("COMPLETE_SOURCE_BOUND_EXPORT_SET")) == "COMPLETE"
    assert export_state_decision(_state("BLOCKED_SOURCE_LANE_ENGINEERING_FAILURE")) == "BLOCK"
    value = _state("WAITING_FOR_COMPLETE_E200")
    value["performance_values_read"] = True
    assert export_state_decision(value) == "BLOCK"
    value = _state("WAITING_FOR_COMPLETE_E200")
    value["confirmation20_opened"] = True
    assert export_state_decision(value) == "BLOCK"


def test_export_recovery_matches_relative_live_script_and_numeric_spellings(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    python = tmp_path / "env" / "bin" / "python"
    expected = render_export_command(python, contract)
    observed = list(expected)
    observed[1] = "operations/paper_aio_export_successor.py"
    observed[observed.index("--timeout-hours") + 1] = "480"
    assert command_matches_contract(
        observed, cwd=Path(contract["control_repo"]), python=python,
        contract=contract,
    )


def test_export_recovery_rejects_changed_lane_or_runtime(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    python = tmp_path / "env" / "bin" / "python"
    command = render_export_command(python, contract)
    command[command.index("--lane") + 1] = "plain"
    assert not command_matches_contract(
        command, cwd=Path(contract["control_repo"]), python=python,
        contract=contract,
    )
    assert not command_matches_contract(
        render_export_command(python, contract), cwd=Path(contract["control_repo"]),
        python=tmp_path / "other" / "python", contract=contract,
    )


def test_adopt_runtime_and_restart_runtime_are_distinct_contract_roles(
    tmp_path: Path,
) -> None:
    contract = _contract(tmp_path)
    restart_python = tmp_path / "verified" / "bin" / "python"
    deleted_argv_python = tmp_path / "deleted" / "bin" / "python"
    live_command = render_export_command(deleted_argv_python, contract)

    assert command_matches_contract(
        live_command,
        cwd=Path(contract["control_repo"]),
        python=deleted_argv_python,
        contract=contract,
    )
    assert not command_matches_contract(
        live_command,
        cwd=Path(contract["control_repo"]),
        python=restart_python,
        contract=contract,
    )
    assert render_export_command(restart_python, contract)[0] == str(
        restart_python.resolve()
    )
