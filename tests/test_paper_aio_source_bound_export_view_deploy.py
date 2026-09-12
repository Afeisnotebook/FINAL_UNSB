from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from operations.paper_aio_source_bound_export_view_deploy import (
    build_export_command,
    build_recovery_command,
    recovery_view_profile,
)
import pytest


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        python=tmp_path / "runtime" / "python",
        repo=tmp_path / "repo",
        source_output=tmp_path / "source",
        view_output=tmp_path / "view",
        destination=tmp_path / "normalized-exports",
        lane="amtnc",
        source_host_label="4090A",
        required_control_git_commit="a" * 40,
        required_training_git_commit="b" * 40,
        required_training_protocol_fingerprint="c" * 64,
        poll_seconds=60,
        timeout_hours=720,
    )


def test_export_command_uses_unique_view_and_physical_host_label(tmp_path: Path):
    args = _args(tmp_path)
    command = build_export_command(args)
    assert str(args.view_output.resolve()) in command
    assert str(args.source_output.resolve()) not in command
    assert command[command.index("--source-host-label") + 1] == "4090A"
    assert command[command.index("--destination") + 1] == str(
        args.destination.resolve()
    )


def test_recovery_command_is_bound_to_view_contract_and_state(tmp_path: Path):
    args = _args(tmp_path)
    contract = args.view_output / "operations" / "contract.json"
    state = args.view_output / "operations" / "state.json"
    command = build_recovery_command(args, export_contract=contract, export_state=state)
    assert command[command.index("--export-contract") + 1] == str(contract.resolve())
    assert command[command.index("--export-state") + 1] == str(state.resolve())
    assert command[command.index("--required-control-git-commit") + 1] == "a" * 40
    assert str(args.view_output.resolve() / "export_recovery") in command


def test_only_registered_recovery_source_views_are_supported():
    assert recovery_view_profile("4090A", "amtnc")["profile"] == (
        "4090A_AMTNC_OVERFLOW_RECOVERY"
    )
    assert recovery_view_profile(
        "5090A", "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"
    )["profile"] == "5090A_STCGR_EXACT_RESUME_RECOVERY"
    with pytest.raises(RuntimeError, match="unsupported recovery source-view pair"):
        recovery_view_profile("5090A", "unregistered-candidate")
