import json
from pathlib import Path

from operations.paper_aio_export_relay_recovery_supervisor import (
    command_matches_contract,
    relay_state_decision,
    render_relay_command,
)


def _relay(tmp_path: Path) -> dict:
    script = tmp_path / "operations" / "paper_aio_export_relay.py"
    script.parent.mkdir()
    script.write_text("pass\n", encoding="utf-8")
    return {
        "schema": "final-unsb-paper-export-relay-contract-v1",
        "status": "FROZEN_WAITING",
        "control_script": str(script),
        "control_script_sha256": "x",
        "relay_id": "5090B_external_v3",
        "source_host_label": "5090B",
        "source_host": "example.invalid",
        "source_port": 44804,
        "source_user": "root",
        "expected_host_key_sha256": "SHA256:test",
        "remote_export_root": "/remote/exports",
        "destination_root": str(tmp_path / "imports"),
        "lanes": ["cut", "cyclegan"],
        "password_env": "FINAL_UNSB_TEST_PASSWORD",
        "poll_seconds": 60,
        "timeout_hours": 480.0,
        "password_persisted": False,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "source_checkpoint_mutation": False,
        "confirmation20_opened": False,
    }


def test_rendered_command_matches_numeric_equivalent_live_form(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    command = render_relay_command(python, relay)
    command[command.index("--timeout-hours") + 1] = "480"
    assert command_matches_contract(command, python, relay)


def test_command_match_rejects_lane_or_source_change(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    command = render_relay_command(python, relay)
    command[command.index("cyclegan")] = "proposal"
    assert not command_matches_contract(command, python, relay)


def test_relay_state_decision_is_metric_blind_and_fail_closed() -> None:
    base = {
        "schema": "final-unsb-paper-export-relay-state-v1",
        "status": "WAITING_FOR_COMPLETE_SOURCE_EXPORTS_OR_TRANSIENT_NETWORK",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert relay_state_decision(base) == "WAIT"
    assert relay_state_decision(
        {**base, "status": "COMPLETE_VERIFIED_IMPORT_SET"}
    ) == "COMPLETE"
    assert relay_state_decision({**base, "status": "FAIL_CLOSED_TEST"}) == "BLOCK"
    assert relay_state_decision({**base, "performance_values_read": True}) == "BLOCK"
    assert relay_state_decision({**base, "confirmation20_opened": True}) == "BLOCK"


def test_password_value_is_not_part_of_rendered_command(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    command = render_relay_command(python, relay)
    assert relay["password_env"] in command
    assert "secret" not in json.dumps(command)
