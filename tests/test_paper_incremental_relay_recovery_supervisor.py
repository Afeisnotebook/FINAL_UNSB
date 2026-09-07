import json
from pathlib import Path

from operations import paper_aio_incremental_relay_recovery_supervisor as recovery


def _relay(tmp_path: Path) -> dict:
    script = tmp_path / "operations" / "paper_aio_incremental_audit_relay.py"
    script.parent.mkdir()
    script.write_text("pass\n", encoding="utf-8")
    return {
        "schema": "final-unsb-paper-incremental-audit-relay-contract-v1",
        "status": "FROZEN_WAITING",
        "control_script": str(script),
        "control_script_sha256": "x",
        "base_relay_script_sha256": "y",
        "relay_id": "incremental_5090A_stcgr_v5",
        "source_host_label": "5090A",
        "source_host": "example.invalid",
        "source_port": 12770,
        "source_user": "root",
        "expected_host_key_sha256": "SHA256:test",
        "remote_export_root": "/remote/exports",
        "destination_root": str(tmp_path / "imports"),
        "lane_id": "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
        "required_training_git_commit": "a" * 40,
        "required_training_protocol_fingerprint": "b" * 64,
        "required_manifest_sha256": "c" * 64,
        "required_epochs": [100, 150, 200],
        "password_env": "FINAL_UNSB_TEST_PASSWORD",
        "poll_seconds": 60,
        "timeout_hours": 720.0,
        "password_persisted": False,
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "source_checkpoint_mutation": False,
        "confirmation20_opened": False,
    }


def test_rendered_incremental_command_matches_numeric_equivalent(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    command = recovery.render_relay_command(python, relay)
    command[command.index("--timeout-hours") + 1] = "720"
    assert recovery.command_matches_contract(command, python, relay)


def test_command_match_rejects_lane_or_identity_change(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    command = recovery.render_relay_command(python, relay)
    command[command.index("--lane") + 1] = "proposal"
    assert not recovery.command_matches_contract(command, python, relay)


def test_relay_state_decision_is_metric_blind_and_fail_closed() -> None:
    base = {
        "schema": "final-unsb-paper-incremental-audit-relay-state-v1",
        "status": "PARTIAL_VERIFIED_INCREMENTAL_AUDIT_IMPORT",
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert recovery.relay_state_decision(base) == "WAIT"
    assert recovery.relay_state_decision(
        {**base, "status": "COMPLETE_VERIFIED_INCREMENTAL_AUDIT_IMPORT"}
    ) == "COMPLETE"
    assert recovery.relay_state_decision({**base, "status": "FAIL_TEST"}) == "BLOCK"
    assert recovery.relay_state_decision(
        {**base, "performance_values_read": True}
    ) == "BLOCK"


def test_matching_processes_requires_exact_command(monkeypatch, tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    good = recovery.render_relay_command(python, relay)
    bad = list(good)
    bad[bad.index("--relay-id") + 1] = "wrong"
    monkeypatch.setattr(recovery, "_process_commands", lambda: [(101, good), (102, bad)])
    assert recovery._matching_processes(python, relay) == [101]


def test_password_value_is_not_persisted_in_command(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    command = recovery.render_relay_command(python, relay)
    assert relay["password_env"] in command
    assert "secret" not in json.dumps(command)


def test_contract_rejects_boundary_violation(tmp_path: Path) -> None:
    relay = _relay(tmp_path)
    recovery._validate_relay_contract(relay)
    relay["confirmation20_opened"] = True
    try:
        recovery._validate_relay_contract(relay)
    except RuntimeError as error:
        assert "frozen boundary" in str(error)
    else:
        raise AssertionError("unsafe incremental relay contract was accepted")
