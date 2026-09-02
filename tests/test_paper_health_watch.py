import json
import os
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations.paper_aio_health_watch import (
    CONTRACT_SCHEMA,
    evaluate_contract,
    evaluate_watch,
    freeze_contract,
    parse_watch,
    proposed_contract,
    process_alive,
)


def _write(path: Path, payload: dict, *, mtime: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    if mtime is not None:
        import os
        os.utime(path, (mtime, mtime))


def test_parse_watch_requires_safe_absolute_state(tmp_path: Path) -> None:
    value = parse_watch(f"plain|123|{tmp_path / 'HEARTBEAT.json'}|3600|60")
    assert value["name"] == "plain"
    assert value["pid"] == 123
    with pytest.raises(ValueError):
        parse_watch("plain|123|relative.json|3600|60")
    with pytest.raises(ValueError):
        parse_watch(f"bad/name|123|{tmp_path / 'x'}|3600|60")
    with pytest.raises(ValueError):
        parse_watch(f"plain|0|{tmp_path / 'x'}|0|60")


def test_process_alive_supports_the_current_platform() -> None:
    assert process_alive(os.getpid()) is True


def test_watch_is_healthy_without_copying_losses(tmp_path: Path) -> None:
    path = tmp_path / "HEARTBEAT.json"
    _write(path, {
        "data_epoch": 10,
        "updates": 85530,
        "losses_last_update": {"secret": 4.2},
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }, mtime=900.0)
    row = evaluate_watch(
        parse_watch(f"plain|123|{path}|200|0"), now=1000.0,
        watch_started=800.0, alive=lambda _pid: True,
    )
    assert row["health"] == "HEALTHY"
    assert "losses_last_update" not in row


def test_watch_detects_stale_dead_and_boundary_states(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    spec = parse_watch(f"plain|123|{path}|100|0")
    _write(path, {"data_epoch": 3}, mtime=800.0)
    assert evaluate_watch(
        spec, now=1000.0, watch_started=700.0, alive=lambda _pid: True,
    )["health"] == "ALERT_STATE_STALE"
    _write(path, {"data_epoch": 3}, mtime=950.0)
    assert evaluate_watch(
        spec, now=1000.0, watch_started=700.0, alive=lambda _pid: False,
    )["health"] == "ALERT_PID_DEAD"
    _write(path, {"data_epoch": 3, "confirmation20_opened": True}, mtime=950.0)
    assert evaluate_watch(
        spec, now=1000.0, watch_started=700.0, alive=lambda _pid: True,
    )["health"] == "ALERT_SCIENTIFIC_BOUNDARY"


def test_terminal_state_allows_supervisor_to_exit(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _write(path, {"data_epoch": 200}, mtime=1.0)
    row = evaluate_watch(
        parse_watch(f"plain|123|{path}|10|0"), now=1000.0,
        watch_started=0.0, alive=lambda _pid: False,
    )
    assert row["health"] == "TERMINAL"


def test_initial_missing_allowance_is_bounded(tmp_path: Path) -> None:
    spec = parse_watch(f"future|0|{tmp_path / 'future.json'}|100|500")
    assert evaluate_watch(
        spec, now=400.0, watch_started=0.0,
    )["health"] == "WAITING_FOR_INITIAL_STATE"
    assert evaluate_watch(
        spec, now=501.0, watch_started=0.0,
    )["health"] == "ALERT_STATE_MISSING"


def test_contract_records_user_capacity_override(tmp_path: Path) -> None:
    args = Namespace(
        output=tmp_path / "watch", host_label="5090A",
        watch=[f"plain|123|{tmp_path / 'state.json'}|3600|0"],
        disk_path=tmp_path, estimated_remaining_write_gib=24.0,
        minimum_headroom_gib=24.0, user_capacity_override=True,
        poll_seconds=60, timeout_hours=480.0, once=False,
    )
    value = proposed_contract(args)
    assert value["schema"] == CONTRACT_SCHEMA
    assert value["user_capacity_override"] is True
    assert value["performance_values_available"] is False


def test_contract_rejects_relative_disk_path(tmp_path: Path) -> None:
    args = Namespace(
        output=tmp_path / "watch", host_label="host",
        watch=[f"plain|123|{tmp_path / 'state.json'}|3600|0"],
        disk_path=Path("relative"), estimated_remaining_write_gib=1.0,
        minimum_headroom_gib=1.0, user_capacity_override=False,
        poll_seconds=60, timeout_hours=480.0, once=False,
    )
    with pytest.raises(ValueError):
        proposed_contract(args)


def test_aggregate_uses_real_remaining_write_requirement(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    _write(path, {"data_epoch": 2}, mtime=950.0)
    contract = {
        "host_label": "host",
        "watches": [parse_watch(f"plain|123|{path}|100|0")],
        "disk_path": str(tmp_path),
        "estimated_remaining_write_gib": 24.0,
        "minimum_headroom_gib": 24.0,
        "user_capacity_override": True,
    }
    healthy = evaluate_contract(
        contract, now=1000.0, watch_started=900.0,
        alive=lambda _pid: True,
        disk_usage=lambda _path: SimpleNamespace(free=100 * 1024 ** 3),
    )
    assert healthy["status"] == "HEALTHY"
    assert healthy["disk"]["required_gib"] == 48.0
    failed = evaluate_contract(
        contract, now=1000.0, watch_started=900.0,
        alive=lambda _pid: True,
        disk_usage=lambda _path: SimpleNamespace(free=47 * 1024 ** 3),
    )
    assert failed["status"] == "ALERT"
    assert failed["disk"]["health"] == "ALERT_REAL_CAPACITY_RISK"


def test_contract_freeze_is_idempotent_and_fail_closed(tmp_path: Path) -> None:
    value = {"schema": CONTRACT_SCHEMA, "host_label": "host"}
    path = freeze_contract(tmp_path, value)
    assert freeze_contract(tmp_path, value) == path
    with pytest.raises(RuntimeError):
        freeze_contract(tmp_path, {**value, "host_label": "changed"})
