import json
from pathlib import Path

import pytest

from research.paper_aio.gates import resolve_capacity_policy


def _receipt(path: Path, **changes) -> Path:
    payload = {
        "schema": "final-unsb-paper-user-capacity-override-v1",
        "status": "USER_CAPACITY_OVERRIDE",
        "allowed_host_labels": ["5090A"],
        "allowed_node_roles": ["training"],
        "estimated_worst_case_incremental_write_gib": 24,
        "safety_multiplier": 2,
        "minimum_operational_floor_gib": 32,
        "effective_minimum_free_gib": 48,
        "no_deletion_authorized": True,
    }
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_user_capacity_override_replaces_operational_floor_only(tmp_path: Path) -> None:
    minimum, record = resolve_capacity_policy(
        protocol_minimum_gib=200,
        override_path=_receipt(tmp_path / "override.json"),
        host_label="5090A",
        node_role="training",
    )
    assert minimum == 48
    assert record["mode"] == "USER_CAPACITY_OVERRIDE"
    assert record["protocol_minimum_free_gib"] == 200
    assert record["no_deletion_authorized"] is True


def test_user_capacity_override_is_host_bound(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not authorize host"):
        resolve_capacity_policy(
            protocol_minimum_gib=200,
            override_path=_receipt(tmp_path / "override.json"),
            host_label="5090B",
            node_role="training",
        )


def test_user_capacity_override_recomputes_declared_floor(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not derivable"):
        resolve_capacity_policy(
            protocol_minimum_gib=200,
            override_path=_receipt(
                tmp_path / "override.json", effective_minimum_free_gib=20,
            ),
            host_label="5090A",
            node_role="training",
        )
