from __future__ import annotations

import copy

import pytest

from operations.local_route1_repaired_portfolio_relay import (
    SCHEMA,
    validate_contract,
)


def _contract(tmp_path):
    return {
        "schema": SCHEMA,
        "source": {
            "host": "source.example",
            "port": 12770,
            "user": "root",
            "path": "/source/portfolio.json",
        },
        "destination": {
            "host": "destination.example",
            "port": 22,
            "user": "worker",
            "path": "/destination/portfolio.imported.json",
        },
        "local_spool": str(tmp_path / "portfolio.json"),
        "state_path": str(tmp_path / "state.json"),
        "poll_seconds": 60,
        "timeout_seconds": 1209600,
        "maximum_consecutive_failures": 3,
        "complete_source_e200_only": True,
        "destination_overwrite_allowed": False,
        "cross_host_checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_relay_contract_contains_no_credentials_and_is_fail_closed(tmp_path):
    contract = _contract(tmp_path)
    validate_contract(contract)
    serialized = __import__("json").dumps(contract).lower()
    assert "password" not in serialized
    changed = copy.deepcopy(contract)
    changed["destination_overwrite_allowed"] = True
    with pytest.raises(RuntimeError, match="destination_overwrite_allowed"):
        validate_contract(changed)


def test_relay_rejects_persisted_password_fields(tmp_path):
    contract = _contract(tmp_path)
    contract["source"]["password"] = "forbidden"
    with pytest.raises(RuntimeError, match="persist a password"):
        validate_contract(contract)

