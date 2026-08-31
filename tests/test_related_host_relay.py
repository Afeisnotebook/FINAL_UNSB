from __future__ import annotations

from argparse import Namespace

import pytest

from operations.local_route1_related_host_relay import (
    _validate_host,
    default_contract,
    validate_contract,
)
from research.local_route1.related_algorithm_adjudication import HOST_SCHEMA


def _host() -> dict:
    return {
        "schema": HOST_SCHEMA,
        "status": "RELATED_HOST_E200_ADJUDICATION_COMPLETE",
        "host_label": "remote5090",
        "ranking": [{}, {}, {}],
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def test_related_host_portable_boundary_requires_complete_three_algorithm_frontier():
    _validate_host(_host())
    invalid = _host()
    invalid["ranking"] = [{}, {}]
    with pytest.raises(RuntimeError, match="frontier"):
        _validate_host(invalid)


def test_related_host_relay_contract_never_persists_password(tmp_path):
    args = Namespace(
        source_host="source",
        source_port=12770,
        source_user="root",
        source_path="/source.json",
        destination_host="destination",
        destination_port=22,
        destination_user="yc",
        destination_path="/destination.json",
        local_spool=tmp_path / "spool.json",
        state=tmp_path / "state.json",
        poll_seconds=60,
        timeout_seconds=1209600,
    )
    contract = default_contract(args)
    validate_contract(contract)
    assert "password" not in str(contract).lower()
    assert contract["checkpoint_transfer"] is False
    assert contract["cross_host_deltas_merged"] is False

