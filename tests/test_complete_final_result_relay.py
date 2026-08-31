from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from operations import local_route1_complete_final_result_relay as relay
from research.local_route1.complete_frontier_final_delivery import (
    POINTER,
    POINTER_SCHEMA,
    PUBLISHED_FILES,
)
from research.local_route1.protocol import file_sha256


def _write(path: Path, value: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def _delivery(root: Path) -> dict:
    selected = "CURRENT-BEST"
    for name in PUBLISHED_FILES:
        if name == "CANDIDATE.json":
            value = {
                "candidate_id": selected,
                "canonical_candidate_is_action_priority_only": True,
                "algorithm_discovery_collapsed_to_single_candidate": False,
            }
        elif name == "RESEARCH_FRONTIER.json":
            value = {
                "action_priority_candidate_id": selected,
                "algorithm_discovery_collapsed_to_single_candidate": False,
                "confirmation20_opened": False,
            }
        elif name.endswith(".json"):
            value = {"name": name}
        else:
            value = "report"
        _write(root / name, value)
    for name in relay.EXTRA_FILES:
        _write(root / name, {"name": name})
    pointer = {
        "schema": POINTER_SCHEMA,
        "status": "COMPLETE_FRONTIER_FINAL_DELIVERY_COMPLETE",
        "selected_candidate_id": selected,
        "final_file_sha256": {
            name: file_sha256(root / name) for name in PUBLISHED_FILES
        },
        "complete_4090_frontier_sha256": file_sha256(root / relay.EXTRA_FILES[0]),
        "portable_5090_frontier_sha256": file_sha256(root / relay.EXTRA_FILES[1]),
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "research_frontier_unique_candidate_count": 3,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(root / POINTER, pointer)
    return pointer


def test_complete_final_result_relay_validates_exact_local_delivery(tmp_path: Path):
    pointer = _delivery(tmp_path)
    assert relay.validate_local_delivery(tmp_path) == pointer
    _write(tmp_path / "RESEARCH_FRONTIER.json", {"changed": True})
    with pytest.raises(RuntimeError, match="relayed final file changed"):
        relay.validate_local_delivery(tmp_path)


def test_complete_final_result_relay_contract_never_persists_credentials(tmp_path: Path):
    contract = relay.default_contract(Namespace(
        source_host="source",
        source_port=22,
        source_user="user",
        source_run_root="/remote/run",
        destination=tmp_path / "delivery",
        state=tmp_path / "state.json",
        poll_seconds=60,
        timeout_seconds=1209600,
    ))
    relay.validate_contract(contract)
    assert contract["credentials_persisted"] is False
    assert contract["checkpoint_transfer"] is False
    assert contract["cross_host_deltas_merged"] is False
    assert "password" not in json.dumps(contract).lower()
