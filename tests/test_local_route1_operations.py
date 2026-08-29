import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations.local_route1_executor import (
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
    atomic_json,
    current_epoch,
    default_contract,
    latest_sidecar,
    validate_contract,
    validate_lane_sidecar,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_atomic_json_never_leaves_temporary(tmp_path):
    path = tmp_path / "state.json"
    atomic_json(path, {"status": "PASS"})
    assert json.loads(path.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not path.with_suffix(".json.tmp").exists()


def test_contract_locks_scripts_and_scientific_identity(tmp_path):
    supervisor = tmp_path / "local_route1_executor.py"
    backfill = tmp_path / "local_route1_metric_backfill.py"
    supervisor.write_text("supervisor", encoding="utf-8")
    backfill.write_text("backfill", encoding="utf-8")
    args = SimpleNamespace(
        main_repo=tmp_path,
        executor_repo=tmp_path,
        run_root=tmp_path,
        train_view=tmp_path,
        data_root=tmp_path,
        manifest=tmp_path / "manifest.csv",
        python=tmp_path / "python.exe",
    )
    # default_contract uses the real module paths; replace them with fixture files.
    contract = default_contract(args)
    contract.update(
        {
            "supervisor_script": str(supervisor),
            "supervisor_sha256": sha(supervisor),
            "backfill_script": str(backfill),
            "backfill_sha256": sha(backfill),
        }
    )
    validate_contract(contract)
    supervisor.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="supervisor script changed"):
        validate_contract(contract)


def test_checkpoint_sidecar_epoch_and_identity(tmp_path):
    lane_root = tmp_path / "anchors" / "plain"
    lane_root.mkdir(parents=True)
    sidecar = {
        "physical_epoch_completed": 100,
        "full_state_sha256": "checkpoint",
        "metadata": {
            "probe_id": "plain",
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        },
    }
    atomic_json(lane_root / "full_state_latest.pt.json", sidecar)
    assert current_epoch(tmp_path, "plain") == 100
    assert latest_sidecar(tmp_path, "plain") == sidecar
    identity = {
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
    }
    validate_lane_sidecar(sidecar, identity, "plain")
    sidecar["metadata"]["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="confirmation20_opened"):
        validate_lane_sidecar(sidecar, identity, "plain")
