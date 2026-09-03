from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import pytest

from operations import paper_aio_cross_host_plain_successor as successor


def _args(tmp_path):
    return argparse.Namespace(
        training_repo=tmp_path / "training",
        training_output=tmp_path / "output",
        manifest=tmp_path / "manifest.csv",
        data_root=tmp_path / "data",
        train_view=tmp_path / "view",
        gpu=0,
        python=tmp_path / "python",
        host_label="5090B_MATCHED",
        peer_runtime_receipt=tmp_path / "peer.json",
        co_resident_supervisor_state=None,
        co_resident_heartbeat=None,
        co_resident_lane_id=None,
        capacity_probe_epochs=0,
        plain_isolated_epoch_seconds=None,
        co_resident_isolated_epoch_seconds=None,
        minimum_makespan_saving_seconds=3600,
        required_training_git_commit="a" * 40,
        required_protocol_fingerprint="b" * 64,
        source_host_label="5090B_MATCHED_PLAIN",
        predecessor_state=tmp_path / "cut.json",
        poll_seconds=60,
        timeout_hours=720,
    )


def test_predecessor_decision_is_metric_blind_and_fail_closed():
    assert successor.predecessor_decision(None, timed_out=False) == "WAIT"
    assert successor.predecessor_decision("CHILD_RUNNING", timed_out=False) == "WAIT"
    assert successor.predecessor_decision("COMPLETE_E200", timed_out=False) == "START"
    assert successor.predecessor_decision("BLOCKED_IO", timed_out=False) == "BLOCK"
    assert successor.predecessor_decision(None, timed_out=True) == "TIMEOUT"


def test_gate_chain_requires_exact_runtime_twin_before_authorization(tmp_path):
    rendered = [" ".join(row) for row in successor.gate_commands(_args(tmp_path))]
    assert len(rendered) == 5
    assert "--stage preflight" in rendered[0]
    assert "--stage resume-gate --lane plain" in rendered[1]
    assert "--stage runtime-twin" in rendered[2]
    assert "--peer-receipt" in rendered[2]
    assert "--stage evaluation-repeat-gate --lane plain" in rendered[3]
    assert "--stage authorize --lane plain" in rendered[4]


def test_runtime_receipt_must_be_exact_and_confirmation_closed():
    receipt = {
        "schema": "final-unsb-paper-runtime-twin-receipt-v1",
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": "5090B_MATCHED",
        "updates": 2000,
        "protocol_fingerprint": "frozen",
        "exact_runtime_equivalence": True,
        "differences": {},
        "confirmation20_opened": False,
    }
    successor.validate_runtime_receipt(
        receipt,
        host_label="5090B_MATCHED",
        required_protocol_fingerprint="frozen",
    )
    for key, value in (
        ("status", "FAIL_HOST_SEPARATED"),
        ("updates", 1999),
        ("exact_runtime_equivalence", False),
        ("confirmation20_opened", True),
    ):
        broken = {**receipt, key: value}
        with pytest.raises(RuntimeError, match="did not pass exactly"):
            successor.validate_runtime_receipt(
                broken,
                host_label="5090B_MATCHED",
                required_protocol_fingerprint="frozen",
            )


def test_capacity_contract_is_all_or_none_and_requires_clean_probe(tmp_path):
    args = _args(tmp_path)
    assert successor.capacity_contract(args) is None
    args.co_resident_lane_id = "cyclegan"
    with pytest.raises(RuntimeError, match="requires all inputs"):
        successor.capacity_contract(args)
    args.co_resident_supervisor_state = tmp_path / "cycle_state.json"
    args.co_resident_heartbeat = tmp_path / "cycle_heartbeat.json"
    args.plain_isolated_epoch_seconds = 2410.47
    args.co_resident_isolated_epoch_seconds = 1588.13
    args.capacity_probe_epochs = 1
    with pytest.raises(RuntimeError, match="at least two probe epochs"):
        successor.capacity_contract(args)
    args.capacity_probe_epochs = 2
    value = successor.capacity_contract(args)
    assert value["capacity_probe_epochs"] == 2


def test_current_5090b_projection_prefers_start_after_cut():
    value = successor.project_colocation_makespan(
        target_epochs=200,
        plain_completed_epochs=2,
        plain_colocated_epoch_seconds=2410.47 * 1.5,
        plain_isolated_epoch_seconds=2410.47,
        co_resident_completed_epochs=136.14,
        co_resident_colocated_epoch_seconds=2334.91,
        co_resident_isolated_epoch_seconds=1588.13,
    )
    assert value["continue_now_saving_seconds"] > 10 * 3600
    assert value["continue_now_seconds"] < value["wait_for_release_seconds"]


def test_projection_waits_when_contention_exceeds_release_cost():
    value = successor.project_colocation_makespan(
        target_epochs=200,
        plain_completed_epochs=2,
        plain_colocated_epoch_seconds=20000,
        plain_isolated_epoch_seconds=2000,
        co_resident_completed_epochs=190,
        co_resident_colocated_epoch_seconds=3000,
        co_resident_isolated_epoch_seconds=1000,
    )
    assert value["continue_now_saving_seconds"] < 0


def test_capacity_gate_uses_only_training_heartbeats_and_preserves_resume(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    companion_state = tmp_path / "cycle_state.json"
    companion_heartbeat = tmp_path / "cycle_heartbeat.json"
    companion_state.write_text(
        json.dumps({"status": "CHILD_RUNNING"}), encoding="utf-8"
    )
    companion_heartbeat.write_text(
        json.dumps(
            {
                "lane_id": "cyclegan",
                "data_epoch": 136,
                "epoch_wall_seconds": 2335.0,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
        ),
        encoding="utf-8",
    )
    contract = {
        "co_resident_supervisor_state": companion_state,
        "co_resident_heartbeat": companion_heartbeat,
        "co_resident_lane_id": "cyclegan",
        "plain_isolated_epoch_seconds": 2410.47,
        "co_resident_isolated_epoch_seconds": 1588.13,
        "capacity_probe_epochs": 2,
        "minimum_makespan_saving_seconds": 3600.0,
    }

    def fake_run_logged(command, *, cwd, log):
        assert "--engineering-stop-after-updates" in command
        heartbeat = args.training_output / "lanes" / "plain" / "HEARTBEAT.json"
        heartbeat.parent.mkdir(parents=True)
        heartbeat.write_text(
            json.dumps(
                {
                    "lane_id": "plain",
                    "updates": 17106,
                    "data_epoch": 2,
                    "epoch_wall_seconds": 3615.0,
                    "paired_controller_access": False,
                    "confirmation20_opened": False,
                }
            ),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(successor, "run_logged", fake_run_logged)
    value = successor.run_capacity_gate(
        args,
        repo=tmp_path,
        output=args.training_output,
        python=tmp_path / "python",
        protocol={
            "training": {
                "steps_per_data_epoch": 8553,
                "target_data_epochs": 200,
                "target_updates": 1710600,
            }
        },
        log=tmp_path / "probe.log",
        contract=contract,
    )
    assert value["decision"] == "CONTINUE_PLAIN_NOW"
    assert value["probe_full_state_preserved_for_exact_resume"] is True
    assert value["performance_values_read"] is False


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _frozen_repo(path: Path, files: dict[str, str]) -> str:
    path.mkdir(parents=True)
    _git(path, "init")
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", ".")
    _git(
        path, "-c", "user.email=test@example.com", "-c", "user.name=test",
        "commit", "-m", "fixture",
    )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True,
    ).strip()


def test_cross_host_successor_freezes_control_manifest_and_peer_receipt(
    tmp_path: Path, monkeypatch,
) -> None:
    control = tmp_path / "control"
    scientific = tmp_path / "scientific"
    _frozen_repo(control, {
        "operations/paper_aio_cross_host_plain_successor.py": "controller\n",
    })
    scientific_commit = _frozen_repo(scientific, {"training.py": "frozen\n"})
    monkeypatch.setattr(
        successor, "__file__",
        str(control / "operations/paper_aio_cross_host_plain_successor.py"),
    )
    args = _args(tmp_path)
    args.training_repo = scientific
    args.required_training_git_commit = scientific_commit
    args.manifest.write_text("manifest\n", encoding="utf-8")
    args.peer_runtime_receipt.write_text(json.dumps({
        "schema": "final-unsb-paper-runtime-twin-receipt-v1",
        "status": "LOCAL_TWIN_COMPLETE",
        "host_label": "5090A",
        "updates": 2000,
        "protocol_fingerprint": args.required_protocol_fingerprint,
        "manifest_sha256": successor.file_sha256(args.manifest),
        "confirmation20_opened": False,
    }) + "\n", encoding="utf-8")
    contract = successor.frozen_contract(args, colocation=None)
    successor.verify_frozen_contract(contract)
    assert contract["peer_host_label"] == "5090A"
    assert contract["fresh_e0_required"] is True
    assert contract["cross_host_checkpoint_resume"] is False

    args.peer_runtime_receipt.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="peer runtime receipt changed"):
        successor.verify_frozen_contract(contract)
