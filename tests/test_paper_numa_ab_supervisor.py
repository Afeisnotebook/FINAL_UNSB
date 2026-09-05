from pathlib import Path

import pytest
import torch

from operations.paper_aio_numa_ab_supervisor import (
    _branch_command,
    component_hashes,
    format_cpu_list,
    net_remaining_saving,
    parse_cpu_list,
    prepare_branch,
)


def test_cpu_list_round_trip_and_canonicalization() -> None:
    cpus = parse_cpu_list("0-3,8,10-11")
    assert cpus == {0, 1, 2, 3, 8, 10, 11}
    assert format_cpu_list(cpus) == "0-3,8,10-11"


def test_net_saving_includes_both_branch_costs() -> None:
    value = net_remaining_saving(
        original_seconds=100.0,
        bound_seconds=80.0,
        updates=1000,
        remaining_updates=100_000,
    )
    assert value["ab_test_cost_seconds"] == 180.0
    assert value["original_remaining_seconds"] == 10_000.0
    assert value["net_saving_seconds"] == 1_820.0
    assert value["net_saving_fraction"] == pytest.approx(0.182)


def test_prepare_regular_lane_branch_copies_only_required_state(tmp_path) -> None:
    source = tmp_path / "source"
    for relative, value in {
        "shared_e0/unsb_common/e0.pt": b"e0",
        "shared_e0/unsb_common/e0.pt.json": b"{}",
        "lanes/proposal/full_state_latest.pt": b"state",
        "lanes/proposal/full_state_latest.pt.json": b"{}",
        "gates/LANE_AUTHORIZATION_proposal.json": b"{}",
        "PAPER_PROTOCOL.json": b"{}",
    }.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    destination = tmp_path / "branch"
    prepare_branch(
        source_output=source, destination=destination,
        lane_id="proposal", candidate_id=None,
    )
    assert (destination / "lanes/proposal/full_state_latest.pt").read_bytes() == b"state"
    assert (destination / "gates/LANE_AUTHORIZATION_proposal.json").is_file()
    assert not (destination / "lanes/proposal/HEARTBEAT.json").exists()


def test_prepare_candidate_branch_copies_lock_and_authority(tmp_path) -> None:
    source = tmp_path / "source"
    candidate = "G4-01"
    for relative in (
        "shared_e0/unsb_common/e0.pt",
        "shared_e0/unsb_common/e0.pt.json",
        f"lanes/{candidate}/full_state_latest.pt",
        f"lanes/{candidate}/full_state_latest.pt.json",
        f"candidate_locks/{candidate}/CANDIDATE_LOCK.json",
        f"gates/CANDIDATE_AUTHORIZATION_{candidate}.json",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    destination = tmp_path / "branch"
    prepare_branch(
        source_output=source, destination=destination,
        lane_id=candidate, candidate_id=candidate,
    )
    assert (destination / f"candidate_locks/{candidate}/CANDIDATE_LOCK.json").is_file()
    assert (destination / f"gates/CANDIDATE_AUTHORIZATION_{candidate}.json").is_file()


def test_branch_command_changes_only_affinity_and_engineering_stop(tmp_path) -> None:
    command = _branch_command(
        python=Path("/env/python"), repo=tmp_path, output=tmp_path / "branch",
        lane_id="proposal", candidate_id=None,
        manifest=tmp_path / "manifest.csv", data_root=tmp_path / "data",
        train_view=tmp_path / "view", gpu=0, stop=99_000,
        cpus={0, 1, 64, 65},
    )
    assert command[:3] == ["taskset", "--cpu-list", "0-1,64-65"]
    assert command[command.index("--lane") + 1] == "proposal"
    assert command[command.index("--engineering-stop-after-updates") + 1] == "99000"
    forbidden = {"--batch-size", "--amp", "--tf32", "--compile", "--num-threads"}
    assert forbidden.isdisjoint(command)


def test_component_hashes_cover_transition_defining_substates(tmp_path) -> None:
    payload = {
        "step": 10,
        "target_steps": 100,
        "model": {
            "networks": {"G": {"w": torch.tensor([1.0])}},
            "optimizers": [{"state": {}, "param_groups": []}],
            "schedulers": [{"last_epoch": 0}],
            "method_state": {"counter": 2},
        },
        "rng": {"python": (3, (1, 2, 3), None)},
        "samplers": {"primary": {"position": 4}, "secondary": {"position": 5}},
    }
    left = tmp_path / "left.pt"
    right = tmp_path / "right.pt"
    torch.save(payload, left)
    torch.save(payload, right)
    left_hashes = component_hashes(left, Path.cwd())
    right_hashes = component_hashes(right, Path.cwd())
    assert left_hashes == right_hashes
    changed = torch.load(right, map_location="cpu", weights_only=False)
    changed["samplers"]["primary"]["position"] = 6
    torch.save(changed, right)
    changed_hashes = component_hashes(right, Path.cwd())
    assert left_hashes["networks"] == changed_hashes["networks"]
    assert left_hashes["samplers"] != changed_hashes["samplers"]
    assert left_hashes["scientific_core"] != changed_hashes["scientific_core"]
