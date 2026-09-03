from pathlib import Path
from types import SimpleNamespace

import subprocess

import pytest

from operations import paper_aio_successor as successor
from operations.paper_aio_successor import (
    frozen_contract,
    gate_commands,
    predecessor_decision,
    verify_frozen_contract,
)


def test_predecessor_decision_is_metric_blind() -> None:
    assert predecessor_decision(None) == "WAIT"
    assert predecessor_decision("CHILD_RUNNING") == "WAIT"
    assert predecessor_decision("WAITING_TO_EXACT_RESUME") == "WAIT"
    assert predecessor_decision("COMPLETE_E200") == "START"
    assert predecessor_decision("BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE") == "BLOCK"


def test_proposal_successor_has_identity_and_same_runtime_gate() -> None:
    commands = gate_commands(
        python="python", output=Path("/run"), manifest=Path("/manifest.csv"),
        data_root=Path("/data"), train_view=Path("/view"),
        successor="proposal", gpu=0,
    )
    rendered = [" ".join(command) for command in commands]
    assert len(commands) == 5
    assert "--stage preflight" in rendered[0]
    assert "--stage resume-gate --lane proposal" in rendered[1]
    assert "--stage zero-intervention-gate" in rendered[2]
    assert "--stage evaluation-repeat-gate --lane proposal" in rendered[3]
    assert "--matched-plain-mode same_runtime_output_root" in rendered[4]


def test_cyclegan_successor_does_not_claim_matched_delta() -> None:
    commands = gate_commands(
        python="python", output=Path("/run"), manifest=Path("/manifest.csv"),
        data_root=Path("/data"), train_view=Path("/view"),
        successor="cyclegan", gpu=0,
    )
    rendered = [" ".join(command) for command in commands]
    assert len(commands) == 4
    assert all("zero-intervention" not in command for command in rendered)
    assert all("matched-plain-mode" not in command for command in rendered)


def test_amtnc_successor_uses_frozen_static_lane_gates() -> None:
    commands = gate_commands(
        python="python", output=Path("/run"), manifest=Path("/manifest.csv"),
        data_root=Path("/data"), train_view=Path("/view"),
        successor="amtnc", gpu=0,
    )
    rendered = [" ".join(command) for command in commands]
    assert len(commands) == 5
    assert "--stage resume-gate --lane amtnc" in rendered[1]
    assert "paper_aio_amtnc_identity_gate.py" in rendered[2]
    assert "--stage evaluation-repeat-gate --lane amtnc" in rendered[3]
    assert "--stage authorize --lane amtnc" in rendered[4]
    assert all("matched-plain-mode" not in command for command in rendered)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def _frozen_repo(path: Path, files: dict[str, str]) -> None:
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


def test_successor_freezes_control_and_scientific_checkouts(
    tmp_path: Path, monkeypatch,
) -> None:
    control = tmp_path / "control"
    scientific = tmp_path / "scientific"
    _frozen_repo(control, {
        "operations/paper_aio_successor.py": "controller\n",
        "operations/paper_aio_amtnc_identity_gate.py": "identity\n",
    })
    _frozen_repo(scientific, {"training.py": "frozen\n"})
    monkeypatch.setattr(
        successor, "__file__", str(control / "operations/paper_aio_successor.py"),
    )
    scientific_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=scientific, text=True,
    ).strip()
    args = SimpleNamespace(
        repo=scientific,
        output=tmp_path / "output",
        manifest=tmp_path / "manifest.csv",
        data_root=tmp_path / "data",
        train_view=tmp_path / "view",
        predecessor="plain",
        successor="amtnc",
        required_git_commit=scientific_commit,
        required_protocol_fingerprint="p" * 64,
        gpu=0,
        poll_seconds=60,
    )
    contract = frozen_contract(args)
    verify_frozen_contract(contract)
    assert contract["scientific_git_commit"] == scientific_commit
    assert set(contract["control_source_sha256"]) == {
        "operations/paper_aio_successor.py",
        "operations/paper_aio_amtnc_identity_gate.py",
    }

    identity = control / "operations/paper_aio_amtnc_identity_gate.py"
    identity.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="control checkout moved"):
        verify_frozen_contract(contract)
