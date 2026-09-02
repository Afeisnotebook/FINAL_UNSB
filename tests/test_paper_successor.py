from pathlib import Path

from operations.paper_aio_successor import gate_commands, predecessor_decision


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
    assert len(commands) == 4
    assert "--stage resume-gate --lane amtnc" in rendered[1]
    assert "--stage evaluation-repeat-gate --lane amtnc" in rendered[2]
    assert "--stage authorize --lane amtnc" in rendered[3]
    assert all("zero-intervention" not in command for command in rendered)
    assert all("matched-plain-mode" not in command for command in rendered)
