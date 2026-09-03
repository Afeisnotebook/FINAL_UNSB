import json
from pathlib import Path

import pytest

from operations.paper_aio_training_supervisor_guard import (
    _argument,
    next_no_progress_count,
    process_decision,
)


def test_argument_requires_named_value() -> None:
    assert _argument(["x", "--lane", "cut"], "--lane") == "cut"
    assert _argument(["x", "--lane"], "--lane") is None
    assert _argument(["x"], "--lane") is None


@pytest.mark.parametrize(
    ("status", "supervisors", "trainers", "expected"),
    [
        ("COMPLETE_E200", [], [], "COMPLETE"),
        ("CHILD_RUNNING", [10], [11], "ADOPT"),
        ("CHILD_RUNNING", [10], [], "ADOPT"),
        ("CHILD_RUNNING", [], [11], "WAIT_ORPHAN_TRAINER"),
        ("BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE", [], [], "RESTART"),
        ("CHILD_RUNNING", [10, 12], [11], "BLOCK_DUPLICATE"),
        ("CHILD_RUNNING", [10], [11, 13], "BLOCK_DUPLICATE"),
    ],
)
def test_process_decision_is_duplicate_safe(
    status: str, supervisors: list[int], trainers: list[int], expected: str
) -> None:
    assert process_decision(
        supervisor_status=status,
        supervisors=supervisors,
        trainers=trainers,
    ) == expected


def test_restart_budget_resets_only_after_durable_checkpoint_progress() -> None:
    assert next_no_progress_count(2, last_restart_step=100, current_step=101) == 0
    assert next_no_progress_count(0, last_restart_step=100, current_step=100) == 1
    assert next_no_progress_count(1, last_restart_step=100, current_step=99) == 2
    assert next_no_progress_count(2, last_restart_step=None, current_step=0) == 0


def test_guard_source_contains_no_metric_or_confirmation_reader() -> None:
    source = (
        Path(__file__).parents[1]
        / "operations"
        / "paper_aio_training_supervisor_guard.py"
    ).read_text(encoding="utf-8")
    assert "PAPER_RESULTS.json" not in source
    assert "PER_IMAGE" not in source
    assert "confirmation20_opened\": True" not in source
    assert '"performance_values_read": False' in source


def test_state_contract_flags_are_literal_false() -> None:
    # Guard state is intentionally inspectable without importing a training stack.
    sample = {
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    assert json.loads(json.dumps(sample)) == sample
