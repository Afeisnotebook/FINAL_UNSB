from operations.paper_aio_first_failure_interlock import should_interlock


def test_interlock_acts_only_after_a_failure() -> None:
    assert not should_interlock({
        "status": "CHILD_RUNNING", "consecutive_failures": 0,
    })
    assert not should_interlock({
        "status": "COMPLETE_E200", "consecutive_failures": 0,
    })
    assert should_interlock({
        "status": "WAITING_TO_EXACT_RESUME", "consecutive_failures": 1,
    })
    assert should_interlock({
        "status": "CHILD_RUNNING", "consecutive_failures": 1,
    })
    assert should_interlock({
        "status": "BLOCKED_AFTER_REPEATED_ENGINEERING_FAILURE",
        "consecutive_failures": 3,
    })
