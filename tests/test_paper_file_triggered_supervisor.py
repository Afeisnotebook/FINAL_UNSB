from operations.paper_aio_file_triggered_supervisor import trigger_decision


def test_file_trigger_decision_is_metric_blind() -> None:
    assert trigger_decision(trigger_exists=False, timed_out=False) == "WAIT"
    assert trigger_decision(trigger_exists=True, timed_out=False) == "START"
    assert trigger_decision(trigger_exists=True, timed_out=True) == "START"
    assert trigger_decision(trigger_exists=False, timed_out=True) == "TIMEOUT"
