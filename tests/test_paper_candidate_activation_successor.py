from operations.paper_aio_candidate_activation_successor import file_trigger_decision


def test_candidate_activation_trigger_is_completion_only() -> None:
    assert file_trigger_decision(all_exist=False, timed_out=False) == "WAIT"
    assert file_trigger_decision(all_exist=True, timed_out=False) == "START"
    assert file_trigger_decision(all_exist=True, timed_out=True) == "START"
    assert file_trigger_decision(all_exist=False, timed_out=True) == "TIMEOUT"
