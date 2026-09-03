from operations.paper_aio_progress_watch import (
    classify_probe,
    effective_progress_age,
    parse_process_stat,
    process_start_unix,
)


def _snapshot(*, reads: int, cpu: int) -> dict:
    return {
        "cpu_ticks": cpu,
        "io": {
            "rchar": reads,
            "wchar": 10,
            "read_bytes": reads,
            "write_bytes": 20,
            "syscr": reads,
            "syscw": 1,
        },
    }


def test_child_restart_resets_effective_stale_age() -> None:
    assert effective_progress_age(
        now=10_000, heartbeat_mtime=1_000, child_started_unix=9_900,
    ) == 100


def test_stale_heartbeat_with_io_progress_is_not_an_alert() -> None:
    status, alert = classify_probe(
        effective_age_seconds=8000, stall_seconds=7200,
        before=_snapshot(reads=100, cpu=20), after=_snapshot(reads=101, cpu=30),
    )
    assert status == "STALE_EPOCH_HEARTBEAT_BUT_PROCESS_IO_PROGRESSING"
    assert alert is False


def test_live_compute_without_io_progress_is_an_alert() -> None:
    status, alert = classify_probe(
        effective_age_seconds=8000, stall_seconds=7200,
        before=_snapshot(reads=100, cpu=20), after=_snapshot(reads=100, cpu=30),
    )
    assert status == "ALERT_LIVE_PROCESS_COMPUTE_WITHOUT_IO_PROGRESS"
    assert alert is True


def test_recent_heartbeat_does_not_alert_even_without_io() -> None:
    status, alert = classify_probe(
        effective_age_seconds=100, stall_seconds=7200,
        before=_snapshot(reads=100, cpu=20), after=_snapshot(reads=100, cpu=20),
    )
    assert status == "HEALTHY_WITHIN_EPOCH_BOUND"
    assert alert is False


def test_linux_process_stat_parser_and_start_time() -> None:
    # Fields after comm start at Linux stat field 3.  The values below make
    # utime=11, stime=12 and starttime=19.
    parsed = parse_process_stat(
        "42 (worker name) R 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
    )
    assert parsed == {"pid": 42, "state": "R", "cpu_ticks": 23, "start_ticks": 19}
    assert process_start_unix(
        200, now=2000.0, uptime_seconds=1000.0, clock_ticks=100,
    ) == 1002.0
