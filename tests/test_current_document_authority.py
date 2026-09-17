import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_document_registry_resolves_current_and_historical_layers():
    registry = _load("configs/DOCUMENT_AUTHORITY_REGISTRY.json")
    assert registry["status"] == (
        "CURRENT_AUTHORITY_EXPLICIT_HISTORICAL_EXECUTION_QUARANTINED"
    )
    assert registry["authority_order"][:4] == [
        "FINAL_HANDOFF_CN.md",
        "FINAL_HANDOFF.json",
        "docs/CURRENT_PROJECT_STATE_CN.md",
        "CLAIM_BOUNDARIES.md",
    ]
    assert registry["hard_interpretation_rules"][
        "old_pid_or_heartbeat_means_job_is_live"
    ] is False
    assert registry["hard_interpretation_rules"][
        "active_token_inside_frozen_protocol_authorizes_execution"
    ] is False
    assert registry["hard_interpretation_rules"]["confirmation20_authorized"] is False


def test_current_entrypoints_contain_no_stale_live_schedule_claims():
    current_files = [
        "README.md",
        "START_HERE_CN.md",
        "CONTEXT_CAPSULE_CN.md",
        "AGENTS.md",
        "WORKFLOW_CN.md",
        "CLAIM_BOUNDARIES.md",
        "docs/CURRENT_PROJECT_STATE_CN.md",
        "prompts/MASTER_CODEX_BOOTSTRAP_CN.md",
        "prompts/SERVER_CODEX_BOOTSTRAP_CN.md",
    ]
    forbidden = [
        "FIRST_WAVE_RUNNING",
        "THREE_ALGORITHM_PATHS_RUNNING",
        "现运行AM-TNC",
        "运行Proposal；",
        "运行full-data ST-CGR",
        "独占运行DCLGAN",
        "GRANTED_FULL_DATA_PAPER_AND_ROUTE1_RECONSTRUCTION",
    ]
    for relative in current_files:
        text = (ROOT / relative).read_text(encoding="utf-8")
        for phrase in forbidden:
            assert phrase not in text, (relative, phrase)


def test_operational_ledgers_have_final_overlays_and_cannot_schedule():
    state = _load("PROJECT_STATE.json")
    portfolio = _load("configs/FULL_DATA_METHOD_PORTFOLIO.json")
    matrix = _load("configs/PAPER_DELIVERY_COMPLETION_MATRIX.json")
    assert state["current_execution_authorized"] is False
    assert state["historical_nested_pid_heartbeat_and_running_fields_are_current"] is False
    assert portfolio["nested_running_pid_and_waiter_fields_are_current"] is False
    assert portfolio["historical_nested_fields_must_not_schedule_or_restart_work"] is True
    assert matrix["nested_waiter_pid_and_running_fields_are_current"] is False
    assert matrix["historical_nested_fields_must_not_schedule_or_restart_work"] is True
    assert portfolio["final_outcome_overlay"]["proposal"] == (
        "pass_preregistered_full_data_gate"
    )


def test_retired_prompts_and_plan_cannot_authorize_execution():
    server = (ROOT / "prompts/SERVER_CODEX_BOOTSTRAP_CN.md").read_text(encoding="utf-8")
    master = (ROOT / "prompts/MASTER_CODEX_BOOTSTRAP_CN.md").read_text(encoding="utf-8")
    plan = (ROOT / "ACTIVE_PAPER_AIO_PLAN_CN.md").read_text(encoding="utf-8")
    assert "RETIRED_NO_LIVE_EXECUTION_AUTHORITY" in server
    assert "不得启动实验" in master
    assert "CLOSED_EXECUTION_PLAN" in plan
