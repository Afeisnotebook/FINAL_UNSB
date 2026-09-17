import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_final_handoff_is_the_first_repository_entrypoint():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert readme.index("FINAL_HANDOFF_CN.md") < readme.index("CURRENT_PROJECT_STATE_CN.md")
    assert agents.index("FINAL_HANDOFF_CN.md") < agents.index("CURRENT_PROJECT_STATE_CN.md")


def test_final_handoff_keeps_scientific_boundaries_sealed():
    handoff = json.loads((ROOT / "FINAL_HANDOFF.json").read_text(encoding="utf-8"))
    assert handoff["status"] == "DISCOVERY_ARCHIVED_CLAIM_REVIEW_PENDING_CONFIRMATION20_SEALED"
    assert handoff["protocol"]["confirmation20_opened"] is False
    assert handoff["protocol"]["best_checkpoint_selection"] is False
    assert handoff["canonical_outcome"]["accepted_algorithms"] == ["proposal"]
    assert handoff["authority_rules"]["restart_old_successors"] is False
    assert handoff["authority_rules"]["auto_import_independent_proofs"] is False


def test_historical_active_plan_is_explicitly_closed():
    plan = (ROOT / "ACTIVE_PAPER_AIO_PLAN_CN.md").read_text(encoding="utf-8")
    start = (ROOT / "START_HERE_CN.md").read_text(encoding="utf-8")
    capsule = (ROOT / "CONTEXT_CAPSULE_CN.md").read_text(encoding="utf-8")
    assert "CLOSED_EXECUTION_PLAN" in plan[:700]
    assert "本文件不再保存会过期的服务器进度" in start[:700]
    assert "不保存旧PID或在线队列" in capsule[:700]


def test_project_state_points_to_final_handoff():
    state = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    assert state["active_control_entrypoint"]["primary"] == "FINAL_HANDOFF.json"
    assert "FINAL_HANDOFF" in state["phase"]
