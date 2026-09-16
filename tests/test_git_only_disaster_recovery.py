import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_git_only_recovery_levels_are_not_overclaimed():
    payload = json.loads(
        (ROOT / "configs" / "GIT_ONLY_RECOVERY_MANIFEST.json").read_text(encoding="utf-8")
    )
    capabilities = payload["git_only_capabilities"]
    assert capabilities["new_codex_paper_discussion"] == "COMPLETE"
    assert capabilities["scientific_state_and_decision_recovery"] == "COMPLETE"
    assert capabilities["trained_weight_recovery"] == "NOT_IN_GIT"
    assert capabilities["exact_checkpoint_resume"] == "NOT_IN_GIT"
    assert payload["data_redistribution"]["image_bytes_in_public_git"] is False
    assert payload["confirmation20_opened"] is False


def test_six_essential_e200_binary_identities_are_preserved():
    payload = json.loads(
        (ROOT / "configs" / "GIT_ONLY_RECOVERY_MANIFEST.json").read_text(encoding="utf-8")
    )
    assets = payload["essential_e200_binary_assets_not_stored_in_git"]
    assert len(assets) == 6
    assert {item["lane"] for item in assets} == {
        "4090A_plain",
        "4090A_amtnc",
        "5090A_stcgr",
        "5090B_matched_plain",
        "5090C_proposal",
        "local_dclgan",
    }
    assert all(len(item["checkpoint_sha256"]) == 64 for item in assets)


def test_fresh_context_bootstrap_cannot_authorize_training():
    prompt = (ROOT / "prompts" / "GIT_ONLY_PAPER_DISCUSSION_BOOTSTRAP_CN.md").read_text(
        encoding="utf-8"
    )
    assert "不要启动实验" in prompt
    assert "verify_git_only_recovery.py" in prompt
