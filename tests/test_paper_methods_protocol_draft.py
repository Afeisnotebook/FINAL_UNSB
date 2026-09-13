from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "PAPER_METHODS_PROTOCOL_DRAFT_CONTRACT.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_methods_draft_and_all_authorities_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bindings = [contract["draft"], *contract["authorities"].values()]
    for binding in bindings:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]


def test_methods_draft_matches_frozen_protocol_constants() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "PAPER_AIO_UNPAIRED_V1.json").read_text(
            encoding="utf-8"
        )
    )
    draft = (ROOT / "research" / "paper_aio" / "METHODS_PROTOCOL_DRAFT_EN.md").read_text(
        encoding="utf-8"
    )
    training = protocol["training"]
    common = protocol["common"]
    unsb = protocol["unsb"]
    evaluation = protocol["evaluation"]

    assert "PRE-RESULT / NO EMPIRICAL CLAIM" in draft
    assert f"{training['steps_per_data_epoch']:,}" in draft
    assert f"{training['target_updates']:,}" in draft
    assert f"{training['target_data_epochs']} data epochs" in draft
    assert f"seed {protocol['seed']}" in draft
    assert f"{common['crop_size']}-by-{common['crop_size']}" in draft
    assert f"\\tau={unsb['tau']}" in draft
    assert f"\\(T={unsb['num_timesteps']}\\)" in draft
    assert f"temperature {unsb['nce_T']}" in draft
    assert f"{unsb['num_patches']} sampled patches" in draft
    assert f"{evaluation['trajectory_discovery_per_domain']} discovery images per domain" in draft
    assert f"{evaluation['terminal_discovery_per_domain']} discovery images per domain" in draft
    assert "e150/e175/e200" in draft
    assert "never the best checkpoint" in draft


def test_methods_draft_contains_no_numeric_performance_claim() -> None:
    draft = (ROOT / "research" / "paper_aio" / "METHODS_PROTOCOL_DRAFT_EN.md").read_text(
        encoding="utf-8"
    )
    forbidden = [
        r"[+-]\d+(?:\.\d+)?\s*dB",
        r"\bachieved\b",
        r"\boutperformed\b",
        r"\bstate-of-the-art performance\b",
    ]
    for pattern in forbidden:
        assert re.search(pattern, draft, flags=re.IGNORECASE) is None

    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    boundaries = contract["hard_boundaries"]
    assert boundaries == {
        "performance_values_read": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "full_data_benefit_claimed": False,
        "multi_seed_stability_claimed": False,
        "terminal_singularity_repair_claimed": False,
        "confirmation20_opened": False,
    }


def test_methods_draft_discloses_both_amtnc_numerical_recovery_stages() -> None:
    draft = (ROOT / "research" / "paper_aio" / "METHODS_PROTOCOL_DRAFT_EN.md").read_text(
        encoding="utf-8"
    )
    required = [
        "epoch 178",
        "epoch 194",
        "raw gradient square",
        "`exp_avg_sq`",
        "epoch-195 precision gate",
        "six such state promotions",
        "`sequential_method_only_recovery_v1`",
        "does not clip a gradient",
        "skip an update",
        "full-state reload preserves",
    ]
    for phrase in required:
        assert phrase in draft


def test_two_stage_disclosure_is_registered_in_current_paper_authorities() -> None:
    project = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    portfolio = json.loads(
        (ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json").read_text(
            encoding="utf-8"
        )
    )
    matrix = json.loads(
        (ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    records = [
        project["paper_aio_20260902"][
            "amtnc_two_stage_pre_result_disclosure_20260913"
        ],
        portfolio["amtnc_two_stage_pre_result_disclosure_20260913"],
        matrix["amtnc_two_stage_pre_result_disclosure_20260913"],
    ]
    expected = {
        "methods_draft_sha256": _sha256(
            ROOT / "research" / "paper_aio" / "METHODS_PROTOCOL_DRAFT_EN.md"
        ),
        "reproducibility_checklist_sha256": _sha256(
            ROOT / "research" / "paper_aio" / "REPRODUCIBILITY_CHECKLIST_EN.md"
        ),
        "limitations_statement_sha256": _sha256(
            ROOT / "research" / "paper_aio" / "LIMITATIONS_PRE_RESULT_EN.md"
        ),
        "manuscript_branching_contract_sha256": _sha256(
            ROOT / "configs" / "PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json"
        ),
    }
    for record in records:
        assert record["recovery_chain_version"] == "sequential_method_only_recovery_v1"
        assert record["recovery_stage_epochs"] == [178, 194]
        for key, digest in expected.items():
            assert record[key] == digest
        assert record["training_queue_changed"] is False
        assert record["performance_values_read"] is False
        assert record["confirmation20_opened"] is False
