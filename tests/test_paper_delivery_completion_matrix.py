import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_delivery_matrix_keeps_dclgan_nonblocking_and_confirmation_sealed() -> None:
    value = json.loads(
        (ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    assert value["schema"] == "final-unsb-paper-delivery-completion-matrix-v1"
    core_ids = {row["id"] for row in value["core_completion_path"]}
    assert core_ids == {
        "full_training",
        "legal_matched_control",
        "dynamic_unified_evaluation",
        "algorithm_dispositions",
        "core_final_portfolio",
    }
    extensions = {row["id"]: row for row in value["nonblocking_extensions"]}
    assert extensions["dclgan"]["critical_path"] is False
    assert extensions["dclgan"]["implementation"].endswith(
        "paper_aio_dclgan_portfolio_addendum_successor.py"
    )
    assert extensions["confirmation20"]["status"] == "sealed"
    assert value["scientific_boundaries"] == {
        "intermediate_performance_controls_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
        "cross_non_equivalent_runtime_delta": False,
        "dclgan_blocks_core_delivery": False,
    }
