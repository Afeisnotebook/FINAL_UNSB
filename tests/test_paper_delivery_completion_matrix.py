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
    assert extensions["higher_resolution_inference"]["status"].startswith(
        "implemented_tested_fail_closed_"
    )
    assert extensions["higher_resolution_inference"]["critical_path"] is False
    assert (
        extensions["higher_resolution_inference"]["training_fingerprint_changed"]
        is False
    )
    assert extensions["higher_resolution_inference"]["confirmation20_opened"] is False
    assert extensions["confirmation20"]["status"].startswith("sealed_")
    assert extensions["confirmation20"]["open_count_allowed"] == 1
    assert extensions["confirmation20"]["same_session_failure_recovery_allowed"] is True
    assert extensions["confirmation20"]["post_completion_reopen_allowed"] is False
    assert value["scientific_boundaries"] == {
        "intermediate_performance_controls_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
        "cross_non_equivalent_runtime_delta": False,
        "dclgan_blocks_core_delivery": False,
        "confirmation_authorized": False,
    }


def test_v7_is_root_authority_and_v6_is_retained_as_history() -> None:
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

    project_paper = project["paper_aio_20260902"]
    project_run = project_paper["runs"]["4090A_amtnc"]
    portfolio_run = portfolio["methods"]["amtnc"]
    project_v6 = project_paper[
        "amtnc_staged_recovery_relation_and_v6_delivery_20260913"
    ]
    matrix_v6 = matrix[
        "amtnc_staged_recovery_relation_and_v6_delivery_20260913"
    ]
    project_v7 = project_paper["amtnc_v7_post_training_chain_rebind_20260913"]
    portfolio_v7 = portfolio["amtnc_v7_post_training_chain_rebind_20260913"]
    matrix_v7 = matrix["amtnc_v7_post_training_chain_rebind_20260913"]
    matrix_core = {row["id"]: row for row in matrix["core_completion_path"]}
    unified = matrix_core["dynamic_unified_evaluation"]
    disposition = matrix_core["algorithm_dispositions"]
    final_delivery = matrix_core["core_final_portfolio"]

    expected_v6_commit = "83d4b062d4c25da4575e6220a14c9bcbfb35a411"
    expected_v6_generations = [
        "V2_2B45F88",
        "V3_A738846",
        "V4_DE1DD82",
        "V5_9214838",
    ]
    expected_v7_commit = "da41652b7ff73bdd9e66db72fc72169bd08374f1"
    expected_v7_supervisors = [125400, 125401, 125402, 125403]
    expected_v7_children = [125436, 125440, 125437, 125444]
    expected_v7_health = 125502

    # V6 remains an immutable provenance record, but no longer owns the root
    # execution path after the current AM-TNC release-state rebind.
    assert project_run["replacement_generation"] == "STAGED_V6_83D4B06"
    assert portfolio_run["replacement_generation"] == "STAGED_V6_83D4B06"
    assert project_run["retired_replacement_generations"] == expected_v6_generations
    assert portfolio_run["retired_replacement_generations"] == expected_v6_generations
    assert project_v6["control_git_commit"] == expected_v6_commit
    assert matrix_v6["control_git_commit"] == expected_v6_commit
    assert project_run["replacement_control_commit"] == expected_v6_commit
    assert portfolio_run["replacement_control_commit"] == expected_v6_commit

    expected_epoch = project_v6["4090a_amtnc_latest_complete_epoch"]
    assert expected_epoch >= 195
    assert matrix_v6["amtnc_latest_complete_epoch"] == expected_epoch
    assert project_run["latest_complete_epoch"] == expected_epoch
    assert portfolio_run["completed_full_data_epoch"] == expected_epoch

    # The three current control entries and the root completion path must all
    # agree on V7.  This prevents a blank control agent from reviving V6.
    for current in (project_v7, portfolio_v7, matrix_v7):
        assert current["control_git_commit"] == expected_v7_commit
        assert current["supervisor_pids"] == expected_v7_supervisors
        assert current["child_pids"] == expected_v7_children
        assert current["health_watcher_pid"] == expected_v7_health
        assert current["old_outputs_deleted"] is False
        assert current["training_processes_changed"] is False
        assert current["performance_values_read"] is False
        assert current["confirmation20_opened"] is False

    assert unified["recovery_control_commit"] == expected_v7_commit
    assert unified["recovery_supervisor_pid"] == expected_v7_supervisors[0]
    assert unified["pid"] == expected_v7_children[0]
    assert unified["health_watcher_pid"] == expected_v7_health
    assert unified["output"].endswith("FINAL_UNSB_PAPER_UNIFIED_EVAL_V7_DA41652")

    assert disposition["amtnc_recovery_supervisor_pid"] == expected_v7_supervisors[1]
    assert disposition["stcgr_recovery_supervisor_pid"] == expected_v7_supervisors[2]
    assert disposition["amtnc_waiter_pid"] == expected_v7_children[1]
    assert disposition["stcgr_waiter_pid"] == expected_v7_children[2]
    assert "V6_83D4B06" in disposition["retired_waiter_versions"]

    assert final_delivery["recovery_supervisor_pid"] == expected_v7_supervisors[3]
    assert final_delivery["pid"] == expected_v7_children[3]
    assert final_delivery["health_watcher_pid"] == expected_v7_health
    assert final_delivery["output"].endswith(
        "FINAL_UNSB_PAPER_FINAL_DELIVERY_V7_DA41652"
    )
