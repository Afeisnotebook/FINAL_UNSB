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


def test_active_amtnc_recovery_authority_is_consistent_across_control_entries() -> None:
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
    project_relation = project_paper[
        "amtnc_recovery_relation_and_v5_delivery_20260913"
    ]
    portfolio_relation = portfolio[
        "amtnc_recovery_relation_and_v5_delivery_20260913"
    ]
    matrix_relation = matrix["amtnc_recovery_relation_and_v5_delivery_20260913"]
    matrix_core = {row["id"]: row for row in matrix["core_completion_path"]}
    disposition = matrix_core["algorithm_dispositions"]
    final_delivery = matrix_core["core_final_portfolio"]

    expected_control_commit = "9214838dec1b1e0ab82ded943c21d247d7ab62a8"
    expected_retired = ["V2_2B45F88", "V3_A738846", "V4_DE1DD82"]
    expected_pids = {
        "evaluation_supervisor": project_relation[
            "v5_amtnc_evaluation_supervisor_pid"
        ],
        "evaluation_child": project_relation["v5_amtnc_evaluation_child_pid"],
        "final_supervisor": project_relation["v5_final_delivery_supervisor_pid"],
        "final_child": project_relation["v5_final_delivery_child_pid"],
        "health": project_relation["v5_health_watcher_pid"],
    }

    assert project_run["replacement_generation"] == "V5_9214838"
    assert portfolio_run["replacement_generation"] == "V5_9214838"
    assert project_run["retired_replacement_generations"] == expected_retired
    assert portfolio_run["retired_replacement_generations"] == expected_retired
    assert disposition["retired_waiter_versions"] == expected_retired

    assert project_relation["control_git_commit"] == expected_control_commit
    assert portfolio_relation["control_git_commit"] == expected_control_commit
    assert matrix_relation["control_git_commit"] == expected_control_commit

    assert project_run["replacement_evaluation_supervisor_pid"] == expected_pids[
        "evaluation_supervisor"
    ]
    assert portfolio_run["replacement_evaluation_supervisor_pid"] == expected_pids[
        "evaluation_supervisor"
    ]
    assert matrix_relation["amtnc_evaluation_supervisor_pid"] == expected_pids[
        "evaluation_supervisor"
    ]
    assert disposition["amtnc_recovery_supervisor_pid"] == expected_pids[
        "evaluation_supervisor"
    ]

    assert project_run["replacement_evaluation_child_pid"] == expected_pids[
        "evaluation_child"
    ]
    assert portfolio_run["replacement_evaluation_child_pid"] == expected_pids[
        "evaluation_child"
    ]
    assert matrix_relation["amtnc_evaluation_child_pid"] == expected_pids[
        "evaluation_child"
    ]
    assert disposition["amtnc_waiter_pid"] == expected_pids["evaluation_child"]

    assert project_run["replacement_final_delivery_supervisor_pid"] == expected_pids[
        "final_supervisor"
    ]
    assert portfolio_run["replacement_final_delivery_supervisor_pid"] == expected_pids[
        "final_supervisor"
    ]
    assert matrix_relation["final_delivery_supervisor_pid"] == expected_pids[
        "final_supervisor"
    ]
    assert final_delivery["recovery_supervisor_pid"] == expected_pids[
        "final_supervisor"
    ]

    assert project_run["replacement_final_delivery_child_pid"] == expected_pids[
        "final_child"
    ]
    assert portfolio_run["replacement_final_delivery_child_pid"] == expected_pids[
        "final_child"
    ]
    assert matrix_relation["final_delivery_child_pid"] == expected_pids["final_child"]
    assert final_delivery["pid"] == expected_pids["final_child"]

    assert project_run["replacement_evaluation_health_watcher_pid"] == expected_pids[
        "health"
    ]
    assert portfolio_run["replacement_evaluation_health_watcher_pid"] == expected_pids[
        "health"
    ]
    assert matrix_relation["health_watcher_pid"] == expected_pids["health"]
    assert final_delivery["health_watcher_pid"] == expected_pids["health"]

    expected_epoch = project_relation["4090a_amtnc_latest_complete_epoch"]
    assert expected_epoch >= 182
    assert portfolio_relation["latest_complete_epoch"] == expected_epoch
    assert matrix_relation["amtnc_latest_complete_epoch"] == expected_epoch
    assert project_run["latest_observed_data_epoch"] == expected_epoch
    assert portfolio_run["completed_full_data_epoch"] == expected_epoch
