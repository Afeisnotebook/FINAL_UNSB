from pathlib import Path

from production import common


def test_suspended_four_lane_provenance_and_hj_physical_window(tmp_path: Path):
    contract = common.load_json("configs/FOUR_LANES.json")
    assert contract["status"] == "SUSPENDED_NOT_CURRENT"
    assert [lane["id"] for lane in contract["lanes"]] == [
        "P0_PLAIN", "P1_HJ_HANDOFF", "P2_HNEK", "P3_MACRO_MARGINAL"
    ]
    argv, lane = common.train_argv(
        lane_id="P1_HJ_HANDOFF", data_view=tmp_path,
        run_root=tmp_path / "runs", gpu_id=0, steps_per_epoch=8553,
    )
    assert lane["resolved_active_updates"] == [13685, 68424]
    assert argv[argv.index("--hj_search_start_step") + 1] == "13685"
    assert argv[argv.index("--hj_search_duration_steps") + 1] == "54739"


def test_macro_is_independent_domain_measure():
    _, lane = common.lane_record("P3_MACRO_MARGINAL")
    method = lane["method"]
    assert method["A_domain_and_B_domain_independent"] is True
    assert method["domain_label_visible_to_model"] is False


def test_active_local_route1_probe_clock_and_scope():
    contract = common.load_json("configs/LOCAL_ROUTE1_PROBES.json")
    assert contract["status"] == "ACTIVE_LOCAL_RESEARCH"
    assert contract["time_unit_for_scientific_decisions"] == "data_epoch"
    assert contract["local_view"]["target_updates_per_lane"] == 150 * 200
    assert [probe["id"] for probe in contract["anchor_probes"]] == [
        "P0_PLAIN_LONG",
        "P1_HJ_CONTINUOUS_LONG",
        "P2_HNEK_LONG",
        "P3_DT_LONG",
    ]
    assert "HJ finite handoff" in contract["not_current_tasks"]
    assert "former four-server/four-frozen-lane validation" in contract["not_current_tasks"]
    assert "unauthorized cross-host delta merging" in contract["not_current_tasks"]
    assert "full-data execution" in contract["not_current_tasks"]


def test_project_level_paper_override_is_explicit_and_bounded():
    project = common.load_json("PROJECT_CONTRACT.json")
    state = common.load_json("PROJECT_STATE.json")
    paper = common.load_json("configs/PAPER_AIO_UNPAIRED_V1.json")
    assert project["status"] == "ACTIVE_FULL_DATA_PAPER_AND_ALGORITHM_RECONSTRUCTION"
    assert project["paper_full_frozen"]["updates_per_lane"] == 8553 * 200
    assert paper["status"] == "ACTIVE_FULL_DATA_PAPER_RESEARCH"
    assert state["phase"] == "PAPER_AIO_MULTI_ALGORITHM_FULL_DATA_PORTFOLIO_RUNNING"
    portfolio = common.load_json("configs/FULL_DATA_METHOD_PORTFOLIO.json")
    assert portfolio["methods"]["proposal"]["status"] == "running"
    assert portfolio["methods"]["amtnc"]["status"] == "running"
    assert portfolio["methods"]["stcgr"]["status"] == "running"
    assert portfolio["controls_and_external"]["plain_5090A"].startswith(
        "paused_by_explicit_user_time_priority_at_e9"
    )
    assert (
        portfolio["methods"]["stcgr"]["matched_delta_status"]
        == "unavailable_until_5090B_e200_and_reviewed_registry_relation"
    )
    assert (
        portfolio["methods"]["proposal"]["matched_delta_status"]
        == "unavailable_until_5090B_e200_and_reviewed_registry_relation"
    )
    assert (
        portfolio["post_training_delivery"]["replacement_lane_sources"]["plain"]
        == "5090B_MATCHED_PLAIN"
    )
    plain_resume = state["paper_aio_20260902"]["runs"]["5090A_plain_resume_after_stcgr"]
    assert plain_resume["status"] == "CANCELED_BY_EXPLICIT_USER_STCGR_ONLY_PRIORITY"
    assert plain_resume["automatic_resume_authorized"] is False
    assert plain_resume["future_resume_requires_new_explicit_decision"] is True
    future_control = portfolio["future_matched_plain_successor"]
    assert future_control["stcgr_relation_status"] == (
        "review_only_two_link_proof_successor_armed_registry_unchanged"
    )
    relation = state["paper_aio_20260902"]["multi_control_runtime_relation_interface"]
    stcgr_relation = relation["stcgr_candidate_control_relation"]
    assert stcgr_relation["registry_edited"] is False
    assert stcgr_relation["comparison_authorized"] is False
    assert portfolio["methods"]["hjcgr"]["status"] == "deferred"
    assert portfolio["methods"]["hjcgr"]["mechanism_falsified"] is False
    authorization = project["authorization_required"]
    assert authorization["status"] == "GRANTED_FULL_DATA_PAPER_AND_ROUTE1_RECONSTRUCTION"
    assert "confirmation20 access" in authorization["excludes"]
    assert "cross-host method-minus-plain comparisons" in authorization["excludes"]
    assert "paired metric training or scheduling control" in authorization["excludes"]
    assert state["paper_aio_20260902"]["confirmation20_opened"] is False


def test_paper_baseline_tiers_cannot_silently_mix_protocols():
    baseline = common.load_json("configs/PAPER_BASELINE_PORTFOLIO.json")
    core = {row["id"]: row for row in baseline["core_controlled_main_table"]}
    assert set(core) == {
        "input", "cyclegan", "cut", "plain_unsb", "proposal_only", "stcgr", "amtnc"
    }
    assert core["cyclegan"]["paper_label"] == (
        "CycleGAN (official-loss, controlled shared backbone)"
    )
    assert "not a matched delta" in core["cut"]["comparison_rule"]
    assert "withheld" in core["proposal_only"]["comparison_rule"]
    assert "withheld" in core["stcgr"]["comparison_rule"]

    external = {row["id"]: row for row in baseline["direct_external_extensions"]}
    assert external["ddsb"]["status"] == "reproduction_incomplete_fail_closed"
    assert external["ddsb"]["main_table_number_allowed"] is False
    assert external["negcut"]["status"] == "deferred_engineering_and_license_not_falsified"

    contextual = {row["id"]: row for row in baseline["domain_specific_context"]}
    assert "never impute missing domains" in contextual["dehazesb"]["reporting_rule"]
    ceilings = {row["id"]: row for row in baseline["paired_ceiling_block"]}
    assert set(ceilings) == {"restorevar", "promptir"}
    assert all("not an unpaired competitor" in row["role"] for row in ceilings.values())
    assert all("no delta" in row["reporting_rule"] for row in ceilings.values())

    hard = baseline["hard_reporting_rules"]
    assert hard["main_table_checkpoint"] == "e200_only"
    assert hard["best_checkpoint_selection"] is False
    assert hard["partial_domain_result_used_as_six_domain_macro"] is False
    assert hard["paired_method_called_unpaired_competitor"] is False
    assert hard["confirmation20_opened"] is False
    assert baseline["priority_and_scheduling"]["current_gpu_queue_changed"] is False
