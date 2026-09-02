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
    assert portfolio["methods"]["amtnc"]["status"] == "queued"
    assert portfolio["methods"]["stcgr"]["status"] == "running"
    assert portfolio["controls_and_external"]["plain_5090A"].startswith(
        "paused_by_explicit_user_time_priority_at_e9"
    )
    assert (
        portfolio["methods"]["stcgr"]["matched_delta_status"]
        == "deferred_until_plain_resumes_and_completes_e200"
    )
    assert portfolio["methods"]["hjcgr"]["status"] == "deferred"
    assert portfolio["methods"]["hjcgr"]["mechanism_falsified"] is False
    authorization = project["authorization_required"]
    assert authorization["status"] == "GRANTED_FULL_DATA_PAPER_AND_ROUTE1_RECONSTRUCTION"
    assert "confirmation20 access" in authorization["excludes"]
    assert "cross-host method-minus-plain comparisons" in authorization["excludes"]
    assert "paired metric training or scheduling control" in authorization["excludes"]
    assert state["paper_aio_20260902"]["confirmation20_opened"] is False
