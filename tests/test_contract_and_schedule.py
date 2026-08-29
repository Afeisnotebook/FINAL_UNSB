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
    assert "former four-server/four-lane execution plan" in contract["not_current_tasks"]
    assert "cross-host method-minus-plain comparison" in contract["not_current_tasks"]
    assert "full-data execution" in contract["not_current_tasks"]
