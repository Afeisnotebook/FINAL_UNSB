from pathlib import Path

from production import common


def test_four_lane_ids_and_hj_physical_window(tmp_path: Path):
    contract = common.load_json("configs/FOUR_LANES.json")
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
