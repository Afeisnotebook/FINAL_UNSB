from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path

import pytest

from operations.paper_aio_evaluation_chain_recovery_deploy import (
    STCGR_LANE,
    _validate_amtnc_release_boundary,
    _state_path,
    build_child_commands,
)


def _args(tmp_path: Path) -> Namespace:
    return Namespace(
        python=tmp_path / "runtime" / "python",
        repo=tmp_path / "repo",
        manifest=tmp_path / "repo" / "manifests" / "FULL_DATA_MANIFEST.csv",
        data_root=tmp_path / "data",
        train_view=tmp_path / "view",
        gpu_lock=tmp_path / "gpu.lock",
        import_root=tmp_path / "imports",
        first_wave_output=tmp_path / "first-wave-recovery",
        amtnc_release_state=tmp_path / "new-amtnc" / "SUPERVISOR_amtnc.json",
        amtnc_export_root=tmp_path / "new-amtnc" / "exports",
        amtnc_source_host="4090A_AMTNC_OVERFLOW_RECOVERY",
        plain_export_root=tmp_path / "plain" / "exports",
        plain_source_host="4090A",
        amtnc_evaluation_output=tmp_path / "amtnc-evaluation-recovery",
        stcgr_candidate_authority=tmp_path / "authority.json",
        stcgr_candidate_metadata_receipt=tmp_path / "metadata.json",
        stcgr_source_host="5090A",
        final_delivery_output=tmp_path / "final-delivery-recovery",
        gpu=0,
        poll_seconds=60,
        timeout_hours=720,
        role=None,
    )


def test_recovery_commands_bind_new_amtnc_paths_and_new_cohort(tmp_path: Path):
    args = _args(tmp_path)
    commands = build_child_commands(args, "a" * 40)

    unified = commands["unified_evaluation"]
    assert str(args.amtnc_release_state.resolve()) in unified
    assert str(args.first_wave_output.resolve()) in unified

    amtnc = commands["amtnc_evaluation"]
    assert "--evaluation-mode" in amtnc
    assert "--mode" not in amtnc
    assert str(args.amtnc_export_root.resolve()) in amtnc
    assert "4090A_AMTNC_OVERFLOW_RECOVERY" in amtnc
    assert str(args.plain_export_root.resolve()) in amtnc

    stcgr = commands["stcgr_evaluation"]
    assert "--evaluation-mode" in stcgr
    assert "--mode" not in stcgr
    assert STCGR_LANE in stcgr
    assert str(
        args.first_wave_output.resolve() / "gates" / "UNIFIED_EVALUATION_COHORT.json"
    ) in stcgr

    final = commands["final_delivery"]
    assert str(_state_path("unified_evaluation", unified)) in final
    assert str(_state_path("amtnc_evaluation", amtnc)) in final
    assert str(_state_path("stcgr_evaluation", stcgr)) in final
    assert str(args.amtnc_export_root.resolve()) in final


def test_recovery_uses_unique_outputs_not_retired_dynamic_output(tmp_path: Path):
    args = _args(tmp_path)
    commands = build_child_commands(args, "b" * 40)
    flattened = "\n".join(value for command in commands.values() for value in command)
    assert "FINAL_UNSB_PAPER_UNIFIED_EVAL_DYNAMIC_V1" not in flattened
    assert "FINAL_UNSB_PAPER_AIO_V1_6a68e2c/exports" not in flattened


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_completed_legacy_release_requires_run_state_and_source_bound_export(tmp_path: Path):
    args = _args(tmp_path)
    _write_json(
        args.amtnc_release_state,
        {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "COMPLETE_E200",
            "lane_id": "amtnc",
            "confirmation20_opened": False,
            "run_state": {
                "status": "COMPLETE_E200",
                "final_data_epoch": 200.0,
                "metadata": {
                    "paired_controller_access": False,
                    "confirmation20_opened": False,
                },
            },
        },
    )
    export_set = args.amtnc_export_root / "amtnc" / "EXPORT_SET.json"
    _write_json(
        export_set,
        {
            "schema": "final-unsb-paper-source-export-set-v1",
            "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
            "lane_id": "amtnc",
            "source_host_label": args.amtnc_source_host,
            "epochs": [100, 125, 150, 175, 200],
            "paired_metric_control": False,
            "performance_values_read": False,
            "confirmation20_opened": False,
        },
    )

    proof = _validate_amtnc_release_boundary(args)

    assert proof["proof"] == "completed_run_state_plus_source_bound_export"
    assert proof["paired_metric_control"] is False
    assert proof["export_set_sha256"]


def test_completed_legacy_release_fails_closed_without_export_proof(tmp_path: Path):
    args = _args(tmp_path)
    _write_json(
        args.amtnc_release_state,
        {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "COMPLETE_E200",
            "confirmation20_opened": False,
            "run_state": {
                "status": "COMPLETE_E200",
                "final_data_epoch": 200.0,
                "metadata": {
                    "paired_controller_access": False,
                    "confirmation20_opened": False,
                },
            },
        },
    )

    with pytest.raises(RuntimeError, match="source-bound export proof"):
        _validate_amtnc_release_boundary(args)


def test_live_release_still_requires_explicit_supervisor_root_invariant(tmp_path: Path):
    args = _args(tmp_path)
    _write_json(
        args.amtnc_release_state,
        {
            "schema": "final-unsb-paper-supervisor-v1",
            "status": "CHILD_RUNNING",
            "confirmation20_opened": False,
            "run_state": {"metadata": {"paired_controller_access": False}},
        },
    )

    with pytest.raises(RuntimeError, match="permits paired control"):
        _validate_amtnc_release_boundary(args)
