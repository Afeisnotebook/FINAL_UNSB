from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from operations.paper_aio_evaluation_chain_recovery_deploy import (
    STCGR_LANE,
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
    )


def test_recovery_commands_bind_new_amtnc_paths_and_new_cohort(tmp_path: Path):
    args = _args(tmp_path)
    commands = build_child_commands(args, "a" * 40)

    unified = commands["unified_evaluation"]
    assert str(args.amtnc_release_state.resolve()) in unified
    assert str(args.first_wave_output.resolve()) in unified

    amtnc = commands["amtnc_evaluation"]
    assert str(args.amtnc_export_root.resolve()) in amtnc
    assert "4090A_AMTNC_OVERFLOW_RECOVERY" in amtnc
    assert str(args.plain_export_root.resolve()) in amtnc

    stcgr = commands["stcgr_evaluation"]
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
