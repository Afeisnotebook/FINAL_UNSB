from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from research.local_route1 import causal_audit
from research.local_route1.candidates import (
    CARD_SCHEMA,
    GATE_SCHEMA,
    IMPLEMENTATION_SCHEMA,
    freeze_candidate_derivation,
    load_candidate_registration,
    validate_candidate_id,
)
from research.local_route1.candidate_gate import _validate_gate_report
from research.local_route1.candidate_runner import (
    late_rolling_drawdown,
    plain_collapse_adjudication,
)
from research.local_route1.causal_audit import (
    AuditCell,
    BranchResult,
    _audit_regimes,
    _initialize_operator_costate,
    _operator_modes,
    _restore_terminal_base_lrs,
    append_unique_rows,
    build_causal_matrix,
    target_blind_signal_screen,
)
from research.local_route1.evaluate import select_discovery70
from research.local_route1.interfaces import CounterfactualAuditor, StateObservation
from research.local_route1.lineage import (
    HISTORICAL_DT_SEMANTIC_HASHES,
    split_lineage_documents,
)
from research.local_route1.protocol import (
    ROOT,
    dt_lambda_for_physical_epoch,
    epoch_to_step,
    load_protocol,
    milestone_steps,
    probe_spec,
    protocol_fingerprint,
    semantic_source_sha256,
    step_to_physical_epoch,
    validate_protocol,
)
from research.local_route1.runtime import file_sha256, full_state_hash, read_manifest
from research.local_route1.seed_validation import (
    MULTI_SEED_ADJUDICATION_SCHEMA,
    SEED_FREEZE_SCHEMA,
    SEED_SUMMARY_SCHEMA,
    _crn_fingerprint,
    _e0_identity,
    _seed_late_rolling_drawdown,
    _seed_plain_collapse_adjudication,
    summarize_multi_seed_validation,
    validate_seed_freeze,
)
from research.local_route1.stages import derive_from_completed_atlas
from research.local_route1.stages import prepare_audit_queue
from research.local_route1.observations import (
    component_directional_derivatives,
    state_dict_delta_cosine,
    state_dict_update_geometry,
)


def test_route1_protocol_uses_physical_epoch_and_cli_aliases():
    protocol = load_protocol()
    assert validate_protocol(protocol) == []
    assert epoch_to_step(200, protocol) == 30_000
    assert step_to_physical_epoch(0, protocol) == 1
    assert step_to_physical_epoch(149, protocol) == 1
    assert step_to_physical_epoch(150, protocol) == 2
    assert milestone_steps(protocol)[-3:] == [22_500, 26_250, 30_000]
    assert probe_spec("hj", protocol).contract_id == "P1_HJ_CONTINUOUS_LONG"
    manifest = ROOT / protocol["manifest"]["repo_path"]
    assert protocol_fingerprint(manifest) == protocol_fingerprint(
        manifest, source_root=ROOT,
    )


def test_lineage_split_emits_every_required_probe_and_mechanism_object_map():
    payload = {
        "git_commit": "commit",
        "protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "baseline": {"scientific_clock": "data_epoch"},
        "historical_to_clean_semantics": {"shared": [], "must_not_be_conflated": []},
        "probes": {probe: {"physical_protocol": probe} for probe in ("dt", "hj", "hnek")},
        "unsb_object_graph": {
            "native_objects": ["G", "D", "E", "F"],
            "dt": ["covariance"],
            "hj": ["PatchNCE risk"],
            "hnek": ["bridge coordinate"],
            "later_mechanisms": {"LBST": {"unsb_object": "rollout distribution"}},
        },
    }
    documents = split_lineage_documents(payload)
    assert set(documents) == {
        "DT_LINEAGE.json", "HJ_LINEAGE.json", "HNEK_LINEAGE.json",
        "MECHANISM_OBJECT_MAP.json",
    }
    assert documents["DT_LINEAGE.json"]["probe"] == "dt"
    assert documents["HJ_LINEAGE.json"]["unsb_objects"] == ["PatchNCE risk"]
    object_map = documents["MECHANISM_OBJECT_MAP.json"]
    assert object_map["later_mechanisms"]["LBST"]["unsb_object"] == "rollout distribution"
    assert object_map["confirmation20_opened"] is False


def test_candidate_late_stability_and_plain_collapse_are_hard_gates():
    stable = [
        {"epoch": 100, "macro_psnr": 18.0, "plain_macro_psnr": 17.9},
        {"epoch": 125, "macro_psnr": 18.1, "plain_macro_psnr": 18.0},
        {"epoch": 150, "macro_psnr": 18.2, "plain_macro_psnr": 18.0},
        {"epoch": 175, "macro_psnr": 18.25, "plain_macro_psnr": 18.05},
        {"epoch": 200, "macro_psnr": 18.3, "plain_macro_psnr": 18.1},
    ]
    assert late_rolling_drawdown(stable) == pytest.approx(0.0)
    assert plain_collapse_adjudication(stable[-3:])["status"] == "PASS_NOT_PLAIN_COLLAPSE"

    collapsing = [
        {"epoch": 150, "macro_psnr": 18.0, "plain_macro_psnr": 18.0},
        {"epoch": 175, "macro_psnr": 17.8, "plain_macro_psnr": 17.6},
        {"epoch": 200, "macro_psnr": 17.6, "plain_macro_psnr": 17.2},
    ]
    assert plain_collapse_adjudication(collapsing)["status"].startswith("FAIL_")
    peaked = stable[:-1] + [
        {"epoch": 200, "macro_psnr": 16.9, "plain_macro_psnr": 18.1},
    ]
    assert late_rolling_drawdown(peaked) > 0.3


def test_dt_core_semantics_and_physical_schedule_are_frozen():
    for name in ("__init__.py", "dtcovmatch.py"):
        path = ROOT / "src" / "models" / "dtcov" / name
        assert semantic_source_sha256(path) == HISTORICAL_DT_SEMANTIC_HASHES[name]
    values = {epoch: dt_lambda_for_physical_epoch(epoch) for epoch in (20, 21, 25, 35, 44, 45, 200)}
    assert values[20] == 0.0
    assert values[21] == pytest.approx(0.0002)
    assert values[25] == pytest.approx(0.001)
    assert values[35] == pytest.approx(0.001)
    assert values[44] > 0.0
    assert values[45] == 0.0
    assert values[200] == 0.0


def test_hj_and_dt_options_do_not_use_total_step_relative_windows():
    hj = probe_spec("hj").method
    dt = probe_spec("dt").method
    assert hj["hj_start_epoch"] == 5
    assert hj["hj_search_start_step"] == -1
    assert hj["hj_search_duration_steps"] == 0
    assert dt["dtcov_search_start_step"] == -1
    assert dt["dtcov_search_duration_steps"] == 0


def test_discovery70_selector_never_reaches_confirmation20():
    rows = read_manifest(ROOT / "manifests" / "frozen" / "legacy_split_manifest.csv")
    selected = select_discovery70(rows)
    assert len(selected) == 420
    assert {row["split"] for row in selected} == {"discovery"}
    with pytest.raises(RuntimeError, match="frozen to discovery70"):
        select_discovery70(rows, count_per_domain=80)


def test_observable_rejects_paired_and_heldout_metric_fields():
    StateObservation(
        step=1, physical_epoch=1.0,
        bridge={"endpoint_dispersion": 0.2},
    ).validate_target_blind()
    for field in ("paired_psnr", "discovery_ssim", "confirmation_lpips"):
        with pytest.raises(ValueError):
            StateObservation(
                step=1, physical_epoch=1.0,
                method_internal={field: 1.0},
            ).validate_target_blind()


def test_counterfactual_auditor_preserves_parent_hash():
    parent = {"tensor": torch.tensor([1.0]), "nested": {"value": 2}}
    expected = full_state_hash(parent)
    result, observed = CounterfactualAuditor().run(
        parent, lambda branch: branch["tensor"].mul_(4.0).item()
    )
    assert result == 4.0
    assert observed == expected == full_state_hash(parent)


def test_actual_update_geometry_and_native_component_derivative():
    before = {"block.weight": torch.tensor([0.0, 0.0])}
    native = {"block.weight": torch.tensor([1.0, 0.0])}
    proposal = {"block.weight": torch.tensor([1.0, 2.0])}
    geometry, blocks = state_dict_update_geometry(before, native, proposal)
    assert geometry["correction_norm"] == pytest.approx(2.0)
    assert geometry["correction_reference_cosine"] == pytest.approx(0.0)
    assert blocks["block"] == geometry
    consensus = state_dict_delta_cosine(
        native, proposal, native, {"block.weight": torch.tensor([1.0, -3.0])}
    )
    assert consensus["cosine"] == pytest.approx(-1.0)
    derivatives = component_directional_derivatives(
        before=before, reference_after=native, proposal_after=proposal,
        native_component_gradients={"SB": {"block.weight": torch.tensor([0.0, -4.0])}},
    )
    assert derivatives["SB"]["gradient_dot_correction"] == pytest.approx(-8.0)


def test_cross_state_operator_costate_is_not_transplanted_from_method_lane():
    class FakeModel:
        def __init__(self):
            self.loaded = None

        def load_extra_training_state(self, state):
            self.loaded = state

    source = {
        "step": 15_000,
        "target_steps": 30_000,
        "model": {"method": {"hj_controller": {"_hj_conflict_ema": 0.7}}},
    }
    model = FakeModel()
    policy = _initialize_operator_costate(
        model, target_probe="hj", source_label="plain", source=source,
    )
    assert policy == "reinitialized_from_source_state"
    assert model.loaded == {"search_global_step": 15_000, "search_total_steps": 30_000}
    policy = _initialize_operator_costate(
        model, target_probe="hj", source_label="hj", source=source,
    )
    assert policy == "matched_historical_costate"
    assert model.loaded == source["model"]["method"]


def test_dt_registered_and_forced_active_diagnostics_are_separate():
    assert _operator_modes("dt", 20) == ("registered", "forced_active_diagnostic")
    assert _operator_modes("dt", 40) == ("registered",)
    assert _operator_modes("dt", 45) == ("registered",)
    assert _operator_modes("dt", 100) == ("registered", "forced_active_diagnostic")
    assert _operator_modes("hj", 200) == ("registered",)


def test_terminal_audit_is_local_and_never_fabricates_future_e200_label():
    registered = _audit_regimes((1, 8, 32, 200), start_step=29_999)
    terminal = _audit_regimes((1, 8, 32, 200), start_step=30_000)
    assert any(horizon == 200 for _regime, horizon, _pulse in registered)
    assert {horizon for _regime, horizon, _pulse in terminal} == {1, 8, 32}
    assert not any(horizon == 200 for _regime, horizon, _pulse in terminal)


def test_terminal_next_native_consensus_allows_internal_h2_without_public_row():
    from research.local_route1.causal_audit import TERMINAL_INTERNAL_HORIZONS

    assert 2 in TERMINAL_INTERNAL_HORIZONS
    assert _audit_regimes((2,), start_step=30_000) == ()


def test_terminal_lr_restore_preserves_moments_and_scheduler_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.Adam([parameter], lr=0.0)
    optimizer.state[parameter]["exp_avg"] = torch.tensor([0.25])
    optimizer.state[parameter]["exp_avg_sq"] = torch.tensor([0.5])
    scheduler = SimpleNamespace(base_lrs=[0.0001], last_epoch=200)
    model = SimpleNamespace(optimizers=[optimizer], schedulers=[scheduler])
    before = {
        name: value.clone() if torch.is_tensor(value) else value
        for name, value in optimizer.state[parameter].items()
    }
    restored = _restore_terminal_base_lrs(model)
    assert restored == (0.0001,)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0001)
    assert scheduler.last_epoch == 200
    for name, value in before.items():
        assert torch.equal(optimizer.state[parameter][name], value)


def test_audit_queue_uses_e175_label_and_terminal_e200_vector_field(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    trajectory = [
        {"epoch": epoch, "macro_psnr_delta": value}
        for epoch, value in ((20, 0.1), (100, 0.2), (150, -0.1), (175, -0.2), (200, -0.3))
    ]
    (evidence / "ANCHOR_TRAJECTORIES.json").write_text(
        __import__("json").dumps({"summaries": [{"probe_id": "hj", "trajectory": trajectory}]}),
        encoding="utf-8",
    )
    for lane in ("plain", "hj"):
        milestone = tmp_path / "anchors" / lane / "milestones"
        milestone.mkdir(parents=True)
        for epoch in (20, 100, 150, 175, 200):
            (milestone / f"e{epoch:03d}.pt").write_bytes(b"checkpoint")
    queue = prepare_audit_queue(tmp_path)
    assert queue["status"] == "READY"
    jobs = {row["data_epoch"]: row for row in queue["jobs"]}
    assert 200 in jobs[175]["branch_horizons_updates"]
    assert jobs[175]["branch_semantics"] == "registered_training_continuation"
    assert jobs[200]["branch_horizons_updates"] == [1, 8, 32]
    assert jobs[200]["branch_semantics"] == "terminal_base_lr_vector_field_no_future_label"


def test_audit_queue_brackets_first_reversal_and_maximum_drawdown(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    trajectory = [
        {"epoch": epoch, "macro_psnr_delta": value}
        for epoch, value in (
            (20, 0.1), (40, 0.5), (60, -0.1), (80, -0.4),
            (100, 0.2), (150, 0.1), (175, 0.0), (200, 0.05),
        )
    ]
    (evidence / "ANCHOR_TRAJECTORIES.json").write_text(
        __import__("json").dumps({"summaries": [{"probe_id": "hj", "trajectory": trajectory}]}),
        encoding="utf-8",
    )
    for lane in ("plain", "hj"):
        milestone = tmp_path / "anchors" / lane / "milestones"
        milestone.mkdir(parents=True)
        for epoch, _value in (
            (20, 0.1), (40, 0.5), (60, -0.1), (80, -0.4),
            (100, 0.2), (150, 0.1), (175, 0.0), (200, 0.05),
        ):
            (milestone / f"e{epoch:03d}.pt").write_bytes(b"checkpoint")

    queue = prepare_audit_queue(tmp_path)
    jobs = {row["data_epoch"]: row for row in queue["jobs"]}
    assert "first_sign_reversal_left" in jobs[40]["selection_reasons"]
    assert "first_sign_reversal_right" in jobs[60]["selection_reasons"]
    assert "first_sign_reversal_after" in jobs[80]["selection_reasons"]
    assert "maximum_benefit" in jobs[40]["selection_reasons"]
    assert "maximum_drawdown_peak" in jobs[40]["selection_reasons"]
    assert "maximum_drawdown_trough" in jobs[80]["selection_reasons"]


def test_audit_queue_ignores_warmup_crossing_before_positive_to_negative_reversal(
    tmp_path,
):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    trajectory = [
        {"epoch": epoch, "macro_psnr_delta": value}
        for epoch, value in (
            (5, -0.1), (10, 0.2), (20, 0.1), (40, 0.5),
            (60, -0.4), (80, 0.3), (100, 0.2), (150, 0.1),
            (175, 0.1), (200, 0.1),
        )
    ]
    (evidence / "ANCHOR_TRAJECTORIES.json").write_text(
        __import__("json").dumps({
            "summaries": [{"probe_id": "hj", "trajectory": trajectory}]
        }),
        encoding="utf-8",
    )
    for lane in ("plain", "hj"):
        milestone = tmp_path / "anchors" / lane / "milestones"
        milestone.mkdir(parents=True)
        for row in trajectory:
            (milestone / f"e{row['epoch']:03d}.pt").write_bytes(b"checkpoint")

    queue = prepare_audit_queue(tmp_path)
    jobs = {row["data_epoch"]: row for row in queue["jobs"]}
    assert "first_positive_onset_left" in jobs[5]["selection_reasons"]
    assert "first_positive_onset_right" in jobs[10]["selection_reasons"]
    assert "first_sign_reversal_left" not in jobs[5]["selection_reasons"]
    assert "first_sign_reversal_right" not in jobs[10]["selection_reasons"]
    assert "first_sign_reversal_left" in jobs[40]["selection_reasons"]
    assert "first_sign_reversal_right" in jobs[60]["selection_reasons"]
    assert "first_sign_reversal_after" in jobs[80]["selection_reasons"]


def test_uncalibrated_proxy_excludes_unrun_dt_without_calling_it_missing(tmp_path):
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    trajectory = [
        {"epoch": epoch, "macro_psnr_delta": -0.1}
        for epoch in (20, 100, 150, 175, 200)
    ]
    (evidence / "ANCHOR_TRAJECTORIES.json").write_text(
        __import__("json").dumps({
            "summaries": [
                {"probe_id": "hj", "trajectory": trajectory, "complete_e200": True},
                {"probe_id": "hnek", "trajectory": trajectory, "complete_e200": True},
                {"probe_id": "dt", "trajectory": [], "complete_e200": False},
            ]
        }),
        encoding="utf-8",
    )
    (evidence / "PROXY_CALIBRATION.json").write_text(
        __import__("json").dumps({"status": "NOT_CALIBRATED_PAUSE"}),
        encoding="utf-8",
    )
    for lane in ("plain", "hj", "hnek"):
        milestone = tmp_path / "anchors" / lane / "milestones"
        milestone.mkdir(parents=True)
        for epoch in (20, 100, 150, 175, 200):
            (milestone / f"e{epoch:03d}.pt").write_bytes(b"checkpoint")

    queue = prepare_audit_queue(tmp_path)
    assert queue["status"] == "READY"
    assert {row["probe"] for row in queue["jobs"]} == {"hj", "hnek"}
    assert queue["missing_or_replay"] == []
    assert queue["excluded_probes"] == [{
        "probe": "dt",
        "reason": "excluded_by_proxy_gate",
        "proxy_status": "NOT_CALIBRATED_PAUSE",
        "scientific_interpretation": (
            "DT was deliberately not run; this is not a missing checkpoint or "
            "mechanism falsification."
        ),
    }]


def test_sampling_variance_uses_actual_correction_fields(monkeypatch, tmp_path):
    before = {"block.weight": torch.tensor([0.0, 0.0])}

    def fake_branch(**kwargs):
        offset = int(kwargs.get("batch_skip", 0))
        native_after = {"block.weight": torch.tensor([1.0, 0.0])}
        if kwargs["target_probe"] == "plain":
            after = native_after
        else:
            correction = 1.0 if offset == 0 else -1.0
            after = {"block.weight": torch.tensor([1.0 + correction, 0.0])}
        observation = StateObservation(
            step=1500, physical_epoch=10.0,
            sampling={f"domain_count::d{offset}": 1.0, "time_count::0": 1.0},
        )
        return BranchResult(
            observation=observation,
            before_g={key: value.clone() for key, value in before.items()},
            after_g=after,
            component_gradients={}, metrics=None,
            scientific_state_after=f"state-{offset}-{kwargs['target_probe']}",
        ), "reinitialized_from_source_state"

    monkeypatch.setattr(causal_audit, "_run_branch", fake_branch)
    cell = AuditCell(
        probe="hj", data_epoch=10,
        plain_state=tmp_path / "plain.pt", method_state=tmp_path / "hj.pt",
    )
    parent = {"step": 1500, "model": {"method": {}}, "samplers": {}, "rng": {}}
    row = causal_audit._sampling_variance_row(
        cell=cell, parent=parent, source_label="plain", operator_mode="registered",
        axis="independent_unpaired_batch", replicates=2, rows=[],
        train_view=tmp_path, work_dir=tmp_path, seed=2026, gpu=0,
        pair_identity={}, identity={},
    )
    assert row["mean_correction_norm"] == pytest.approx(0.0)
    assert row["correction_variance_fraction"] == pytest.approx(1.0)
    assert row["same_batch_native_cosine_mean"] == pytest.approx(0.0)
    assert row["next_independent_batch_native_cosine_mean"] == pytest.approx(0.0)
    assert row["parent_state_sha256_before"] == row["parent_state_sha256_after"]


def test_dt_preferred_operator_mode_preserves_registered_active_age():
    assert causal_audit._preferred_operator_mode("hj", 20) == "registered"
    assert causal_audit._preferred_operator_mode("hnek", 200) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 19) == "forced_active_diagnostic"
    assert causal_audit._preferred_operator_mode("dt", 20) == "forced_active_diagnostic"
    assert causal_audit._preferred_operator_mode("dt", 21) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 40) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 44) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 45) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 46) == "forced_active_diagnostic"
    assert causal_audit._operator_modes("dt", 20) == (
        "registered", "forced_active_diagnostic",
    )
    assert causal_audit._operator_modes("dt", 45) == ("registered",)


def test_causal_matrix_refuses_to_rank_until_every_registered_row_exists(tmp_path):
    audit = tmp_path / "audit"
    audit.mkdir()
    queue = {
        "jobs": [{"probe": "hj", "data_epoch": 100}],
        "confirmation20_opened": False,
    }
    (audit / "AUDIT_QUEUE.json").write_text(__import__("json").dumps(queue), encoding="utf-8")
    rows = []
    for source in ("plain", "hj"):
        for regime, horizon, intervention_steps in _audit_regimes((1, 8, 32, 200)):
            rows.append({
                "schema": "final-unsb-local-route1-reversal-atlas-row-v1",
                "row_id": f"{source}-{regime}-{horizon}",
                "probe": "hj",
                "data_epoch": 100,
                "source_state": source,
                "operator_mode": "registered",
                "branch_regime": regime,
                "intervention_steps": intervention_steps,
                "horizon": horizon,
                "update_geometry": {
                    "correction_norm": 0.5,
                    "reference_norm": 1.0,
                    "correction_reference_cosine": -0.4,
                },
                "next_independent_native_consensus": (
                    {"cosine": -0.4}
                    if horizon == 1 and regime == "continuous_intervention" else None
                ),
                "reference_observation": {
                    "bridge": {"rollout_velocity_l2": 1.0},
                },
                "proposal_observation": {
                    "bridge": {
                        "rollout_velocity_l2": 0.5 if source == "plain" else 2.0,
                    },
                },
                "post_branch_development_label": (
                    {
                        "macro_psnr_delta": 0.2 if source == "plain" else -0.1,
                        "domain_psnr_delta": {
                            f"d{index}": 0.2 if source == "plain" else -0.1
                            for index in range(6)
                        },
                    }
                    if horizon == 200 else None
                ),
            })
    append_unique_rows(audit / "LONG_REVERSAL_ATLAS.jsonl", rows[:-1])
    partial = build_causal_matrix(tmp_path)
    assert partial["status"] == "PARTIAL_CAUSAL_AUDIT"
    assert partial["ranked_failure_mechanisms"] == []
    append_unique_rows(audit / "LONG_REVERSAL_ATLAS.jsonl", rows[-1:])
    variance_rows = []
    for source in ("plain", "hj"):
        for axis in ("independent_unpaired_batch", "latent_time_bridge_rng"):
            variance_rows.append({
                "schema": "final-unsb-local-route1-sampling-variance-row-v1",
                "row_id": f"variance-{source}-{axis}",
                "probe": "hj",
                "data_epoch": 100,
                "source_state": source,
                "operator_mode": "registered",
                "axis": axis,
                "replicates": 8,
                "correction_variance_fraction": 0.8,
                "mean_correction_norm": 0.2,
                "bridge_time_summary": (
                    {
                        "0": {"n": 2.0, "correction_norm_mean": 0.01},
                        "1": {"n": 2.0, "correction_norm_mean": 0.01},
                        "2": {"n": 2.0, "correction_norm_mean": 1.0},
                    }
                    if axis == "latent_time_bridge_rng" else {}
                ),
            })
    append_unique_rows(audit / "SAMPLING_VARIANCE_ATLAS.jsonl", variance_rows)
    complete = build_causal_matrix(tmp_path)
    assert complete["status"] == "COMPLETE_CAUSAL_AUDIT"
    failure_types = {row["failure_type"] for row in complete["ranked_failure_mechanisms"]}
    assert "correction_sign_reversal" in failure_types
    assert "state_feedback_missing" in failure_types
    assert "sampling_variance" in failure_types
    assert "rollout_distribution_speed" in failure_types
    assert "coordinate_horizon_imbalance" in failure_types


def test_parallel_audit_row_merges_are_lossless_and_canonical(tmp_path):
    path = tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl"

    def write_group(group: int) -> None:
        append_unique_rows(path, [
            {
                "row_id": f"row-{group:02d}-{index:02d}",
                "probe": "hj",
                "data_epoch": group * 10 + index,
                "source_state": "plain",
                "operator_mode": "registered",
                "branch_regime": "continuous_intervention",
                "horizon": 1,
            }
            for index in range(8)
        ])

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write_group, range(8)))
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 64
    assert len({row["row_id"] for row in rows}) == 64
    assert [row["data_epoch"] for row in rows] == sorted(
        row["data_epoch"] for row in rows
    )


def test_cross_process_audit_row_merges_do_not_drop_a_worker(tmp_path):
    path = tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    code = """
import json
import sys
from pathlib import Path
from research.local_route1.causal_audit import append_unique_rows

path = Path(sys.argv[1])
group = int(sys.argv[2])
append_unique_rows(path, [{
    'row_id': f'process-{group}-{index}',
    'probe': 'hnek',
    'data_epoch': group * 20 + index,
    'source_state': 'plain',
    'operator_mode': 'registered',
    'branch_regime': 'continuous_intervention',
    'horizon': 1,
} for index in range(12)])
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", code, str(path), str(group)],
            cwd=ROOT,
        )
        for group in range(2)
    ]
    assert [process.wait(timeout=60) for process in processes] == [0, 0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24
    assert len({row["row_id"] for row in rows}) == 24


def test_target_blind_signal_screen_uses_offline_labels_without_fitting_thresholds():
    rows = []
    for probe in ("dt", "hj", "hnek"):
        for epoch, score in ((20, 1.0), (100, -1.0)):
            mode = causal_audit._preferred_operator_mode(probe, epoch)
            common = {
                "probe": probe,
                "data_epoch": epoch,
                "source_state": "plain",
                "operator_mode": mode,
                "branch_regime": "continuous_intervention",
            }
            rows.append({
                **common,
                "horizon": 1,
                "update_geometry": {
                    "correction_reference_cosine": score,
                    "correction_norm": 0.2,
                    "reference_norm": 1.0,
                },
                "next_independent_native_consensus": {"cosine": score},
                "native_component_directional_derivatives": {},
                "reference_observation": {
                    "bridge": {"rollout_velocity_l2": 1.0},
                },
                "proposal_observation": {
                    "bridge": {"rollout_velocity_l2": 0.5 if score > 0 else 2.0},
                },
            })
            rows.append({
                **common,
                "horizon": 200,
                "post_branch_development_label": {
                    "macro_psnr_delta": score,
                    "domain_psnr_delta": {f"d{index}": score for index in range(6)},
                },
            })
    screen = target_blind_signal_screen(rows, [])
    assert screen["status"] == "ELIGIBLE_SIGNALS_FOUND"
    assert "correction_next_native_cosine" in screen["eligible_driver_signals"]
    assert "rollout_speed_stability_margin" in screen["eligible_driver_signals"]
    signal = next(
        row for row in screen["signals"]
        if row["feature"] == "correction_next_native_cosine"
    )
    assert signal["leave_one_method_out_future_sign_accuracy"] == pytest.approx(1.0)
    assert signal["future_200_step_delta_spearman"] == pytest.approx(1.0)
    assert signal["shared_driver_eligible"] is True
    assert screen["eligible_shared_driver_signals"]


def test_target_blind_screen_retains_method_specific_signal_without_unifying_it():
    rows = []
    for epoch, score in ((20, 1.0), (40, 0.5), (60, -0.5), (100, -1.0)):
        common = {
            "probe": "hj",
            "data_epoch": epoch,
            "source_state": "plain",
            "operator_mode": "registered",
            "branch_regime": "continuous_intervention",
        }
        rows.append({
            **common,
            "horizon": 1,
            "update_geometry": {
                "correction_reference_cosine": score,
                "correction_norm": 0.2,
                "reference_norm": 1.0,
            },
            "next_independent_native_consensus": {"cosine": score},
            "native_component_directional_derivatives": {},
            "reference_observation": {"bridge": {"rollout_velocity_l2": 1.0}},
            "proposal_observation": {"bridge": {"rollout_velocity_l2": 1.0}},
        })
        rows.append({
            **common,
            "horizon": 200,
            "post_branch_development_label": {
                "macro_psnr_delta": score,
                "domain_psnr_delta": {f"d{index}": score for index in range(6)},
            },
        })
    screen = target_blind_signal_screen(rows, [])
    assert screen["status"] == "ELIGIBLE_SIGNALS_FOUND"
    assert screen["eligible_shared_driver_signals"] == []
    assert "correction_next_native_cosine" in screen[
        "eligible_method_specific_driver_signals"
    ]["hj"]
    signal = next(
        row for row in screen["signals"]
        if row["feature"] == "correction_next_native_cosine"
    )
    assert signal["shared_driver_eligible"] is False
    assert signal["method_specific_driver_eligible_for"] == ["hj"]
    assert signal["mean_domain_sign_agreement_of_six"] == pytest.approx(6.0)
    assert signal["paired_label_available_to_controller"] is False


def test_shared_signal_requires_every_probe_to_pass_not_only_the_mean():
    rows = []
    for probe, correct in (("dt", True), ("hj", True), ("hnek", False)):
        for epoch, score in ((20, 1.0), (40, 0.5), (60, -0.5), (100, -1.0)):
            label = score if correct else -score
            common = {
                "probe": probe, "data_epoch": epoch, "source_state": probe,
                "operator_mode": causal_audit._preferred_operator_mode(probe, epoch),
                "branch_regime": "continuous_intervention",
            }
            rows.append({
                **common, "horizon": 1,
                "update_geometry": {
                    "correction_reference_cosine": score,
                    "correction_norm": 0.2,
                    "reference_norm": 1.0,
                },
                "next_independent_native_consensus": {"cosine": score},
                "native_component_directional_derivatives": {},
                "reference_observation": {"bridge": {}},
                "proposal_observation": {"bridge": {}},
            })
            rows.append({
                **common, "horizon": 200,
                "post_branch_development_label": {
                    "macro_psnr_delta": label,
                    "domain_psnr_delta": {
                        f"d{index}": label for index in range(6)
                    },
                },
            })
    screen = target_blind_signal_screen(rows, [])
    signal = next(
        row for row in screen["signals"]
        if row["feature"] == "correction_next_native_cosine"
    )
    assert signal["mean_per_method_future_sign_accuracy"] == pytest.approx(2 / 3)
    assert signal["leave_one_method_out_future_sign_accuracy"] == pytest.approx(0.0)
    assert signal["shared_driver_eligible"] is False
    assert signal["method_specific_driver_eligible_for"] == ["dt", "hj"]


def test_signal_records_expose_low_dimensional_block_domain_and_time_consensus():
    common = {
        "probe": "hj",
        "data_epoch": 20,
        "source_state": "hj",
        "operator_mode": "registered",
        "branch_regime": "continuous_intervention",
    }
    rows = [{
        **common,
        "horizon": 1,
        "update_geometry": {
            "correction_reference_cosine": 0.2,
            "correction_norm": 0.2,
            "reference_norm": 1.0,
        },
        "block_geometry": {
            "model": {"correction_reference_cosine": 0.4, "correction_norm": 0.2},
            "model_res": {"correction_reference_cosine": -0.3, "correction_norm": 0.1},
            "zero_block": {"correction_reference_cosine": -1.0, "correction_norm": 0.0},
        },
        "next_independent_native_consensus": {"cosine": 0.1},
        "native_component_directional_derivatives": {},
        "reference_observation": {
            "bridge": {
                "independent_endpoint_separation_l2": 1.0,
                "bridge_kdd_critic_loss": -2.0,
            },
            "gradient": {"diagnostics": {"generator_grad_norm": 2.0}},
            "game_balance": {"d_to_g_loss_ratio": 1.0, "e_to_g_loss_ratio": 2.0},
        },
        "proposal_observation": {
            "bridge": {
                "independent_endpoint_separation_l2": 1.2,
                "bridge_kdd_critic_loss": -1.0,
            },
            "gradient": {"diagnostics": {
                "generator_grad_norm": 3.0,
                "adam_moment_gradient_cosine": -0.1,
            }},
            "game_balance": {"d_to_g_loss_ratio": 2.0, "e_to_g_loss_ratio": 1.0},
        },
    }, {
        **common,
        "horizon": 200,
        "post_branch_development_label": {
            "macro_psnr_delta": 0.1,
            "domain_psnr_delta": {f"d{index}": 0.1 for index in range(6)},
        },
    }]
    variance_rows = []
    for axis, replicates in (
        ("independent_unpaired_batch", [
            {"domain": "d0", "bridge_time": "0", "correction_norm": 0.2, "same_batch_native_cosine": 0.2},
            {"domain": "d1", "bridge_time": "1", "correction_norm": 0.2, "same_batch_native_cosine": -0.4},
        ]),
        ("latent_time_bridge_rng", [
            {"domain": "d0", "bridge_time": "0", "correction_norm": 0.2, "same_batch_native_cosine": 0.5},
            {"domain": "d0", "bridge_time": "1", "correction_norm": 0.2, "same_batch_native_cosine": -0.2},
        ]),
    ):
        variance_rows.append({
            "probe": "hj", "data_epoch": 20, "source_state": "hj",
            "operator_mode": "registered", "axis": axis,
            "mean_correction_norm": 0.2,
            "expected_correction_norm_sq": 0.04,
            "correction_variance_fraction": 0.2,
            "next_independent_batch_native_cosine_mean": 0.1,
            "block_stable_mean_energy": {
                "model": {"variance_fraction": 0.1},
                "model_res": {"variance_fraction": 0.6},
            },
            "bridge_time_summary": {},
            "replicate_records": replicates,
        })
    records = causal_audit._signal_records(rows, variance_rows)
    assert records["minimum_block_correction_native_cosine"][0]["score"] == pytest.approx(-0.3)
    assert records["minimum_domain_correction_native_cosine"][0]["score"] == pytest.approx(-0.4)
    assert records["minimum_time_correction_native_cosine"][0]["score"] == pytest.approx(-0.2)
    assert records["low_max_block_variance_margin"][0]["score"] == pytest.approx(0.15)
    assert records["endpoint_dispersion_stability_margin"][0]["score"] == pytest.approx(-0.2)
    assert records["bridge_kdd_magnitude_stability_margin"][0]["score"] == pytest.approx(0.5)
    assert records["generator_gradient_scale_margin"][0]["score"] == pytest.approx(-0.5)
    assert records["adam_moment_gradient_alignment"][0]["score"] == pytest.approx(-0.1)
    assert records["d_to_g_balance_stability_margin"][0]["score"] == pytest.approx(-1.0)
    assert records["e_to_g_balance_stability_margin"][0]["score"] == pytest.approx(0.5)
    for feature in (
        "minimum_block_correction_native_cosine",
        "minimum_domain_correction_native_cosine",
        "minimum_time_correction_native_cosine",
    ):
        assert records[feature][0]["paired_label_available_to_controller"] is False


def test_signal_records_do_not_call_an_inactive_correction_low_variance():
    common = {
        "probe": "dt", "data_epoch": 20, "source_state": "plain",
        "operator_mode": "forced_active_diagnostic",
        "branch_regime": "continuous_intervention",
    }
    rows = [{
        **common, "horizon": 1,
        "update_geometry": {
            "correction_reference_cosine": 0.0,
            "correction_norm": 0.0,
            "reference_norm": 1.0,
        },
        "block_geometry": {
            "model": {"correction_reference_cosine": 0.0, "correction_norm": 0.0},
        },
        "native_component_directional_derivatives": {},
        "reference_observation": {"bridge": {}},
        "proposal_observation": {"bridge": {}},
    }, {
        **common, "horizon": 200,
        "post_branch_development_label": {
            "macro_psnr_delta": -0.1,
            "domain_psnr_delta": {f"d{index}": -0.1 for index in range(6)},
        },
    }]
    variance_rows = [{
        **{key: common[key] for key in (
            "probe", "data_epoch", "source_state", "operator_mode",
        )},
        "axis": "independent_unpaired_batch",
        "mean_correction_norm": 0.0,
        "expected_correction_norm_sq": 0.0,
        "correction_variance_fraction": 0.0,
        "next_independent_batch_native_cosine_mean": 0.0,
        "block_stable_mean_energy": {"model": {"variance_fraction": 0.0}},
        "replicate_records": [{
            "domain": "d0", "bridge_time": "0", "correction_norm": 0.0,
            "same_batch_native_cosine": 0.0,
        }],
    }]
    records = causal_audit._signal_records(rows, variance_rows)
    assert records == {}
    assert "low_batch_variance_margin" not in records
    assert "replicated_next_batch_consensus" not in records
    assert "low_max_block_variance_margin" not in records
    assert "minimum_domain_correction_native_cosine" not in records
    assert "minimum_block_correction_native_cosine" not in records


def test_signal_records_consume_probe_internal_defects_at_mathematical_zeroes():
    rows = []
    for epoch, mismatch in ((20, 2.0), (40, 1.0)):
        common = {
            "probe": "dt", "data_epoch": epoch, "source_state": "dt",
            "operator_mode": causal_audit._preferred_operator_mode("dt", epoch),
            "branch_regime": "continuous_intervention",
        }
        rows.extend([{
            **common, "horizon": 1,
            "update_geometry": {
                "correction_reference_cosine": 0.1,
                "correction_norm": 0.2,
                "reference_norm": 1.0,
            },
            "native_component_directional_derivatives": {},
            "reference_observation": {"bridge": {}},
            "proposal_observation": {
                "bridge": {},
                "method_internal": {"dt_loss_u_match": mismatch},
            },
        }, {
            **common, "horizon": 200,
            "post_branch_development_label": {
                "macro_psnr_delta": 0.1,
                "domain_psnr_delta": {f"d{index}": 0.1 for index in range(6)},
            },
        }])
    common = {
        "probe": "hj", "data_epoch": 20, "source_state": "hj",
        "operator_mode": "registered",
        "branch_regime": "continuous_intervention",
    }
    rows.extend([{
        **common, "horizon": 1,
        "update_geometry": {
            "correction_reference_cosine": 0.1,
            "correction_norm": 0.2,
            "reference_norm": 1.0,
        },
        "native_component_directional_derivatives": {},
        "reference_observation": {"bridge": {}},
        "proposal_observation": {
            "bridge": {},
            "method_internal": {
                "hj_active": 1.0,
                "hj_probe_sum": 1.5,
                "hj_risk_sum": 0.5,
            },
        },
    }, {
        **common, "horizon": 200,
        "post_branch_development_label": {
            "macro_psnr_delta": 0.1,
            "domain_psnr_delta": {f"d{index}": 0.1 for index in range(6)},
        },
    }])

    records = causal_audit._signal_records(rows, [])
    assert [
        item["score"]
        for item in records["dt_covariance_mismatch_applicability"]
    ] == pytest.approx([2.0, 1.0])
    assert records["dt_covariance_mismatch_descent_margin"][0][
        "score"
    ] == pytest.approx(0.5)
    assert records["hj_supported_structure_conflict_margin"][0][
        "score"
    ] == pytest.approx(0.2)
    screen = target_blind_signal_screen(rows, [])
    coverage = screen["probe_internal_observable_coverage"]
    assert "dt_covariance_mismatch_descent_margin" in coverage["dt"][
        "screened_features"
    ]
    assert "hj_supported_structure_conflict_margin" in coverage["hj"][
        "screened_features"
    ]
    assert "low_time_conditioning_spread_margin" in coverage["hnek"][
        "screened_features"
    ]
    assert screen["paired_metrics_accessed_by_controller"] is False


def test_variance_summary_excludes_inactive_zero_correction_rows():
    rows = [{
        "probe": "hj", "data_epoch": 20, "source_state": "plain",
        "operator_mode": "registered", "axis": "independent_unpaired_batch",
        "mean_correction_norm": 0.0,
        "expected_correction_norm_sq": 0.0,
        "correction_variance_fraction": 0.0,
    }, {
        "probe": "hj", "data_epoch": 100, "source_state": "hj",
        "operator_mode": "registered", "axis": "independent_unpaired_batch",
        "mean_correction_norm": 0.2,
        "expected_correction_norm_sq": 0.5,
        "correction_variance_fraction": 0.8,
    }]
    summary = causal_audit._variance_summary(rows, "hj")
    batch = summary["axes"]["independent_unpaired_batch"]
    assert batch["recorded_rows"] == 2
    assert batch["rows"] == 1
    assert batch["inactive_zero_correction_rows"] == 1
    assert batch["variance_dominated_rows"] == 1


def test_probe_classification_does_not_hide_late_sign_or_scale_failure_in_mean():
    rows = []
    for epoch, cosine, ratio in ((20, 1.0, 0.2), (100, -0.2, 0.2), (150, 0.4, 2.0)):
        rows.append({
            "probe": "hj", "data_epoch": epoch, "source_state": "hj",
            "operator_mode": "registered", "branch_regime": "continuous_intervention",
            "horizon": 1,
            "update_geometry": {
                "correction_norm": ratio, "reference_norm": 1.0,
            },
            "next_independent_native_consensus": {"cosine": cosine},
        })
    summary = causal_audit._classify_probe(rows, "hj")
    assert summary["next_batch_consensus_mean"] > 0.0
    assert summary["next_batch_consensus_negative_rows"] == 1
    assert len(summary["next_batch_consensus_sign_changes"]) == 2
    assert summary["correction_to_native_norm_ratio_mean"] < 1.0
    assert summary["correct_direction_overscale_rows"] == 1
    screen = {
        "eligible_shared_driver_signals": [
            "correction_next_native_cosine",
            "correction_within_native_scale_margin",
        ],
        "eligible_driver_signals": [
            "correction_next_native_cosine",
            "correction_within_native_scale_margin",
        ],
        "eligible_method_specific_driver_signals": {},
        "signals": [],
    }
    ranked = causal_audit._rank_failure_mechanisms(
        [summary], [], screen, rows, [],
    )
    failure_types = {row["failure_type"] for row in ranked}
    assert "correction_sign_reversal" in failure_types
    assert "correct_direction_unstable_magnitude" in failure_types
    sign = next(
        row for row in ranked
        if row["failure_type"] == "correction_sign_reversal"
    )
    assert sign["construction_route"] == (
        "independent_future_native_gradient_consensus_or_one_sided_constraint"
    )
    assert sign["search005_fbcmp_equivalent_forbidden"] is True
    assert "previous and current auxiliary correction" in sign[
        "equivalence_boundary"
    ]


def test_both_state_harm_routes_to_mathematical_rewrite_not_exit_schedule():
    summary = {
        "probe": "hj",
        "next_batch_consensus_mean": 0.2,
        "next_batch_consensus_negative_rows": 0,
        "correction_to_native_norm_ratio_mean": 0.2,
        "correct_direction_overscale_rows": 0,
        "case_counts": {"harmful_on_both_states": 2},
    }
    screen = {
        "eligible_shared_driver_signals": ["adam_moment_gradient_alignment"],
        "eligible_driver_signals": ["adam_moment_gradient_alignment"],
        "eligible_method_specific_driver_signals": {},
        "signals": [],
    }
    ranked = causal_audit._rank_failure_mechanisms(
        [summary], [], screen, [], [],
    )
    mechanism = next(
        row for row in ranked
        if row["failure_type"] == "state_independent_late_bias"
    )
    assert mechanism["candidate_generation_eligible"] is True
    assert mechanism["construction_route"] == "current_state_rate_or_curvature_reformulation"
    assert mechanism["fixed_exit_or_handoff_forbidden"] is True
    assert mechanism["fixed_annealing_forbidden"] is True


def test_derive_initializes_evidence_bound_hypothesis_ledger(tmp_path):
    import json

    audit = tmp_path / "audit"
    audit.mkdir(parents=True)
    atlas = audit / "LONG_REVERSAL_ATLAS.jsonl"
    atlas.write_text(json.dumps({"row_id": "causal-row"}) + "\n", encoding="utf-8")
    matrix = audit / "LONG_CAUSAL_MATRIX.json"
    matrix.write_text(json.dumps({
        "status": "COMPLETE_CAUSAL_AUDIT",
        "ranked_failure_mechanisms": [{
            "failure_type": "sampling_variance",
            "candidate_generation_eligible": True,
            "cross_probe_support": 2,
        }],
        "target_blind_signal_screen": {"eligible_driver_signals": []},
    }), encoding="utf-8")
    result = derive_from_completed_atlas(tmp_path)
    assert result["status"] == "DERIVATION_CARDS_REQUIRED"
    ledger_path = tmp_path / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["status"] == "ACTIVE_DERIVATION"
    assert ledger["records"][0]["candidate_id"] == "G1-01-SAMPLING-VARIANCE"
    assert ledger["records"][0]["status"] == "DERIVATION_REQUIRED"
    assert ledger["generation_policy"]["maximum_revisions_per_mechanism"] == 1
    assert ledger["paired_controller_access"] is False
    repeated = derive_from_completed_atlas(tmp_path)
    assert repeated["hypothesis_ledger"]["sha256"] == result["hypothesis_ledger"]["sha256"]


def test_rollout_growth_signal_uses_only_past_and_current_unpaired_state():
    rows = []
    for probe in ("dt", "hj", "hnek"):
        for source_state, future_score, current_velocity in (
            ("plain", 0.3, 0.5),
            (probe, -0.3, 2.0),
        ):
            for epoch, velocity in ((20, 1.0), (100, current_velocity)):
                mode = causal_audit._preferred_operator_mode(probe, epoch)
                common = {
                    "probe": probe,
                    "data_epoch": epoch,
                    "source_state": source_state,
                    "operator_mode": mode,
                    "branch_regime": "continuous_intervention",
                }
                rows.append({
                    **common,
                    "horizon": 1,
                    "update_geometry": {
                        "correction_reference_cosine": 0.0,
                        "correction_norm": 0.2,
                        "reference_norm": 1.0,
                    },
                    "native_component_directional_derivatives": {},
                    "reference_observation": {
                        "bridge": {"rollout_velocity_l2": velocity},
                    },
                    "proposal_observation": {
                        "bridge": {"rollout_velocity_l2": velocity},
                    },
                })
                rows.append({
                    **common,
                    "horizon": 200,
                    "post_branch_development_label": {
                        "macro_psnr_delta": future_score,
                        "domain_psnr_delta": {
                            f"d{index}": future_score for index in range(6)
                        },
                    },
                })
    screen = target_blind_signal_screen(rows, [])
    assert "rollout_velocity_growth_margin" in screen["eligible_driver_signals"]
    signal = next(
        row for row in screen["signals"]
        if row["feature"] == "rollout_velocity_growth_margin"
    )
    assert signal["records"] == 6
    assert signal["leave_one_method_out_future_sign_accuracy"] == pytest.approx(1.0)
    assert signal["paired_label_available_to_controller"] is False


def test_failure_mechanism_ranking_uses_all_discovery_contract_dimensions():
    mechanisms = [
        {
            "failure_type": "sampling_variance",
            "supporting_probes": ["hj", "hnek"],
            "cross_probe_support": 2,
            "candidate_generation_eligible": True,
        },
        {
            "failure_type": "correction_sign_reversal",
            "supporting_probes": ["hj", "hnek"],
            "cross_probe_support": 2,
            "candidate_generation_eligible": True,
            "eligible_target_blind_driver_signals": [
                "correction_next_native_cosine"
            ],
            "eligible_method_specific_driver_signals_by_probe": {},
        },
    ]
    screen = {
        "signals": [{
            "feature": "correction_next_native_cosine",
            "reversal_precursor_lead_fraction": 1.0,
            "reversal_precursor_cases": [{"probe": "hj"}],
            "mean_domain_sign_agreement_of_six": 5.0,
            "per_method_performance": {},
        }],
    }
    rows = []
    for probe in ("hj", "hnek"):
        rows.append({
            "probe": probe,
            "data_epoch": 20,
            "source_state": probe,
            "operator_mode": "registered",
            "branch_regime": "continuous_intervention",
            "horizon": 200,
            "update_geometry": {"correction_norm": 0.2},
            "post_branch_development_label": {"macro_psnr_delta": 0.2},
        })
    ranked = causal_audit._rank_mechanisms_by_discovery_evidence(
        mechanisms, screen, rows,
    )
    assert [row["failure_type"] for row in ranked] == [
        "correction_sign_reversal", "sampling_variance",
    ]
    evidence = ranked[0]["discovery_ranking_evidence"]
    assert evidence["target_blind_precursor_lead_fraction"] == pytest.approx(1.0)
    assert evidence["target_blind_domain_sign_agreement_of_six"] == pytest.approx(5.0)
    assert evidence["short_counterfactual"]["records"] == 2
    assert evidence["short_counterfactual"]["positive_fraction"] == pytest.approx(1.0)
    assert evidence["short_counterfactual"]["paired_label_available_to_controller"] is False
    assert "minimum_route_complexity_prior" in evidence


def test_matrix_can_route_endpoint_and_game_failures_without_changing_endpoint_law():
    summaries = [{
        "probe": "hj",
        "next_batch_consensus_mean": None,
        "correction_to_native_norm_ratio_mean": None,
        "case_counts": {},
    }]
    common = {
        "probe": "hj", "data_epoch": 20, "source_state": "hj",
        "operator_mode": "registered", "branch_regime": "continuous_intervention",
    }
    rows = [{
        **common, "horizon": 1,
        "update_geometry": {"correction_norm": 0.2},
        "reference_observation": {
            "bridge": {
                "rollout_velocity_l2": 1.0,
                "independent_endpoint_separation_l2": 1.0,
                "bridge_kdd_critic_loss": -1.0,
            },
            "gradient": {"diagnostics": {"adam_moment_gradient_cosine": 0.2}},
            "game_balance": {"d_to_g_loss_ratio": 1.0, "e_to_g_loss_ratio": 1.0},
        },
        "proposal_observation": {
            "bridge": {
                "rollout_velocity_l2": 1.0,
                "independent_endpoint_separation_l2": 1.5,
                "bridge_kdd_critic_loss": -2.0,
            },
            "gradient": {"diagnostics": {"adam_moment_gradient_cosine": -0.2}},
            "game_balance": {"d_to_g_loss_ratio": 1.5, "e_to_g_loss_ratio": 1.0},
        },
    }, {
        **common, "horizon": 200,
        "post_branch_development_label": {
            "macro_psnr_delta": -0.2,
            "domain_psnr_delta": {f"d{index}": -0.2 for index in range(6)},
        },
    }]
    features = [
        "endpoint_dispersion_stability_margin",
        "bridge_kdd_magnitude_stability_margin",
        "d_to_g_balance_stability_margin",
        "adam_moment_gradient_alignment",
    ]
    screen = {
        "eligible_shared_driver_signals": features,
        "eligible_driver_signals": features,
        "eligible_method_specific_driver_signals": {},
        "signals": [],
    }
    ranked = causal_audit._rank_failure_mechanisms(
        summaries, [], screen, rows, [],
    )
    by_type = {row["failure_type"]: row for row in ranked}
    assert by_type["endpoint_dispersion_instability"]["candidate_generation_eligible"] is True
    assert by_type["endpoint_dispersion_instability"]["endpoint_law_change_forbidden"] is True
    assert by_type["game_balance_instability"]["candidate_generation_eligible"] is True
    assert by_type["rollout_distribution_speed"]["candidate_generation_eligible"] is True


def _write_candidate_registration_fixture(
    tmp_path: Path, candidate_id: str = "G1-TEST", *, freeze_ledger: bool = True,
):
    import json

    audit = tmp_path / "audit"
    cards = tmp_path / "derive" / "cards"
    implementations = tmp_path / "derive" / "implementations"
    shared_e0 = tmp_path / "shared_e0"
    for path in (audit, cards, implementations, shared_e0):
        path.mkdir(parents=True, exist_ok=True)
    matrix = {"status": "COMPLETE_CAUSAL_AUDIT", "ranked_failure_mechanisms": []}
    matrix_path = audit / "LONG_CAUSAL_MATRIX.json"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    atlas_path = audit / "LONG_REVERSAL_ATLAS.jsonl"
    atlas_path.write_text(json.dumps({"row_id": "evidence-1"}) + "\n", encoding="utf-8")
    card_path = cards / f"{candidate_id}.json"
    card = {
        "schema": CARD_SCHEMA,
        "candidate_id": candidate_id,
        "parent_evidence": {"failure_type": "sampling_variance"},
        "lineage_evidence": ["LONG_CAUSAL_MATRIX:sampling_variance"],
        "prior_equivalence_audit": {
            "compared_implementations": ["AEB", "BCAVP"],
            "material_difference": "estimate a gradient without changing the endpoint law",
            "equivalent_rerun": False,
        },
        "unsb_object": "unpaired stochastic gradient estimator",
        "formula": "E[g_new] = E[g_UNSB]",
        "identity_or_unbiased_condition": "unbiased under registered measure",
        "objective_change": False,
        "estimator_change": True,
        "coordinate_change": False,
        "endpoint_law_change": False,
        "target_inaccessibility_proof": "only unpaired training tensors are visible",
        "expected_applicable_state": {"condition": "sampling variance dominates"},
        "falsifying_experiment": "estimator expectation differs from plain",
        "compute_cost": "two samples per update",
        "memory_cost": "one additional gradient buffer",
        "recovery_state_cost": "stateless estimator; RNG streams are checkpointed",
        "algorithm_hyperparameters": {},
        "algorithm_state_variables": [],
        "ablation_definitions": {
            "proposal_only": "estimator without the target-blind observable",
            "observable_only": "record variance without changing the update",
            "projected_or_full": "full unbiased estimator versus plain update",
        },
        "historical_evidence_index_sha256": file_sha256(
            ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl"
        ),
        "mechanism_object_map_sha256": file_sha256(
            ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json"
        ),
        "reuse_boundary_sha256": file_sha256(
            ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json"
        ),
        "construction_authority": "independent_unbiased_reparameterization",
        "unbiased_proof": "paired antithetic draws preserve the marginal expectation",
        "paired_target_available_to_training": False,
        "causal_matrix_sha256": file_sha256(matrix_path),
        "reversal_atlas_sha256": file_sha256(atlas_path),
    }
    card_path.write_text(json.dumps(card), encoding="utf-8")
    source = ROOT / "src" / "models" / "sb_model.py"
    implementation = {
        "schema": IMPLEMENTATION_SCHEMA,
        "candidate_id": candidate_id,
        "status": "FROZEN_FOR_GATES",
        "derivation_card_sha256": file_sha256(card_path),
        "model": "sb",
        "method": {},
        "training_target_access": "unpaired_only",
        "paired_controller_access": False,
        "state_contract": {
            "full_state_restorable": True,
            "zero_intervention_identity_test": True,
            "parent_state_isolation_test": True,
        },
        "source_files": [{
            "path": "src/models/sb_model.py",
            "sha256": file_sha256(source),
        }],
    }
    implementation_path = implementations / f"{candidate_id}.json"
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    e0 = {
        "schema": "fixture-e0",
        "metadata": {"protocol_fingerprint": "base-crn-fingerprint"},
        "model": {"networks": {}, "optimizers": [], "schedulers": [], "method": {}},
        "rng": {},
        "samplers": {},
    }
    e0_path = shared_e0 / "e0.pt"
    torch.save(e0, e0_path)
    (shared_e0 / "e0.pt.json").write_text(json.dumps({
        "checkpoint_sha256": file_sha256(e0_path),
        "scientific_state_sha256": full_state_hash(e0),
    }), encoding="utf-8")
    ledger_path = tmp_path / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger_path.write_text(json.dumps({
        "schema": "final-unsb-route1-hypothesis-ledger-v1",
        "status": "ACTIVE_DERIVATION",
        "evidence_identity": {
            "causal_matrix_sha256": file_sha256(matrix_path),
            "reversal_atlas_sha256": file_sha256(atlas_path),
            "historical_evidence_index_sha256": file_sha256(
                ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl"
            ),
            "mechanism_object_map_sha256": file_sha256(
                ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json"
            ),
            "reuse_boundary_sha256": file_sha256(
                ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json"
            ),
        },
        "generation_policy": {
            "maximum_generation1_candidates": 3,
            "maximum_revisions_per_mechanism": 1,
            "maximum_components_per_composition": 2,
            "fixed_window_or_hyperparameter_grid_forbidden": True,
        },
        "records": [{
            "candidate_id": candidate_id,
            "generation": 1,
            "parent_candidate_id": None,
            "parent_evidence": {"failure_type": "sampling_variance"},
            "construction_route": "unbiased estimator",
            "status": "DERIVATION_REQUIRED",
            "revision_count": 0,
            "experiments": [],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    if freeze_ledger:
        freeze_candidate_derivation(tmp_path, candidate_id)
    return card_path, implementation_path


def test_candidate_registration_binds_evidence_card_code_e0_and_gate(tmp_path):
    import json

    _write_candidate_registration_fixture(tmp_path)
    registration = load_candidate_registration(tmp_path, "G1-TEST")
    assert registration.spec.model == "sb"
    assert registration.base_protocol_fingerprint == "base-crn-fingerprint"
    assert registration.gate is None
    gate_path = tmp_path / "derive" / "gates" / "G1-TEST.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps({
        "schema": GATE_SCHEMA,
        "status": "PASS_LONG_RUN",
        "candidate_fingerprint": registration.candidate_fingerprint,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "checks": {
            "mathematical_invariants": True,
            "zero_intervention_identity": True,
            "resume_exact": True,
            "cross_state_counterfactual": True,
            "target_blind_observable": True,
            "micro_engineering": True,
            "base_unsb_semantics_preserved": True,
            "shared_e0_load_exact": True,
        },
        "paired_metric_used_for_promotion": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    gated = load_candidate_registration(tmp_path, "G1-TEST", require_gate=True)
    assert gated.gate["status"] == "PASS_LONG_RUN"


def test_candidate_registration_rejects_unsafe_ids_and_stale_source(tmp_path):
    import json

    with pytest.raises(ValueError, match="unsafe candidate id"):
        validate_candidate_id("../escape")
    _, implementation_path = _write_candidate_registration_fixture(tmp_path)
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["source_files"][0]["sha256"] = "0" * 64
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source hash mismatch"):
        load_candidate_registration(tmp_path, "G1-TEST")


def test_candidate_registration_rejects_incomplete_discovery_card(tmp_path):
    import json

    card_path, implementation_path = _write_candidate_registration_fixture(tmp_path)
    card = json.loads(card_path.read_text(encoding="utf-8"))
    del card["ablation_definitions"]
    card_path.write_text(json.dumps(card), encoding="utf-8")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["derivation_card_sha256"] = file_sha256(card_path)
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="incomplete derivation card"):
        load_candidate_registration(tmp_path, "G1-TEST")


def test_signal_driven_candidate_requires_matrix_eligible_driver(tmp_path):
    import json

    card_path, implementation_path = _write_candidate_registration_fixture(tmp_path)
    matrix_path = tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json"
    matrix = {
        "status": "COMPLETE_CAUSAL_AUDIT",
        "ranked_failure_mechanisms": [{
            "failure_type": "sampling_variance",
            "candidate_generation_eligible": True,
        }],
        "target_blind_signal_screen": {
            "eligible_driver_signals": ["correction_next_native_cosine"],
        },
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card.update({
        "construction_authority": "eligible_target_blind_signal",
        "target_blind_driver_signal": "not_registered",
        "causal_matrix_sha256": file_sha256(matrix_path),
    })
    card_path.write_text(json.dumps(card), encoding="utf-8")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["derivation_card_sha256"] = file_sha256(card_path)
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not an eligible target-blind signal"):
        load_candidate_registration(tmp_path, "G1-TEST")


def test_method_specific_candidate_cannot_borrow_another_probes_signal(tmp_path):
    import json

    card_path, implementation_path = _write_candidate_registration_fixture(tmp_path)
    matrix_path = tmp_path / "audit" / "LONG_CAUSAL_MATRIX.json"
    matrix = {
        "status": "COMPLETE_CAUSAL_AUDIT",
        "ranked_failure_mechanisms": [{
            "failure_type": "sampling_variance",
            "candidate_generation_eligible": True,
            "supporting_probes": ["hj", "hnek"],
        }],
        "target_blind_signal_screen": {
            "eligible_shared_driver_signals": [],
            "eligible_method_specific_driver_signals": {
                "hj": ["correction_next_native_cosine"],
            },
        },
    }
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card.update({
        "construction_authority": "eligible_method_specific_signal",
        "target_blind_driver_signal": "correction_next_native_cosine",
        "target_blind_driver_probe": "hnek",
        "causal_matrix_sha256": file_sha256(matrix_path),
    })
    card_path.write_text(json.dumps(card), encoding="utf-8")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["derivation_card_sha256"] = file_sha256(card_path)
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="declared method-specific probe"):
        load_candidate_registration(tmp_path, "G1-TEST")


def test_algorithm_fingerprint_is_seed_e0_independent_but_execution_is_not(tmp_path):
    import json

    first = tmp_path / "seed2026"
    second = tmp_path / "seed2027"
    _write_candidate_registration_fixture(first)
    _write_candidate_registration_fixture(second)
    e0_path = second / "shared_e0" / "e0.pt"
    e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    e0["rng"] = {"synthetic_seed": 2027}
    torch.save(e0, e0_path)
    (second / "shared_e0" / "e0.pt.json").write_text(json.dumps({
        "checkpoint_sha256": file_sha256(e0_path),
        "scientific_state_sha256": full_state_hash(e0),
    }), encoding="utf-8")
    registration_2026 = load_candidate_registration(first, "G1-TEST")
    registration_2027 = load_candidate_registration(second, "G1-TEST")
    assert registration_2026.algorithm_fingerprint == registration_2027.algorithm_fingerprint
    assert registration_2026.candidate_fingerprint != registration_2027.candidate_fingerprint


def test_algorithm_definition_is_host_evidence_independent_but_registration_is_not(tmp_path):
    import json

    first = tmp_path / "local1660"
    second = tmp_path / "remote4090"
    _write_candidate_registration_fixture(first)
    card_path, implementation_path = _write_candidate_registration_fixture(
        second, freeze_ledger=False,
    )
    matrix_path = second / "audit" / "LONG_CAUSAL_MATRIX.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["host_evidence"] = "remote4090"
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["causal_matrix_sha256"] = file_sha256(matrix_path)
    card["parent_evidence"] = {
        "failure_type": "sampling_variance", "host": "remote4090",
    }
    card_path.write_text(json.dumps(card), encoding="utf-8")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["derivation_card_sha256"] = file_sha256(card_path)
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    ledger_path = second / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["evidence_identity"]["causal_matrix_sha256"] = file_sha256(matrix_path)
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    freeze_candidate_derivation(second, "G1-TEST")
    local = load_candidate_registration(first, "G1-TEST")
    remote = load_candidate_registration(second, "G1-TEST")
    assert local.algorithm_fingerprint == remote.algorithm_fingerprint
    assert local.candidate_fingerprint != remote.candidate_fingerprint


def test_candidate_registration_requires_frozen_hypothesis_ledger(tmp_path):
    card_path, implementation_path = _write_candidate_registration_fixture(
        tmp_path, freeze_ledger=False,
    )
    with pytest.raises(RuntimeError, match="not frozen for gates"):
        load_candidate_registration(tmp_path, "G1-TEST")
    frozen = freeze_candidate_derivation(tmp_path, "G1-TEST")
    repeated = freeze_candidate_derivation(tmp_path, "G1-TEST")
    assert frozen.hypothesis_ledger_sha256 == repeated.hypothesis_ledger_sha256
    import json

    ledger_path = tmp_path / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["records"].append({
        "candidate_id": "G1-SIBLING",
        "generation": 1,
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    after_sibling = load_candidate_registration(tmp_path, "G1-TEST")
    assert after_sibling.candidate_fingerprint == frozen.candidate_fingerprint
    assert after_sibling.hypothesis_ledger_sha256 != frozen.hypothesis_ledger_sha256

    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["formula"] = "E[g_new] = E[g_UNSB] + 0"
    card_path.write_text(json.dumps(card), encoding="utf-8")
    implementation = json.loads(implementation_path.read_text(encoding="utf-8"))
    implementation["derivation_card_sha256"] = file_sha256(card_path)
    implementation_path.write_text(json.dumps(implementation), encoding="utf-8")
    with pytest.raises(RuntimeError, match="may not be silently rewritten"):
        freeze_candidate_derivation(tmp_path, "G1-TEST")


def test_causal_revision_requires_negative_e200_and_is_once_per_mechanism(tmp_path):
    import json

    from research.local_route1.candidates import register_candidate_revision

    parent_card_path, parent_implementation_path = _write_candidate_registration_fixture(
        tmp_path
    )
    candidate_root = tmp_path / "candidates" / "G1-TEST"
    candidate_root.mkdir(parents=True)
    trajectory_path = candidate_root / "CANDIDATE_TRAJECTORY.json"
    trajectory_path.write_text(json.dumps({
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "candidate_id": "G1-TEST",
        "status": "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION",
        "trajectory": [{"epoch": 200, "macro_psnr_delta": -0.1}],
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    defect_path = candidate_root / "DEFECT_ADJUDICATION.json"
    defect_path.write_text(json.dumps({
        "schema": "final-unsb-route1-candidate-defect-adjudication-v1",
        "candidate_id": "G1-TEST",
        "data_epoch_adjudicated": 200,
        "target_blind_defect_reduced": True,
        "long_horizon_benefit_reversed": True,
        "new_causal_failure_reason": "the safe direction remained biased in one block",
        "target_blind_defect_measurement": {
            "observable": "block correction variance",
            "desired_direction": "decrease",
            "reference_value": 1.0,
            "candidate_value": 0.4,
        },
        "paired_target_used_to_compute_defect": False,
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    revisions = tmp_path / "derive" / "revisions"
    revisions.mkdir(parents=True)

    def write_request(candidate_id: str):
        path = revisions / f"{candidate_id}.json"
        path.write_text(json.dumps({
            "schema": "final-unsb-route1-causal-revision-request-v1",
            "parent_candidate_id": "G1-TEST",
            "revision_candidate_id": candidate_id,
            "source_candidate_trajectory_sha256": file_sha256(trajectory_path),
            "defect_evidence_path": "candidates/G1-TEST/DEFECT_ADJUDICATION.json",
            "defect_evidence_sha256": file_sha256(defect_path),
            "new_causal_failure_reason": "the safe direction remained biased in one block",
            "mathematical_change_from_parent": "replace biased block removal with an unbiased estimator",
            "construction_route": "causal revision of the estimator",
            "fixed_window_or_handoff": False,
            "hyperparameter_grid_search": False,
            "paired_target_available_to_revision": False,
            "confirmation20_opened": False,
        }), encoding="utf-8")

    write_request("G2-TEST")
    result = register_candidate_revision(tmp_path, "G1-TEST", "G2-TEST")
    assert result["status"] == "DERIVATION_REQUIRED"
    revision_card = json.loads(parent_card_path.read_text(encoding="utf-8"))
    revision_card.update({
        "candidate_id": "G2-TEST",
        "parent_candidate_id": "G1-TEST",
        "revision_request_sha256": file_sha256(revisions / "G2-TEST.json"),
        "causal_revision_reason": "the safe direction remained biased in one block",
    })
    revision_card_path = tmp_path / "derive" / "cards" / "G2-TEST.json"
    revision_card_path.write_text(json.dumps(revision_card), encoding="utf-8")
    revision_implementation = json.loads(
        parent_implementation_path.read_text(encoding="utf-8")
    )
    revision_implementation.update({
        "candidate_id": "G2-TEST",
        "derivation_card_sha256": file_sha256(revision_card_path),
    })
    (tmp_path / "derive" / "implementations" / "G2-TEST.json").write_text(
        json.dumps(revision_implementation), encoding="utf-8"
    )
    frozen_revision = freeze_candidate_derivation(tmp_path, "G2-TEST")
    assert frozen_revision.candidate_id == "G2-TEST"
    repeated = register_candidate_revision(tmp_path, "G1-TEST", "G2-TEST")
    assert repeated["record"]["candidate_id"] == "G2-TEST"
    write_request("G2-SECOND")
    with pytest.raises(RuntimeError, match="already used its one revision"):
        register_candidate_revision(tmp_path, "G1-TEST", "G2-SECOND")


def _write_passed_candidate_gate(output_root: Path):
    import json

    _write_candidate_registration_fixture(output_root)
    registration = load_candidate_registration(output_root, "G1-TEST")
    gate_path = output_root / "derive" / "gates" / "G1-TEST.json"
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(json.dumps({
        "schema": GATE_SCHEMA,
        "status": "PASS_LONG_RUN",
        "candidate_fingerprint": registration.candidate_fingerprint,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "checks": {
            "mathematical_invariants": True,
            "zero_intervention_identity": True,
            "resume_exact": True,
            "cross_state_counterfactual": True,
            "target_blind_observable": True,
            "micro_engineering": True,
            "base_unsb_semantics_preserved": True,
            "shared_e0_load_exact": True,
        },
        "paired_metric_used_for_promotion": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    return load_candidate_registration(output_root, "G1-TEST", require_gate=True)


def test_seed_validation_freezes_algorithm_and_requires_sign_authorization(tmp_path):
    import json

    registration = _write_passed_candidate_gate(tmp_path)
    candidate_root = tmp_path / "candidates" / "G1-TEST"
    candidate_root.mkdir(parents=True)
    trajectory_path = candidate_root / "CANDIDATE_TRAJECTORY.json"
    trajectory_path.write_text(json.dumps({
        "status": "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
    }), encoding="utf-8")
    freeze_path = candidate_root / "SEED_VALIDATION_FREEZE.json"
    freeze = {
        "schema": SEED_FREEZE_SCHEMA,
        "status": "FROZEN_FOR_SEED_VALIDATION",
        "candidate_id": "G1-TEST",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2026_candidate_fingerprint": registration.candidate_fingerprint,
        "seed2026_trajectory_sha256": file_sha256(trajectory_path),
        "plain_collapse_adjudication": "PASS_NOT_PLAIN_COLLAPSE",
        "authorized_seeds": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    assert validate_seed_freeze(tmp_path, registration, 2027) == freeze
    with pytest.raises(RuntimeError, match="seed2028 requires"):
        validate_seed_freeze(tmp_path, registration, 2028)
    seed2027_root = tmp_path / "seed_validation" / "seed2027"
    seed2027_root.mkdir(parents=True)
    summary_path = seed2027_root / "SEED_VALIDATION_SUMMARY.json"
    summary_path.write_text(json.dumps({"late_sign": "nonpositive"}), encoding="utf-8")
    (seed2027_root / "SEED2028_AUTHORIZATION.json").write_text(json.dumps({
        "status": "AUTHORIZED_SIGN_INCONSISTENCY",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2027_summary_sha256": file_sha256(summary_path),
        "seed2027_late_sign": "nonpositive",
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    assert validate_seed_freeze(tmp_path, registration, 2028) == freeze
    freeze["algorithm_fingerprint"] = "changed"
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    with pytest.raises(RuntimeError, match="algorithm changed"):
        validate_seed_freeze(tmp_path, registration, 2027)


def test_seed_validation_absolute_guards_match_candidate_policy():
    stable = [
        {"epoch": 150, "macro_psnr": 20.0, "plain_macro_psnr": 19.5},
        {"epoch": 175, "macro_psnr": 20.1, "plain_macro_psnr": 19.6},
        {"epoch": 200, "macro_psnr": 20.2, "plain_macro_psnr": 19.7},
    ]
    assert _seed_late_rolling_drawdown(stable) == pytest.approx(0.0)
    assert (
        _seed_plain_collapse_adjudication(stable)["status"]
        == "PASS_NOT_PLAIN_COLLAPSE"
    )
    collapsing = [
        {"epoch": 150, "macro_psnr": 20.0, "plain_macro_psnr": 20.0},
        {"epoch": 175, "macro_psnr": 19.7, "plain_macro_psnr": 19.6},
        {"epoch": 200, "macro_psnr": 19.5, "plain_macro_psnr": 19.5},
    ]
    assert _seed_plain_collapse_adjudication(collapsing)["status"].startswith("FAIL_")


def test_multi_seed_adjudication_enforces_all_route1_guardrails(tmp_path):
    registration = _write_passed_candidate_gate(tmp_path)
    candidate_root = tmp_path / "candidates" / "G1-TEST"
    candidate_root.mkdir(parents=True, exist_ok=True)
    trajectory_path = candidate_root / "CANDIDATE_TRAJECTORY.json"
    trajectory = {
        "schema": "final-unsb-route1-candidate-trajectory-v1",
        "status": "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
        "candidate_id": "G1-TEST",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "trajectory": [
            {
                "epoch": epoch, "macro_psnr": 20.0 + epoch / 1000.0,
                "plain_macro_psnr": 19.8 + epoch / 1000.0,
                "positive_domains": 5,
            }
            for epoch in (150, 175, 200)
        ],
        "late_three_mean_macro_psnr_delta": 0.2,
        "e200_macro_psnr_delta": 0.2,
        "late_average_worst_domain_delta": -0.1,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
        "candidate_best_to_terminal_three_point_rolling_drawdown": 0.0,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }
    trajectory_path.write_text(json.dumps(trajectory), encoding="utf-8")
    freeze = {
        "schema": SEED_FREEZE_SCHEMA,
        "status": "FROZEN_FOR_SEED_VALIDATION",
        "candidate_id": "G1-TEST",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2026_candidate_fingerprint": registration.candidate_fingerprint,
        "seed2026_trajectory_sha256": file_sha256(trajectory_path),
        "plain_collapse_adjudication": "PASS_NOT_PLAIN_COLLAPSE",
        "authorized_seeds": [2027, 2028],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    (candidate_root / "SEED_VALIDATION_FREEZE.json").write_text(
        json.dumps(freeze), encoding="utf-8",
    )
    waiting = summarize_multi_seed_validation(tmp_path, "G1-TEST")
    assert waiting["status"] == "WAITING_FOR_SEED2027"

    seed2027_root = tmp_path / "seed_validation" / "seed2027"
    seed2027_root.mkdir(parents=True)
    seed2027 = {
        "schema": SEED_SUMMARY_SCHEMA,
        "status": "COMPLETE",
        "seed": 2027,
        "candidate_id": "G1-TEST",
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "late_three_mean_macro_psnr_delta": 0.2,
        "late_sign": "positive",
        "e200_macro_psnr_delta": 0.2,
        "late_average_positive_domains": 5.0,
        "late_average_worst_domain_delta": -0.1,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
        "candidate_best_to_terminal_three_point_rolling_drawdown": 0.0,
        "plain_collapse_adjudication": {"status": "PASS_NOT_PLAIN_COLLAPSE"},
        "numeric_gate_pass": True,
        "paired_metric_changed_algorithm": False,
        "confirmation20_opened": False,
    }
    (seed2027_root / "SEED_VALIDATION_SUMMARY.json").write_text(
        json.dumps(seed2027), encoding="utf-8",
    )
    result = summarize_multi_seed_validation(tmp_path, "G1-TEST")
    assert result["schema"] == MULTI_SEED_ADJUDICATION_SCHEMA
    assert result["status"] == "ROUTE1_SUSTAINED_LOCAL"
    assert result["classification"] == "route1_sustained_local"
    assert result["included_seeds"] == [2026, 2027]
    assert result["failed_checks"] == []


def test_seed_validation_e0_and_crn_are_candidate_independent(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_candidate_registration_fixture(first, candidate_id="G1-FIRST")
    _write_candidate_registration_fixture(second, candidate_id="G1-SECOND")
    registration_first = load_candidate_registration(first, "G1-FIRST")
    registration_second = load_candidate_registration(second, "G1-SECOND")
    manifest = tmp_path / "manifest.csv"
    manifest.write_bytes(b"frozen-manifest\r\n")
    assert _e0_identity(seed=2027, manifest_path=manifest) == _e0_identity(
        seed=2027, manifest_path=manifest,
    )
    assert _crn_fingerprint(registration_first, 2027) == _crn_fingerprint(
        registration_second, 2027,
    )


def test_candidate_executable_gate_rejects_short_micro_or_fake_resume():
    report = {
        "checks": {
            "mathematical_invariants": True,
            "zero_intervention_identity": True,
            "resume_exact": True,
            "cross_state_counterfactual": True,
            "target_blind_observable": True,
            "micro_engineering": True,
            "base_unsb_semantics_preserved": True,
            "shared_e0_load_exact": True,
        },
        "mathematical_invariant_evidence": [
            {"name": "identity", "status": "PASS", "observed": "exact zero"},
        ],
        "zero_intervention_evidence": {
            "candidate_state_sha256": "same", "plain_state_sha256": "same",
        },
        "resume_evidence": {
            "continuous_state_sha256": "same", "resumed_state_sha256": "same",
        },
        "cross_state_evidence": {
            "data_epochs": [20, 100, 200],
            "all_parent_state_hashes_preserved": True,
        },
        "target_blind_evidence": {
            "paired_fields_observed": [], "paired_target_available": False,
        },
        "micro_engineering_evidence": {
            "updates": 400, "finite": True,
            "paired_metric_used_for_promotion": False,
        },
        "paired_metric_used_for_promotion": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    assert _validate_gate_report(report) == report
    report["micro_engineering_evidence"]["updates"] = 399
    with pytest.raises(RuntimeError, match="400--800"):
        _validate_gate_report(report)
    report["micro_engineering_evidence"]["updates"] = 400
    report["resume_evidence"]["resumed_state_sha256"] = "different"
    with pytest.raises(RuntimeError, match="resume is not exact"):
        _validate_gate_report(report)
