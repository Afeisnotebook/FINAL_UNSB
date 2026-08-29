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
    SEED_FREEZE_SCHEMA,
    _crn_fingerprint,
    _e0_identity,
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
    assert _operator_modes("dt", 100) == ("registered", "forced_active_diagnostic")
    assert _operator_modes("hj", 200) == ("registered",)


def test_terminal_audit_is_local_and_never_fabricates_future_e200_label():
    registered = _audit_regimes((1, 8, 32, 200), start_step=29_999)
    terminal = _audit_regimes((1, 8, 32, 200), start_step=30_000)
    assert any(horizon == 200 for _regime, horizon, _pulse in registered)
    assert {horizon for _regime, horizon, _pulse in terminal} == {1, 8, 32}
    assert not any(horizon == 200 for _regime, horizon, _pulse in terminal)


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
    assert "maximum_benefit" in jobs[40]["selection_reasons"]
    assert "maximum_drawdown_peak" in jobs[40]["selection_reasons"]
    assert "maximum_drawdown_trough" in jobs[80]["selection_reasons"]


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
    assert causal_audit._preferred_operator_mode("dt", 20) == "forced_active_diagnostic"
    assert causal_audit._preferred_operator_mode("dt", 21) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 40) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 45) == "registered"
    assert causal_audit._preferred_operator_mode("dt", 46) == "forced_active_diagnostic"


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
        mode = "forced_active_diagnostic" if probe == "dt" else "registered"
        for epoch, score in ((20, 1.0), (100, -1.0)):
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
    assert signal["mean_domain_sign_agreement_of_six"] == pytest.approx(6.0)
    assert signal["paired_label_available_to_controller"] is False


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
        mode = "forced_active_diagnostic" if probe == "dt" else "registered"
        for source_state, future_score, current_velocity in (
            ("plain", 0.3, 0.5),
            (probe, -0.3, 2.0),
        ):
            for epoch, velocity in ((20, 1.0), (100, current_velocity)):
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
