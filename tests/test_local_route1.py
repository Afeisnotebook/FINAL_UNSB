from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.local_route1 import causal_audit
from research.local_route1.causal_audit import (
    AuditCell,
    BranchResult,
    _audit_regimes,
    _initialize_operator_costate,
    _operator_modes,
    append_unique_rows,
    build_causal_matrix,
    target_blind_signal_screen,
)
from research.local_route1.evaluate import select_discovery70
from research.local_route1.interfaces import CounterfactualAuditor, StateObservation
from research.local_route1.lineage import HISTORICAL_DT_SEMANTIC_HASHES
from research.local_route1.protocol import (
    ROOT,
    dt_lambda_for_physical_epoch,
    epoch_to_step,
    load_protocol,
    milestone_steps,
    probe_spec,
    semantic_source_sha256,
    step_to_physical_epoch,
    validate_protocol,
)
from research.local_route1.runtime import full_state_hash, read_manifest
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
            })
    append_unique_rows(audit / "SAMPLING_VARIANCE_ATLAS.jsonl", variance_rows)
    complete = build_causal_matrix(tmp_path)
    assert complete["status"] == "COMPLETE_CAUSAL_AUDIT"
    failure_types = {row["failure_type"] for row in complete["ranked_failure_mechanisms"]}
    assert "correction_sign_reversal" in failure_types
    assert "state_feedback_missing" in failure_types
    assert "sampling_variance" in failure_types


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
    signal = next(
        row for row in screen["signals"]
        if row["feature"] == "correction_next_native_cosine"
    )
    assert signal["leave_one_method_out_future_sign_accuracy"] == pytest.approx(1.0)
    assert signal["future_200_step_delta_spearman"] == pytest.approx(1.0)
    assert signal["mean_domain_sign_agreement_of_six"] == pytest.approx(6.0)
    assert signal["paired_label_available_to_controller"] is False
