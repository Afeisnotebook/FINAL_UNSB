from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.local_route1.causal_audit import (
    _initialize_operator_costate,
    _operator_modes,
    append_unique_rows,
    build_causal_matrix,
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
        for horizon in (1, 8, 32, 200):
            rows.append({
                "schema": "final-unsb-local-route1-reversal-atlas-row-v1",
                "row_id": f"{source}-{horizon}",
                "probe": "hj",
                "data_epoch": 100,
                "source_state": source,
                "operator_mode": "registered",
                "horizon": horizon,
                "update_geometry": {"correction_norm": 0.5, "reference_norm": 1.0},
                "next_independent_native_consensus": (
                    {"cosine": -0.4} if horizon == 1 else None
                ),
                "post_branch_development_label": (
                    {"macro_psnr_delta": 0.2 if source == "plain" else -0.1}
                    if horizon == 200 else None
                ),
            })
    append_unique_rows(audit / "LONG_REVERSAL_ATLAS.jsonl", rows[:-1])
    partial = build_causal_matrix(tmp_path)
    assert partial["status"] == "PARTIAL_CAUSAL_AUDIT"
    assert partial["ranked_failure_mechanisms"] == []
    append_unique_rows(audit / "LONG_REVERSAL_ATLAS.jsonl", rows[-1:])
    complete = build_causal_matrix(tmp_path)
    assert complete["status"] == "COMPLETE_CAUSAL_AUDIT"
    failure_types = {row["failure_type"] for row in complete["ranked_failure_mechanisms"]}
    assert "correction_sign_reversal" in failure_types
    assert "state_feedback_missing" in failure_types
