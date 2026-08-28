from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
