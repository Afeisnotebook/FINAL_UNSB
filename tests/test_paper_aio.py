from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research.paper_aio.adjudicate import adjudicate
from research.paper_aio.evaluate import (
    aggregate_metric_rows,
    replicate_stochasticity,
    select_discovery,
)
from research.paper_aio.gates import external_gate_status, scientific_core
from research.paper_aio.protocol import (
    EXPECTED_MANIFEST_SHA256,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    epoch_to_step,
    evaluation_bundle_fingerprint,
    lane_spec,
    load_protocol,
    steps_per_epoch,
    validate_protocol,
)
from research.paper_aio.runtime import manifest_report, option_args
from src.util.image_pool import ImagePool


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_paper_protocol_and_full_manifest() -> None:
    protocol = load_protocol()
    assert validate_protocol(protocol) == []
    assert protocol["manifest"]["sha256"] == EXPECTED_MANIFEST_SHA256
    assert steps_per_epoch(protocol) == 8553
    assert epoch_to_step(200, protocol) == 1_710_600
    assert evaluation_bundle_fingerprint(protocol) == FROZEN_EVALUATION_BUNDLE_FINGERPRINT
    report = manifest_report(ROOT / "manifests" / "FULL_DATA_MANIFEST.csv")
    assert report["counts"] == {"train": 8553, "discovery": 480, "confirmation": 120}
    assert len(report["domains"]) == 6


def test_main_sampler_never_enables_macro_marginal(tmp_path: Path) -> None:
    spec = lane_spec("plain")
    args = option_args(spec, dataroot=tmp_path, option_root=tmp_path, seed=2026, gpu=-1)
    assert "--macro_marginal" not in args
    assert args[args.index("--batch_size") + 1] == "1"
    assert args[args.index("--n_epochs") + 1] == "200"
    proposal = lane_spec("proposal")
    assert proposal.method == {
        "route1_ablation_enable": True,
        "pcrsmg_ablation_role": "proposal_only",
    }


def test_discovery_selector_cannot_address_confirmation() -> None:
    rows = []
    for domain in ("a", "b"):
        rows.extend(
            {"domain": domain, "split": "discovery", "order": str(index), "stem": str(index)}
            for index in range(80)
        )
        rows.extend(
            {"domain": domain, "split": "confirmation", "order": str(index), "stem": f"c{index}"}
            for index in range(20)
        )
    selected = select_discovery(rows, 70)
    assert len(selected) == 140
    assert all(row["split"] == "discovery" for row in selected)
    with pytest.raises(RuntimeError, match="not frozen"):
        select_discovery(rows, 81)


def test_image_pool_roundtrip_is_exact() -> None:
    pool = ImagePool(2)
    pool.num_imgs = 2
    pool.images = [torch.randn(1, 3, 4, 4), torch.randn(1, 3, 4, 4)]
    state = pool.state_dict()
    restored = ImagePool(2)
    restored.load_state_dict(state, device=torch.device("cpu"))
    assert restored.num_imgs == 2
    assert all(torch.equal(left, right) for left, right in zip(pool.images, restored.images))
    bad = dict(state)
    bad["num_imgs"] = 1
    with pytest.raises(RuntimeError, match="inconsistent"):
        restored.load_state_dict(bad)


def test_terminal_rollout_stochasticity_is_macro_and_explicit() -> None:
    first = aggregate_metric_rows([
        {"domain": "a", "psnr": 10.0, "ssim": 0.5, "lpips": 0.4},
        {"domain": "b", "psnr": 20.0, "ssim": 0.7, "lpips": 0.2},
    ])
    second = aggregate_metric_rows([
        {"domain": "a", "psnr": 12.0, "ssim": 0.6, "lpips": 0.3},
        {"domain": "b", "psnr": 22.0, "ssim": 0.8, "lpips": 0.1},
    ])
    stochasticity = replicate_stochasticity([first, second])
    assert first["macro_psnr"] == 15.0
    assert second["macro_psnr"] == 17.0
    assert stochasticity == {
        "macro_psnr_std": 1.0,
        "macro_ssim_std": pytest.approx(0.05),
        "macro_lpips_std": pytest.approx(0.05),
        "ddof": 0,
        "replicate_count": 2,
    }


def test_runtime_core_excludes_only_host_metadata() -> None:
    payload = {
        "step": 10,
        "target_steps": 20,
        "model": {"value": torch.tensor([1.0])},
        "rng": {"x": 1},
        "samplers": {"x": 2},
        "metadata": {"host": "one"},
        "lane": {"id": "plain"},
    }
    first = scientific_core(payload)
    payload["metadata"]["host"] = "two"
    assert scientific_core(payload) == first
    payload["step"] = 11
    assert scientific_core(payload) != first


def test_external_lanes_fail_closed(tmp_path: Path) -> None:
    ddsb = external_gate_status(tmp_path, "ddsb")
    assert lane_spec("cut").backend == "internal"
    assert ddsb["fallback_lane"] == "cyclegan"


def _metric(value: float) -> dict:
    domains = {
        domain: {"psnr": value, "ssim": value / 100, "lpips": 0.5 - value / 100}
        for domain in ("a", "b", "c", "d", "e", "f")
    }
    return {
        "macro_psnr": value,
        "macro_ssim": value / 100,
        "macro_lpips": 0.5 - value / 100,
        "domains": domains,
        "protocol_fingerprint": "bundle",
        "evaluation_input_sha256": "inputs",
        "images": [{
            "domain": "a", "stem": "one", "order": 0,
            "replicate": 0, "nfe": 5, "crn_bundle_sha256": "crn",
        }],
        "confirmation20_opened": False,
    }


def test_adjudication_is_terminal_and_incomplete_safe(tmp_path: Path) -> None:
    result = adjudicate(tmp_path)
    assert result["results"]["status"] == "FIRST_WAVE_INCOMPLETE"
    assert result["results"]["unified_evaluation_cohort_pass"] is False
    initial = {row["lane_id"]: row for row in result["results"]["lanes"]}
    assert initial["ddsb"]["status"] == "REPRODUCTION_INCOMPLETE"
    assert initial["hjcgr"]["status"] == "DEFERRED_NOT_FIRST_WAVE"
    assert result["algorithm_set"]["reproduction_incomplete"] == ["ddsb"]
    assert result["algorithm_set"]["paper_claims_frozen"] is False
    for epoch in (150, 175, 200):
        plain = tmp_path / "lanes" / "plain" / "metrics" / f"e{epoch:03d}.json"
        proposal = tmp_path / "lanes" / "proposal" / "metrics" / f"e{epoch:03d}.json"
        plain.parent.mkdir(parents=True, exist_ok=True)
        proposal.parent.mkdir(parents=True, exist_ok=True)
        plain.write_text(json.dumps(_metric(10.0)), encoding="utf-8")
        proposal.write_text(json.dumps(_metric(10.2)), encoding="utf-8")
    result = adjudicate(tmp_path)
    proposal_row = next(
        row for row in result["results"]["lanes"] if row["lane_id"] == "proposal"
    )
    assert proposal_row["scientific_gate"]["status"] == "PASS"
    assert result["algorithm_set"]["accepted_algorithms"] == ["proposal"]


def test_first_wave_completion_requires_unified_cohort_and_four_lanes(tmp_path: Path) -> None:
    input_path = tmp_path / "lanes" / "input" / "metrics" / "e200.json"
    input_path.parent.mkdir(parents=True, exist_ok=True)
    input_path.write_text(json.dumps(_metric(8.0)), encoding="utf-8")
    for lane in ("plain", "proposal"):
        for epoch in (150, 175, 200):
            path = tmp_path / "lanes" / lane / "metrics" / f"e{epoch:03d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(_metric(10.0 + (lane == "proposal") * 0.2)), encoding="utf-8")
    for lane in ("cut", "cyclegan"):
        path = tmp_path / "lanes" / lane / "metrics" / "e200.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_metric(9.0)), encoding="utf-8")
    cohort = tmp_path / "gates" / "UNIFIED_EVALUATION_COHORT.json"
    cohort.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_text(json.dumps({
        "schema": "final-unsb-paper-unified-evaluation-cohort-v1",
        "status": "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT",
        "required_lanes": ["input", "plain", "proposal", "cut", "cyclegan"],
        "cross_host_training_delta_merged": False,
        "confirmation20_opened": False,
    }), encoding="utf-8")
    result = adjudicate(tmp_path)
    assert result["results"]["status"] == "FIRST_WAVE_COMPLETE"
    assert result["results"]["unified_evaluation_cohort_pass"] is True
    assert result["algorithm_set"]["status"] == (
        "FIRST_WAVE_EVIDENCE_READY_CANDIDATES_PENDING"
    )
    assert result["algorithm_set"]["confirmation_authorized"] is False


def test_adjudication_includes_evidence_locked_dynamic_candidate(tmp_path: Path) -> None:
    candidate_id = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"
    lock = tmp_path / "candidate_locks" / candidate_id / "CANDIDATE_LOCK.json"
    lock.parent.mkdir(parents=True)
    lock.write_text(json.dumps({
        "schema": "final-unsb-paper-candidate-lock-v1",
        "status": "PASS_FULL_DATA_CANDIDATE_LOCK",
        "candidate_id": candidate_id,
        "full_data_authorized": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "parent_paper": {"parent_output": str(tmp_path)},
    }), encoding="utf-8")
    for epoch in (150, 175, 200):
        plain = tmp_path / "lanes" / "plain" / "metrics" / f"e{epoch:03d}.json"
        candidate = tmp_path / "lanes" / candidate_id / "metrics" / f"e{epoch:03d}.json"
        plain.parent.mkdir(parents=True, exist_ok=True)
        candidate.parent.mkdir(parents=True, exist_ok=True)
        plain.write_text(json.dumps(_metric(10.0 + epoch / 1000)), encoding="utf-8")
        candidate.write_text(json.dumps(_metric(10.3 + epoch / 1000)), encoding="utf-8")
    result = adjudicate(tmp_path)
    row = next(
        value for value in result["results"]["lanes"]
        if value["lane_id"] == candidate_id
    )
    assert row["comparison_scope"] == "same_host_cross_code_runtime_gate"
    assert row["scientific_gate"]["status"] == "PASS"
    assert row["scientific_gate"]["crn_exact_at_all_late_points"] is True
    assert candidate_id in result["algorithm_set"]["accepted_algorithms"]
