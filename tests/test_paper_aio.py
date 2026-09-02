from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research.paper_aio.adjudicate import adjudicate
from research.paper_aio.evaluate import select_discovery
from research.paper_aio.gates import external_gate_status, scientific_core
from research.paper_aio.protocol import (
    EXPECTED_MANIFEST_SHA256,
    epoch_to_step,
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
    }


def test_adjudication_is_terminal_and_incomplete_safe(tmp_path: Path) -> None:
    result = adjudicate(tmp_path)
    assert result["results"]["status"] == "FIRST_WAVE_INCOMPLETE"
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
