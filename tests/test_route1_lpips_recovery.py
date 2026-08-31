from __future__ import annotations

import copy
import json

import pytest

from operations.local_route1_lpips_recovery import (
    validate_complete_lpips,
    validate_incomplete_lpips,
    without_lpips,
)
from research.local_route1.candidate_runner import validate_matched_plain_late_metrics


def _metric(*, available: bool) -> dict:
    lpips = 0.2 if available else None
    return {
        "schema": "local-route1-discovery70-crn-single-rollout-v1",
        "split": "discovery",
        "count_per_domain": 70,
        "replicates": 1,
        "probe_id": "plain",
        "epoch": 150,
        "updates": 22500,
        "data_epoch": 150,
        "lpips_requested": True,
        "lpips_available": available,
        "macro_lpips": lpips,
        "macro_psnr": 17.0,
        "confirmation20_opened": False,
        "domains": {
            f"d{index}": {"n": 70, "psnr": 17.0, "ssim": 0.5, "lpips": lpips}
            for index in range(6)
        },
        "images": [
            {"domain": f"d{index % 6}", "psnr": 17.0, "ssim": 0.5, "lpips": lpips}
            for index in range(420)
        ],
    }


def test_lpips_projection_preserves_and_compares_all_other_fields() -> None:
    missing = _metric(available=False)
    complete = _metric(available=True)
    assert without_lpips(missing) == without_lpips(complete)
    complete["images"][0]["psnr"] = 18.0
    assert without_lpips(missing) != without_lpips(complete)


def test_incomplete_and_complete_lpips_shapes_are_fail_closed() -> None:
    validate_incomplete_lpips(_metric(available=False), lane="plain", epoch=150)
    validate_complete_lpips(_metric(available=True))
    broken = copy.deepcopy(_metric(available=False))
    broken["images"][0]["lpips"] = 0.2
    with pytest.raises(RuntimeError, match="420 nulls"):
        validate_incomplete_lpips(broken, lane="plain", epoch=150)
    broken = copy.deepcopy(_metric(available=True))
    broken["domains"]["d0"]["lpips"] = None
    with pytest.raises(RuntimeError, match="domain payload"):
        validate_complete_lpips(broken)


def test_candidate_admission_rejects_missing_plain_late_lpips(tmp_path) -> None:
    root = tmp_path
    metrics = root / "anchors" / "plain" / "metrics"
    metrics.mkdir(parents=True)
    for epoch in (150, 175, 200):
        payload = _metric(available=True)
        payload.update({"epoch": epoch, "updates": epoch * 150, "data_epoch": epoch})
        (metrics / f"e{epoch:03d}.json").write_text(
            json.dumps(payload), encoding="utf-8",
        )
    hashes = validate_matched_plain_late_metrics(root)
    assert set(hashes) == {150, 175, 200}
    broken = _metric(available=False)
    broken.update({"epoch": 175, "updates": 26250, "data_epoch": 175})
    (metrics / "e175.json").write_text(
        json.dumps(broken), encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="LPIPS authority is unavailable at e175"):
        validate_matched_plain_late_metrics(root)
