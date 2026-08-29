from __future__ import annotations

import pytest

from operations.local_route1_verify_milestone import (
    DOMAINS,
    EXPECTED_CRN,
    EXPECTED_PROTOCOL,
    validate_metric,
)


def metric_payload() -> dict:
    images = []
    domains = {}
    for domain in DOMAINS:
        domains[domain] = {"n": 70, "psnr": 20.0, "ssim": 0.7, "lpips": 0.2}
        images.extend({
            "domain": domain,
            "stem": f"{index:04d}",
            "crn_bundle_sha256": f"crn-{domain}-{index}",
        } for index in range(70))
    return {
        "schema": "local-route1-discovery70-crn-single-rollout-v1",
        "split": "discovery",
        "count_per_domain": 70,
        "replicates": 1,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "evaluation_input_sha256": EXPECTED_CRN,
        "confirmation20_opened": False,
        "probe_id": "plain",
        "epoch": 200,
        "updates": 30_000,
        "data_epoch": 200,
        "lpips_requested": True,
        "lpips_available": True,
        "macro_lpips": 0.2,
        "images": images,
        "domains": domains,
    }


def test_milestone_metric_requires_exact_discovery70_and_lpips():
    payload = metric_payload()
    result = validate_metric(payload, lane="plain", epoch=200, require_lpips=True)
    assert set(result) == set(DOMAINS)
    payload["images"].pop()
    with pytest.raises(RuntimeError, match="420 images"):
        validate_metric(payload, lane="plain", epoch=200, require_lpips=True)


def test_milestone_metric_rejects_confirmation_and_wrong_crn():
    payload = metric_payload()
    payload["confirmation20_opened"] = True
    with pytest.raises(RuntimeError, match="confirmation20_opened"):
        validate_metric(payload, lane="plain", epoch=200, require_lpips=True)
    payload = metric_payload()
    payload["evaluation_input_sha256"] = "wrong"
    with pytest.raises(RuntimeError, match="evaluation_input_sha256"):
        validate_metric(payload, lane="plain", epoch=200, require_lpips=True)


def test_milestone_metric_rejects_missing_lpips():
    payload = metric_payload()
    payload["lpips_available"] = False
    payload["macro_lpips"] = None
    with pytest.raises(RuntimeError, match="LPIPS"):
        validate_metric(payload, lane="plain", epoch=200, require_lpips=True)
