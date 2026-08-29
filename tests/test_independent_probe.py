from __future__ import annotations

import pytest

from operations.local_route1_independent_probe import (
    EXPECTED_E0_FILE,
    EXPECTED_E0_SCIENTIFIC,
    EXPECTED_MANIFEST,
    EXPECTED_PROTOCOL,
    EXPECTED_TRAINING_COMMIT,
    validate_e0_sidecar,
    validate_plain_sidecar,
)


def test_independent_probe_accepts_only_exact_plain_e200_identity():
    sidecar = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "step": 30_000,
        "physical_epoch_completed": 200,
        "metadata": {
            "probe_id": "plain",
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": False,
        },
    }
    validate_plain_sidecar(sidecar)
    sidecar["step"] = 29_999
    with pytest.raises(RuntimeError, match="plain e200"):
        validate_plain_sidecar(sidecar)


def test_independent_probe_rejects_confirmation_unlock():
    sidecar = {
        "schema": "final-unsb-local-route1-full-state-v1",
        "step": 30_000,
        "physical_epoch_completed": 200,
        "metadata": {
            "probe_id": "plain",
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "confirmation20_opened": True,
        },
    }
    with pytest.raises(RuntimeError, match="confirmation20_opened"):
        validate_plain_sidecar(sidecar)


def test_independent_probe_requires_exact_shared_e0():
    sidecar = {
        "schema": "final-unsb-local-route1-shared-e0-v1",
        "checkpoint_sha256": EXPECTED_E0_FILE,
        "scientific_state_sha256": EXPECTED_E0_SCIENTIFIC,
        "metadata": {
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
        },
    }
    validate_e0_sidecar(sidecar)
    sidecar["checkpoint_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="file identity"):
        validate_e0_sidecar(sidecar)
