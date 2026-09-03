from __future__ import annotations

import pytest

from operations.paper_aio_host_identity_gate import (
    REGISTRY_SCHEMA,
    classify_identity,
    validate_registry,
)


def _registry() -> dict:
    return {
        "schema": REGISTRY_SCHEMA,
        "hosts": {
            "5090A": {"gpu_uuid": "GPU-a", "hostname": "host-a"},
            "5090B": {"gpu_uuid": "GPU-b", "hostname": "host-b"},
        },
    }


def test_registered_host_identity_matches() -> None:
    result = classify_identity(
        requested_label="5090A", identity={"gpu_uuid": "GPU-a"},
        registry=_registry(),
    )
    assert result["outcome"] == "REGISTERED_HOST_MATCH"
    assert result["registered_label"] == "5090A"
    assert result["long_training_launch_allowed_by_identity_gate"] is True


def test_endpoint_alias_is_not_counted_as_a_new_gpu() -> None:
    result = classify_identity(
        requested_label="5090D", identity={"gpu_uuid": "GPU-b"},
        registry=_registry(),
    )
    assert result["outcome"] == "DUPLICATE_ENDPOINT_OF_REGISTERED_HOST"
    assert result["registered_label"] == "5090B"
    assert result["physical_gpu_count_may_increase"] is False
    assert result["long_training_launch_allowed_by_identity_gate"] is False


def test_existing_label_cannot_silently_change_physical_gpu() -> None:
    result = classify_identity(
        requested_label="5090A", identity={"gpu_uuid": "GPU-new"},
        registry=_registry(),
    )
    assert result["outcome"] == "LABEL_COLLISION_DIFFERENT_GPU"
    assert result["long_training_launch_allowed_by_identity_gate"] is False


def test_unseen_label_and_uuid_are_a_new_candidate() -> None:
    result = classify_identity(
        requested_label="5090D", identity={"gpu_uuid": "GPU-new"},
        registry=_registry(),
    )
    assert result["outcome"] == "NEW_PHYSICAL_GPU_CANDIDATE"
    assert result["physical_gpu_count_may_increase"] is True
    assert result["long_training_launch_allowed_by_identity_gate"] is True


def test_registry_rejects_duplicate_gpu_uuid() -> None:
    registry = _registry()
    registry["hosts"]["5090C"] = {"gpu_uuid": "GPU-a"}
    with pytest.raises(RuntimeError, match="duplicates GPU UUID"):
        validate_registry(registry)
