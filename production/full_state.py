"""Atomic epoch-boundary full-state checkpointing for one lane."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from production import common


def assert_finite_tensors(value, path: str = "state") -> None:
    if torch.is_tensor(value):
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite tensor in {path}")
        return
    if isinstance(value, dict):
        for key, child in value.items():
            assert_finite_tensors(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            assert_finite_tensors(child, f"{path}[{index}]")


def capture(model, *, metadata: dict) -> dict:
    networks = {}
    for name in model.model_names:
        if isinstance(name, str):
            networks[name] = {
                key: value.detach().cpu()
                for key, value in common.unwrap(getattr(model, "net" + name)).state_dict().items()
            }
    payload = {
        "schema_version": 1,
        "metadata": dict(metadata),
        "networks": networks,
        "optimizers": [optimizer.state_dict() for optimizer in model.optimizers],
        "schedulers": [scheduler.state_dict() for scheduler in model.schedulers],
        "rng": common.capture_rng_state(),
        "method_state": model.get_extra_training_state(),
    }
    assert_finite_tensors(payload["networks"], "networks")
    assert_finite_tensors(payload["optimizers"], "optimizers")
    return payload


def save(path: Path, model, *, metadata: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = capture(model, metadata=metadata)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)
    sidecar = {
        **metadata,
        "checkpoint": path.name,
        "checkpoint_sha256": common.file_sha256(path),
        "network_state_sha256": common.state_tensor_sha256(
            {name: getattr(model, "net" + name) for name in model.model_names}
        ),
    }
    common.atomic_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def load(path: Path, model, *, expected: dict) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(
                f"checkpoint identity mismatch for {key}: {metadata.get(key)!r} != {value!r}"
            )
    for name, state in payload["networks"].items():
        common.unwrap(getattr(model, "net" + name)).load_state_dict(state, strict=True)
    if len(payload["optimizers"]) != len(model.optimizers):
        raise RuntimeError("optimizer count mismatch")
    if len(payload["schedulers"]) != len(model.schedulers):
        raise RuntimeError("scheduler count mismatch")
    for optimizer, state in zip(model.optimizers, payload["optimizers"]):
        optimizer.load_state_dict(state)
    for scheduler, state in zip(model.schedulers, payload["schedulers"]):
        scheduler.load_state_dict(state)
    model.load_extra_training_state(payload.get("method_state", {}))
    common.restore_rng_state(payload["rng"])
    return metadata
