"""Target-blind, checkpoint-read-only paper complexity profiling."""

from __future__ import annotations

import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from production.metrics import build_rollout_bundle
from research.local_route1.runtime import capture_rng, restore_rng, write_json

from .evaluate import _prediction, read_image
from .gates import environment_record
from .protocol import (
    LaneSpec,
    evaluation_bundle_fingerprint,
    file_sha256,
    load_protocol,
    protocol_fingerprint,
)
from .runtime import optimizer_step


SCHEMA = "final-unsb-paper-complexity-profile-v1"
INFERENCE_WARMUP = 5
INFERENCE_REPEATS = 30
TRAIN_WARMUP = 3
TRAIN_REPEATS = 10


def summarize_milliseconds(samples: list[float]) -> dict[str, float | int]:
    if not samples or any(value < 0 for value in samples):
        raise ValueError("latency samples must be nonempty and nonnegative")
    ordered = sorted(float(value) for value in samples)

    def percentile(fraction: float) -> float:
        index = int(round(fraction * (len(ordered) - 1)))
        return ordered[index]

    return {
        "repeats": len(ordered),
        "median_ms": float(statistics.median(ordered)),
        "mean_ms": float(statistics.fmean(ordered)),
        "p10_ms": percentile(0.10),
        "p90_ms": percentile(0.90),
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
    }


def parameter_inventory(model) -> dict[str, Any]:
    networks: dict[str, Any] = {}
    optimizer_owned = {
        id(parameter)
        for optimizer in model.optimizers
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    seen: set[int] = set()
    total = 0
    owned_total = 0
    requires_grad_total = 0
    bytes_total = 0
    for name in model.model_names:
        network = getattr(model, "net" + name)
        parameters = list(network.parameters())
        network_total = sum(parameter.numel() for parameter in parameters)
        network_owned = sum(
            parameter.numel() for parameter in parameters
            if id(parameter) in optimizer_owned
        )
        network_requires_grad = sum(
            parameter.numel() for parameter in parameters
            if parameter.requires_grad
        )
        network_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        )
        networks[str(name)] = {
            "parameters": int(network_total),
            "optimizer_owned_parameters": int(network_owned),
            "requires_grad_parameters_at_measurement": int(network_requires_grad),
            "parameter_bytes": int(network_bytes),
        }
        for parameter in parameters:
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            total += parameter.numel()
            owned_total += parameter.numel() if identity in optimizer_owned else 0
            requires_grad_total += parameter.numel() if parameter.requires_grad else 0
            bytes_total += parameter.numel() * parameter.element_size()
    return {
        "networks": networks,
        "unique_parameters": int(total),
        "unique_optimizer_owned_parameters": int(owned_total),
        "unique_requires_grad_parameters_at_measurement": int(requires_grad_total),
        "unique_parameter_bytes": int(bytes_total),
    }


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_calls(
    call: Callable[[], Any], *, device: torch.device, warmup: int, repeats: int,
) -> list[float]:
    for _ in range(int(warmup)):
        call()
    _synchronize(device)
    samples = []
    for _ in range(int(repeats)):
        started = time.perf_counter()
        call()
        _synchronize(device)
        samples.append((time.perf_counter() - started) * 1000.0)
    return samples


def profile_model(
    *, model, spec: LaneSpec, rows: list[dict], primary, secondary,
    data_root: Path, checkpoint: Path, checkpoint_metadata: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    """Profile a disposable loaded model without reading a paired target."""
    destination = Path(destination).resolve()
    if destination.exists():
        raise RuntimeError(f"complexity receipt already exists: {destination}")
    checkpoint = Path(checkpoint).resolve()
    source_hash_before = file_sha256(checkpoint)
    protocol = load_protocol()
    candidates = sorted(
        (row for row in rows if row.get("split") == "train"),
        key=lambda row: (str(row.get("domain")), int(row.get("order", 0))),
    )
    if not candidates:
        raise RuntimeError("complexity profiling requires at least one training row")
    row = candidates[0]
    source = read_image(Path(data_root) / row["input_relpath"]).to(model.device)
    latent_dim = 4 * int(getattr(model.opt, "ngf", 64))
    bundle = build_rollout_bundle(
        protocol_hash=evaluation_bundle_fingerprint(), domain=str(row["domain"]),
        stem=str(row["stem"]), replicate=0, latent_dim=latent_dim,
        height=128, width=128, num_timesteps=5,
    )
    nfe_values = list(range(1, 6)) if spec.family == "unsb" else [1]
    saved_rng = capture_rng()
    saved_modes = {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    inference: dict[str, Any] = {}
    training: dict[str, Any]
    try:
        model.eval()
        with torch.inference_mode():
            for nfe in nfe_values:
                if model.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(model.device)
                samples = _time_calls(
                    lambda nfe=nfe: _prediction(
                        model, spec, source, bundle, nfe=int(nfe),
                    ),
                    device=model.device, warmup=INFERENCE_WARMUP,
                    repeats=INFERENCE_REPEATS,
                )
                inference[str(nfe)] = {
                    **summarize_milliseconds(samples),
                    "peak_allocated_bytes": (
                        int(torch.cuda.max_memory_allocated(model.device))
                        if model.device.type == "cuda" else None
                    ),
                    "peak_reserved_bytes": (
                        int(torch.cuda.max_memory_reserved(model.device))
                        if model.device.type == "cuda" else None
                    ),
                }

        for name in model.model_names:
            getattr(model, "net" + name).train()
        if model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(model.device)
        train_samples = _time_calls(
            lambda: optimizer_step(model, spec, primary, secondary),
            device=model.device, warmup=TRAIN_WARMUP, repeats=TRAIN_REPEATS,
        )
        training = {
            **summarize_milliseconds(train_samples),
            "warmup_steps": TRAIN_WARMUP,
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(model.device))
                if model.device.type == "cuda" else None
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(model.device))
                if model.device.type == "cuda" else None
            ),
        }
        parameters = parameter_inventory(model)
    finally:
        for name, was_training in saved_modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)

    source_hash_after = file_sha256(checkpoint)
    if source_hash_after != source_hash_before:
        raise RuntimeError("complexity profiling changed the source checkpoint")
    result = {
        "schema": SCHEMA,
        "status": "PASS_TARGET_BLIND_CHECKPOINT_READ_ONLY_PROFILE",
        "protocol_fingerprint": protocol_fingerprint(),
        "evaluation_bundle_fingerprint": evaluation_bundle_fingerprint(protocol),
        "lane_id": spec.id,
        "lane": spec.to_dict(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_hash_before,
        "checkpoint_unchanged": True,
        "checkpoint_metadata": dict(checkpoint_metadata),
        "environment": environment_record(),
        "image_shape": [1, 3, 128, 128],
        "batch_size": int(protocol["training"]["batch_size"]),
        "parameters": parameters,
        "inference": {
            "warmup_repeats": INFERENCE_WARMUP,
            "measured_repeats": INFERENCE_REPEATS,
            "nfe": inference,
        },
        "training_step": training,
        "flops": {
            "reported": False,
            "reason": (
                "custom stochastic bridge and lazy PatchNCE operators are not "
                "fully covered by a single audited FLOP counter; measured latency, "
                "NFE and peak memory are reported instead"
            ),
        },
        "source_input": {
            "domain": str(row["domain"]),
            "stem": str(row["stem"]),
            "target_path_read": False,
        },
        "disposable_in_memory_model_mutated_for_training_timing": True,
        "performance_metric_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(destination, result)
    return result
