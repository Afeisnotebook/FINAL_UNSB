"""Source-bound checkpoint export and one-container paper evaluation cohort."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from research.local_route1.runtime import full_state_hash, write_json

from .candidate_runtime import load_candidate_spec
from .evaluate import evaluate_model
from .gates import environment_record
from .protocol import (
    EVALUATION_SCHEMA,
    FULL_STATE_SCHEMA,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    LaneSpec,
    file_sha256,
    lane_spec,
    load_protocol,
    object_sha256,
    protocol_fingerprint,
)
from .runtime import load_full_state, prepare_lane


EXPORT_SCHEMA = "final-unsb-paper-checkpoint-export-v1"
UNIFIED_RECEIPT_SCHEMA = "final-unsb-paper-unified-evaluation-receipt-v1"
UNIFIED_COHORT_SCHEMA = "final-unsb-paper-unified-evaluation-cohort-v1"
UNIFIED_EPOCHS = (100, 125, 150, 175, 200)
REQUIRED_FIRST_WAVE = ("plain", "proposal", "cut", "cyclegan")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _expected_step(epoch: int) -> int:
    if int(epoch) not in UNIFIED_EPOCHS:
        raise RuntimeError(f"unified paper evaluation epoch is not frozen: {epoch}")
    return int(epoch) * 8553


def export_checkpoint_receipt(
    *, checkpoint: Path, sidecar: Path, lane_id: str, epoch: int,
    host_label: str, destination: Path,
) -> dict[str, Any]:
    """Bind one immutable training checkpoint without copying or evaluating it."""
    checkpoint = Path(checkpoint).resolve()
    sidecar = Path(sidecar).resolve()
    if not checkpoint.is_file() or not sidecar.is_file():
        raise RuntimeError("checkpoint export requires checkpoint and sidecar files")
    metadata = _read_json(sidecar)
    expected_step = _expected_step(epoch)
    if (
        metadata.get("schema") != FULL_STATE_SCHEMA
        or metadata.get("lane_id") != lane_id
        or int(metadata.get("step", -1)) != expected_step
        or int(metadata.get("physical_epoch_completed", -1)) != int(epoch)
    ):
        raise RuntimeError("checkpoint export sidecar identity mismatch")
    if file_sha256(checkpoint) != metadata.get("full_state_sha256"):
        raise RuntimeError("checkpoint export file hash mismatch")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != FULL_STATE_SCHEMA
        or payload.get("lane", {}).get("id") != lane_id
        or int(payload.get("step", -1)) != expected_step
        or full_state_hash(payload) != metadata.get("scientific_state_sha256")
    ):
        raise RuntimeError("checkpoint export scientific state mismatch")
    training = payload.get("metadata", {})
    if training.get("paired_controller_access") is not False:
        raise RuntimeError("checkpoint export reports paired controller access")
    if training.get("confirmation20_opened") is not False:
        raise RuntimeError("checkpoint export opened confirmation20")
    result = {
        "schema": EXPORT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT",
        "lane_id": lane_id,
        "epoch": int(epoch),
        "updates": expected_step,
        "source_host_label": str(host_label),
        "source_checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "source_sidecar": str(sidecar),
        "sidecar_sha256": file_sha256(sidecar),
        "scientific_state_sha256": metadata["scientific_state_sha256"],
        "lane": payload["lane"],
        "training_git_commit": training.get("git_commit"),
        "training_protocol_fingerprint": training.get("protocol_fingerprint"),
        "manifest_sha256": training.get("manifest_sha256"),
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(Path(destination).resolve(), result)
    return result


def _spec_from_export(
    *, output_root: Path, lane: dict[str, Any], candidate_id: str | None,
) -> LaneSpec:
    lane_id = str(lane.get("id", ""))
    if candidate_id is None:
        spec = lane_spec(lane_id)
    else:
        if lane_id != candidate_id:
            raise RuntimeError("candidate export lane id differs from --candidate-id")
        spec, _ = load_candidate_spec(output_root, candidate_id)
    if lane != spec.to_dict():
        raise RuntimeError("imported checkpoint lane semantics differ from current evaluator")
    return spec


def evaluate_imported_checkpoint(
    *, output_root: Path, export_receipt: Path, copied_checkpoint: Path,
    train_view: Path, data_root: Path, manifest_path: Path, gpu: int,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    """Evaluate a copied model read-only inside one common runtime."""
    output_root = Path(output_root).resolve()
    export_path = Path(export_receipt).resolve()
    checkpoint = Path(copied_checkpoint).resolve()
    if not export_path.is_file() or not checkpoint.is_file():
        raise RuntimeError("unified evaluation requires export receipt and copied checkpoint")
    export = _read_json(export_path)
    if (
        export.get("schema") != EXPORT_SCHEMA
        or export.get("status") != "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT"
        or export.get("paired_metric_control") is not False
        or export.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("unified checkpoint export receipt is invalid")
    if file_sha256(checkpoint) != export.get("checkpoint_sha256"):
        raise RuntimeError("copied unified checkpoint differs from source export")
    lane_id = str(export["lane_id"])
    epoch = int(export["epoch"])
    spec = _spec_from_export(
        output_root=output_root, lane=export["lane"], candidate_id=candidate_id,
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (
        payload.get("schema") != FULL_STATE_SCHEMA
        or payload.get("lane") != export["lane"]
        or int(payload.get("step", -1)) != _expected_step(epoch)
        or full_state_hash(payload) != export.get("scientific_state_sha256")
    ):
        raise RuntimeError("copied unified checkpoint scientific identity mismatch")
    model, primary, secondary, rows = prepare_lane(
        output_root=output_root / "unified_runtime" / lane_id,
        train_view=Path(train_view).resolve(), manifest_path=Path(manifest_path).resolve(),
        spec=spec, gpu=int(gpu),
    )
    load_full_state(
        checkpoint, model=model, spec=spec, primary=primary, secondary=secondary,
        expected_metadata=payload["metadata"],
    )
    protocol = load_protocol()
    terminal = epoch == 200
    nfe_values = (
        list(protocol["evaluation"]["reported_unsb_nfes"])
        if spec.family == "unsb" and epoch in protocol["training"]["nfe_epochs"]
        else [5 if spec.family == "unsb" else 1]
    )
    result = evaluate_model(
        model=model, spec=spec, rows=rows, data_root=Path(data_root).resolve(),
        protocol_hash=FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        count_per_domain=(80 if terminal else 70),
        replicates=(5 if terminal else 1), nfe_values=nfe_values,
        include_lpips=epoch in protocol["training"]["lpips_epochs"],
    )
    result.update({
        "epoch": epoch,
        "updates": _expected_step(epoch),
        "training_protocol_fingerprint": export.get("training_protocol_fingerprint"),
        "unified_evaluator_protocol_fingerprint": protocol_fingerprint(manifest_path),
        "source_export_receipt_sha256": file_sha256(export_path),
        "source_checkpoint_sha256": export["checkpoint_sha256"],
        "source_host_label": export["source_host_label"],
        "unified_environment": environment_record(),
        "training_checkpoint_read_only": True,
        "cross_host_training_delta_merged": False,
    })
    metric_path = output_root / "lanes" / lane_id / "metrics" / f"e{epoch:03d}.json"
    if metric_path.is_file():
        existing = _read_json(metric_path)
        if object_sha256(existing) != object_sha256(result):
            raise RuntimeError(
                f"unified metric already exists with different evidence: {metric_path}"
            )
    else:
        write_json(metric_path, result)
    receipt = {
        "schema": UNIFIED_RECEIPT_SCHEMA,
        "status": "PASS_UNIFIED_READ_ONLY_EVALUATION",
        "lane_id": lane_id,
        "epoch": epoch,
        "source_host_label": export["source_host_label"],
        "source_export_receipt": str(export_path),
        "source_export_receipt_sha256": file_sha256(export_path),
        "source_checkpoint_sha256": export["checkpoint_sha256"],
        "metric": str(metric_path.resolve()),
        "metric_sha256": file_sha256(metric_path),
        "evaluation_schema": EVALUATION_SCHEMA,
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "unified_evaluator_protocol_fingerprint": protocol_fingerprint(manifest_path),
        "unified_environment": environment_record(),
        "training_checkpoint_read_only": True,
        "paired_metric_control": False,
        "cross_host_training_delta_merged": False,
        "confirmation20_opened": False,
    }
    receipt_path = output_root / "gates" / f"UNIFIED_EVALUATION_{lane_id}_e{epoch:03d}.json"
    if receipt_path.is_file():
        existing = _read_json(receipt_path)
        if object_sha256(existing) != object_sha256(receipt):
            raise RuntimeError(
                f"unified evaluation receipt already exists and differs: {receipt_path}"
            )
    else:
        write_json(receipt_path, receipt)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return receipt


def _expected_evaluation(epoch: int, family: str) -> dict[str, Any]:
    return {
        "count_per_domain": 80 if epoch == 200 else 70,
        "replicates": 5 if epoch == 200 else 1,
        "nfe_values": (
            [1, 2, 3, 4, 5]
            if family == "unsb" and epoch in (100, 150, 200)
            else [5 if family == "unsb" else 1]
        ),
    }


def lock_unified_evaluation_cohort(output_root: Path) -> dict[str, Any]:
    """Require every first-wave model in one evaluator before adjudication."""
    output_root = Path(output_root).resolve()
    receipts = []
    environment = None
    evaluator_fingerprint = None
    for lane_id in REQUIRED_FIRST_WAVE:
        family = lane_spec(lane_id).family
        for epoch in UNIFIED_EPOCHS:
            receipt_path = output_root / "gates" / f"UNIFIED_EVALUATION_{lane_id}_e{epoch:03d}.json"
            if not receipt_path.is_file():
                raise RuntimeError(f"unified first-wave receipt is missing: {receipt_path}")
            receipt = _read_json(receipt_path)
            metric_path = Path(receipt.get("metric", ""))
            if (
                receipt.get("schema") != UNIFIED_RECEIPT_SCHEMA
                or receipt.get("status") != "PASS_UNIFIED_READ_ONLY_EVALUATION"
                or receipt.get("lane_id") != lane_id
                or int(receipt.get("epoch", -1)) != epoch
                or not metric_path.is_file()
                or file_sha256(metric_path) != receipt.get("metric_sha256")
                or receipt.get("evaluation_bundle_fingerprint")
                != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
                or receipt.get("training_checkpoint_read_only") is not True
                or receipt.get("paired_metric_control") is not False
                or receipt.get("cross_host_training_delta_merged") is not False
                or receipt.get("confirmation20_opened") is not False
            ):
                raise RuntimeError(f"invalid unified evaluation receipt: {receipt_path}")
            metric = _read_json(metric_path)
            expected = _expected_evaluation(epoch, family)
            if any(metric.get(key) != value for key, value in expected.items()):
                raise RuntimeError(f"unified metric protocol mismatch: {lane_id} e{epoch}")
            if metric.get("protocol_fingerprint") != FROZEN_EVALUATION_BUNDLE_FINGERPRINT:
                raise RuntimeError("unified metric CRN bundle identity changed")
            if (
                metric.get("training_checkpoint_read_only") is not True
                or metric.get("cross_host_training_delta_merged") is not False
                or metric.get("confirmation20_opened") is not False
            ):
                raise RuntimeError("unified metric violates read-only or sealed-evaluation policy")
            current_environment = receipt.get("unified_environment")
            current_evaluator = receipt.get("unified_evaluator_protocol_fingerprint")
            if environment is None:
                environment = current_environment
                evaluator_fingerprint = current_evaluator
            if current_environment != environment or current_evaluator != evaluator_fingerprint:
                raise RuntimeError("unified evaluations did not use one runtime and evaluator")
            receipts.append({
                "lane_id": lane_id, "epoch": epoch,
                "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path),
                "source_host_label": receipt.get("source_host_label"),
            })
    if evaluator_fingerprint != protocol_fingerprint():
        raise RuntimeError("unified evaluator fingerprint is stale for the current checkout")
    if environment != environment_record():
        raise RuntimeError("unified cohort lock must be issued inside the evaluation runtime")
    result = {
        "schema": UNIFIED_COHORT_SCHEMA,
        "status": "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT",
        "required_lanes": list(REQUIRED_FIRST_WAVE),
        "epochs": list(UNIFIED_EPOCHS),
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "unified_evaluator_protocol_fingerprint": evaluator_fingerprint,
        "unified_environment": environment,
        "receipts": receipts,
        "training_hosts_remain_separate": True,
        "cross_host_training_delta_merged": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "gates" / "UNIFIED_EVALUATION_COHORT.json", result)
    return result
