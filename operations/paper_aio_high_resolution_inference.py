"""Post-freeze 256px inference sensitivity for fixed paper checkpoints.

This supplementary path is deliberately outside the scientific training
fingerprint.  It cannot run before the committed algorithm/baseline/claim
freeze, evaluates only the fixed e200 checkpoint, never retains images, and
cannot be used to select a lane or checkpoint.  The controlled 128px table
remains primary; this module asks only whether its conclusions survive a
single preregistered 256px full-convolution inference setting.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from production.metrics import (
    METRIC_SEMANTICS,
    build_rollout_bundle,
    bundle_hash,
    psnr_unit,
    ssim_unit,
    to_unit,
)
from research.local_route1.runtime import capture_rng, restore_rng, write_json
from research.paper_aio.distribution import committed_freeze_identity
from research.paper_aio.evaluate import (
    _lpips,
    _prediction,
    aggregate_metric_rows,
    evaluation_input_hash,
    read_image,
    replicate_stochasticity,
    select_discovery,
)
from research.paper_aio.gates import environment_record
from research.paper_aio.protocol import (
    LaneSpec,
    evaluation_bundle_fingerprint,
    file_sha256,
    git_commit,
    lane_spec,
    load_protocol,
    object_sha256,
    protocol_fingerprint,
)
from research.paper_aio.runtime import (
    _annotated_rows,
    load_full_state,
    manifest_report,
    prepare_lane,
)
from research.paper_aio.unified import candidate_spec_from_portable_authority


SCHEMA = "final-unsb-paper-post-freeze-high-resolution-v1"
COHORT_SCHEMA = "final-unsb-paper-post-freeze-high-resolution-cohort-v1"
STATUS = "PASS_POST_FREEZE_DISCOVERY80_256PX_FULL_CONVOLUTION"
COHORT_STATUS = "PASS_COMPLETE_FROZEN_256PX_SUPPLEMENTARY_COHORT"
IMAGE_SIZE = 256
COUNT_PER_DOMAIN = 80
UNSB_REPLICATES = 5
NUM_TIMESTEPS = 5
SCRIPT_PATH = Path(__file__).resolve()
SPATIAL_POLICY = {
    "schema": "final-unsb-paper-high-resolution-spatial-policy-v1",
    "training_resolution": 128,
    "inference_resolution": IMAGE_SIZE,
    "source_and_target_resize": "PIL_RGB_bicubic_square_256",
    "execution": "whole_image_fully_convolutional",
    "tiling": False,
    "retraining": False,
    "checkpoint_selection": False,
    "interpretation": (
        "supplementary fixed-resolution sensitivity; not native-resolution "
        "restoration and not part of the controlled 128px primary table"
    ),
}
HIGH_RES_BUNDLE_FINGERPRINT = object_sha256({
    "base_bundle": evaluation_bundle_fingerprint(),
    "spatial_policy": SPATIAL_POLICY,
    "namespace": "FINAL_UNSB_HIGH_RES_256_V1",
})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _source_identity() -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "script": SCRIPT_PATH.relative_to(SCRIPT_PATH.parents[1]).as_posix(),
        "script_sha256": file_sha256(SCRIPT_PATH),
    }


def _metric_semantics() -> dict[str, Any]:
    result = dict(METRIC_SEMANTICS)
    result.update({
        "base_schema": METRIC_SEMANTICS["schema"],
        "schema": "final-unsb-paper-high-resolution-metric-semantics-v1",
        "spatial_policy": SPATIAL_POLICY,
    })
    return result


def _validate_e200_checkpoint(
    *, model, checkpoint: Path | None, checkpoint_step: int | None,
    checkpoint_metadata: dict[str, Any] | None,
) -> tuple[Path | None, str | None]:
    if model is None:
        if checkpoint is not None or checkpoint_step is not None or checkpoint_metadata is not None:
            raise RuntimeError("Input supplementary evaluation cannot name a checkpoint")
        return None, None
    protocol = load_protocol()
    if (
        checkpoint is None
        or not Path(checkpoint).resolve().is_file()
        or int(checkpoint_step or -1) != int(protocol["training"]["target_updates"])
        or checkpoint_metadata is None
        or checkpoint_metadata.get("paired_controller_access") is not False
        or checkpoint_metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("high-resolution inference requires a safe fixed e200 checkpoint")
    path = Path(checkpoint).resolve()
    return path, file_sha256(path)


def validate_receipt(value: dict[str, Any], *, expected_lane: str | None = None) -> None:
    lane = str(value.get("lane_id", ""))
    lane_spec_value = value.get("lane")
    family = "input" if lane == "input" else str(
        (lane_spec_value or {}).get("family", "")
    )
    images = value.get("images")
    replicates = int(value.get("replicate_count", -1))
    expected_replicates = UNSB_REPLICATES if family == "unsb" else 1
    expected_nfe = 0 if lane == "input" else (NUM_TIMESTEPS if family == "unsb" else 1)
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != STATUS
        or not lane
        or (expected_lane is not None and lane != expected_lane)
        or int(value.get("primary_epoch", -1)) != 200
        or int(value.get("image_size", -1)) != IMAGE_SIZE
        or int(value.get("count_per_domain", -1)) != COUNT_PER_DOMAIN
        or int(value.get("domain_count", -1)) != 6
        or replicates != expected_replicates
        or int(value.get("primary_nfe", -1)) != expected_nfe
        or (lane == "input" and lane_spec_value is not None)
        or (lane != "input" and not isinstance(lane_spec_value, dict))
        or value.get("spatial_policy") != SPATIAL_POLICY
        or value.get("high_resolution_bundle_fingerprint")
        != HIGH_RES_BUNDLE_FINGERPRINT
        or value.get("supplementary_only") is not True
        or value.get("main_table") is not False
        or value.get("checkpoint_unchanged") is not True
        or value.get("generated_images_retained") is not False
        or value.get("target_path_read_only_after_committed_freeze") is not True
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("confirmation20_opened") is not False
        or not isinstance(images, list)
        or len(images) != 6 * COUNT_PER_DOMAIN * replicates
    ):
        raise RuntimeError("invalid post-freeze high-resolution receipt")
    seen = set()
    domains = set()
    identities: dict[str, set[tuple[str, int]]] = {}
    replicate_counts: dict[tuple[str, int], int] = {}
    for row in images:
        try:
            key = (
                str(row["domain"]), str(row["stem"]), int(row["order"]),
                int(row["replicate"]), int(row["nfe"]),
            )
            psnr = float(row["psnr"])
            ssim = float(row["ssim"])
            lpips = float(row["lpips"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("malformed high-resolution image evidence") from error
        if key in seen or not torch.isfinite(torch.tensor([psnr, ssim, lpips])).all():
            raise RuntimeError("duplicate or non-finite high-resolution image evidence")
        if psnr > 120.0 + 1e-9 or not -1.01 <= ssim <= 1.01:
            raise RuntimeError("high-resolution metric is outside its numerical range")
        if not 0 <= key[3] < replicates:
            raise RuntimeError("high-resolution replicate is outside the fixed range")
        seen.add(key)
        domains.add(key[0])
        identities.setdefault(key[0], set()).add((key[1], key[2]))
        replicate_key = (key[0], key[3])
        replicate_counts[replicate_key] = replicate_counts.get(replicate_key, 0) + 1
        if key[4] != expected_nfe:
            raise RuntimeError("high-resolution image evidence changes fixed NFE")
        digest = row.get("crn_bundle_sha256")
        if lane == "input":
            if digest is not None:
                raise RuntimeError("Input high-resolution row cannot claim CRN")
        elif not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("high-resolution model row lacks CRN identity")
    if len(domains) != 6:
        raise RuntimeError("high-resolution receipt does not cover six domains")
    for domain in domains:
        if (
            len(identities.get(domain, set())) != COUNT_PER_DOMAIN
            or {order for _, order in identities[domain]} != set(range(COUNT_PER_DOMAIN))
        ):
            raise RuntimeError("high-resolution receipt has incomplete discovery identities")
        for replicate in range(replicates):
            if replicate_counts.get((domain, replicate)) != COUNT_PER_DOMAIN:
                raise RuntimeError("high-resolution receipt has an incomplete replicate cell")


def _verify_selected_content(
    selected: list[dict[str, Any]], *, data_root: Path,
) -> dict[str, Any]:
    """Bind every paired metric input to the already frozen manifest bytes."""
    checked = 0
    root = Path(data_root).resolve()
    for row in selected:
        for rel_key, bytes_key, hash_key in (
            ("input_relpath", "input_bytes", "input_sha256"),
            ("target_relpath", "target_bytes", "target_sha256"),
        ):
            path = root / str(row[rel_key])
            try:
                expected_bytes = int(row[bytes_key])
                expected_hash = str(row[hash_key])
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError("high-resolution manifest row lacks content identity") from error
            if (
                not path.is_file()
                or path.stat().st_size != expected_bytes
                or file_sha256(path) != expected_hash
            ):
                raise RuntimeError(f"high-resolution discovery content differs: {path}")
            checked += 1
    return {
        "split": "discovery",
        "selected_images": len(selected),
        "content_hash_files": checked,
        "manifest_bytes_and_sha256_verified": True,
        "confirmation20_opened": False,
    }


@torch.no_grad()
def evaluate_high_resolution(
    *, model, spec: LaneSpec | None, rows: list[dict[str, Any]], data_root: Path,
    destination: Path, freeze_receipt: Path, checkpoint: Path | None,
    checkpoint_step: int | None, checkpoint_metadata: dict[str, Any] | None,
    gpu: int,
) -> dict[str, Any]:
    """Evaluate exactly one frozen lane at the preregistered 256px setting."""
    if model is not None and spec is None:
        raise RuntimeError("model high-resolution evaluation requires a lane spec")
    lane_id = "input" if model is None else str(spec.id)
    freeze, freeze_commit = committed_freeze_identity(
        Path(freeze_receipt).resolve(), lane_id=lane_id,
    )
    destination = Path(destination).resolve()
    if destination.exists():
        raise RuntimeError(f"high-resolution receipt already exists: {destination}")
    selected = select_discovery(rows, COUNT_PER_DOMAIN)
    domains = sorted({str(row["domain"]) for row in selected})
    if len(domains) != 6:
        raise RuntimeError("high-resolution inference requires six discovery domains")
    discovery_content = _verify_selected_content(
        selected, data_root=Path(data_root).resolve(),
    )
    checkpoint_path, checkpoint_hash_before = _validate_e200_checkpoint(
        model=model, checkpoint=checkpoint, checkpoint_step=checkpoint_step,
        checkpoint_metadata=checkpoint_metadata,
    )
    family = "input" if model is None else str(spec.family)
    replicates = UNSB_REPLICATES if family == "unsb" else 1
    primary_nfe = 0 if model is None else (NUM_TIMESTEPS if family == "unsb" else 1)
    device = (
        torch.device(
            f"cuda:{int(gpu)}"
            if int(gpu) >= 0 and torch.cuda.is_available() else "cpu"
        )
        if model is None else torch.device(model.device)
    )
    modes = {} if model is None else {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    saved_rng = capture_rng()
    image_rows: list[dict[str, Any]] = []
    try:
        if model is not None:
            model.eval()
        perceptual = _lpips(device)
        for row in selected:
            source = read_image(
                Path(data_root).resolve() / row["input_relpath"], size=IMAGE_SIZE,
            ).to(device)
            target = read_image(
                Path(data_root).resolve() / row["target_relpath"], size=IMAGE_SIZE,
            ).to(device)
            target_unit = to_unit(target)
            for replicate in range(replicates):
                if model is None:
                    prediction = source
                    crn_sha256 = None
                else:
                    bundle = build_rollout_bundle(
                        protocol_hash=HIGH_RES_BUNDLE_FINGERPRINT,
                        domain=str(row["domain"]), stem=str(row["stem"]),
                        replicate=replicate,
                        latent_dim=4 * int(getattr(model.opt, "ngf", 64)),
                        height=IMAGE_SIZE, width=IMAGE_SIZE,
                        num_timesteps=NUM_TIMESTEPS,
                    )
                    prediction = _prediction(
                        model, spec, source, bundle, nfe=primary_nfe,
                    ).clamp(-1.0, 1.0)
                    crn_sha256 = bundle_hash(bundle)
                image_rows.append({
                    "domain": str(row["domain"]), "stem": str(row["stem"]),
                    "order": int(row["order"]), "replicate": replicate,
                    "nfe": primary_nfe,
                    "psnr": psnr_unit(to_unit(prediction), target_unit),
                    "ssim": ssim_unit(to_unit(prediction), target_unit),
                    "lpips": float(perceptual(prediction, target).item()),
                    "crn_bundle_sha256": crn_sha256,
                })
    finally:
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)
    checkpoint_hash_after = (
        None if checkpoint_path is None else file_sha256(checkpoint_path)
    )
    if checkpoint_hash_after != checkpoint_hash_before:
        raise RuntimeError("high-resolution inference changed the source checkpoint")
    aggregate = aggregate_metric_rows(image_rows)
    replicate_cells = [
        {
            "replicate": replicate,
            **aggregate_metric_rows([
                row for row in image_rows if int(row["replicate"]) == replicate
            ]),
        }
        for replicate in range(replicates)
    ]
    semantics = _metric_semantics()
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "lane_id": lane_id,
        "lane": None if spec is None else spec.to_dict(),
        "primary_epoch": 200,
        "primary_nfe": primary_nfe,
        "image_size": IMAGE_SIZE,
        "count_per_domain": COUNT_PER_DOMAIN,
        "domain_count": len(domains),
        "replicate_count": replicates,
        "spatial_policy": SPATIAL_POLICY,
        "metric_semantics": semantics,
        "metric_semantics_sha256": object_sha256(semantics),
        "base_protocol_fingerprint": protocol_fingerprint(),
        "base_evaluation_bundle_fingerprint": evaluation_bundle_fingerprint(),
        "high_resolution_bundle_fingerprint": HIGH_RES_BUNDLE_FINGERPRINT,
        "evaluation_input_sha256": evaluation_input_hash(
            selected, HIGH_RES_BUNDLE_FINGERPRINT,
        ),
        "discovery_content": discovery_content,
        "freeze_receipt": str(Path(freeze_receipt).resolve()),
        "freeze_receipt_sha256": file_sha256(Path(freeze_receipt).resolve()),
        "freeze_receipt_object_sha256": object_sha256(freeze),
        "freeze_git_commit": freeze_commit,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash_before,
        "checkpoint_sha256_after_evaluation": checkpoint_hash_after,
        "checkpoint_unchanged": checkpoint_hash_before == checkpoint_hash_after,
        "source_identity": _source_identity(),
        "macro_psnr": aggregate["macro_psnr"],
        "macro_ssim": aggregate["macro_ssim"],
        "macro_lpips": aggregate["macro_lpips"],
        "domains": aggregate["domains"],
        "replicate_cells": replicate_cells,
        "stochasticity": replicate_stochasticity(replicate_cells),
        "images": image_rows,
        "environment": environment_record(),
        "supplementary_only": True,
        "main_table": False,
        "generated_images_retained": False,
        "target_path_read_only_after_committed_freeze": True,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    validate_receipt(result, expected_lane=lane_id)
    write_json(destination, result)
    return result


def lock_high_resolution_cohort(
    *, freeze_receipt: Path, receipts: list[Path], destination: Path,
) -> dict[str, Any]:
    """Require every frozen lane to share one evaluator and one 256px policy."""
    freeze, freeze_commit = committed_freeze_identity(
        Path(freeze_receipt).resolve(), lane_id="input",
    )
    paths = [Path(path).resolve() for path in receipts]
    if not paths or len(paths) != len(set(paths)):
        raise RuntimeError("high-resolution cohort requires unique receipts")
    values: dict[str, dict[str, Any]] = {}
    common_environment = None
    common_input = None
    common_source = None
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"high-resolution receipt is missing: {path}")
        value = _read_json(path)
        lane = str(value.get("lane_id", ""))
        if not lane or lane in values:
            raise RuntimeError("high-resolution cohort has duplicate lanes")
        validate_receipt(value, expected_lane=lane)
        if (
            Path(value.get("freeze_receipt", "")).resolve()
            != Path(freeze_receipt).resolve()
            or value.get("freeze_receipt_sha256")
            != file_sha256(Path(freeze_receipt).resolve())
            or value.get("freeze_receipt_object_sha256") != object_sha256(freeze)
            or value.get("freeze_git_commit") != freeze_commit
        ):
            raise RuntimeError("high-resolution receipt uses another freeze")
        if common_environment is None:
            common_environment = value.get("environment")
            common_input = value.get("evaluation_input_sha256")
            common_source = value.get("source_identity")
        elif (
            value.get("environment") != common_environment
            or value.get("evaluation_input_sha256") != common_input
            or value.get("source_identity") != common_source
        ):
            raise RuntimeError("high-resolution receipts do not share one evaluator")
        values[lane] = {
            "lane_id": lane,
            "receipt": str(path),
            "receipt_sha256": file_sha256(path),
            "receipt_object_sha256": object_sha256(value),
        }
    expected = set(str(lane) for lane in freeze["distribution_lanes"])
    if set(values) != expected:
        raise RuntimeError("high-resolution cohort does not cover the frozen lane set")
    result = {
        "schema": COHORT_SCHEMA,
        "status": COHORT_STATUS,
        "freeze_receipt": str(Path(freeze_receipt).resolve()),
        "freeze_receipt_sha256": file_sha256(Path(freeze_receipt).resolve()),
        "freeze_git_commit": freeze_commit,
        "lanes": sorted(values),
        "receipts": [values[lane] for lane in sorted(values)],
        "image_size": IMAGE_SIZE,
        "spatial_policy": SPATIAL_POLICY,
        "high_resolution_bundle_fingerprint": HIGH_RES_BUNDLE_FINGERPRINT,
        "evaluation_input_sha256": common_input,
        "environment": common_environment,
        "source_identity": common_source,
        "supplementary_only": True,
        "main_table": False,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    destination = Path(destination).resolve()
    if destination.is_file():
        if object_sha256(_read_json(destination)) != object_sha256(result):
            raise RuntimeError("high-resolution cohort already exists and differs")
    else:
        write_json(destination, result)
    return result


def _load_standard_model(args):
    checkpoint = Path(args.checkpoint).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if args.lane == "candidate":
        if not args.candidate_id or not args.candidate_authority:
            raise SystemExit("candidate requires --candidate-id and --candidate-authority")
        metadata = payload.get("metadata") or {}
        spec, _ = candidate_spec_from_portable_authority(
            authority_path=Path(args.candidate_authority).resolve(),
            candidate_id=args.candidate_id, exported_lane=payload.get("lane"),
            training_git_commit=str(metadata.get("git_commit", "")),
            training_protocol_fingerprint=str(metadata.get("protocol_fingerprint", "")),
        )
    else:
        spec = lane_spec(args.lane)
    model, primary, secondary, rows = prepare_lane(
        output_root=Path(args.output).resolve(),
        train_view=Path(args.train_view).resolve(),
        manifest_path=Path(args.manifest).resolve(), spec=spec, gpu=int(args.gpu),
    )
    load_full_state(
        checkpoint, model=model, spec=spec, primary=primary, secondary=secondary,
        expected_metadata=payload["metadata"],
    )
    return model, spec, rows, payload


def _load_dclgan_model(args):
    if args.upstream_root is None:
        raise SystemExit("DCLGAN requires --upstream-root")
    from operations.paper_aio_dclgan_adapter import (
        _load_evaluation_runtime,
        annotated_manifest_rows,
        dclgan_lane_spec,
    )
    model, _stream, payload = _load_evaluation_runtime(
        upstream_root=Path(args.upstream_root).resolve(),
        manifest_path=Path(args.manifest).resolve(),
        train_view=Path(args.train_view).resolve(),
        output_root=Path(args.output).resolve(),
        checkpoint=Path(args.checkpoint).resolve(), gpu=int(args.gpu),
    )
    return model, dclgan_lane_spec(), annotated_manifest_rows(Path(args.manifest).resolve()), payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--mode", required=True, choices=["evaluate", "lock"])
    value.add_argument("--lane")
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--candidate-id")
    value.add_argument("--candidate-authority", type=Path)
    value.add_argument("--upstream-root", type=Path)
    value.add_argument("--output", type=Path, default=Path("runs/FINAL_UNSB_HIGH_RES_256"))
    value.add_argument("--manifest", type=Path, default=Path("manifests/FULL_DATA_MANIFEST.csv"))
    value.add_argument("--data-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--freeze-receipt", type=Path, required=True)
    value.add_argument("--receipt-output", type=Path, required=True)
    value.add_argument("--high-resolution-receipt", action="append", type=Path, default=[])
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.mode == "lock":
        if args.lane or args.checkpoint or not args.high_resolution_receipt:
            raise SystemExit("lock requires only repeated --high-resolution-receipt inputs")
        result = lock_high_resolution_cohort(
            freeze_receipt=args.freeze_receipt,
            receipts=args.high_resolution_receipt,
            destination=args.receipt_output,
        )
    else:
        if not args.lane:
            raise SystemExit("evaluate requires --lane")
        if args.data_root is None:
            raise SystemExit("evaluate requires --data-root")
        if args.lane == "input":
            if args.checkpoint is not None:
                raise SystemExit("Input evaluation cannot use --checkpoint")
            manifest_report(Path(args.manifest).resolve(), data_root=Path(args.data_root).resolve())
            model = spec = payload = None
            rows = _annotated_rows(Path(args.manifest).resolve())
        else:
            if args.checkpoint is None or args.train_view is None:
                raise SystemExit("model evaluation requires --checkpoint and --train-view")
            if args.lane == "dclgan":
                model, spec, rows, payload = _load_dclgan_model(args)
            else:
                model, spec, rows, payload = _load_standard_model(args)
        result = evaluate_high_resolution(
            model=model, spec=spec, rows=rows,
            data_root=Path(args.data_root).resolve(),
            destination=Path(args.receipt_output).resolve(),
            freeze_receipt=Path(args.freeze_receipt).resolve(),
            checkpoint=None if args.checkpoint is None else Path(args.checkpoint).resolve(),
            checkpoint_step=None if payload is None else int(payload.get("step", -1)),
            checkpoint_metadata=None if payload is None else payload.get("metadata"),
            gpu=int(args.gpu),
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
