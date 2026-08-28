#!/usr/bin/env python3
"""Evaluate one full-state checkpoint on a frozen paired development split."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from production import common, metrics


def read_image(path: Path, size: int = 128) -> torch.Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous().unsqueeze(0)


def selected_rows(manifest: Path, split: str) -> list[dict]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["split"] == split]
    if not rows:
        raise RuntimeError(f"manifest has no {split!r} rows")
    return rows


def build_model(lane_id: str, gpu_id: int, scratch: Path):
    from models import create_model
    from options.train_options import TrainOptions

    contract, lane = common.lane_record(lane_id)
    argv = [
        "--dataroot", str(scratch), "--name", "read_only_eval",
        "--checkpoints_dir", str(scratch), "--model", lane["model"],
        "--dataset_mode", "unaligned", "--gpu_ids", str(gpu_id),
        "--seed", str(contract["seed"]), "--batch_size", "1",
        "--num_threads", "0", "--load_size", "128", "--crop_size", "128",
        "--preprocess", "resize_and_crop", "--num_timesteps", "5",
        "--tau", "0.01", "--no_flip", "--no_html", "--display_id", "-1",
    ]
    if lane_id == "P2_HNEK":
        method = lane["method"]
        argv += [
            "--hnek_gamma", str(method["hnek_gamma"]),
            "--hnek_coord", method["hnek_coord"],
            "--hnek_horizon_mode", method["hnek_horizon_mode"],
            "--hnek_partial", method["hnek_partial"],
        ]
    model = create_model(TrainOptions(cmd_line=argv).parse())
    return model


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["domain"]].append(row)
    per_domain = {}
    for domain in sorted(grouped):
        per_domain[domain] = {
            "n_stem_replicates": len(grouped[domain]),
            "psnr": float(np.mean([row["psnr"] for row in grouped[domain]])),
            "ssim": float(np.mean([row["ssim"] for row in grouped[domain]])),
        }
    return {
        "per_domain": per_domain,
        "macro_psnr": float(np.mean([value["psnr"] for value in per_domain.values()])),
        "macro_ssim": float(np.mean([value["ssim"] for value in per_domain.values()])),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=("discovery", "confirmation"), default="discovery")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--limit-per-domain", type=int)
    args = parser.parse_args()

    if args.split == "confirmation":
        unlock = common.ROOT / "decisions" / "CONFIRMATION_UNLOCK.json"
        if not unlock.is_file():
            raise RuntimeError("confirmation is sealed until a committed freeze/unlock exists")
        unlock_payload = common.load_json(unlock)
        if not unlock_payload.get("authorized"):
            raise RuntimeError("confirmation is sealed until a committed freeze/unlock exists")
        checkpoint_hash = common.file_sha256(args.checkpoint)
        confirmation_checks = {
            "protocol_fingerprint": common.protocol_fingerprint(),
            "data_manifest_sha256": common.file_sha256(args.manifest),
        }
        for key, value in confirmation_checks.items():
            if unlock_payload.get(key) != value:
                raise RuntimeError(f"confirmation unlock mismatch for {key}")
        if args.lane not in unlock_payload.get("authorized_lanes", []):
            raise RuntimeError(f"lane {args.lane} is not confirmation-authorized")
        if unlock_payload.get("checkpoint_sha256_by_lane", {}).get(args.lane) != checkpoint_hash:
            raise RuntimeError("checkpoint is not the frozen confirmation checkpoint")
    common.apply_determinism(2026)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = payload["metadata"]
    if metadata.get("lane_id") != args.lane:
        raise RuntimeError("checkpoint lane does not match requested lane")
    if metadata.get("manifest_sha256") != common.file_sha256(args.manifest):
        raise RuntimeError("checkpoint and evaluation manifest identities differ")
    if metadata.get("protocol_fingerprint") != common.protocol_fingerprint():
        raise RuntimeError("checkpoint protocol differs from the current repository")
    if metadata.get("project_id") != "FINAL_UNSB_FOUR_LANE_E200" or metadata.get("seed") != 2026:
        raise RuntimeError("checkpoint project/seed identity is invalid")
    contract, _ = common.lane_record(args.lane)
    _, resolved_lane = common.train_argv(
        lane_id=args.lane, data_view=Path("IDENTITY_ONLY"),
        run_root=Path("IDENTITY_ONLY"), gpu_id=args.gpu_id,
        steps_per_epoch=int(metadata["steps_per_epoch"]),
    )
    expected_config_hash = common.object_sha256({"contract": contract, "lane": resolved_lane})
    if metadata.get("lane_config_sha256") != expected_config_hash:
        raise RuntimeError("checkpoint lane configuration differs from the repository")

    model = build_model(args.lane, args.gpu_id, args.output.parent / ".eval_scratch")
    net_g = common.unwrap(model.netG)
    net_g.load_state_dict(payload["networks"]["G"], strict=True)
    net_g.eval()
    for parameter in net_g.parameters():
        parameter.requires_grad_(False)

    rows = selected_rows(args.manifest, args.split)
    if args.limit_per_domain:
        counts = defaultdict(int)
        limited = []
        for row in rows:
            if counts[row["domain"]] < args.limit_per_domain:
                limited.append(row)
                counts[row["domain"]] += 1
        rows = limited
    protocol_hash = common.object_sha256({
        "schema": "FINAL_UNSB_EVAL_V1", "manifest": common.file_sha256(args.manifest),
        "split": args.split, "replicates": args.replicates, "timesteps": 5,
        "tau": 0.01, "resize": "PIL_BICUBIC_128",
    })
    raw = []
    device = next(net_g.parameters()).device
    for row in rows:
        source = read_image(args.data_root / row["input_relpath"]).to(device)
        target = metrics.to_unit(read_image(args.data_root / row["target_relpath"]).to(device))
        for replicate in range(args.replicates):
            bundle = metrics.build_rollout_bundle(
                protocol_hash=protocol_hash, domain=row["domain"], stem=row["stem"],
                replicate=replicate,
            )
            prediction = metrics.to_unit(metrics.rollout_endpoint(net_g, source, bundle))
            raw.append({
                "domain": row["domain"], "stem": row["stem"],
                "replicate": replicate, "bundle_sha256": metrics.bundle_hash(bundle),
                "psnr": metrics.psnr_unit(prediction, target),
                "ssim": metrics.ssim_unit(prediction, target),
            })
    result = {
        "schema_version": 1, "lane_id": args.lane, "split": args.split,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": common.file_sha256(args.checkpoint),
        "checkpoint_metadata": metadata, "protocol_sha256": protocol_hash,
        "replicates": args.replicates, "summary": aggregate(raw), "rows": raw,
    }
    common.atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("lane_id", "split", "summary")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
