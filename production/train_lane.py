#!/usr/bin/env python3
"""Train one frozen lane with exact epoch-boundary resume and identity binding."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

from production import common, full_state


def manifest_counts(path: Path) -> tuple[int, str]:
    with path.open(encoding="utf-8", newline="") as handle:
        counts = sum(1 for row in csv.DictReader(handle) if row["split"] == "train")
    return counts, common.file_sha256(path)


def build(args, steps_per_epoch: int):
    from data import create_dataset
    from models import create_model
    from options.train_options import TrainOptions

    argv, resolved_lane = common.train_argv(
        lane_id=args.lane, data_view=args.data_view, run_root=args.run_root,
        gpu_id=args.gpu_id, steps_per_epoch=steps_per_epoch,
    )
    opt = TrainOptions(cmd_line=argv).parse()
    dataset = create_dataset(opt)
    dataset2 = create_dataset(opt)
    if len(dataset) != steps_per_epoch or len(dataset2) != steps_per_epoch:
        raise RuntimeError(
            f"materialized train view length mismatch: {len(dataset)}, {len(dataset2)} "
            f"!= {steps_per_epoch}"
        )
    model = create_model(opt)
    return opt, dataset, dataset2, model, resolved_lane


def initialize_disposable(model, dataset, dataset2, opt) -> None:
    dataset.set_epoch(1)
    dataset2.set_epoch(1)
    data = next(iter(dataset))
    data2 = next(iter(dataset2))
    model.data_dependent_initialize(data, data2)
    model.setup(opt)
    model.parallelize()


def metadata_base(args, manifest_hash: str, resolved_lane: dict, config_hash: str) -> dict:
    return {
        "project_id": "FINAL_UNSB_FOUR_LANE_E200",
        "lane_id": args.lane,
        "seed": 2026,
        "git_commit": common.git_commit(),
        "protocol_fingerprint": common.protocol_fingerprint(),
        "manifest_sha256": manifest_hash,
        "lane_config_sha256": config_hash,
        "resolved_lane": resolved_lane,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--data-view", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--e0-only", action="store_true")
    parser.add_argument("--local-smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()

    contract, _ = common.lane_record(args.lane)
    steps_per_epoch, manifest_hash = manifest_counts(args.manifest)
    expected_steps = 8553
    if not args.local_smoke and steps_per_epoch != expected_steps:
        raise RuntimeError(f"production train count {steps_per_epoch} != {expected_steps}")
    stop_epoch = args.stop_after_epoch or int(contract["common"]["stop_after_epoch"])
    authorization = None
    if not args.local_smoke and not args.e0_only:
        authorization = common.ROOT / "decisions/RUN_AUTHORIZATION.json"
        if not authorization.is_file():
            raise RuntimeError("production run is not authorized")
        authorization = common.load_json(authorization)
        if not authorization.get("authorized"):
            raise RuntimeError("production run is not authorized")
        if common.git_status():
            raise RuntimeError("production requires a clean Git worktree")
        expected_auth = {
            "protocol_fingerprint": common.protocol_fingerprint(),
            "data_manifest_sha256": manifest_hash,
        }
        for key, value in expected_auth.items():
            if authorization.get(key) != value:
                raise RuntimeError(f"authorization mismatch for {key}")
        if args.lane not in authorization.get("lanes", []):
            raise RuntimeError(f"lane {args.lane} is absent from authorization")

    environment = common.apply_determinism(int(contract["seed"]))
    if authorization is not None:
        for key, value in authorization.get("runtime", {}).items():
            if environment.get(key) != value:
                raise RuntimeError(
                    f"runtime authorization mismatch for {key}: "
                    f"{environment.get(key)!r} != {value!r}"
                )
    opt, dataset, dataset2, model, resolved_lane = build(args, steps_per_epoch)
    config_hash = common.object_sha256({"contract": contract, "lane": resolved_lane})
    base = metadata_base(args, manifest_hash, resolved_lane, config_hash)
    lane_root = args.run_root / args.lane
    lane_root.mkdir(parents=True, exist_ok=True)
    latest = lane_root / "full_state_latest.pt"
    trace_path = lane_root / "TRAIN_TRACE.jsonl"
    heartbeat_path = lane_root / "HEARTBEAT.json"
    start_epoch = 1
    global_step = 0
    initialized = False

    if args.resume:
        if not latest.is_file():
            raise RuntimeError(f"resume checkpoint missing: {latest}")
        initialize_disposable(model, dataset, dataset2, opt)
        initialized = True
        restored = full_state.load(
            latest, model,
            expected={
                "project_id": base["project_id"],
                "lane_id": args.lane,
                "seed": 2026,
                "manifest_sha256": manifest_hash,
                "lane_config_sha256": config_hash,
                "protocol_fingerprint": base["protocol_fingerprint"],
            },
        )
        start_epoch = int(restored["epoch_completed"]) + 1
        global_step = int(restored["global_step"])

    start_wall = time.time()
    last_heartbeat = 0.0
    for epoch in range(start_epoch, stop_epoch + 1):
        dataset.set_epoch(epoch)
        dataset2.set_epoch(epoch)
        model.set_train_epoch(epoch)
        epoch_start = time.time()
        for index, (data, data2) in enumerate(zip(dataset, dataset2)):
            if not initialized:
                model.data_dependent_initialize(data, data2)
                model.setup(opt)
                model.parallelize()
                initialized = True
                e0_networks = {
                    name: getattr(model, "net" + name) for name in model.model_names
                }
                e0 = {
                    **base,
                    "steps_per_epoch": steps_per_epoch,
                    "e0_network_state_sha256": common.state_tensor_sha256(e0_networks),
                    "environment": environment,
                    "resolved_hj_updates": resolved_lane.get("resolved_active_updates"),
                }
                if authorization is not None:
                    expected_e0 = authorization.get("e0_network_state_sha256")
                    if e0["e0_network_state_sha256"] != expected_e0:
                        raise RuntimeError(
                            "fresh e0 does not match the four-server authorization"
                        )
                common.atomic_json(lane_root / "E0_IDENTITY.json", e0)
                if args.e0_only:
                    print(json.dumps(e0, ensure_ascii=False, indent=2))
                    return 0
            model.set_search_step(global_step, steps_per_epoch * stop_epoch)
            model.set_input(data, data2)
            model.optimize_parameters()
            global_step += int(data["A"].size(0))
            now = time.time()
            if global_step % 100 == 0:
                losses = model.get_current_losses()
                if not all(math.isfinite(value) for value in losses.values()):
                    raise RuntimeError(f"non-finite loss at global step {global_step}: {losses}")
                entry = {
                    "epoch": epoch,
                    "step_in_epoch": index + 1,
                    "global_step": global_step,
                    "losses": losses,
                    "elapsed_hours": (now - start_wall) / 3600.0,
                }
                with trace_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            if now - last_heartbeat >= 300:
                common.atomic_json(
                    heartbeat_path,
                    {
                        **base,
                        "epoch": epoch,
                        "step_in_epoch": index + 1,
                        "global_step": global_step,
                        "elapsed_hours": (now - start_wall) / 3600.0,
                    },
                )
                last_heartbeat = now

        model.update_learning_rate()
        epoch_meta = {
            **base,
            "epoch_completed": epoch,
            "global_step": global_step,
            "steps_per_epoch": steps_per_epoch,
            "elapsed_hours": (time.time() - start_wall) / 3600.0,
            "last_epoch_seconds": time.time() - epoch_start,
        }
        full_state.save(latest, model, metadata=epoch_meta)
        if epoch in contract["common"]["checkpoint_epochs"] or epoch == stop_epoch:
            milestone = lane_root / "milestones" / f"full_state_e{epoch}.pt"
            full_state.save(milestone, model, metadata=epoch_meta)
            model.save_networks(epoch)

    summary = {
        **base,
        "status": "complete",
        "epoch_completed": stop_epoch,
        "global_step": global_step,
        "steps_per_epoch": steps_per_epoch,
        "elapsed_hours": (time.time() - start_wall) / 3600.0,
        "latest_checkpoint_sha256": common.file_sha256(latest),
        "method_state": model.get_extra_training_state(),
    }
    common.atomic_json(lane_root / "TRAIN_SUMMARY.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
