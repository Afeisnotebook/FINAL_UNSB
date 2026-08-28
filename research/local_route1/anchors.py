"""Shared-e0 anchor training, milestone evaluation and proxy calibration."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .evaluate import compare_to_plain, evaluate_model
from .protocol import (
    ProbeSpec,
    epoch_to_step,
    git_commit,
    load_protocol,
    milestone_steps,
    probe_spec,
    protocol_fingerprint,
    step_to_physical_epoch,
    steps_per_epoch,
)
from .runtime import (
    FULL_STATE_SCHEMA,
    SerializableDataStream,
    atomic_torch_save,
    build_datasets,
    build_model,
    build_options,
    capture_rng,
    cpu_clone,
    file_sha256,
    full_state_hash,
    load_full_state,
    load_model_state,
    model_state,
    read_manifest,
    restore_rng,
    runtime_identity,
    save_full_state,
    seed_everything,
    write_json,
)


E0_SCHEMA = "final-unsb-local-route1-shared-e0-v1"


def _streams(opt, rows: list[dict], per_domain: int, seed: int):
    primary_data, secondary_data = build_datasets(opt, rows, per_domain)
    return (
        SerializableDataStream(primary_data, seed=seed + 101, label="primary"),
        SerializableDataStream(secondary_data, seed=seed + 202, label="secondary"),
    )


def create_shared_e0(
    *, output_root: Path, train_view: Path, manifest_path: Path, gpu: int,
) -> dict:
    protocol = load_protocol()
    seed = int(protocol["seed"])
    per_domain = int(protocol["local_view"]["train_per_domain"])
    e0_path = output_root / "shared_e0" / "e0.pt"
    identity = {
        "project_id": protocol["project_id"],
        "seed": seed,
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "git_commit": git_commit(),
    }
    if e0_path.is_file():
        payload = torch.load(e0_path, map_location="cpu", weights_only=False)
        if payload.get("schema") != E0_SCHEMA or payload.get("metadata") != identity:
            raise RuntimeError("shared e0 identity mismatch; use a new output root")
        return payload

    rows = read_manifest(manifest_path)
    spec = probe_spec("plain", protocol)
    seed_everything(seed)
    opt = build_options(
        spec, dataroot=train_view, option_root=output_root / "option_records",
        seed=seed, gpu=gpu, diagnostic_root=None,
    )
    primary, secondary = _streams(opt, rows, per_domain, seed)
    model = build_model(opt, primary.next(), secondary.next())
    payload = {
        "schema": E0_SCHEMA,
        "metadata": identity,
        "model": model_state(model),
        "rng": capture_rng(),
        "samplers": {
            "primary": primary.state_dict(),
            "secondary": secondary.state_dict(),
        },
        "steps_per_epoch": steps_per_epoch(protocol),
        "target_steps": int(protocol["local_view"]["target_updates_per_lane"]),
    }
    atomic_torch_save(e0_path, payload)
    write_json(Path(str(e0_path) + ".json"), {
        "schema": E0_SCHEMA,
        "metadata": identity,
        "checkpoint_sha256": file_sha256(e0_path),
        "scientific_state_sha256": full_state_hash(payload),
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def prepare_probe(
    *, spec: ProbeSpec, output_root: Path, train_view: Path, manifest_path: Path,
    gpu: int, e0: dict | None = None,
):
    protocol = load_protocol()
    seed = int(protocol["seed"])
    rows = read_manifest(manifest_path)
    per_domain = int(protocol["local_view"]["train_per_domain"])
    e0 = e0 or create_shared_e0(
        output_root=output_root, train_view=train_view,
        manifest_path=manifest_path, gpu=gpu,
    )
    seed_everything(seed)
    lane_root = output_root / "anchors" / spec.id
    opt = build_options(
        spec, dataroot=train_view, option_root=output_root / "option_records",
        seed=seed, gpu=gpu, diagnostic_root=lane_root / "diagnostics",
    )
    primary, secondary = _streams(opt, rows, per_domain, seed)
    model = build_model(opt, primary.next(), secondary.next())
    load_model_state(model, e0["model"], load_method=False)
    primary.load_state_dict(e0["samplers"]["primary"])
    secondary.load_state_dict(e0["samplers"]["secondary"])
    restore_rng(e0["rng"])
    model.set_search_step(0, int(protocol["local_view"]["target_updates_per_lane"]))
    return model, primary, secondary, rows


def _read_latest_step(path: Path) -> int | None:
    if not path.is_file():
        return None
    return int(torch.load(path, map_location="cpu", weights_only=False)["step"])


def assert_anchor_order(output_root: Path, probe_id: str) -> None:
    protocol = load_protocol()
    target = int(protocol["local_view"]["target_updates_per_lane"])
    predecessors = {"plain": [], "hj": ["plain"], "hnek": ["plain", "hj"], "dt": ["plain", "hj", "hnek"]}
    for predecessor in predecessors[probe_id]:
        latest = output_root / "anchors" / predecessor / "full_state_latest.pt"
        if _read_latest_step(latest) != target:
            raise RuntimeError(f"{probe_id} is blocked until {predecessor} reaches e200")
    if probe_id == "dt":
        calibration = output_root / "evidence" / "PROXY_CALIBRATION.json"
        if not calibration.is_file():
            raise RuntimeError("DT is blocked until proxy calibration is evaluated")
        status = json.loads(calibration.read_text(encoding="utf-8")).get("status")
        if status != "CALIBRATED":
            raise RuntimeError(f"DT is blocked because proxy status is {status}")


def _trace(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_anchor(
    *, probe_id: str, output_root: Path, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, resume: bool,
    engineering_stop_after_epoch: int | None = None,
) -> dict:
    protocol = load_protocol()
    spec = probe_spec(probe_id, protocol)
    assert_anchor_order(output_root, probe_id)
    target_steps = int(protocol["local_view"]["target_updates_per_lane"])
    if engineering_stop_after_epoch is not None:
        stop_steps = epoch_to_step(int(engineering_stop_after_epoch), protocol)
        if stop_steps <= 0 or stop_steps > target_steps:
            raise ValueError("engineering stop epoch must be in [1, 200]")
    else:
        stop_steps = target_steps
    lane_root = output_root / "anchors" / probe_id
    latest = lane_root / "full_state_latest.pt"
    if latest.is_file() and not resume:
        raise RuntimeError(f"existing state requires --resume: {latest}")

    e0 = create_shared_e0(
        output_root=output_root, train_view=train_view,
        manifest_path=manifest_path, gpu=gpu,
    )
    model, primary, secondary, rows = prepare_probe(
        spec=spec, output_root=output_root, train_view=train_view,
        manifest_path=manifest_path, gpu=gpu, e0=e0,
    )
    metadata = {
        **runtime_identity(manifest_path, spec),
        "git_commit": git_commit(),
        "train_view": str(train_view.resolve()),
        "data_root": str(data_root.resolve()),
        "data_epochs_target": int(protocol["local_view"]["target_epochs"]),
        "steps_per_data_epoch": steps_per_epoch(protocol),
        "confirmation20_opened": False,
    }
    start_step = 0
    if resume and latest.is_file():
        restored = load_full_state(
            latest, model=model, spec=spec, primary=primary, secondary=secondary,
            expected_metadata={
                "project_id": metadata["project_id"],
                "probe_id": probe_id,
                "seed": metadata["seed"],
                "manifest_sha256": metadata["manifest_sha256"],
                "protocol_fingerprint": metadata["protocol_fingerprint"],
                "git_commit": metadata["git_commit"],
            },
        )
        start_step = int(restored["step"])
    if start_step >= stop_steps:
        return {"status": "ALREADY_AT_REQUESTED_STOP", "probe_id": probe_id, "step": start_step}

    milestone_set = set(milestone_steps(protocol))
    lpips_epochs = set(int(value) for value in protocol["local_view"]["lpips_epochs"])
    protocol_hash = metadata["protocol_fingerprint"]
    started = time.time()
    epoch_started = time.time()
    last_losses = None
    for zero_step in range(start_step, stop_steps):
        physical_epoch = step_to_physical_epoch(zero_step, protocol)
        model.set_train_epoch(physical_epoch)
        model.set_search_step(zero_step, target_steps)
        model.set_input(primary.next(), secondary.next())
        model.optimize_parameters()
        completed = zero_step + 1
        last_losses = model.get_current_losses()
        if completed % steps_per_epoch(protocol) != 0:
            continue

        completed_epoch = completed // steps_per_epoch(protocol)
        model.update_learning_rate()
        sidecar = save_full_state(
            latest, model=model, spec=spec, step=completed,
            target_steps=target_steps, primary=primary, secondary=secondary,
            metadata=metadata,
        )
        if completed in milestone_set:
            save_full_state(
                lane_root / "milestones" / f"e{completed_epoch:03d}.pt",
                model=model, spec=spec, step=completed, target_steps=target_steps,
                primary=primary, secondary=secondary, metadata=metadata,
            )
            metric_path = lane_root / "metrics" / f"e{completed_epoch:03d}.json"
            if not metric_path.is_file():
                metrics = evaluate_model(
                    model, rows=rows, data_root=data_root,
                    protocol_hash=protocol_hash,
                    include_lpips=completed_epoch in lpips_epochs,
                )
                metrics.update({
                    "probe_id": probe_id, "epoch": completed_epoch,
                    "updates": completed, "data_epoch": completed_epoch,
                })
                write_json(metric_path, metrics)
        epoch_seconds = time.time() - epoch_started
        trace_row = {
            "probe_id": probe_id,
            "updates": completed,
            "data_epoch": completed_epoch,
            "epoch_wall_seconds": epoch_seconds,
            "losses_last_update": last_losses,
            "latest_scientific_state_sha256": sidecar["scientific_state_sha256"],
            "confirmation20_opened": False,
        }
        _trace(lane_root / "TRAIN_TRACE.jsonl", trace_row)
        write_json(lane_root / "HEARTBEAT.json", {
            **trace_row,
            "target_updates": target_steps,
            "target_data_epochs": 200,
            "wall_seconds_this_call": time.time() - started,
        })
        epoch_started = time.time()

    status = "COMPLETE_E200" if stop_steps == target_steps else "ENGINEERING_PAUSE"
    result = {
        "status": status,
        "probe_id": probe_id,
        "start_updates": start_step,
        "final_updates": stop_steps,
        "final_data_epoch": stop_steps // steps_per_epoch(protocol),
        "target_updates": target_steps,
        "target_data_epochs": 200,
        "wall_seconds_this_call": time.time() - started,
        "metadata": metadata,
        "confirmation20_opened": False,
    }
    write_json(lane_root / "RUN_STATE.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _metric(output_root: Path, probe_id: str, epoch: int) -> dict | None:
    path = output_root / "anchors" / probe_id / "metrics" / f"e{epoch:03d}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def summarize_anchors(output_root: Path) -> dict:
    protocol = load_protocol()
    epochs = [int(value) for value in protocol["local_view"]["trajectory_epochs"]]
    late = [int(value) for value in protocol["local_view"]["late_epochs"]]
    plain_by_epoch = {epoch: _metric(output_root, "plain", epoch) for epoch in epochs}
    summaries = []
    for probe_id in ("hj", "hnek", "dt"):
        trajectory = []
        for epoch in epochs:
            method, plain = _metric(output_root, probe_id, epoch), plain_by_epoch[epoch]
            if method is not None and plain is not None:
                trajectory.append(compare_to_plain(method, plain, epoch=epoch))
        late_rows = [row for row in trajectory if row["epoch"] in late]
        summaries.append({
            "probe_id": probe_id,
            "trajectory": trajectory,
            "late_three_mean_macro_psnr_delta": (
                None if len(late_rows) != 3 else float(np.mean([row["macro_psnr_delta"] for row in late_rows]))
            ),
            "late_points_with_four_of_six_positive_domains": sum(row["positive_domains"] >= 4 for row in late_rows),
            "complete_e200": len(late_rows) == 3 and late_rows[-1]["epoch"] == 200,
        })
    evidence = {
        "schema": "local-route1-anchor-summary-v1",
        "time_unit": "data_epoch",
        "summaries": summaries,
        "confirmation20_opened": False,
    }
    write_json(output_root / "evidence" / "ANCHOR_TRAJECTORIES.json", evidence)

    proxy_rows = [row for row in summaries if row["probe_id"] in ("hj", "hnek")]
    complete = all(row["complete_e200"] for row in proxy_rows)
    passing = [
        row for row in proxy_rows
        if row["complete_e200"]
        and row["late_three_mean_macro_psnr_delta"] > 0.0
        and row["late_points_with_four_of_six_positive_domains"] >= 2
    ]
    status = "INCOMPLETE" if not complete else ("CALIBRATED" if passing else "NOT_CALIBRATED_PAUSE")
    calibration = {
        "schema": "local-route1-proxy-calibration-v1",
        "status": status,
        "passing_probes": [row["probe_id"] for row in passing],
        "rule": protocol["proxy_calibration"],
        "interpretation": (
            "proxy supports long-horizon route-1 discovery"
            if status == "CALIBRATED" else
            "do not call mechanisms dead; diagnose lineage/batch1-small25 proxy before DT or new long runs"
            if status == "NOT_CALIBRATED_PAUSE" else
            "HJ and HNEK e200 trajectories are not both complete"
        ),
        "confirmation20_opened": False,
    }
    write_json(output_root / "evidence" / "PROXY_CALIBRATION.json", calibration)
    return {"anchor_summary": evidence, "proxy_calibration": calibration}
