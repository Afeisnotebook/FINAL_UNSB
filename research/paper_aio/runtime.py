"""Deterministic full-data runtime with exact interruption recovery."""

from __future__ import annotations

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

from research.local_route1.runtime import (
    SerializableDataStream,
    assert_finite,
    atomic_torch_save,
    capture_rng,
    file_sha256,
    full_state_hash,
    load_model_state,
    model_state,
    read_manifest,
    restore_rng,
    seed_everything,
    write_json,
)

from .protocol import (
    E0_SCHEMA,
    FULL_STATE_SCHEMA,
    ROOT,
    LaneSpec,
    epoch_to_step,
    evaluation_bundle_fingerprint,
    git_commit,
    lane_spec,
    load_protocol,
    milestone_steps,
    object_sha256,
    protocol_fingerprint,
    step_to_epoch,
    steps_per_epoch,
)


SRC = ROOT / "src"


def install_import_paths() -> None:
    for path in (str(SRC), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def _as_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def option_args(
    spec: LaneSpec, *, dataroot: Path, option_root: Path, seed: int, gpu: int,
) -> list[str]:
    protocol = load_protocol()
    common = protocol["common"]
    schedule = protocol["unsb"] if spec.family == "unsb" else protocol["external"]
    args = [
        "--dataroot", str(dataroot),
        "--name", f"paper_{spec.id}",
        "--checkpoints_dir", str(option_root),
        "--model", spec.model,
        "--phase", "train",
        "--dataset_mode", str(common["dataset_mode"]),
        "--direction", str(common["direction"]),
        "--gpu_ids", str(gpu),
        "--seed", str(seed),
        "--batch_size", str(common["batch_size"]),
        "--num_threads", str(common["num_threads"]),
        "--n_epochs", str(schedule["n_epochs"]),
        "--n_epochs_decay", str(schedule["n_epochs_decay"]),
        "--lr", str(schedule["lr"]),
        "--beta1", str(schedule["beta1"]),
        "--beta2", str(schedule["beta2"]),
        "--load_size", str(common["load_size"]),
        "--crop_size", str(common["crop_size"]),
        "--preprocess", str(common["preprocess"]),
        "--display_id", "-1", "--no_html", "--no_flip",
        "--print_freq", "100000000", "--save_latest_freq", "100000000",
        "--save_epoch_freq", "100000000",
    ]
    if spec.family == "unsb":
        args += [
            "--mode", str(schedule["mode"]),
            "--lambda_GAN", str(schedule["lambda_GAN"]),
            "--lambda_SB", str(schedule["lambda_SB"]),
            "--lambda_NCE", str(schedule["lambda_NCE"]),
            "--tau", str(schedule["tau"]),
            "--num_timesteps", str(schedule["num_timesteps"]),
            "--nce_T", str(schedule["nce_T"]),
            "--nce_idt", _as_cli(schedule["nce_idt"]),
            "--nce_layers", str(schedule["nce_layers"]),
            "--num_patches", str(schedule["num_patches"]),
            "--netF", str(schedule["netF"]),
            "--netG", str(schedule["netG"]),
            "--netD", str(schedule["netD"]),
        ]
    for key, value in spec.method.items():
        args += [f"--{key}", _as_cli(value)]
    return args


def build_options(spec: LaneSpec, **kwargs):
    install_import_paths()
    from options.train_options import TrainOptions

    return TrainOptions(cmd_line=option_args(spec, **kwargs)).parse()


def _annotated_rows(path: Path) -> list[dict]:
    rows = read_manifest(path)
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["domain"], row["split"])
        row["order"] = str(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
    return rows


def manifest_report(path: Path, *, data_root: Path | None = None) -> dict:
    protocol = load_protocol()
    actual = file_sha256(path)
    if actual != protocol["manifest"]["sha256"]:
        raise RuntimeError(f"full manifest SHA256 mismatch: {actual}")
    rows = _annotated_rows(path)
    counts = {
        split: sum(row["split"] == split for row in rows)
        for split in ("train", "discovery", "confirmation")
    }
    if counts != {key: int(value) for key, value in protocol["manifest"]["counts"].items()}:
        raise RuntimeError(f"full manifest split counts changed: {counts}")
    domains = sorted({row["domain"] for row in rows})
    if len(domains) != int(protocol["manifest"]["domains"]):
        raise RuntimeError(f"expected six domains, got {domains}")
    missing = []
    hash_mismatch = []
    if data_root is not None:
        for row in rows:
            for rel_key, hash_key in (
                ("input_relpath", "input_sha256"),
                ("target_relpath", "target_sha256"),
            ):
                candidate = Path(data_root) / row[rel_key]
                if not candidate.is_file():
                    missing.append(str(candidate))
                elif file_sha256(candidate) != row[hash_key]:
                    hash_mismatch.append(str(candidate))
    if missing or hash_mismatch:
        raise RuntimeError(
            f"full data integrity failed: missing={len(missing)}, "
            f"hash_mismatch={len(hash_mismatch)}"
        )
    return {
        "path": str(Path(path).resolve()),
        "sha256": actual,
        "counts": counts,
        "domains": domains,
        "content_hashes_verified": data_root is not None,
    }


def build_datasets(opt, *, expected_train: int):
    install_import_paths()
    from data.unaligned_dataset import UnalignedDataset

    primary = UnalignedDataset(opt)
    secondary = UnalignedDataset(opt)
    for label, dataset in (("primary", primary), ("secondary", secondary)):
        if dataset.A_size != expected_train or dataset.B_size != expected_train:
            raise RuntimeError(
                f"{label} full view mismatch: A={dataset.A_size}, B={dataset.B_size}, "
                f"expected={expected_train}"
            )
        if bool(getattr(dataset, "_macro_enabled", False)):
            raise RuntimeError("main paper lane accidentally enabled macro_marginal")
    return primary, secondary


def build_streams(opt, *, seed: int):
    expected = int(load_protocol()["manifest"]["counts"]["train"])
    primary_data, secondary_data = build_datasets(opt, expected_train=expected)
    return (
        SerializableDataStream(primary_data, seed=seed + 101, label="primary"),
        SerializableDataStream(secondary_data, seed=seed + 202, label="secondary"),
    )


def _build_model(opt, spec: LaneSpec, primary: SerializableDataStream, secondary: SerializableDataStream):
    install_import_paths()
    from models import create_model

    model = create_model(opt)
    if spec.family == "unsb":
        model.data_dependent_initialize(primary.next(), secondary.next())
    elif spec.id == "cyclegan":
        model.data_dependent_initialize()
    elif spec.id == "cut":
        model.data_dependent_initialize(primary.next())
    else:
        raise RuntimeError(f"no internal initialization adapter for lane {spec.id}")
    model.setup(opt)
    model.parallelize()
    return model


def _e0_path(output_root: Path, spec: LaneSpec) -> Path:
    family = "unsb_common" if spec.family == "unsb" else spec.id
    return Path(output_root) / "shared_e0" / family / "e0.pt"


def create_e0(
    *, output_root: Path, train_view: Path, manifest_path: Path, spec: LaneSpec,
    gpu: int,
) -> dict:
    protocol = load_protocol()
    seed = int(protocol["seed"])
    if spec.family == "unsb":
        initializer = lane_spec("plain", protocol)
    else:
        initializer = spec
    path = _e0_path(output_root, spec)
    identity = {
        "schema": E0_SCHEMA,
        "project_id": protocol["project_id"],
        "family": "unsb_common" if spec.family == "unsb" else spec.id,
        "seed": seed,
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "git_commit": git_commit(),
    }
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != E0_SCHEMA or payload.get("metadata") != identity:
            raise RuntimeError("paper e0 identity mismatch; use a new output root")
        return payload
    seed_everything(seed)
    opt = build_options(
        initializer, dataroot=train_view, option_root=Path(output_root) / "option_records",
        seed=seed, gpu=gpu,
    )
    primary, secondary = build_streams(opt, seed=seed)
    model = _build_model(opt, initializer, primary, secondary)
    payload = {
        "schema": E0_SCHEMA,
        "metadata": identity,
        "model": model_state(model),
        "rng": capture_rng(),
        "samplers": {"primary": primary.state_dict(), "secondary": secondary.state_dict()},
        "steps_per_epoch": steps_per_epoch(protocol),
        "target_steps": int(protocol["training"]["target_updates"]),
    }
    atomic_torch_save(path, payload)
    write_json(Path(str(path) + ".json"), {
        "schema": E0_SCHEMA,
        "metadata": identity,
        "checkpoint_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
    })
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def prepare_lane(
    *, output_root: Path, train_view: Path, manifest_path: Path, spec: LaneSpec,
    gpu: int, e0: dict | None = None,
):
    if spec.backend != "internal":
        raise RuntimeError(
            f"{spec.id} is fail-closed until its external source/full-state adapter is locked"
        )
    protocol = load_protocol()
    seed = int(protocol["seed"])
    e0 = e0 or create_e0(
        output_root=output_root, train_view=train_view, manifest_path=manifest_path,
        spec=spec, gpu=gpu,
    )
    seed_everything(seed)
    opt = build_options(
        spec, dataroot=train_view, option_root=Path(output_root) / "option_records",
        seed=seed, gpu=gpu,
    )
    primary, secondary = build_streams(opt, seed=seed)
    model = _build_model(opt, spec, primary, secondary)
    load_model_state(model, e0["model"], load_method=False)
    primary.load_state_dict(e0["samplers"]["primary"])
    secondary.load_state_dict(e0["samplers"]["secondary"])
    restore_rng(e0["rng"])
    model.set_search_step(0, int(protocol["training"]["target_updates"]))
    return model, primary, secondary, _annotated_rows(manifest_path)


def _capture_full_state(
    *, model, spec: LaneSpec, step: int, primary: SerializableDataStream,
    secondary: SerializableDataStream, metadata: dict,
) -> dict:
    protocol = load_protocol()
    payload = {
        "schema": FULL_STATE_SCHEMA,
        "lane": spec.to_dict(),
        "step": int(step),
        "physical_epoch_completed": int(step) // steps_per_epoch(protocol),
        "target_steps": int(protocol["training"]["target_updates"]),
        "model": model_state(model),
        "rng": capture_rng(),
        "samplers": {"primary": primary.state_dict(), "secondary": secondary.state_dict()},
        "metadata": copy.deepcopy(metadata),
    }
    assert_finite(payload["model"])
    return payload


def save_full_state(
    path: Path, *, model, spec: LaneSpec, step: int,
    primary: SerializableDataStream, secondary: SerializableDataStream,
    metadata: dict,
) -> dict:
    payload = _capture_full_state(
        model=model, spec=spec, step=step, primary=primary, secondary=secondary,
        metadata=metadata,
    )
    atomic_torch_save(path, payload)
    sidecar = {
        "schema": FULL_STATE_SCHEMA,
        "lane_id": spec.id,
        "step": int(step),
        "physical_epoch_completed": int(step) // steps_per_epoch(),
        "target_steps": int(load_protocol()["training"]["target_updates"]),
        "full_state_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": metadata,
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def load_full_state(
    path: Path, *, model, spec: LaneSpec, primary: SerializableDataStream,
    secondary: SerializableDataStream, expected_metadata: dict,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FULL_STATE_SCHEMA:
        raise RuntimeError("paper full-state schema mismatch")
    if payload.get("lane") != spec.to_dict():
        raise RuntimeError("paper lane identity mismatch")
    for key, expected in expected_metadata.items():
        if payload.get("metadata", {}).get(key) != expected:
            raise RuntimeError(f"paper checkpoint metadata mismatch for {key}")
    load_model_state(model, payload["model"])
    primary.load_state_dict(payload["samplers"]["primary"])
    secondary.load_state_dict(payload["samplers"]["secondary"])
    restore_rng(payload["rng"])
    return payload


def lane_metadata(
    *, spec: LaneSpec, manifest_path: Path, e0: dict, train_view: Path,
    data_root: Path,
) -> dict:
    protocol = load_protocol()
    return {
        "project_id": protocol["project_id"],
        "lane_id": spec.id,
        "lane_config_sha256": object_sha256(spec.to_dict()),
        "seed": int(protocol["seed"]),
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "e0_scientific_state_sha256": full_state_hash(e0),
        "git_commit": git_commit(),
        "train_view": str(Path(train_view).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "steps_per_data_epoch": steps_per_epoch(protocol),
        "target_updates": int(protocol["training"]["target_updates"]),
        "batch_size": 1,
        "sampling_measure": "official_image_proportional_unpaired",
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def optimizer_step(model, spec: LaneSpec, primary: SerializableDataStream, secondary: SerializableDataStream) -> None:
    first = primary.next()
    if spec.family == "unsb":
        model.set_input(first, secondary.next())
    elif spec.id in ("cyclegan", "cut"):
        model.set_input(first)
    else:
        raise RuntimeError(f"no optimizer adapter for {spec.id}")
    model.optimize_parameters()


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def train_spec(
    *, spec: LaneSpec, output_root: Path, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, resume: bool,
    engineering_stop_after_updates: int | None = None,
    gate_context: bool = False,
    authorization_kind: str = "lane",
) -> dict:
    """Train one frozen paper spec, including a dynamically locked candidate."""
    from .evaluate import evaluate_live_model

    protocol = load_protocol()
    lane_id = spec.id
    if spec.backend != "internal":
        raise RuntimeError(
            f"{lane_id} is blocked: external source/formula/full-state adapter is not locked"
        )
    target = int(protocol["training"]["target_updates"])
    stop = target if engineering_stop_after_updates is None else int(engineering_stop_after_updates)
    if stop <= 0 or stop > target:
        raise ValueError(f"engineering stop must be in [1,{target}]")
    output_root = Path(output_root).resolve()
    if not gate_context:
        if authorization_kind == "lane":
            from .gates import require_lane_authorization
            require_lane_authorization(output_root, lane_id)
        elif authorization_kind == "candidate":
            from .candidate_runtime import require_candidate_authorization
            require_candidate_authorization(output_root, lane_id)
        else:
            raise RuntimeError(f"unknown paper authorization kind: {authorization_kind}")
    lane_root = output_root / "lanes" / lane_id
    latest = lane_root / "full_state_latest.pt"
    if latest.is_file() and not resume:
        raise RuntimeError(f"existing paper lane requires --resume: {latest}")
    e0 = create_e0(
        output_root=output_root, train_view=train_view, manifest_path=manifest_path,
        spec=spec, gpu=gpu,
    )
    model, primary, secondary, rows = prepare_lane(
        output_root=output_root, train_view=train_view, manifest_path=manifest_path,
        spec=spec, gpu=gpu, e0=e0,
    )
    metadata = lane_metadata(
        spec=spec, manifest_path=manifest_path, e0=e0, train_view=train_view,
        data_root=data_root,
    )
    start = 0
    if resume and latest.is_file():
        payload = load_full_state(
            latest, model=model, spec=spec, primary=primary, secondary=secondary,
            expected_metadata={
                key: metadata[key]
                for key in (
                    "project_id", "lane_id", "lane_config_sha256", "seed",
                    "manifest_sha256", "protocol_fingerprint",
                    "e0_scientific_state_sha256", "git_commit",
                )
            },
        )
        start = int(payload["step"])
    if start >= stop:
        return {"status": "ALREADY_AT_REQUESTED_STOP", "lane_id": lane_id, "step": start}

    milestones = set(milestone_steps(protocol))
    started = time.time()
    epoch_started = time.time()
    last_losses = None
    for zero_step in range(start, stop):
        model.set_train_epoch(step_to_epoch(zero_step, protocol))
        model.set_search_step(zero_step, target)
        optimizer_step(model, spec, primary, secondary)
        completed = zero_step + 1
        last_losses = model.get_current_losses()
        at_epoch = completed % steps_per_epoch(protocol) == 0
        at_stop = completed == stop
        if not at_epoch and not at_stop:
            continue
        if at_epoch:
            model.update_learning_rate()
        sidecar = save_full_state(
            latest, model=model, spec=spec, step=completed,
            primary=primary, secondary=secondary, metadata=metadata,
        )
        completed_epoch = completed / steps_per_epoch(protocol)
        if completed in milestones:
            milestone_path = lane_root / "milestones" / f"e{int(completed_epoch):03d}.pt"
            save_full_state(
                milestone_path, model=model, spec=spec, step=completed,
                primary=primary, secondary=secondary, metadata=metadata,
            )
            evaluate_live_model(
                model=model, spec=spec, rows=rows, data_root=data_root,
                protocol_hash=evaluation_bundle_fingerprint(protocol),
                epoch=int(completed_epoch), lane_root=lane_root,
            )
        trace = {
            "lane_id": lane_id,
            "updates": completed,
            "data_epoch": completed_epoch,
            "epoch_wall_seconds": time.time() - epoch_started,
            "losses_last_update": last_losses,
            "scientific_state_sha256": sidecar["scientific_state_sha256"],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        append_jsonl(lane_root / "TRAIN_TRACE.jsonl", trace)
        write_json(lane_root / "HEARTBEAT.json", {
            **trace, "target_updates": target,
            "wall_seconds_this_call": time.time() - started,
        })
        epoch_started = time.time()

    result = {
        "status": "COMPLETE_E200" if stop == target else "ENGINEERING_PAUSE",
        "lane_id": lane_id,
        "start_updates": start,
        "final_updates": stop,
        "final_data_epoch": stop / steps_per_epoch(protocol),
        "target_updates": target,
        "wall_seconds_this_call": time.time() - started,
        "metadata": metadata,
        "confirmation20_opened": False,
    }
    write_json(lane_root / "RUN_STATE.json", result)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def train_lane(
    *, lane_id: str, output_root: Path, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, resume: bool,
    engineering_stop_after_updates: int | None = None,
    gate_context: bool = False,
) -> dict:
    return train_spec(
        spec=lane_spec(lane_id, load_protocol()), output_root=output_root,
        train_view=train_view, data_root=data_root, manifest_path=manifest_path,
        gpu=gpu, resume=resume,
        engineering_stop_after_updates=engineering_stop_after_updates,
        gate_context=gate_context, authorization_kind="lane",
    )
