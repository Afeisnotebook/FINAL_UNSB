"""Deterministic model/data runtime and exact full-state persistence."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .protocol import (
    FULL_STATE_SCHEMA,
    ROOT,
    ProbeSpec,
    file_sha256,
    load_protocol,
    object_sha256,
    protocol_fingerprint,
)


SRC = ROOT / "src"


def install_import_paths() -> None:
    for path in (str(SRC), str(ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def seed_everything(seed: int) -> dict:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    return {
        "seed": int(seed),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def capture_rng(*, include_cuda: bool = True) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all()
            if include_cuda and torch.cuda.is_available() else None
        ),
    }


def restore_rng(state: dict, *, include_cuda: bool = True) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if include_cuda and state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def isolated_cpu_rng(seed: int) -> dict:
    saved = capture_rng(include_cuda=False)
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    result = capture_rng(include_cuda=False)
    restore_rng(saved, include_cuda=False)
    return result


class SerializableDataStream:
    """Independent permutation and transform/unpaired-B RNG for one sample stream."""

    def __init__(self, dataset, *, seed: int, label: str):
        self.dataset = dataset
        self.seed = int(seed)
        self.label = str(label)
        self.order_rng = np.random.default_rng(self.seed)
        self.order: list[int] = []
        self.cursor = 0
        self.epoch = 0
        self.data_rng = isolated_cpu_rng(self.seed + 1_000_003)

    def _reshuffle(self) -> None:
        self.order = self.order_rng.permutation(len(self.dataset)).tolist()
        self.cursor = 0
        self.epoch += 1
        self.dataset.current_epoch = self.epoch

    @staticmethod
    def _batch(item: dict) -> dict:
        return {
            key: (value.unsqueeze(0) if torch.is_tensor(value) else [value])
            for key, value in item.items()
        }

    def next(self) -> dict:
        if self.cursor >= len(self.order):
            self._reshuffle()
        index = self.order[self.cursor]
        self.cursor += 1
        main = capture_rng(include_cuda=False)
        restore_rng(self.data_rng, include_cuda=False)
        try:
            item = self.dataset[index]
            self.data_rng = capture_rng(include_cuda=False)
        finally:
            restore_rng(main, include_cuda=False)
        return self._batch(item)

    def state_dict(self) -> dict:
        return {
            "label": self.label,
            "seed": self.seed,
            "order_rng": copy.deepcopy(self.order_rng.bit_generator.state),
            "order": list(self.order),
            "cursor": int(self.cursor),
            "epoch": int(self.epoch),
            "data_rng": self.data_rng,
        }

    def load_state_dict(self, state: dict) -> None:
        if state.get("label") != self.label or int(state["seed"]) != self.seed:
            raise RuntimeError(f"data-stream identity mismatch: {self.label}")
        self.order_rng.bit_generator.state = copy.deepcopy(state["order_rng"])
        self.order = list(state["order"])
        self.cursor = int(state["cursor"])
        self.epoch = int(state["epoch"])
        self.data_rng = state["data_rng"]
        self.dataset.current_epoch = self.epoch


def read_manifest(path: Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def manifest_report(path: Path, protocol: dict | None = None) -> dict:
    protocol = protocol or load_protocol()
    actual_hash = file_sha256(path)
    expected_hash = str(protocol["manifest"]["sha256"]).lower()
    if actual_hash.lower() != expected_hash:
        raise RuntimeError(f"manifest SHA256 mismatch: {actual_hash}")
    rows = read_manifest(path)
    expected = protocol["manifest"]["expected_per_domain"]
    domains = sorted({row["domain"] for row in rows})
    if len(domains) != int(protocol["local_view"]["domains"]):
        raise RuntimeError(f"expected six domains, got {domains}")
    counts = {}
    for domain in domains:
        counts[domain] = {
            split: sum(row["domain"] == domain and row["split"] == split for row in rows)
            for split in ("train", "discovery", "confirmation")
        }
        if counts[domain] != {key: int(value) for key, value in expected.items()}:
            raise RuntimeError(f"manifest split mismatch for {domain}: {counts[domain]}")
    return {"path": str(Path(path).resolve()), "sha256": actual_hash, "counts": counts}


def selected_train_names(rows: list[dict], per_domain: int) -> set[str]:
    selected: set[str] = set()
    for domain in sorted({row["domain"] for row in rows}):
        candidates = sorted(
            (row for row in rows if row["domain"] == domain and row["split"] == "train"),
            key=lambda row: int(row["order"]),
        )[: int(per_domain)]
        if len(candidates) != int(per_domain):
            raise RuntimeError(f"{domain}: expected {per_domain} train identities")
        selected.update(f'{domain}__{row["stem"]}' for row in candidates)
    return selected


def restrict_dataset(dataset, selected: set[str]) -> None:
    keep = lambda path: Path(path).stem in selected
    dataset.A_paths = [path for path in dataset.A_paths if keep(path)]
    dataset.B_paths = [path for path in dataset.B_paths if keep(path)]
    dataset.A_size = len(dataset.A_paths)
    dataset.B_size = len(dataset.B_paths)
    if dataset.A_size != len(selected) or dataset.B_size != len(selected):
        raise RuntimeError(
            f"small25 materialized view mismatch: A={dataset.A_size}, "
            f"B={dataset.B_size}, expected={len(selected)}"
        )


def _as_cli(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def option_args(
    spec: ProbeSpec, *, dataroot: Path, option_root: Path, seed: int, gpu: int,
    diagnostic_root: Path | None = None,
) -> list[str]:
    protocol = load_protocol()
    common = protocol["common"]
    args = [
        "--dataroot", str(dataroot),
        "--name", f"route1_{spec.id}",
        "--checkpoints_dir", str(option_root),
        "--model", spec.model,
        "--mode", str(common["mode"]),
        "--phase", "train",
        "--dataset_mode", str(common["dataset_mode"]),
        "--direction", str(common["direction"]),
        "--gpu_ids", str(gpu),
        "--seed", str(seed),
        "--batch_size", str(common["batch_size"]),
        "--num_threads", str(common["num_threads"]),
        "--n_epochs", str(common["n_epochs"]),
        "--n_epochs_decay", str(common["n_epochs_decay"]),
        "--lr", str(common["lr"]),
        "--beta1", str(common["beta1"]),
        "--beta2", str(common["beta2"]),
        "--lambda_GAN", str(common["lambda_GAN"]),
        "--lambda_SB", str(common["lambda_SB"]),
        "--lambda_NCE", str(common["lambda_NCE"]),
        "--tau", str(common["tau"]),
        "--num_timesteps", str(common["num_timesteps"]),
        "--nce_T", str(common["nce_T"]),
        "--nce_idt", _as_cli(common["nce_idt"]),
        "--nce_layers", str(common["nce_layers"]),
        "--num_patches", str(common["num_patches"]),
        "--netF", str(common["netF"]),
        "--netG", str(common["netG"]),
        "--netD", str(common["netD"]),
        "--load_size", str(common["load_size"]),
        "--crop_size", str(common["crop_size"]),
        "--preprocess", str(common["preprocess"]),
        "--display_id", "-1", "--no_html", "--no_flip",
        "--print_freq", "100000000", "--save_latest_freq", "100000000",
        "--save_epoch_freq", "100000000",
    ]
    ignored = {"physical_support_epochs"}
    for key, value in spec.method.items():
        if key in ignored:
            continue
        args.extend([f"--{key}", _as_cli(value)])
    if diagnostic_root is not None and spec.id == "hj":
        args += ["--hj_diag_out", str(diagnostic_root / "HJ_INTERNAL.jsonl")]
    if diagnostic_root is not None and spec.id == "dt":
        args += ["--dtcov_diag_out", str(diagnostic_root / "DT_INTERNAL.jsonl")]
    return args


def build_options(spec: ProbeSpec, **kwargs):
    install_import_paths()
    from options.train_options import TrainOptions

    return TrainOptions(cmd_line=option_args(spec, **kwargs)).parse()


def build_datasets(opt, rows: list[dict], per_domain: int):
    install_import_paths()
    from data.unaligned_dataset import UnalignedDataset

    selected = selected_train_names(rows, per_domain)
    primary = UnalignedDataset(opt)
    secondary = UnalignedDataset(opt)
    restrict_dataset(primary, selected)
    restrict_dataset(secondary, selected)
    return primary, secondary


def build_model(opt, ddi_primary: dict, ddi_secondary: dict):
    install_import_paths()
    from models import create_model

    model = create_model(opt)
    model.data_dependent_initialize(ddi_primary, ddi_secondary)
    model.setup(opt)
    model.parallelize()
    return model


def inner(net):
    return net.module if isinstance(net, torch.nn.DataParallel) else net


def cpu_clone(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_clone(item) for item in value)
    return copy.deepcopy(value)


def model_state(model) -> dict:
    return {
        "networks": {
            name: cpu_clone(inner(getattr(model, "net" + name)).state_dict())
            for name in model.model_names
        },
        "optimizers": [cpu_clone(optimizer.state_dict()) for optimizer in model.optimizers],
        "schedulers": [copy.deepcopy(scheduler.state_dict()) for scheduler in model.schedulers],
        "method": cpu_clone(model.get_extra_training_state()),
    }


def _optimizer_to(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_model_state(model, state: dict, *, load_method: bool = True) -> None:
    for name in model.model_names:
        inner(getattr(model, "net" + name)).load_state_dict(state["networks"][name], strict=True)
    if len(model.optimizers) != len(state["optimizers"]):
        raise RuntimeError("optimizer count mismatch")
    for optimizer, saved in zip(model.optimizers, state["optimizers"]):
        optimizer.load_state_dict(saved)
        _optimizer_to(optimizer, model.device)
    if len(model.schedulers) != len(state["schedulers"]):
        raise RuntimeError("scheduler count mismatch")
    for scheduler, saved in zip(model.schedulers, state["schedulers"]):
        scheduler.load_state_dict(saved)
    if load_method:
        model.load_extra_training_state(state.get("method", {}))


def assert_finite(value, path: str = "state") -> None:
    if torch.is_tensor(value):
        if torch.is_floating_point(value) and not torch.isfinite(value).all():
            raise RuntimeError(f"non-finite tensor in {path}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def _hash_update(digest, value: Any) -> None:
    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        digest.update(b"ndarray")
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _hash_update(digest, key)
            _hash_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _hash_update(digest, item)
    else:
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode("utf-8"))


def full_state_hash(payload: dict) -> str:
    digest = hashlib.sha256()
    _hash_update(digest, payload)
    return digest.hexdigest()


def capture_full_state(
    *, model, spec: ProbeSpec, step: int, target_steps: int,
    primary: SerializableDataStream, secondary: SerializableDataStream,
    metadata: dict,
) -> dict:
    payload = {
        "schema": FULL_STATE_SCHEMA,
        "probe": spec.to_dict(),
        "step": int(step),
        "physical_epoch_completed": int(step) // 150,
        "target_steps": int(target_steps),
        "model": model_state(model),
        "rng": capture_rng(),
        "samplers": {
            "primary": primary.state_dict(),
            "secondary": secondary.state_dict(),
        },
        "metadata": copy.deepcopy(metadata),
    }
    assert_finite(payload["model"])
    return payload


def atomic_torch_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def save_full_state(
    path: Path, *, model, spec: ProbeSpec, step: int, target_steps: int,
    primary: SerializableDataStream, secondary: SerializableDataStream,
    metadata: dict,
) -> dict:
    payload = capture_full_state(
        model=model, spec=spec, step=step, target_steps=target_steps,
        primary=primary, secondary=secondary, metadata=metadata,
    )
    atomic_torch_save(path, payload)
    sidecar = {
        "schema": FULL_STATE_SCHEMA,
        "probe_id": spec.id,
        "step": int(step),
        "physical_epoch_completed": int(step) // 150,
        "target_steps": int(target_steps),
        "full_state_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": metadata,
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def load_full_state(
    path: Path, *, model, spec: ProbeSpec,
    primary: SerializableDataStream, secondary: SerializableDataStream,
    expected_metadata: dict | None = None,
) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FULL_STATE_SCHEMA:
        raise RuntimeError("full-state schema mismatch")
    if payload.get("probe") != spec.to_dict():
        raise RuntimeError("probe identity mismatch")
    for key, expected in (expected_metadata or {}).items():
        if payload.get("metadata", {}).get(key) != expected:
            raise RuntimeError(f"checkpoint metadata mismatch for {key}")
    load_model_state(model, payload["model"])
    primary.load_state_dict(payload["samplers"]["primary"])
    secondary.load_state_dict(payload["samplers"]["secondary"])
    restore_rng(payload["rng"])
    return payload


def runtime_identity(manifest_path: Path, spec: ProbeSpec) -> dict:
    protocol = load_protocol()
    return {
        "project_id": protocol["project_id"],
        "probe_id": spec.id,
        "seed": int(protocol["seed"]),
        "manifest_sha256": file_sha256(manifest_path),
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "probe_config_sha256": object_sha256(spec.to_dict()),
    }
