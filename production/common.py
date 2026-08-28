"""Shared contract, identity and deterministic-runtime helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def load_json(path: Path | str) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(payload) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_source_sha256(path: Path) -> str:
    """Hash source text identically on Windows and Linux checkouts."""
    data = Path(path).read_bytes()
    # Source/control files do not always have a conventional text suffix
    # (for example .gitignore, *.example and .gitkeep).  Treat every
    # NUL-free file as text so the fingerprint is stable across Git's CRLF
    # conversion, while preserving binary artifacts byte-for-byte.
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNCOMMITTED"


def git_status() -> str:
    try:
        return subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "UNKNOWN"


def protocol_fingerprint() -> str:
    """Hash every executable/contract input while excluding run decisions.

    An authorization file can therefore be committed after the fingerprint is
    calculated without creating a self-referential commit-hash problem.
    """
    exact = {
        "PROJECT_CONTRACT.json", "DATA_CONTRACT.json", "COMPUTE_BUDGET.json",
        "CLAIM_BOUNDARIES.md", "HYPOTHESIS_LEDGER.json",
        "decisions/DEC-0001-FOUR-LANE-FREEZE.md",
    }
    roots = {"configs", "production", "src", "tools", "scripts", "environment", "manifests"}
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(ROOT).parts
        if any(part in {"__pycache__", ".pytest_cache"} for part in parts):
            continue
        if path.suffix in {".pyc", ".pyo"} or ".local." in path.name:
            continue
        if relative not in exact and parts[0] not in roots:
            continue
        rows.append((relative, portable_source_sha256(path)))
    return object_sha256(rows)


def apply_determinism(seed: int) -> dict:
    import torch

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
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
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def capture_rng_state() -> dict:
    import torch

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def unwrap(network):
    import torch

    return network.module if isinstance(network, torch.nn.DataParallel) else network


def state_tensor_sha256(networks: dict[str, object]) -> str:
    """Hash state tensors without depending on torch serialization metadata."""
    import torch

    digest = hashlib.sha256()
    for component in sorted(networks):
        digest.update(component.encode("utf-8"))
        state = unwrap(networks[component]).state_dict()
        for key in sorted(state):
            tensor = state[key].detach().cpu().contiguous()
            if torch.is_floating_point(tensor) and not torch.isfinite(tensor).all():
                raise RuntimeError(f"non-finite network tensor: {component}.{key}")
            digest.update(key.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def lane_record(lane_id: str) -> tuple[dict, dict]:
    contract = load_json("configs/FOUR_LANES.json")
    matches = [lane for lane in contract["lanes"] if lane["id"] == lane_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate lane: {lane_id}")
    return contract, matches[0]


def train_argv(
    *, lane_id: str, data_view: Path, run_root: Path, gpu_id: int,
    steps_per_epoch: int,
) -> tuple[list[str], dict]:
    contract, lane = lane_record(lane_id)
    common = contract["common"]
    args = [
        "--dataroot", str(data_view),
        "--name", lane_id,
        "--checkpoints_dir", str(run_root / lane_id / "network_checkpoints"),
        "--model", lane["model"],
        "--seed", str(contract["seed"]),
        "--gpu_ids", str(gpu_id),
        "--phase", "train",
        "--dataset_mode", "unaligned",
        "--num_threads", str(common["num_threads"]),
        "--batch_size", str(common["batch_size"]),
        "--load_size", str(common["load_size"]),
        "--crop_size", str(common["crop_size"]),
        "--preprocess", common["preprocess"],
        "--n_epochs", str(common["n_epochs"]),
        "--n_epochs_decay", str(common["n_epochs_decay"]),
        "--lr", str(common["lr"]),
        "--beta1", str(common["beta1"]),
        "--beta2", str(common["beta2"]),
        "--num_timesteps", str(common["num_timesteps"]),
        "--tau", str(common["tau"]),
        "--lambda_GAN", str(common["lambda_GAN"]),
        "--lambda_SB", str(common["lambda_SB"]),
        "--lambda_NCE", str(common["lambda_NCE"]),
        "--nce_idt", "true",
        "--nce_layers", common["nce_layers"],
        "--num_patches", str(common["num_patches"]),
        "--netF", common["netF"],
        "--netG", common["netG"],
        "--netD", common["netD"],
        "--no_flip",
        "--no_html",
        "--display_id", "-1",
        "--print_freq", "100",
        "--save_epoch_freq", "10000",
    ]
    method = lane["method"]
    if lane_id == "P1_HJ_HANDOFF":
        start = int(round(float(method["active_start_data_epoch"]) * steps_per_epoch))
        end = int(round(float(method["active_end_data_epoch"]) * steps_per_epoch))
        args.extend([
            "--hj_enable", "true",
            "--hj_layers", method["hj_layers"],
            "--hj_direction", method["hj_direction"],
            "--hj_scales", method["hj_scales"],
            "--hj_step", str(method["hj_step"]),
            "--hj_quantile", str(method["hj_quantile"]),
            "--hj_gate_quantile", str(method["hj_gate_quantile"]),
            "--hj_strength", str(method["hj_strength"]),
            "--hj_boundary_scale", str(method["hj_boundary_scale"]),
            "--hj_min_risk", str(method["hj_min_risk"]),
            "--hj_min_delta", str(method["hj_min_delta"]),
            "--hj_probe_mode", method["hj_probe_mode"],
            "--hj_control", method["hj_control"],
            "--hj_amplitude", method["hj_amplitude"],
            "--hj_update_mode", method["hj_update_mode"],
            "--hj_search_start_step", str(start),
            "--hj_search_duration_steps", str(end - start),
        ])
        lane = {**lane, "resolved_active_updates": [start, end]}
    elif lane_id == "P2_HNEK":
        args.extend([
            "--hnek_gamma", str(method["hnek_gamma"]),
            "--hnek_coord", method["hnek_coord"],
            "--hnek_horizon_mode", method["hnek_horizon_mode"],
            "--hnek_partial", method["hnek_partial"],
        ])
    elif lane_id == "P3_MACRO_MARGINAL":
        args.extend(["--macro_marginal", "true"])
    return args, lane


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)
