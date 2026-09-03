"""Source-bound deterministic adapter for the official DCLGAN implementation.

The upstream repository is intentionally not copied into FINAL_UNSB.  This
module verifies a separately cloned checkout against the frozen source lock,
then supplies the deterministic data stream and full-state persistence missing
from the original training script.  It does not authorize a GPU run; the
separate 1000-update GPU/capacity gate remains mandatory.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.local_route1.runtime import (  # noqa: E402
    SerializableDataStream,
    assert_finite,
    atomic_torch_save,
    capture_rng,
    cpu_clone,
    file_sha256,
    full_state_hash,
    read_manifest,
    restore_rng,
    seed_everything,
    write_json,
)
from research.paper_aio.protocol import (  # noqa: E402
    LaneSpec,
    evaluation_bundle_fingerprint,
    git_commit,
    load_protocol,
    object_sha256,
)
from research.paper_aio.runtime import append_jsonl  # noqa: E402


SOURCE_GATE_PATH = ROOT / "configs" / "PAPER_DCLGAN_NEGCUT_SOURCE_GATE.json"
FULL_STATE_SCHEMA = "final-unsb-paper-dclgan-full-state-v1"
E0_SCHEMA = "final-unsb-paper-dclgan-e0-v1"
ADAPTER_RECEIPT_SCHEMA = "final-unsb-paper-dclgan-adapter-receipt-v1"
LANE_ID = "dclgan"
EXPECTED_MANIFEST_SHA256 = (
    "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
)
ENGINEERING_MAX_UPDATES = 1_000
EXPECTED_REFLECTION_PAD_MODULES = 50


class DeterministicReflectionPad2d(torch.nn.Module):
    """Forward-equivalent reflection padding with deterministic CUDA backward.

    PyTorch 2.6 rejects ``ReflectionPad2d`` backward under deterministic CUDA
    execution. Slicing, flipping and concatenation produce the identical
    reflected tensor while retaining a deterministic autograd path.
    """

    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            values = (padding, padding, padding, padding)
        else:
            values = tuple(int(value) for value in padding)
            if len(values) == 2:
                values = (values[0], values[1], values[0], values[1])
            if len(values) != 4:
                raise ValueError(f"unsupported reflection padding: {padding!r}")
        if any(value < 0 for value in values):
            raise ValueError("reflection padding must be non-negative")
        self.padding = values

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.padding
        if left >= value.shape[-1] or right >= value.shape[-1]:
            raise RuntimeError("reflection width padding exceeds the input")
        horizontal = torch.cat((
            value[..., 1:left + 1].flip(-1) if left else value[..., :0],
            value,
            value[..., -right - 1:-1].flip(-1) if right else value[..., :0],
        ), dim=-1)
        if top >= horizontal.shape[-2] or bottom >= horizontal.shape[-2]:
            raise RuntimeError("reflection height padding exceeds the input")
        return torch.cat((
            horizontal[..., 1:top + 1, :].flip(-2)
            if top else horizontal[..., :0, :],
            horizontal,
            horizontal[..., -bottom - 1:-1, :].flip(-2)
            if bottom else horizontal[..., :0, :],
        ), dim=-2)

    def extra_repr(self) -> str:
        return f"padding={self.padding}"


def replace_reflection_padding(module: torch.nn.Module) -> int:
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, torch.nn.ReflectionPad2d):
            setattr(module, name, DeterministicReflectionPad2d(child.padding))
            replaced += 1
        else:
            replaced += replace_reflection_padding(child)
    return replaced


def _gate() -> dict[str, Any]:
    return json.loads(SOURCE_GATE_PATH.read_text(encoding="utf-8"))


def _git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.STDOUT,
    ).strip()


def verify_upstream(upstream_root: Path) -> dict[str, Any]:
    """Fail closed unless the external checkout is exactly the locked source."""
    upstream_root = Path(upstream_root).resolve()
    if not (upstream_root / ".git").exists():
        raise RuntimeError(f"DCLGAN upstream is not a git checkout: {upstream_root}")
    lock = _gate()["dclgan"]["source"]
    commit = _git("rev-parse", "HEAD", cwd=upstream_root)
    if commit != lock["commit"]:
        raise RuntimeError(f"DCLGAN commit mismatch: {commit}")
    dirty = _git("status", "--porcelain", "--untracked-files=no", cwd=upstream_root)
    if dirty:
        raise RuntimeError("DCLGAN tracked source is dirty")
    paths = {
        "readme_sha256": "README.md",
        "model_sha256": "models/dcl_model.py",
        "networks_sha256": "models/networks.py",
        "dataset_sha256": "data/unaligned_dataset.py",
        "train_sha256": "train.py",
        "license_file_sha256": "LICENSE",
    }
    observed: dict[str, str] = {}
    for key, relative in paths.items():
        path = upstream_root / relative
        if not path.is_file():
            raise RuntimeError(f"locked DCLGAN source missing: {relative}")
        observed[key] = file_sha256(path)
        if observed[key] != lock[key]:
            raise RuntimeError(f"DCLGAN source hash mismatch: {relative}")
    return {
        "repository": lock["repository"],
        "authority": lock["authority"],
        "commit": commit,
        "tracked_source_clean": True,
        "hashes": observed,
        "upstream_root": str(upstream_root),
    }


def verify_manifest_and_view(
    manifest_path: Path, train_view: Path, *, verify_content: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    train_view = Path(train_view).resolve()
    actual_manifest = file_sha256(manifest_path)
    if actual_manifest != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError(f"full manifest mismatch: {actual_manifest}")
    rows = [row for row in read_manifest(manifest_path) if row["split"] == "train"]
    if len(rows) != 8553:
        raise RuntimeError(f"DCLGAN manifest training count is {len(rows)}, expected 8553")
    counts: dict[str, int] = {}
    verified_files = 0
    for name, rel_key, bytes_key, hash_key in (
        ("trainA", "input_relpath", "input_bytes", "input_sha256"),
        ("trainB", "target_relpath", "target_bytes", "target_sha256"),
    ):
        directory = train_view / name
        if not directory.is_dir():
            raise RuntimeError(f"DCLGAN training view missing {name}: {directory}")
        actual = {path.name: path for path in directory.iterdir() if path.is_file()}
        expected = {
            f'{row["domain"]}__{row["stem"]}{Path(row[rel_key]).suffix}': row
            for row in rows
        }
        if len(expected) != len(rows) or set(actual) != set(expected):
            raise RuntimeError(f"DCLGAN {name} identity set differs from the manifest")
        counts[name] = len(actual)
        if counts[name] != 8553:
            raise RuntimeError(f"DCLGAN {name} count is {counts[name]}, expected 8553")
        for filename, row in expected.items():
            path = actual[filename]
            if path.stat().st_size != int(row[bytes_key]):
                raise RuntimeError(f"DCLGAN {name} byte count differs: {filename}")
            if verify_content:
                if file_sha256(path) != row[hash_key]:
                    raise RuntimeError(f"DCLGAN {name} content differs: {filename}")
                verified_files += 1
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_manifest,
        "train_view": str(train_view),
        "counts": counts,
        "identity_and_size_verified": True,
        "content_hashes_verified": bool(verify_content),
        "content_hash_files": verified_files,
        "confirmation_directory_enumerated": False,
        "paired_target_read": False,
    }


def adapter_fingerprint(
    *, upstream_receipt: dict[str, Any], manifest_path: Path,
) -> str:
    adaptation = _gate()["dclgan"]["controlled_paper_adaptation"]
    # Installation paths are host-local and cannot define a scientific
    # protocol.  Source authority, commit and file hashes remain bound.
    canonical_upstream = {
        key: value for key, value in upstream_receipt.items()
        if key != "upstream_root"
    }
    payload = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "adapter_source_sha256": file_sha256(Path(__file__)),
        "source_gate_sha256": file_sha256(SOURCE_GATE_PATH),
        "upstream": canonical_upstream,
        "manifest_sha256": file_sha256(manifest_path),
        "controlled_adaptation": adaptation,
        "full_state_schema": FULL_STATE_SCHEMA,
        "e0_schema": E0_SCHEMA,
    }
    return object_sha256(payload)


def dclgan_lane_spec() -> LaneSpec:
    return LaneSpec(
        id=LANE_ID,
        backend="author_source_bound_adapter",
        family="external",
        model="dcl",
        role="dual-contrastive unpaired image-translation baseline",
        method={},
        first_wave=False,
    )


def runtime_host_identity(gpu: int) -> dict[str, Any]:
    identity = {
        "hostname": platform.node(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_index": int(gpu),
        "gpu_name": None,
        "gpu_uuid": None,
    }
    if int(gpu) >= 0:
        if not torch.cuda.is_available():
            raise RuntimeError("DCLGAN GPU runtime requested but CUDA is unavailable")
        properties = torch.cuda.get_device_properties(int(gpu))
        identity["gpu_name"] = properties.name
        identity["gpu_uuid"] = str(properties.uuid)
    return identity


def annotated_manifest_rows(manifest_path: Path) -> list[dict[str, Any]]:
    """Add stable per-domain/per-split order without opening any image."""
    if file_sha256(manifest_path) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("DCLGAN evaluation manifest changed")
    counts: dict[tuple[str, str], int] = {}
    annotated = []
    for raw in read_manifest(manifest_path):
        row = dict(raw)
        key = (str(row["domain"]), str(row["split"]))
        row["order"] = str(counts.get(key, 0))
        counts[key] = counts.get(key, 0) + 1
        annotated.append(row)
    return annotated


def select_dclgan_evaluation_rows(
    rows: list[dict[str, Any]], *, split: str, count_per_domain: int,
) -> list[dict[str, Any]]:
    """Expose only the frozen discovery interface; confirmation fails first."""
    if split != "discovery":
        raise RuntimeError("DCLGAN confirmation20 access blocked")
    from research.paper_aio.evaluate import select_discovery

    return select_discovery(rows, int(count_per_domain))


def _install_upstream_imports(upstream_root: Path) -> None:
    upstream_root = Path(upstream_root).resolve()
    for prefix in ("data", "models", "options", "util"):
        existing = sys.modules.get(prefix)
        existing_file = Path(getattr(existing, "__file__", "")).resolve() if existing else None
        if existing_file and upstream_root not in existing_file.parents:
            raise RuntimeError(
                f"module namespace {prefix!r} was imported before DCLGAN source binding"
            )
    value = str(upstream_root)
    if value not in sys.path:
        sys.path.insert(0, value)


def build_options(
    *, upstream_root: Path, train_view: Path, option_root: Path, gpu: int,
):
    _install_upstream_imports(upstream_root)
    from options.train_options import TrainOptions

    args = [
        "--dataroot", str(Path(train_view).resolve()),
        "--name", "paper_dclgan",
        "--checkpoints_dir", str(Path(option_root).resolve()),
        "--model", "dcl",
        # Do not pass --DCL_mode: upstream declares choices='DCL', which argparse
        # interprets as three one-character choices.  Its default is DCL and
        # selects exactly the documented branch.
        "--phase", "train",
        "--dataset_mode", "unaligned",
        "--direction", "AtoB",
        "--gpu_ids", str(int(gpu)),
        "--batch_size", "1",
        "--num_threads", "0",
        "--n_epochs", "100",
        "--n_epochs_decay", "100",
        "--lr", "0.0002",
        "--beta1", "0.5",
        "--beta2", "0.999",
        "--gan_mode", "hinge",
        "--lambda_GAN", "1.0",
        "--lambda_NCE", "2.0",
        "--lambda_IDT", "1.0",
        "--nce_idt", "true",
        "--nce_layers", "4,8,12,16",
        "--nce_T", "0.07",
        "--num_patches", "256",
        "--netF", "mlp_sample",
        "--netG", "resnet_9blocks",
        "--netD", "basic",
        "--load_size", "128",
        "--crop_size", "128",
        "--preprocess", "resize_and_crop",
        "--display_id", "-1",
        "--no_html",
        "--no_flip",
        "--print_freq", "100000000",
        "--save_latest_freq", "100000000",
        "--save_epoch_freq", "100000000",
    ]
    # The locked upstream DCL model calls ``parser.parse_known_args()`` without
    # forwarding BaseOptions.cmd_line.  When this adapter is invoked as a CLI,
    # that call otherwise consumes the adapter's own flags (for example,
    # ``--output`` is abbreviated to upstream ``--output_nc``).  Isolate only
    # that broken ambient argv channel; BaseOptions still parses the complete
    # controlled argument list supplied above.  The upstream checkout remains
    # byte-for-byte source locked.
    ambient_argv = sys.argv
    try:
        sys.argv = [ambient_argv[0]]
        opt = TrainOptions(cmd_line=" ".join(args)).parse()
    finally:
        sys.argv = ambient_argv
    if opt.DCL_mode != "DCL" or not opt.nce_idt or opt.pool_size != 0:
        raise RuntimeError("DCLGAN upstream defaults no longer match the source lock")
    return opt


def build_stream(opt, *, seed: int) -> SerializableDataStream:
    from data.unaligned_dataset import UnalignedDataset

    dataset = UnalignedDataset(opt)
    if dataset.A_size != 8553 or dataset.B_size != 8553:
        raise RuntimeError(
            f"DCLGAN view mismatch: A={dataset.A_size}, B={dataset.B_size}"
        )
    return SerializableDataStream(dataset, seed=int(seed) + 101, label="dclgan_primary")


def _inner(network):
    return network.module if isinstance(network, torch.nn.DataParallel) else network


def capture_model_state(model) -> dict[str, Any]:
    state = {
        "networks": {
            name: cpu_clone(_inner(getattr(model, "net" + name)).state_dict())
            for name in model.model_names
        },
        "optimizers": [cpu_clone(optimizer.state_dict()) for optimizer in model.optimizers],
        "schedulers": [copy.deepcopy(scheduler.state_dict()) for scheduler in model.schedulers],
    }
    assert_finite(state)
    return state


def _optimizer_to(optimizer, device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_captured_model_state(model, state: dict[str, Any]) -> None:
    expected_names = list(model.model_names)
    if set(state["networks"]) != set(expected_names):
        raise RuntimeError("DCLGAN network-name mismatch")
    for name in expected_names:
        _inner(getattr(model, "net" + name)).load_state_dict(
            state["networks"][name], strict=True,
        )
    if len(model.optimizers) != len(state["optimizers"]):
        raise RuntimeError("DCLGAN optimizer-count mismatch")
    for optimizer, saved in zip(model.optimizers, state["optimizers"]):
        optimizer.load_state_dict(saved)
        _optimizer_to(optimizer, model.device)
    if len(model.schedulers) != len(state["schedulers"]):
        raise RuntimeError("DCLGAN scheduler-count mismatch")
    for scheduler, saved in zip(model.schedulers, state["schedulers"]):
        scheduler.load_state_dict(saved)


def build_model(opt, ddi_batch: dict[str, Any]):
    from models import create_model

    model = create_model(opt)
    replaced = sum(
        replace_reflection_padding(getattr(model, "net" + name))
        for name in model.model_names
    )
    if replaced != EXPECTED_REFLECTION_PAD_MODULES:
        raise RuntimeError(
            "DCLGAN deterministic reflection-pad coverage changed: "
            f"{replaced} != {EXPECTED_REFLECTION_PAD_MODULES}"
        )
    model._final_unsb_deterministic_reflection_pad_count = replaced
    model.data_dependent_initialize(ddi_batch)
    model.setup(opt)
    # Upstream's ``parallelize`` unconditionally constructs DataParallel and
    # crashes for its documented CPU mode (gpu_ids=[]).  Preserve the author
    # path on GPU and leave the same initialized modules unwrapped for the CPU
    # engineering gate.
    if opt.gpu_ids:
        model.parallelize()
    # The official implementation must retain its otherwise-surprising F
    # optimizer defaults (Adam lr=1e-3, betas=.9/.999).  Record and verify them
    # instead of silently normalizing them to the G/D optimizer.
    if len(model.optimizers) != 3:
        raise RuntimeError("DCLGAN expected G, D and F optimizers")
    feature_group = model.optimizers[2].param_groups[0]
    if feature_group["lr"] != 0.001 or tuple(feature_group["betas"]) != (0.9, 0.999):
        raise RuntimeError("DCLGAN upstream feature-optimizer defaults changed")
    return model


def _identity(
    *, upstream_receipt: dict[str, Any], manifest_path: Path, train_view: Path,
    gpu: int,
) -> dict[str, Any]:
    return {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "lane_id": LANE_ID,
        "seed": 2026,
        "manifest_sha256": file_sha256(manifest_path),
        "adapter_fingerprint": adapter_fingerprint(
            upstream_receipt=upstream_receipt, manifest_path=manifest_path,
        ),
        "upstream_commit": upstream_receipt["commit"],
        "adapter_git_commit": git_commit(),
        "train_view": str(Path(train_view).resolve()),
        "runtime_host": runtime_host_identity(gpu),
        "steps_per_data_epoch": 8553,
        "target_updates": 1710600,
        "sampling_measure": "official_image_proportional_unpaired",
        "sampler_initialization_policy": (
            "reuse_DDI_batch_for_first_optimizer_update"
        ),
        "deterministic_reflection_pad_adapter": {
            "operator": "slice_flip_concat_forward_equivalent_v1",
            "expected_replaced_modules": EXPECTED_REFLECTION_PAD_MODULES,
        },
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def create_e0(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    output_root: Path, gpu: int,
) -> dict[str, Any]:
    upstream_receipt = verify_upstream(upstream_root)
    verify_manifest_and_view(manifest_path, train_view)
    identity = _identity(
        upstream_receipt=upstream_receipt, manifest_path=manifest_path,
        train_view=train_view, gpu=gpu,
    )
    path = Path(output_root).resolve() / "shared_e0" / LANE_ID / "e0.pt"
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != E0_SCHEMA or payload.get("metadata") != identity:
            raise RuntimeError("DCLGAN e0 identity mismatch; use a new output root")
        return payload
    seed_everything(2026)
    opt = build_options(
        upstream_root=upstream_root, train_view=train_view,
        option_root=Path(output_root) / "option_records", gpu=gpu,
    )
    stream = build_stream(opt, seed=2026)
    sampler_before_ddi = copy.deepcopy(stream.state_dict())
    model = build_model(opt, stream.next())
    payload = {
        "schema": E0_SCHEMA,
        "metadata": identity,
        "model": capture_model_state(model),
        "rng": capture_rng(),
        "sampler": sampler_before_ddi,
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


def prepare(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    output_root: Path, gpu: int, e0: dict[str, Any],
):
    seed_everything(2026)
    opt = build_options(
        upstream_root=upstream_root, train_view=train_view,
        option_root=Path(output_root) / "option_records", gpu=gpu,
    )
    stream = build_stream(opt, seed=2026)
    model = build_model(opt, stream.next())
    load_captured_model_state(model, e0["model"])
    stream.load_state_dict(e0["sampler"])
    restore_rng(e0["rng"])
    return model, stream


def capture_full_state(
    *, model, stream: SerializableDataStream, step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": FULL_STATE_SCHEMA,
        "lane_id": LANE_ID,
        "step": int(step),
        "physical_epoch_completed": int(step) // 8553,
        "target_updates": 1710600,
        "model": capture_model_state(model),
        "rng": capture_rng(),
        "sampler": stream.state_dict(),
        "metadata": copy.deepcopy(metadata),
    }
    assert_finite(payload["model"])
    return payload


def save_full_state(
    path: Path, *, model, stream: SerializableDataStream, step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = capture_full_state(
        model=model, stream=stream, step=step, metadata=metadata,
    )
    atomic_torch_save(path, payload)
    sidecar = {
        "schema": FULL_STATE_SCHEMA,
        "lane_id": LANE_ID,
        "step": int(step),
        "physical_epoch_completed": int(step) // 8553,
        "target_updates": 1710600,
        "full_state_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": metadata,
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def load_full_state(
    path: Path, *, model, stream: SerializableDataStream,
    expected_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != FULL_STATE_SCHEMA or payload.get("lane_id") != LANE_ID:
        raise RuntimeError("DCLGAN full-state schema or lane mismatch")
    for key, value in expected_metadata.items():
        if payload.get("metadata", {}).get(key) != value:
            raise RuntimeError(f"DCLGAN checkpoint metadata mismatch: {key}")
    load_captured_model_state(model, payload["model"])
    stream.load_state_dict(payload["sampler"])
    restore_rng(payload["rng"])
    return payload


def train(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    output_root: Path, gpu: int, resume: bool, stop_after_updates: int,
) -> dict[str, Any]:
    stop = int(stop_after_updates)
    if stop <= 0 or stop > 1710600:
        raise ValueError("DCLGAN stop_after_updates must be in [1,1710600]")
    output_root = Path(output_root).resolve()
    upstream_receipt = verify_upstream(upstream_root)
    if stop > ENGINEERING_MAX_UPDATES:
        authorization_path = (
            output_root / "gates" / "DCLGAN_LONG_TRAINING_AUTHORIZATION.json"
        )
        if not authorization_path.is_file():
            raise RuntimeError(
                "DCLGAN long training is blocked until the 1000-update GPU, "
                "exact-resume, repeated-evaluation and confirmation gates authorize it"
            )
        authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
        if (
            authorization.get("status") != "PASS_LONG_TRAINING_AUTHORIZED"
            or authorization.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
            or authorization.get("adapter_git_commit") != git_commit()
            or authorization.get("adapter_fingerprint")
            != adapter_fingerprint(
                upstream_receipt=upstream_receipt, manifest_path=manifest_path,
            )
            or authorization.get("upstream_commit")
            != upstream_receipt["commit"]
            or authorization.get("runtime_host") != runtime_host_identity(gpu)
            or authorization.get("confirmation20_opened") is not False
        ):
            raise RuntimeError("DCLGAN long-training authorization is invalid")
    lane_root = output_root / "lanes" / LANE_ID
    latest = lane_root / "full_state_latest.pt"
    if latest.is_file() and not resume:
        raise RuntimeError(f"existing DCLGAN lane requires --resume: {latest}")
    e0 = create_e0(
        upstream_root=upstream_root, manifest_path=manifest_path,
        train_view=train_view, output_root=output_root, gpu=gpu,
    )
    metadata = _identity(
        upstream_receipt=upstream_receipt, manifest_path=manifest_path,
        train_view=train_view, gpu=gpu,
    )
    model, stream = prepare(
        upstream_root=upstream_root, manifest_path=manifest_path,
        train_view=train_view, output_root=output_root, gpu=gpu, e0=e0,
    )
    start = 0
    if resume and latest.is_file():
        saved = load_full_state(
            latest, model=model, stream=stream,
            expected_metadata={
                key: metadata[key]
                for key in (
                    "lane_id", "seed", "manifest_sha256", "adapter_fingerprint",
                    "upstream_commit", "adapter_git_commit", "train_view",
                    "runtime_host",
                )
            },
        )
        start = int(saved["step"])
    if start >= stop:
        return {"status": "ALREADY_AT_REQUESTED_STOP", "step": start}
    started = time.time()
    if gpu >= 0 and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(gpu)
    epoch_started = time.time()
    last_losses = None
    for zero_step in range(start, stop):
        model.set_input(stream.next())
        model.optimize_parameters()
        completed = zero_step + 1
        last_losses = model.get_current_losses()
        at_epoch = completed % 8553 == 0
        at_stop = completed == stop
        if not at_epoch and not at_stop:
            continue
        if at_epoch:
            model.update_learning_rate()
        sidecar = save_full_state(
            latest, model=model, stream=stream, step=completed, metadata=metadata,
        )
        if at_epoch and completed // 8553 in {1, 5, 10, 20, 40, 60, 80, 100, 125, 150, 175, 200}:
            save_full_state(
                lane_root / "milestones" / f"e{completed // 8553:03d}.pt",
                model=model, stream=stream, step=completed, metadata=metadata,
            )
        heartbeat = {
            "lane_id": LANE_ID,
            "updates": completed,
            "data_epoch": completed / 8553,
            "epoch_wall_seconds": time.time() - epoch_started,
            "losses_last_update": last_losses,
            "scientific_state_sha256": sidecar["scientific_state_sha256"],
            "paired_controller_access": False,
            "confirmation20_opened": False,
            "target_updates": 1710600,
        }
        append_jsonl(lane_root / "TRAIN_TRACE.jsonl", heartbeat)
        write_json(lane_root / "HEARTBEAT.json", heartbeat)
        epoch_started = time.time()
    result = {
        "status": "COMPLETE_E200" if stop == 1710600 else "ENGINEERING_PAUSE",
        "lane_id": LANE_ID,
        "start_updates": start,
        "final_updates": stop,
        "final_data_epoch": stop / 8553,
        "wall_seconds_this_call": time.time() - started,
        "updates_per_second": (stop - start) / max(time.time() - started, 1e-9),
        "runtime": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_index": int(gpu),
            "gpu_name": (
                torch.cuda.get_device_name(gpu)
                if gpu >= 0 and torch.cuda.is_available() else None
            ),
            "peak_allocated_bytes": (
                int(torch.cuda.max_memory_allocated(gpu))
                if gpu >= 0 and torch.cuda.is_available() else None
            ),
            "peak_reserved_bytes": (
                int(torch.cuda.max_memory_reserved(gpu))
                if gpu >= 0 and torch.cuda.is_available() else None
            ),
        },
        "metadata": metadata,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(lane_root / "RUN_STATE.json", result)
    return result


def exact_resume_gate(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    output_root: Path, gpu: int, total_updates: int, split_updates: int,
) -> dict[str, Any]:
    total_updates = int(total_updates)
    split_updates = int(split_updates)
    if not (0 < split_updates < total_updates):
        raise ValueError("DCLGAN exact-resume split must be inside total updates")
    # Establish the expensive per-image training-view proof once before the
    # three transition branches.  Each branch still rechecks manifest, names
    # and byte sizes before it touches a sample.
    upstream_receipt = verify_upstream(upstream_root)
    view_receipt = verify_manifest_and_view(
        manifest_path, train_view, verify_content=True,
    )
    root = Path(output_root).resolve() / "gates" / "DCLGAN_EXACT_RESUME"
    continuous = root / "continuous"
    resumed = root / "resumed"
    continuous_latest = continuous / "lanes" / LANE_ID / "full_state_latest.pt"
    train(
        upstream_root=upstream_root, manifest_path=manifest_path,
        train_view=train_view, output_root=continuous, gpu=gpu,
        resume=continuous_latest.is_file(),
        stop_after_updates=total_updates,
    )
    resumed_latest = resumed / "lanes" / LANE_ID / "full_state_latest.pt"
    resumed_step = 0
    if resumed_latest.is_file():
        resumed_sidecar_path = Path(str(resumed_latest) + ".json")
        resumed_step = int(json.loads(
            resumed_sidecar_path.read_text(encoding="utf-8")
        )["step"])
    if resumed_step < split_updates:
        train(
            upstream_root=upstream_root, manifest_path=manifest_path,
            train_view=train_view, output_root=resumed, gpu=gpu,
            resume=resumed_latest.is_file(), stop_after_updates=split_updates,
        )
    train(
        upstream_root=upstream_root, manifest_path=manifest_path,
        train_view=train_view, output_root=resumed, gpu=gpu, resume=True,
        stop_after_updates=total_updates,
    )
    continuous_sidecar = json.loads(
        (continuous / "lanes" / LANE_ID / "full_state_latest.pt.json").read_text(
            encoding="utf-8"
        )
    )
    resumed_sidecar = json.loads(
        (resumed / "lanes" / LANE_ID / "full_state_latest.pt.json").read_text(
            encoding="utf-8"
        )
    )
    exact = (
        continuous_sidecar["scientific_state_sha256"]
        == resumed_sidecar["scientific_state_sha256"]
    )
    receipt = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "status": "PASS_EXACT_RESUME" if exact else "FAIL_EXACT_RESUME",
        "total_updates": total_updates,
        "split_updates": split_updates,
        "continuous_scientific_state_sha256": continuous_sidecar[
            "scientific_state_sha256"
        ],
        "resumed_scientific_state_sha256": resumed_sidecar[
            "scientific_state_sha256"
        ],
        "exact": exact,
        "training_view_content_hashes_verified": view_receipt[
            "content_hashes_verified"
        ],
        "training_view_content_hash_files": view_receipt["content_hash_files"],
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "upstream_commit": upstream_receipt["commit"],
        "adapter_git_commit": git_commit(),
        "adapter_fingerprint": adapter_fingerprint(
            upstream_receipt=upstream_receipt, manifest_path=manifest_path,
        ),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(root / "EXACT_RESUME_RECEIPT.json", receipt)
    if not exact:
        raise RuntimeError("DCLGAN exact resume mismatch")
    return receipt


def _load_evaluation_runtime(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    output_root: Path, checkpoint: Path, gpu: int,
):
    upstream_receipt = verify_upstream(upstream_root)
    verify_manifest_and_view(manifest_path, train_view)
    expected = _identity(
        upstream_receipt=upstream_receipt, manifest_path=manifest_path,
        train_view=train_view, gpu=gpu,
    )
    seed_everything(2026)
    opt = build_options(
        upstream_root=upstream_root, train_view=train_view,
        option_root=Path(output_root) / "evaluation_option_records", gpu=gpu,
    )
    stream = build_stream(opt, seed=2026)
    model = build_model(opt, stream.next())
    payload = load_full_state(
        Path(checkpoint).resolve(), model=model, stream=stream,
        expected_metadata={
            key: expected[key]
            for key in (
                "lane_id", "seed", "manifest_sha256", "adapter_fingerprint",
                "upstream_commit", "adapter_git_commit",
            )
        },
    )
    metadata = payload.get("metadata", {})
    if (
        metadata.get("paired_controller_access") is not False
        or metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("DCLGAN checkpoint violates the paper access boundary")
    return model, stream, payload


def _verify_discovery_content(
    *, rows: list[dict[str, Any]], data_root: Path, count_per_domain: int,
) -> dict[str, Any]:
    selected = select_dclgan_evaluation_rows(
        rows, split="discovery", count_per_domain=count_per_domain,
    )
    checked = 0
    for row in selected:
        for rel_key, bytes_key, hash_key in (
            ("input_relpath", "input_bytes", "input_sha256"),
            ("target_relpath", "target_bytes", "target_sha256"),
        ):
            path = Path(data_root).resolve() / row[rel_key]
            if not path.is_file() or path.stat().st_size != int(row[bytes_key]):
                raise RuntimeError(f"DCLGAN discovery file missing or wrong size: {path}")
            if file_sha256(path) != row[hash_key]:
                raise RuntimeError(f"DCLGAN discovery content differs: {path}")
            checked += 1
    return {
        "selected_images": len(selected),
        "content_hash_files": checked,
        "split": "discovery",
        "confirmation20_opened": False,
    }


def _evaluate_once(
    *, model, rows: list[dict[str, Any]], data_root: Path,
    count_per_domain: int, include_lpips: bool,
) -> dict[str, Any]:
    from research.paper_aio.evaluate import evaluate_model

    return evaluate_model(
        model, spec=dclgan_lane_spec(), rows=rows,
        data_root=Path(data_root).resolve(),
        protocol_hash=evaluation_bundle_fingerprint(load_protocol()),
        count_per_domain=int(count_per_domain), replicates=1,
        nfe_values=[1], include_lpips=bool(include_lpips),
    )


def evaluate_checkpoint(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    data_root: Path, output_root: Path, checkpoint: Path, gpu: int,
    count_per_domain: int, include_lpips: bool,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    checkpoint_sha256_before = file_sha256(checkpoint)
    rows = annotated_manifest_rows(manifest_path)
    discovery = _verify_discovery_content(
        rows=rows, data_root=data_root, count_per_domain=count_per_domain,
    )
    model, stream, payload = _load_evaluation_runtime(
        upstream_root=upstream_root, manifest_path=manifest_path,
        train_view=train_view, output_root=output_root,
        checkpoint=checkpoint, gpu=gpu,
    )
    state_before = full_state_hash(capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    result = _evaluate_once(
        model=model, rows=rows, data_root=data_root,
        count_per_domain=count_per_domain, include_lpips=include_lpips,
    )
    state_after = full_state_hash(capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    checkpoint_sha256_after = file_sha256(checkpoint)
    if state_before != state_after or checkpoint_sha256_before != checkpoint_sha256_after:
        raise RuntimeError("DCLGAN evaluation mutated its source scientific state")
    epoch = int(payload["step"]) // 8553
    metric_path = (
        Path(output_root).resolve() / "lanes" / LANE_ID / "metrics"
        / f"e{epoch:03d}_discovery{int(count_per_domain)}.json"
    )
    write_json(metric_path, result)
    receipt = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "status": "PASS_SOURCE_BOUND_EVALUATION",
        "lane_id": LANE_ID,
        "epoch": epoch,
        "updates": int(payload["step"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256_before,
        "scientific_state_sha256_before": state_before,
        "scientific_state_sha256_after": state_after,
        "evaluation_result": str(metric_path),
        "evaluation_result_sha256": file_sha256(metric_path),
        "evaluation_input_sha256": result["evaluation_input_sha256"],
        "discovery_content": discovery,
        "performance_values_used_for_training_or_scheduling": False,
        "confirmation20_opened": False,
    }
    write_json(
        Path(output_root).resolve() / "gates" / "DCLGAN_EVALUATION_RECEIPT.json",
        receipt,
    )
    return receipt


def evaluation_repeat_gate(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    data_root: Path, output_root: Path, checkpoint: Path, gpu: int,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint).resolve()
    checkpoint_sha256_before = file_sha256(checkpoint)
    rows = annotated_manifest_rows(manifest_path)
    discovery = _verify_discovery_content(
        rows=rows, data_root=data_root, count_per_domain=70,
    )
    model, stream, payload = _load_evaluation_runtime(
        upstream_root=upstream_root, manifest_path=manifest_path,
        train_view=train_view, output_root=output_root,
        checkpoint=checkpoint, gpu=gpu,
    )
    state_before = full_state_hash(capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    first = _evaluate_once(
        model=model, rows=rows, data_root=data_root,
        count_per_domain=70, include_lpips=False,
    )
    state_middle = full_state_hash(capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    second = _evaluate_once(
        model=model, rows=rows, data_root=data_root,
        count_per_domain=70, include_lpips=False,
    )
    state_after = full_state_hash(capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    first_hash = object_sha256(first)
    second_hash = object_sha256(second)
    checkpoint_sha256_after = file_sha256(checkpoint)
    exact = (
        first_hash == second_hash
        and state_before == state_middle == state_after
        and checkpoint_sha256_before == checkpoint_sha256_after
    )
    receipt = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "status": "PASS_EVALUATION_REPEAT_EXACT" if exact else "FAIL_EVALUATION_REPEAT",
        "lane_id": LANE_ID,
        "updates": int(payload["step"]),
        "checkpoint_sha256": checkpoint_sha256_before,
        "first_result_sha256": first_hash,
        "second_result_sha256": second_hash,
        "scientific_state_sha256_before": state_before,
        "scientific_state_sha256_middle": state_middle,
        "scientific_state_sha256_after": state_after,
        "evaluation_input_sha256": first["evaluation_input_sha256"],
        "evaluation_bundle_fingerprint": evaluation_bundle_fingerprint(
            load_protocol()
        ),
        "manifest_sha256": payload["metadata"]["manifest_sha256"],
        "upstream_commit": payload["metadata"]["upstream_commit"],
        "adapter_git_commit": payload["metadata"]["adapter_git_commit"],
        "adapter_fingerprint": payload["metadata"]["adapter_fingerprint"],
        "discovery_content": discovery,
        "paired_target_access_scope": "determinism_gate_only",
        "performance_values_used_for_training_or_scheduling": False,
        "confirmation20_opened": False,
        "exact": exact,
    }
    write_json(
        Path(output_root).resolve() / "gates" / "DCLGAN_EVALUATION_REPEAT.json",
        receipt,
    )
    if not exact:
        raise RuntimeError("DCLGAN repeated evaluation or read-only state differs")
    return receipt


def confirmation_lock_gate(*, output_root: Path) -> dict[str, Any]:
    rejected = False
    try:
        select_dclgan_evaluation_rows(
            [], split="confirmation", count_per_domain=20,
        )
    except RuntimeError as error:
        rejected = "confirmation20 access blocked" in str(error)
    receipt = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "status": "PASS_CONFIRMATION20_UNADDRESSABLE" if rejected else "FAIL",
        "attempt_rejected_before_manifest_or_image_access": rejected,
        "adapter_git_commit": git_commit(),
        "adapter_source_sha256": file_sha256(Path(__file__)),
        "source_gate_sha256": file_sha256(SOURCE_GATE_PATH),
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(
        Path(output_root).resolve() / "gates" / "DCLGAN_CONFIRMATION_LOCK.json",
        receipt,
    )
    if not rejected:
        raise RuntimeError("DCLGAN confirmation20 interface did not fail closed")
    return receipt


def authorize_long_training(
    *, upstream_root: Path, manifest_path: Path, output_root: Path,
) -> dict[str, Any]:
    """Authorize e200 only after every source-bound GPU gate is complete."""
    output_root = Path(output_root).resolve()
    upstream_receipt = verify_upstream(upstream_root)
    expected_fingerprint = adapter_fingerprint(
        upstream_receipt=upstream_receipt, manifest_path=manifest_path,
    )
    paths = {
        "preflight": output_root / "gates" / "DCLGAN_PREFLIGHT.json",
        "resume": (
            output_root / "gates" / "DCLGAN_EXACT_RESUME"
            / "EXACT_RESUME_RECEIPT.json"
        ),
        "evaluation": output_root / "gates" / "DCLGAN_EVALUATION_REPEAT.json",
        "confirmation": output_root / "gates" / "DCLGAN_CONFIRMATION_LOCK.json",
        "run": output_root / "lanes" / LANE_ID / "RUN_STATE.json",
        "checkpoint": output_root / "lanes" / LANE_ID / "full_state_latest.pt",
    }
    missing = [label for label, path in paths.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"DCLGAN authorization artifacts missing: {missing}")
    values = {
        label: json.loads(path.read_text(encoding="utf-8"))
        for label, path in paths.items()
        if label != "checkpoint"
    }
    failures = []
    preflight = values["preflight"]
    resume = values["resume"]
    evaluation = values["evaluation"]
    confirmation = values["confirmation"]
    run = values["run"]
    if (
        preflight.get("status") != "PASS_SOURCE_AND_CONTROLLED_DATA_PREFLIGHT"
        or preflight.get("adapter_fingerprint") != expected_fingerprint
        or preflight.get("data", {}).get("content_hashes_verified") is not True
        or int(preflight.get("data", {}).get("content_hash_files", -1)) != 17_106
    ):
        failures.append("source/data preflight is absent or stale")
    if (
        resume.get("status") != "PASS_EXACT_RESUME"
        or resume.get("exact") is not True
        or int(resume.get("total_updates", -1)) != 1_000
        or int(resume.get("split_updates", -1)) != 500
        or resume.get("adapter_fingerprint") != expected_fingerprint
    ):
        failures.append("1000-vs-500+resume GPU gate is absent or stale")
    if (
        evaluation.get("status") != "PASS_EVALUATION_REPEAT_EXACT"
        or evaluation.get("exact") is not True
        or int(evaluation.get("updates", -1)) != 1_000
        or evaluation.get("adapter_fingerprint") != expected_fingerprint
    ):
        failures.append("1000-update repeated evaluation gate is absent or stale")
    if (
        confirmation.get("status") != "PASS_CONFIRMATION20_UNADDRESSABLE"
        or confirmation.get("adapter_git_commit") != git_commit()
        or confirmation.get("adapter_source_sha256") != file_sha256(Path(__file__))
        or confirmation.get("confirmation20_opened") is not False
    ):
        failures.append("confirmation20 lock is absent or stale")
    runtime = run.get("runtime", {})
    run_runtime_host = run.get("metadata", {}).get("runtime_host")
    if (
        run.get("status") != "ENGINEERING_PAUSE"
        or int(run.get("final_updates", -1)) != 1_000
        or run.get("metadata", {}).get("adapter_fingerprint") != expected_fingerprint
        or run_runtime_host != runtime_host_identity(int(runtime.get("gpu_index", -1)))
        or runtime.get("cuda_available") is not True
        or not isinstance(runtime.get("peak_allocated_bytes"), int)
        or not math.isfinite(float(run.get("updates_per_second", float("nan"))))
        or float(run.get("updates_per_second", 0.0)) <= 0.0
    ):
        failures.append("1000-update GPU throughput/capacity run is invalid")
    checkpoint_gib = paths["checkpoint"].stat().st_size / (1024 ** 3)
    completed_epoch = int(run["final_updates"]) // 8553
    remaining_milestones = sum(
        int(epoch) > completed_epoch
        for epoch in load_protocol()["training"]["milestone_epochs"]
    )
    estimated_remaining_write_gib = (
        checkpoint_gib * (remaining_milestones + 2) * 1.25 + 5.0
    )
    free_gib = shutil.disk_usage(output_root).free / (1024 ** 3)
    if free_gib <= estimated_remaining_write_gib:
        failures.append("actual free disk cannot cover worst-case remaining writes")
    authorization = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "status": "PASS_LONG_TRAINING_AUTHORIZED" if not failures else "FAIL",
        "lane_id": LANE_ID,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "upstream_commit": upstream_receipt["commit"],
        "adapter_git_commit": git_commit(),
        "adapter_fingerprint": expected_fingerprint,
        "gate_sha256": {
            label: file_sha256(path)
            for label, path in paths.items()
            if label != "checkpoint"
        },
        "checkpoint_sha256": file_sha256(paths["checkpoint"]),
        "gpu_runtime": runtime,
        "runtime_host": run_runtime_host,
        "updates_per_second": run.get("updates_per_second"),
        "disk": {
            "free_gib": free_gib,
            "estimated_remaining_write_gib": estimated_remaining_write_gib,
            "fixed_200_gib_threshold_used": False,
        },
        "failures": failures,
        "performance_values_used_for_training_or_scheduling": False,
        "confirmation20_opened": False,
    }
    path = output_root / "gates" / "DCLGAN_LONG_TRAINING_AUTHORIZATION.json"
    write_json(path, authorization)
    if failures:
        raise RuntimeError(f"DCLGAN long-training authorization failed: {failures}")
    return authorization


def preflight(
    *, upstream_root: Path, manifest_path: Path, train_view: Path,
    output_root: Path,
) -> dict[str, Any]:
    source = verify_upstream(upstream_root)
    data = verify_manifest_and_view(
        manifest_path, train_view, verify_content=True,
    )
    receipt = {
        "schema": ADAPTER_RECEIPT_SCHEMA,
        "status": "PASS_SOURCE_AND_CONTROLLED_DATA_PREFLIGHT",
        "source": source,
        "data": data,
        "adapter_git_commit": git_commit(),
        "adapter_fingerprint": adapter_fingerprint(
            upstream_receipt=source, manifest_path=manifest_path,
        ),
        "training_authorized": False,
        "gpu_gate_required": True,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_root).resolve() / "gates" / "DCLGAN_PREFLIGHT.json", receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "preflight", "train", "exact-resume-gate", "evaluate",
            "evaluation-repeat-gate", "confirmation-lock-gate", "authorize",
        ),
        required=True,
    )
    parser.add_argument("--upstream-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int, default=1710600)
    parser.add_argument("--gate-total-updates", type=int, default=4)
    parser.add_argument("--gate-split-updates", type=int, default=2)
    parser.add_argument("--count-per-domain", type=int, choices=(70, 80), default=70)
    parser.add_argument("--include-lpips", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    common = {
        "upstream_root": args.upstream_root,
        "manifest_path": args.manifest,
        "train_view": args.train_view,
        "output_root": args.output,
    }
    if args.stage == "preflight":
        result = preflight(**common)
    elif args.stage == "train":
        result = train(
            **common, gpu=args.gpu, resume=args.resume,
            stop_after_updates=args.stop_after_updates,
        )
    elif args.stage == "exact-resume-gate":
        result = exact_resume_gate(
            **common, gpu=args.gpu, total_updates=args.gate_total_updates,
            split_updates=args.gate_split_updates,
        )
    elif args.stage == "confirmation-lock-gate":
        result = confirmation_lock_gate(output_root=args.output)
    elif args.stage == "authorize":
        result = authorize_long_training(
            upstream_root=args.upstream_root,
            manifest_path=args.manifest,
            output_root=args.output,
        )
    else:
        if args.data_root is None:
            raise RuntimeError(f"DCLGAN {args.stage} requires --data-root")
        checkpoint = args.checkpoint or (
            args.output / "lanes" / LANE_ID / "full_state_latest.pt"
        )
        if args.stage == "evaluate":
            result = evaluate_checkpoint(
                **common, data_root=args.data_root, checkpoint=checkpoint,
                gpu=args.gpu, count_per_domain=args.count_per_domain,
                include_lpips=args.include_lpips,
            )
        else:
            result = evaluation_repeat_gate(
                **common, data_root=args.data_root, checkpoint=checkpoint,
                gpu=args.gpu,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
