"""Durably evaluate imported DCLGAN milestones in the common paper runtime.

The training adapter remains frozen at its own Git commit.  This controller
loads that exact adapter source, waits for the verified checkpoint relay and
the first-wave evaluation cohort, then evaluates e100/125/150/175/200 under
the shared GPU lock.  Metrics are never exposed to training or scheduling.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from operations import paper_aio_export_relay as relay  # noqa: E402


CONTRACT_SCHEMA = "final-unsb-paper-dclgan-evaluation-contract-v1"
STATE_SCHEMA = "final-unsb-paper-dclgan-evaluation-successor-v1"
RESULT_SCHEMA = "final-unsb-paper-dclgan-result-v1"
UNIFIED_RECEIPT_SCHEMA = "final-unsb-paper-unified-evaluation-receipt-v1"
IMPORT_LANE_SCHEMA = "final-unsb-paper-imported-lane-v1"
COHORT_SCHEMA = "final-unsb-paper-unified-evaluation-cohort-v1"
COMPLETE_STATUS = "COMPLETE_DCLGAN_FIXED_EVALUATION_SET"
EPOCHS = (100, 125, 150, 175, 200)
STEPS_PER_EPOCH = 8553


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def immutable_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.is_file():
        if read_json(path) != value:
            raise RuntimeError(f"immutable DCLGAN artifact differs: {path}")
        return
    atomic_json(path, value)


def git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True, stderr=subprocess.STDOUT,
    ).strip()


def inside(path: Path, root: Path, label: str) -> Path:
    path = Path(path).resolve()
    root = Path(root).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} escapes frozen import root: {path}") from error
    return path


def expected_schedule(epoch: int) -> dict[str, Any]:
    if int(epoch) not in EPOCHS:
        raise ValueError(f"unexpected DCLGAN paper epoch: {epoch}")
    return {
        "count_per_domain": 80 if int(epoch) == 200 else 70,
        "replicates": 5 if int(epoch) == 200 else 1,
        "nfe_values": [1],
        "include_lpips": True,
    }


def validate_import_lane(
    import_root: Path, *, source_host_label: str,
    required_training_commit: str, required_adapter_fingerprint: str,
) -> list[dict[str, Any]]:
    import_root = Path(import_root).resolve()
    lane_path = import_root / "sources" / source_host_label / "dclgan" / "IMPORT_LANE.json"
    value = read_json(lane_path)
    if (
        value.get("schema") != IMPORT_LANE_SCHEMA
        or value.get("status") != "COMPLETE_VERIFIED_IMPORTED_LANE"
        or value.get("source_host_label") != source_host_label
        or value.get("lane_id") != "dclgan"
        or value.get("epochs") != list(EPOCHS)
        or value.get("checkpoint_copy_performed") is not True
        or value.get("source_checkpoint_mutation") is not False
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("invalid imported DCLGAN lane")
    rows = value.get("imports")
    if not isinstance(rows, list) or len(rows) != len(EPOCHS):
        raise RuntimeError("incomplete imported DCLGAN epoch set")
    by_epoch: dict[int, dict[str, Any]] = {}
    for row in rows:
        epoch = int(row.get("epoch", -1))
        if epoch in by_epoch or epoch not in EPOCHS:
            raise RuntimeError("duplicate or unexpected imported DCLGAN epoch")
        receipt_path = inside(row["export_receipt"], import_root, "export receipt")
        checkpoint = inside(row["checkpoint"], import_root, "checkpoint")
        sidecar = inside(row["sidecar"], import_root, "sidecar")
        if (
            not receipt_path.is_file()
            or file_sha256(receipt_path) != row.get("export_receipt_sha256")
            or not checkpoint.is_file()
            or file_sha256(checkpoint) != row.get("checkpoint_sha256")
            or not sidecar.is_file()
            or file_sha256(sidecar) != row.get("sidecar_sha256")
        ):
            raise RuntimeError(f"imported DCLGAN files changed at e{epoch}")
        receipt = read_json(receipt_path)
        relay.validate_export_receipt(
            receipt, lane_id="dclgan", epoch=epoch,
            source_host_label=source_host_label,
        )
        if (
            receipt.get("training_git_commit") != required_training_commit
            or receipt.get("training_protocol_fingerprint")
            != required_adapter_fingerprint
            or receipt.get("scientific_state_sha256")
            != row.get("scientific_state_sha256")
        ):
            raise RuntimeError(f"imported DCLGAN identity differs at e{epoch}")
        by_epoch[epoch] = {
            "epoch": epoch,
            "export_receipt": receipt_path,
            "checkpoint": checkpoint,
            "checkpoint_sha256": row["checkpoint_sha256"],
            "scientific_state_sha256": row["scientific_state_sha256"],
        }
    if tuple(sorted(by_epoch)) != EPOCHS:
        raise RuntimeError("imported DCLGAN epochs differ from the frozen set")
    return [by_epoch[epoch] for epoch in EPOCHS]


def load_frozen_adapter(adapter_repo: Path):
    adapter_repo = Path(adapter_repo).resolve()
    source = adapter_repo / "operations" / "paper_aio_dclgan_adapter.py"
    if not source.is_file():
        raise RuntimeError("frozen DCLGAN adapter source is missing")
    if str(adapter_repo) not in sys.path:
        sys.path.insert(0, str(adapter_repo))
    name = "_final_unsb_frozen_dclgan_adapter"
    existing = sys.modules.get(name)
    if existing is not None:
        if Path(existing.__file__).resolve() != source.resolve():
            raise RuntimeError("a different frozen DCLGAN adapter is already loaded")
        return existing
    specification = importlib.util.spec_from_file_location(name, source)
    if specification is None or specification.loader is None:
        raise RuntimeError("cannot load frozen DCLGAN adapter")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    if Path(module.ROOT).resolve() != adapter_repo:
        raise RuntimeError("frozen DCLGAN adapter resolved a different repository")
    # The frozen adapter imports ``git_commit`` from its own paper protocol.
    # When it is loaded into a newer evaluation-control process, that protocol
    # module may already be cached from the control checkout.  Without this
    # explicit source binding, the adapter then reports the control commit and
    # rejects a checkpoint whose metadata correctly names the frozen adapter
    # commit.  Bind only the identity helper to the checkout that supplied the
    # already hash-verified adapter source; model and evaluation semantics are
    # unchanged.
    adapter_commit = git(adapter_repo, "rev-parse", "HEAD")
    module.git_commit = lambda commit=adapter_commit: commit
    module._final_unsb_adapter_git_identity_source = str(adapter_repo)
    return module


def reference_cohort_ready(reference_output: Path) -> bool:
    path = Path(reference_output) / "gates" / "UNIFIED_EVALUATION_COHORT.json"
    if not path.is_file():
        return False
    value = read_json(path)
    if (
        value.get("schema") != COHORT_SCHEMA
        or value.get("status") != "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT"
        or value.get("cross_host_training_delta_merged") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("first-wave reference cohort is invalid")
    return True


def crn_identity(metric: dict[str, Any]) -> list[tuple[Any, ...]]:
    by_key: dict[tuple[Any, ...], str] = {}
    for row in metric.get("images", []):
        key = (
            row.get("domain"), row.get("stem"), int(row.get("order", -1)),
            int(row.get("replicate", -1)),
        )
        bundle = str(row.get("crn_bundle_sha256", ""))
        if key in by_key and by_key[key] != bundle:
            raise RuntimeError("one sample/replicate has inconsistent CRN bundles")
        by_key[key] = bundle
    return [(*key, value) for key, value in sorted(by_key.items())]


def validate_common_reference(
    metric: dict[str, Any], *, reference_output: Path, epoch: int,
) -> None:
    reference_path = (
        Path(reference_output) / "lanes" / "plain" / "metrics" / f"e{epoch:03d}.json"
    )
    reference = read_json(reference_path)
    if (
        metric.get("evaluation_input_sha256")
        != reference.get("evaluation_input_sha256")
        or metric.get("protocol_fingerprint") != reference.get("protocol_fingerprint")
        or metric.get("unified_environment") != reference.get("unified_environment")
        or metric.get("unified_evaluator_protocol_fingerprint")
        != reference.get("unified_evaluator_protocol_fingerprint")
        or crn_identity(metric) != crn_identity(reference)
    ):
        raise RuntimeError(f"DCLGAN/common evaluator identity differs at e{epoch}")


def evaluate_one(
    *, adapter, row: dict[str, Any], upstream_root: Path, manifest: Path,
    train_view: Path, data_root: Path, output: Path, reference_output: Path,
    source_host_label: str, gpu: int,
) -> dict[str, Any]:
    epoch = int(row["epoch"])
    schedule = expected_schedule(epoch)
    checkpoint = Path(row["checkpoint"])
    checkpoint_before = file_sha256(checkpoint)
    rows = adapter.annotated_manifest_rows(manifest)
    adapter._verify_discovery_content(
        rows=rows, data_root=data_root,
        count_per_domain=schedule["count_per_domain"],
    )
    model, stream, payload = adapter._load_evaluation_runtime(
        upstream_root=upstream_root, manifest_path=manifest, train_view=train_view,
        output_root=output / "frozen_adapter_runtime", checkpoint=checkpoint, gpu=gpu,
    )
    before = adapter.full_state_hash(adapter.capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    from research.paper_aio.evaluate import evaluate_model, validate_evaluation_result
    from research.paper_aio.protocol import (
        EVALUATION_SCHEMA,
        evaluation_bundle_fingerprint,
        load_protocol,
        protocol_fingerprint,
    )
    from research.paper_aio.unified import _deterministic_unified_environment

    environment = _deterministic_unified_environment()
    result = evaluate_model(
        model=model, spec=adapter.dclgan_lane_spec(), rows=rows,
        data_root=data_root,
        protocol_hash=evaluation_bundle_fingerprint(load_protocol()),
        count_per_domain=schedule["count_per_domain"],
        replicates=schedule["replicates"], nfe_values=[1],
        include_lpips=schedule["include_lpips"],
    )
    after = adapter.full_state_hash(adapter.capture_full_state(
        model=model, stream=stream, step=int(payload["step"]),
        metadata=payload["metadata"],
    ))
    checkpoint_after = file_sha256(checkpoint)
    if before != after or checkpoint_before != checkpoint_after:
        raise RuntimeError(f"DCLGAN evaluation mutated e{epoch} state")
    if checkpoint_before != row["checkpoint_sha256"]:
        raise RuntimeError(f"DCLGAN imported checkpoint changed at e{epoch}")
    result.update({
        "epoch": epoch,
        "updates": epoch * STEPS_PER_EPOCH,
        "training_protocol_fingerprint": payload["metadata"]["adapter_fingerprint"],
        "manifest_sha256": payload["metadata"]["manifest_sha256"],
        "unified_evaluator_protocol_fingerprint": protocol_fingerprint(manifest),
        "source_export_receipt_sha256": file_sha256(row["export_receipt"]),
        "source_checkpoint_sha256": checkpoint_before,
        "source_checkpoint_sha256_after_evaluation": checkpoint_after,
        "source_host_label": source_host_label,
        "upstream_commit": payload["metadata"]["upstream_commit"],
        "adapter_git_commit": payload["metadata"]["adapter_git_commit"],
        "unified_environment": environment,
        "training_checkpoint_read_only": True,
        "training_checkpoint_read_only_verified_by_rehash": True,
        "cross_host_training_delta_merged": False,
    })
    validate_evaluation_result(
        result, lane_id="dclgan", family="external",
        count_per_domain=schedule["count_per_domain"],
        replicates=schedule["replicates"], nfe_values=[1], include_lpips=True,
    )
    validate_common_reference(result, reference_output=reference_output, epoch=epoch)
    metric_path = output / "lanes" / "dclgan" / "metrics" / f"e{epoch:03d}.json"
    immutable_json(metric_path, result)
    receipt = {
        "schema": UNIFIED_RECEIPT_SCHEMA,
        "status": "PASS_UNIFIED_READ_ONLY_EVALUATION",
        "lane_id": "dclgan",
        "epoch": epoch,
        "source_host_label": source_host_label,
        "source_export_receipt": str(Path(row["export_receipt"]).resolve()),
        "source_export_receipt_sha256": file_sha256(row["export_receipt"]),
        "source_checkpoint_sha256": checkpoint_before,
        "source_checkpoint_sha256_after_evaluation": checkpoint_after,
        "training_protocol_fingerprint": payload["metadata"]["adapter_fingerprint"],
        "manifest_sha256": payload["metadata"]["manifest_sha256"],
        "metric": str(metric_path.resolve()),
        "metric_sha256": file_sha256(metric_path),
        "evaluation_schema": EVALUATION_SCHEMA,
        "evaluation_bundle_fingerprint": result["protocol_fingerprint"],
        "unified_evaluator_protocol_fingerprint": protocol_fingerprint(manifest),
        "unified_environment": environment,
        "training_checkpoint_read_only": True,
        "training_checkpoint_read_only_verified_by_rehash": True,
        "paired_metric_control": False,
        "cross_host_training_delta_merged": False,
        "confirmation20_opened": False,
    }
    receipt_path = output / "gates" / f"UNIFIED_EVALUATION_dclgan_e{epoch:03d}.json"
    immutable_json(receipt_path, receipt)
    del model
    adapter.torch.cuda.empty_cache()
    return {
        "epoch": epoch,
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": file_sha256(receipt_path),
        "metric": str(metric_path.resolve()),
        "metric_sha256": file_sha256(metric_path),
        "checkpoint_sha256": checkpoint_before,
    }


def build_result(output: Path, evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory = []
    for row in evaluations:
        metric = read_json(row["metric"])
        trajectory.append({
            "epoch": row["epoch"],
            "macro_psnr": metric["macro_psnr"],
            "macro_ssim": metric["macro_ssim"],
            "macro_lpips": metric["macro_lpips"],
            "stochasticity": metric["stochasticity"],
            "domains": metric["domains"],
            "metric_sha256": row["metric_sha256"],
        })
    terminal = next(row for row in trajectory if row["epoch"] == 200)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "COMPLETE_FIXED_E200_EXTERNAL_BASELINE",
        "lane_id": "dclgan",
        "paper_role": "official_source_controlled_exposure_external_comparator",
        "primary_epoch": 200,
        "fixed_epochs": list(EPOCHS),
        "trajectory": trajectory,
        "terminal": terminal,
        "evaluation_receipts": evaluations,
        "comparison_scope": "standalone_fixed_protocol_no_matched_delta_claim",
        "included_in_first_wave_cohort": False,
        "performance_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "cross_non_equivalent_runtime_delta": False,
        "confirmation20_opened": False,
    }
    path = Path(output) / "DCLGAN_PAPER_RESULT.json"
    immutable_json(path, result)
    return {**result, "path": str(path.resolve()), "sha256": file_sha256(path)}


class Heartbeat:
    def __init__(self, path: Path, base: dict[str, Any], interval: int):
        self.path = path
        self.value = dict(base)
        self.interval = int(interval)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def update(self, **values: Any) -> None:
        with self.lock:
            self.value.update(values)
            self.value["captured_unix_time"] = time.time()
            atomic_json(self.path, self.value)

    def _run(self) -> None:
        while not self.stop.wait(self.interval):
            self.update()

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def contract(args: argparse.Namespace) -> dict[str, Any]:
    for name in (
        "repo", "adapter_repo", "upstream_root", "import_root", "reference_output",
        "output", "manifest", "data_root", "train_view", "gpu_lock",
    ):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if git(args.repo, "rev-parse", "HEAD") != args.required_control_git_commit:
        raise RuntimeError("DCLGAN evaluation control checkout moved")
    if git(args.repo, "status", "--porcelain"):
        raise RuntimeError("DCLGAN evaluation control checkout is dirty")
    if git(args.adapter_repo, "rev-parse", "HEAD") != args.required_adapter_git_commit:
        raise RuntimeError("frozen DCLGAN adapter checkout moved")
    if git(args.adapter_repo, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("frozen DCLGAN adapter tracked source is dirty")
    if git(args.upstream_root, "rev-parse", "HEAD") != args.required_upstream_commit:
        raise RuntimeError("DCLGAN upstream checkout moved")
    if git(args.upstream_root, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("DCLGAN upstream tracked source is dirty")
    adapter_source = args.adapter_repo / "operations" / "paper_aio_dclgan_adapter.py"
    source_gate = args.adapter_repo / "configs" / "PAPER_DCLGAN_NEGCUT_SOURCE_GATE.json"
    if file_sha256(adapter_source) != args.required_adapter_source_sha256:
        raise RuntimeError("frozen DCLGAN adapter source hash differs")
    if file_sha256(args.manifest) != args.required_manifest_sha256:
        raise RuntimeError("DCLGAN evaluation manifest differs")
    if not 30 <= int(args.poll_seconds) <= 600 or float(args.timeout_hours) < 24:
        raise ValueError("unsafe DCLGAN evaluation waiting policy")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(args.repo),
        "control_git_commit": args.required_control_git_commit,
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__)),
        "adapter_repo": str(args.adapter_repo),
        "adapter_git_commit": args.required_adapter_git_commit,
        "adapter_source_sha256": args.required_adapter_source_sha256,
        "adapter_source_gate_sha256": file_sha256(source_gate),
        "adapter_fingerprint": args.required_adapter_fingerprint,
        "upstream_root": str(args.upstream_root),
        "upstream_commit": args.required_upstream_commit,
        "import_root": str(args.import_root),
        "source_host_label": args.source_host_label,
        "reference_output": str(args.reference_output),
        "output": str(args.output),
        "manifest": str(args.manifest),
        "manifest_sha256": args.required_manifest_sha256,
        "data_root": str(args.data_root),
        "train_view": str(args.train_view),
        "gpu_lock": str(args.gpu_lock),
        "gpu": int(args.gpu),
        "epochs": list(EPOCHS),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "performance_values_available_to_scheduler": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def verify_contract(value: dict[str, Any]) -> None:
    for repo_key, commit_key in (
        ("control_repo", "control_git_commit"),
        ("adapter_repo", "adapter_git_commit"),
        ("upstream_root", "upstream_commit"),
    ):
        repo = Path(value[repo_key])
        if (
            git(repo, "rev-parse", "HEAD") != value[commit_key]
            or git(repo, "status", "--porcelain", "--untracked-files=no")
        ):
            raise RuntimeError(f"DCLGAN evaluation {repo_key} changed")
    if (
        file_sha256(Path(value["control_script"]))
        != value["control_script_sha256"]
        or file_sha256(Path(value["adapter_repo"]) / "operations" / "paper_aio_dclgan_adapter.py")
        != value["adapter_source_sha256"]
        or file_sha256(Path(value["adapter_repo"]) / "configs" / "PAPER_DCLGAN_NEGCUT_SOURCE_GATE.json")
        != value["adapter_source_gate_sha256"]
        or file_sha256(Path(value["manifest"])) != value["manifest_sha256"]
    ):
        raise RuntimeError("DCLGAN evaluation frozen source changed")


def run(args: argparse.Namespace) -> dict[str, Any]:
    value = contract(args)
    output = Path(value["output"])
    contract_path = output / "operations" / "DCLGAN_EVALUATION_CONTRACT.json"
    state_path = output / "operations" / "DCLGAN_EVALUATION_STATE.json"
    lock_path = output / "operations" / "DCLGAN_EVALUATION.lock"
    if contract_path.is_file():
        if read_json(contract_path) != value:
            raise RuntimeError("DCLGAN evaluation contract changed")
    else:
        atomic_json(contract_path, value)
    base = {
        "schema": STATE_SCHEMA,
        "pid": os.getpid(),
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "performance_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with lock_path.open("a+", encoding="utf-8") as process_lock:
        import fcntl

        fcntl.flock(process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        heartbeat = Heartbeat(state_path, base, value["poll_seconds"])
        heartbeat.start()
        try:
            while True:
                verify_contract(value)
                imported = (
                    Path(value["import_root"]) / "sources" / value["source_host_label"]
                    / "dclgan" / "IMPORT_LANE.json"
                ).is_file()
                reference = reference_cohort_ready(Path(value["reference_output"]))
                if imported and reference:
                    break
                if time.time() - started > value["timeout_hours"] * 3600:
                    raise TimeoutError("DCLGAN evaluation successor timed out")
                heartbeat.update(
                    status="WAITING_FOR_DCLGAN_IMPORT_AND_FIRST_WAVE_COHORT",
                    imported_lane_ready=imported, reference_cohort_ready=reference,
                    completed_evaluations=0,
                )
                time.sleep(value["poll_seconds"])
            rows = validate_import_lane(
                Path(value["import_root"]),
                source_host_label=value["source_host_label"],
                required_training_commit=value["adapter_git_commit"],
                required_adapter_fingerprint=value["adapter_fingerprint"],
            )
            adapter = load_frozen_adapter(Path(value["adapter_repo"]))
            upstream = adapter.verify_upstream(Path(value["upstream_root"]))
            fingerprint = adapter.adapter_fingerprint(
                upstream_receipt=upstream, manifest_path=Path(value["manifest"]),
            )
            if fingerprint != value["adapter_fingerprint"]:
                raise RuntimeError("frozen DCLGAN evaluation fingerprint differs")
            gpu_lock = Path(value["gpu_lock"])
            gpu_lock.parent.mkdir(parents=True, exist_ok=True)
            with gpu_lock.open("a+", encoding="utf-8") as gpu_handle:
                while True:
                    try:
                        fcntl.flock(gpu_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        heartbeat.update(
                            status="WAITING_FOR_SHARED_EVALUATION_GPU",
                            completed_evaluations=0,
                        )
                        time.sleep(value["poll_seconds"])
                    else:
                        break
                evaluations = []
                for row in rows:
                    heartbeat.update(
                        status="EVALUATING_FIXED_DCLGAN_CHECKPOINT",
                        current_epoch=row["epoch"],
                        completed_evaluations=len(evaluations),
                    )
                    evaluations.append(evaluate_one(
                        adapter=adapter, row=row,
                        upstream_root=Path(value["upstream_root"]),
                        manifest=Path(value["manifest"]),
                        train_view=Path(value["train_view"]),
                        data_root=Path(value["data_root"]), output=output,
                        reference_output=Path(value["reference_output"]),
                        source_host_label=value["source_host_label"],
                        gpu=value["gpu"],
                    ))
            result = build_result(output, evaluations)
            final = {
                **base,
                "status": COMPLETE_STATUS,
                "completed_evaluations": len(evaluations),
                "result": result["path"],
                "result_sha256": result["sha256"],
                "performance_values_generated": True,
                "performance_values_in_control_state": False,
            }
            heartbeat.update(**final)
            return final
        except Exception as error:
            heartbeat.update(
                status="FAIL_CLOSED_REQUIRES_CODEX_AUDIT",
                error_type=type(error).__name__, error_message=str(error),
            )
            raise
        finally:
            heartbeat.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--adapter-repo", type=Path, required=True)
    value.add_argument("--required-adapter-git-commit", required=True)
    value.add_argument("--required-adapter-source-sha256", required=True)
    value.add_argument("--required-adapter-fingerprint", required=True)
    value.add_argument("--upstream-root", type=Path, required=True)
    value.add_argument("--required-upstream-commit", required=True)
    value.add_argument("--import-root", type=Path, required=True)
    value.add_argument("--source-host-label", required=True)
    value.add_argument("--reference-output", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--required-manifest-sha256", required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
