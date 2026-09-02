"""Fail-closed CPU/GPU and runtime-equivalence gates."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import time
from pathlib import Path

import torch

from research.local_route1.runtime import full_state_hash, write_json

from .protocol import (
    ROOT,
    file_sha256,
    lane_spec,
    load_protocol,
    protocol_fingerprint,
    validate_protocol,
)
from .runtime import (
    create_e0,
    manifest_report,
    optimizer_step,
    prepare_lane,
    train_lane,
)
from research.local_route1.runtime import seed_everything
from .protocol import LaneSpec


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scientific_core(payload: dict) -> dict:
    """Drop host/path metadata while retaining every transition-defining value."""
    return {
        "step": int(payload["step"]),
        "target_steps": int(payload["target_steps"]),
        "model": payload["model"],
        "rng": payload["rng"],
        "samplers": payload["samplers"],
    }


def scientific_core_hash(checkpoint: Path) -> str:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    return full_state_hash(scientific_core(payload))


def transition_core(*, model, primary, secondary) -> dict:
    """State which can affect the next native optimizer transition.

    Method bookkeeping is deliberately excluded for the zero-intervention
    identity gate: a disabled wrapper may carry a diagnostic label, but its
    networks, optimizers, schedulers, samplers and RNG must be byte-identical
    to plain UNSB.
    """
    from research.local_route1.runtime import capture_rng, model_state

    state = model_state(model)
    return {
        "networks": state["networks"],
        "optimizers": state["optimizers"],
        "schedulers": state["schedulers"],
        "rng": capture_rng(),
        "samplers": {
            "primary": primary.state_dict(),
            "secondary": secondary.state_dict(),
        },
    }


def environment_record() -> dict:
    cuda = torch.cuda.is_available()
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_count": torch.cuda.device_count() if cuda else 0,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }


def run_preflight(
    *, output_root: Path, manifest_path: Path, data_root: Path | None,
    train_view: Path | None, node_role: str = "training",
) -> dict:
    seed_everything(int(load_protocol()["seed"]))
    errors = validate_protocol()
    if errors:
        raise RuntimeError("paper protocol invalid: " + "; ".join(errors))
    manifest = manifest_report(manifest_path, data_root=data_root)
    protocol = load_protocol()
    view = None
    if train_view is not None:
        view = {
            split: len(list((Path(train_view) / split).glob("*")))
            for split in ("trainA", "trainB", "discoveryA", "discoveryB", "confirmationA", "confirmationB")
        }
        expected = {
            "trainA": 8553, "trainB": 8553,
            "discoveryA": 480, "discoveryB": 480,
            "confirmationA": 120, "confirmationB": 120,
        }
        if view != expected:
            raise RuntimeError(f"materialized full view counts differ: {view}")
    disk_path = Path(output_root).resolve()
    disk_path.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(disk_path).free / (1024 ** 3)
    if node_role not in ("training", "audit_only"):
        raise ValueError(f"unknown paper node role: {node_role}")
    minimum = (
        int(protocol["resource_policy"]["minimum_free_disk_gib"])
        if node_role == "training" else 50
    )
    if free_gib < minimum:
        raise RuntimeError(f"paper run needs at least {minimum} GiB free; found {free_gib:.1f}")
    result = {
        "schema": "final-unsb-paper-preflight-v1",
        "status": "PASS",
        "manifest": manifest,
        "view_counts": view,
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "environment": environment_record(),
        "free_disk_gib": free_gib,
        "minimum_free_disk_gib": minimum,
        "node_role": node_role,
        "sampling_measure": protocol["training"]["sampling_measure"],
        "batch_size": 1,
        "target_updates": 1_710_600,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if result["environment"]["tf32_matmul"] or result["environment"]["tf32_cudnn"]:
        raise RuntimeError("paper training requires TF32 disabled before preflight passes")
    write_json(Path(output_root) / "gates" / "PREFLIGHT.json", result)
    write_json(Path(output_root) / "PAPER_PROTOCOL.json", {
        **protocol,
        "protocol_fingerprint": result["protocol_fingerprint"],
        "confirmation20_opened": False,
    })
    return result


def external_gate_status(output_root: Path, lane_id: str) -> dict:
    spec = lane_spec(lane_id)
    if spec.backend != "external_locked":
        return {"lane_id": lane_id, "status": "NOT_EXTERNAL_LOCKED"}
    lock = Path(output_root) / "external_locks" / f"{lane_id.upper()}_SOURCE_LOCK.json"
    if not lock.is_file():
        return {
            "lane_id": lane_id,
            "status": "BLOCKED_SOURCE_OR_FORMULA_LOCK_MISSING",
            "required_lock": str(lock),
            "fallback_lane": "cyclegan" if lane_id == "ddsb" else None,
            "confirmation20_opened": False,
        }
    payload = _read_json(lock)
    required = {
        "status": "PASS",
        "lane_id": lane_id,
        "full_state_resume_gate": True,
        "formula_or_source_audit": True,
        "confirmation20_opened": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"{lane_id} external source lock invalid for {key}")
    payload["source_lock_sha256"] = file_sha256(lock)
    return payload


def create_runtime_twin_receipt(
    *, output_root: Path, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, host_label: str, peer_receipt: Path | None,
) -> dict:
    protocol = load_protocol()
    steps = int(protocol["resource_policy"]["runtime_twin_updates"])
    twin_root = Path(output_root) / "runtime_twin" / host_label
    run = train_lane(
        lane_id="plain", output_root=twin_root, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu,
        resume=(twin_root / "lanes" / "plain" / "full_state_latest.pt").is_file(),
        engineering_stop_after_updates=steps,
        gate_context=True,
    )
    checkpoint = twin_root / "lanes" / "plain" / "full_state_latest.pt"
    e0 = twin_root / "shared_e0" / "unsb_common" / "e0.pt"
    e0_payload = torch.load(e0, map_location="cpu", weights_only=False)
    receipt = {
        "schema": "final-unsb-paper-runtime-twin-receipt-v1",
        "status": "LOCAL_TWIN_COMPLETE",
        "host_label": host_label,
        "updates": steps,
        "e0_core_sha256": full_state_hash({
            "model": e0_payload["model"], "rng": e0_payload["rng"],
            "samplers": e0_payload["samplers"],
        }),
        "step_core_sha256": scientific_core_hash(checkpoint),
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "environment": environment_record(),
        "wall_seconds": run.get("wall_seconds_this_call"),
        "updates_per_second": (
            None if not run.get("wall_seconds_this_call")
            else float(steps) / float(run["wall_seconds_this_call"])
        ),
        "confirmation20_opened": False,
    }
    if peer_receipt is not None:
        peer = _read_json(peer_receipt)
        identity_keys = (
            "updates", "e0_core_sha256", "step_core_sha256",
            "protocol_fingerprint", "manifest_sha256",
        )
        differences = {
            key: {"local": receipt.get(key), "peer": peer.get(key)}
            for key in identity_keys if receipt.get(key) != peer.get(key)
        }
        receipt["peer_receipt"] = str(Path(peer_receipt).resolve())
        receipt["peer_host_label"] = peer.get("host_label")
        receipt["exact_runtime_equivalence"] = not differences
        receipt["differences"] = differences
        receipt["status"] = "PASS_EXACT_RUNTIME_COHORT" if not differences else "FAIL_HOST_SEPARATED"
    path = Path(output_root) / "gates" / f"RUNTIME_TWIN_{host_label}.json"
    write_json(path, receipt)
    return receipt


def run_resume_gate(
    *, output_root: Path, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, lane_id: str = "plain",
) -> dict:
    policy = load_protocol()["resource_policy"]
    total = int(policy["resume_gate_updates"])
    split = int(policy["resume_split_updates"])
    spec = lane_spec(lane_id)
    if spec.backend != "internal":
        raise RuntimeError(f"resume gate unavailable for blocked lane {lane_id}")
    base = Path(output_root) / "resume_gate" / lane_id
    continuous = base / "continuous"
    resumed = base / "resumed"
    train_lane(
        lane_id=lane_id, output_root=continuous, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu, resume=False,
        engineering_stop_after_updates=total, gate_context=True,
    )
    train_lane(
        lane_id=lane_id, output_root=resumed, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu, resume=False,
        engineering_stop_after_updates=split, gate_context=True,
    )
    train_lane(
        lane_id=lane_id, output_root=resumed, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu, resume=True,
        engineering_stop_after_updates=total, gate_context=True,
    )
    left = scientific_core_hash(continuous / "lanes" / lane_id / "full_state_latest.pt")
    right = scientific_core_hash(resumed / "lanes" / lane_id / "full_state_latest.pt")
    result = {
        "schema": "final-unsb-paper-resume-gate-v1",
        "status": "PASS" if left == right else "FAIL",
        "lane_id": lane_id,
        "continuous_core_sha256": left,
        "resumed_core_sha256": right,
        "total_updates": total,
        "split_updates": split,
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "confirmation20_opened": False,
    }
    write_json(Path(output_root) / "gates" / f"RESUME_GATE_{lane_id}.json", result)
    if left != right:
        raise RuntimeError("paper continuous and resumed states differ")
    return result


def run_zero_intervention_gate(
    *, output_root: Path, train_view: Path, manifest_path: Path, gpu: int,
    updates: int = 10,
) -> dict:
    """Prove that a disabled Proposal wrapper is exactly plain UNSB."""
    if int(updates) < 1:
        raise ValueError("zero-intervention gate needs at least one update")
    protocol = load_protocol()
    plain = lane_spec("plain", protocol)
    disabled = LaneSpec(
        id="proposal_zero", backend="internal", family="unsb",
        model="route1_pcrsmg_ablation",
        role="engineering-only zero-intervention identity witness",
        method={
            "route1_ablation_enable": False,
            "pcrsmg_ablation_role": "proposal_only",
        },
    )
    e0 = create_e0(
        output_root=output_root, train_view=train_view,
        manifest_path=manifest_path, spec=plain, gpu=gpu,
    )
    hashes = {}
    for label, spec in (("plain", plain), ("proposal_zero", disabled)):
        model, primary, secondary, _ = prepare_lane(
            output_root=output_root, train_view=train_view,
            manifest_path=manifest_path, spec=spec, gpu=gpu, e0=e0,
        )
        for step in range(int(updates)):
            model.set_train_epoch(1)
            model.set_search_step(step, int(protocol["training"]["target_updates"]))
            optimizer_step(model, spec, primary, secondary)
        hashes[label] = full_state_hash(
            transition_core(model=model, primary=primary, secondary=secondary)
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    result = {
        "schema": "final-unsb-paper-zero-intervention-gate-v1",
        "status": "PASS" if hashes["plain"] == hashes["proposal_zero"] else "FAIL",
        "updates": int(updates),
        "plain_transition_sha256": hashes["plain"],
        "proposal_zero_transition_sha256": hashes["proposal_zero"],
        "protocol_fingerprint": protocol_fingerprint(manifest_path),
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_root) / "gates" / "ZERO_INTERVENTION_PROPOSAL.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("disabled Proposal wrapper is not identical to plain UNSB")
    return result


def authorize_lane(
    *, output_root: Path, lane_id: str, matched_plain_mode: str | None = None,
    runtime_receipt: Path | None = None,
) -> dict:
    """Issue a fail-closed long-training authorization from existing receipts."""
    output_root = Path(output_root)
    preflight_path = output_root / "gates" / "PREFLIGHT.json"
    resume_path = output_root / "gates" / f"RESUME_GATE_{lane_id}.json"
    evaluation_path = output_root / "gates" / f"EVALUATION_REPEAT_{lane_id}.json"
    if not preflight_path.is_file() or not resume_path.is_file() or not evaluation_path.is_file():
        raise RuntimeError(
            f"{lane_id}: preflight, lane resume and repeated-evaluation receipts are required"
        )
    preflight = _read_json(preflight_path)
    resume = _read_json(resume_path)
    evaluation = _read_json(evaluation_path)
    failures = []
    if preflight.get("status") != "PASS" or preflight.get("node_role") != "training":
        failures.append("training preflight did not pass")
    if not preflight.get("manifest", {}).get("content_hashes_verified"):
        failures.append("full data content hashes were not verified")
    if resume.get("status") != "PASS" or resume.get("lane_id") != lane_id:
        failures.append("lane-specific exact resume did not pass")
    if resume.get("protocol_fingerprint") != protocol_fingerprint():
        failures.append("lane-specific resume receipt is stale")
    if evaluation.get("status") != "PASS" or evaluation.get("lane_id") != lane_id:
        failures.append("lane-specific repeated evaluation did not pass")
    if evaluation.get("protocol_fingerprint") != protocol_fingerprint():
        failures.append("lane-specific repeated evaluation receipt is stale")
    if preflight.get("protocol_fingerprint") != protocol_fingerprint():
        failures.append("protocol fingerprint changed after preflight")
    if lane_id == "proposal":
        zero_path = output_root / "gates" / "ZERO_INTERVENTION_PROPOSAL.json"
        zero = _read_json(zero_path) if zero_path.is_file() else {}
        if (
            zero.get("status") != "PASS"
            or zero.get("protocol_fingerprint") != protocol_fingerprint()
        ):
            failures.append("Proposal zero-intervention identity did not pass")
        if matched_plain_mode == "same_runtime_output_root":
            comparison = {"mode": matched_plain_mode, "runtime_receipt": None}
        elif matched_plain_mode == "exact_cross_4090_cohort" and runtime_receipt:
            twin = _read_json(runtime_receipt)
            if twin.get("status") != "PASS_EXACT_RUNTIME_COHORT":
                failures.append("cross-4090 exact runtime cohort did not pass")
            comparison = {
                "mode": matched_plain_mode,
                "runtime_receipt": str(Path(runtime_receipt).resolve()),
                "runtime_receipt_sha256": file_sha256(runtime_receipt),
            }
        else:
            failures.append("Proposal needs same-runtime plain or exact cross-4090 cohort")
            comparison = {"mode": matched_plain_mode}
    else:
        comparison = {"mode": "standalone_fixed_protocol"}
    result = {
        "schema": "final-unsb-paper-lane-authorization-v1",
        "status": "PASS" if not failures else "FAIL",
        "lane_id": lane_id,
        "issued_unix_time": time.time(),
        "protocol_fingerprint": protocol_fingerprint(),
        "preflight_sha256": file_sha256(preflight_path),
        "resume_gate_sha256": file_sha256(resume_path),
        "evaluation_repeat_gate_sha256": file_sha256(evaluation_path),
        "comparison": comparison,
        "failures": failures,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    path = output_root / "gates" / f"LANE_AUTHORIZATION_{lane_id}.json"
    write_json(path, result)
    if failures:
        raise RuntimeError(f"{lane_id} authorization failed: {'; '.join(failures)}")
    return result


def require_lane_authorization(output_root: Path, lane_id: str) -> dict:
    path = Path(output_root) / "gates" / f"LANE_AUTHORIZATION_{lane_id}.json"
    if not path.is_file():
        raise RuntimeError(f"long training blocked; missing {path}")
    payload = _read_json(path)
    if payload.get("status") != "PASS" or payload.get("lane_id") != lane_id:
        raise RuntimeError(f"long training blocked; invalid authorization for {lane_id}")
    if payload.get("protocol_fingerprint") != protocol_fingerprint():
        raise RuntimeError("long training blocked; authorization fingerprint is stale")
    return payload


def run_evaluation_repeat_gate(
    *, output_root: Path, model, spec: LaneSpec, rows: list[dict],
    data_root: Path, protocol_hash: str,
) -> dict:
    """Execute, rather than cache, the same lane-blind evaluation twice."""
    from .evaluate import evaluate_model
    from .protocol import object_sha256

    kwargs = {
        "model": model,
        "spec": spec,
        "rows": rows,
        "data_root": Path(data_root),
        "protocol_hash": protocol_hash,
        "count_per_domain": int(
            load_protocol()["evaluation"]["trajectory_discovery_per_domain"]
        ),
        "replicates": 1,
        "nfe_values": [5 if spec.family == "unsb" else 1],
        "include_lpips": False,
    }
    first = evaluate_model(**kwargs)
    second = evaluate_model(**kwargs)
    left = object_sha256(first)
    right = object_sha256(second)
    result = {
        "schema": "final-unsb-paper-evaluation-repeat-gate-v1",
        "status": "PASS" if left == right else "FAIL",
        "lane_id": spec.id,
        "first_result_sha256": left,
        "second_result_sha256": right,
        "evaluation_input_sha256": first["evaluation_input_sha256"],
        "protocol_fingerprint": protocol_hash,
        "split": "discovery",
        "confirmation20_opened": False,
    }
    write_json(
        Path(output_root) / "gates" / f"EVALUATION_REPEAT_{spec.id}.json",
        result,
    )
    if result["status"] != "PASS":
        raise RuntimeError(f"{spec.id} repeated evaluation differs")
    return result
