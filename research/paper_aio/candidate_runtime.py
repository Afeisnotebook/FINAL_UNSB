"""Cross-code runtime gate and full-data adapter for evidence-locked candidates."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from research.local_route1.runtime import full_state_hash, write_json

from .candidate_lock import (
    LOCK_SCHEMA,
    RUNTIME_GATE_SCHEMA,
    _validate_parent_plain,
    _validate_small25_evidence,
    safe_candidate_id,
)
from .gates import (
    environment_record,
    run_evaluation_repeat_gate,
    run_preflight,
    scientific_core_hash,
    transition_core,
)
from .protocol import (
    LaneSpec,
    file_sha256,
    evaluation_bundle_fingerprint,
    git_commit,
    lane_spec,
    load_protocol,
    protocol_fingerprint,
)
from .runtime import (
    create_e0,
    load_full_state,
    optimizer_step,
    prepare_lane,
    train_lane,
    train_spec,
)


AUTHORIZATION_SCHEMA = "final-unsb-paper-candidate-authorization-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require_clean_checkout() -> str:
    root = Path(__file__).resolve().parents[2]
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True,
    ).strip()
    if dirty:
        raise RuntimeError("candidate runtime gate requires a clean frozen checkout")
    commit = git_commit()
    if len(commit) != 40:
        raise RuntimeError("candidate runtime gate requires a committed Git identity")
    return commit


def _spec(candidate_id: str, implementation: dict[str, Any], *, zero: bool = False) -> LaneSpec:
    method = dict(implementation["method"])
    if zero:
        method.update(dict(implementation["zero_intervention"]))
    return LaneSpec(
        id=(candidate_id if not zero else f"{candidate_id}.zero"),
        backend="internal", family="unsb", model=str(implementation["model"]),
        role=(
            "evidence-locked full-data paper candidate" if not zero else
            "engineering-only zero-intervention candidate witness"
        ),
        method=method, first_wave=False,
    )


def _e0_core(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": payload["model"],
        "rng": payload["rng"],
        "samplers": payload["samplers"],
    }


def _transition_after_updates(
    *, spec: LaneSpec, e0: dict[str, Any], output_root: Path, train_view: Path,
    manifest_path: Path, gpu: int, updates: int,
) -> str:
    model, primary, secondary, _ = prepare_lane(
        output_root=output_root, train_view=train_view, manifest_path=manifest_path,
        spec=spec, gpu=gpu, e0=e0,
    )
    target = int(load_protocol()["training"]["target_updates"])
    for step in range(int(updates)):
        model.set_train_epoch(1)
        model.set_search_step(step, target)
        optimizer_step(model, spec, primary, secondary)
    result = full_state_hash(
        transition_core(model=model, primary=primary, secondary=secondary)
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_candidate_runtime_gate(
    *, output_root: Path, candidate_id: str, terminal_receipt: Path,
    trajectory: Path, derivation_card: Path, implementation: Path,
    parent_output: Path, parent_runtime_receipt: Path, parent_e0: Path,
    parent_scientific_git_commit: str, parent_protocol_fingerprint: str,
    train_view: Path, data_root: Path, manifest_path: Path, gpu: int,
    capacity_override: Path | None = None, host_label: str = "local",
) -> dict[str, Any]:
    """Prove cross-code equivalence without treating source changes as matched."""
    candidate_id = safe_candidate_id(candidate_id)
    candidate_commit = _require_clean_checkout()
    output_root = Path(output_root).resolve()
    train_view = Path(train_view).resolve()
    data_root = Path(data_root).resolve()
    manifest_path = Path(manifest_path).resolve()
    parent_output = Path(parent_output).resolve()
    receipt, evidence = _validate_small25_evidence(
        candidate_id=candidate_id,
        receipt_path=Path(terminal_receipt).resolve(),
        trajectory_path=Path(trajectory).resolve(),
        card_path=Path(derivation_card).resolve(),
        implementation_path=Path(implementation).resolve(),
    )
    _validate_parent_plain(
        parent_output=parent_output,
        required_commit=str(parent_scientific_git_commit),
        required_protocol_fingerprint=str(parent_protocol_fingerprint),
    )
    current_fingerprint = protocol_fingerprint(manifest_path)
    if (
        candidate_commit == str(parent_scientific_git_commit)
        and current_fingerprint == str(parent_protocol_fingerprint)
    ):
        raise RuntimeError("candidate runtime gate requires an explicit cross-code identity")
    preflight = run_preflight(
        output_root=output_root, manifest_path=manifest_path, data_root=data_root,
        train_view=train_view, node_role="training",
        capacity_override=capacity_override, host_label=host_label,
    )

    parent_twin_path = Path(parent_runtime_receipt).resolve()
    parent_e0_path = Path(parent_e0).resolve()
    if not parent_twin_path.is_file() or not parent_e0_path.is_file():
        raise RuntimeError("parent runtime receipt and e0 checkpoint are required")
    parent_twin = _read_json(parent_twin_path)
    if parent_twin.get("status") not in (
        "LOCAL_TWIN_COMPLETE", "PASS_EXACT_RUNTIME_COHORT",
    ):
        raise RuntimeError("parent paper runtime twin did not complete")
    if int(parent_twin.get("updates", -1)) != 2000:
        raise RuntimeError("parent paper runtime twin is not the frozen 2000-update gate")
    if parent_twin.get("protocol_fingerprint") != str(parent_protocol_fingerprint):
        raise RuntimeError("parent runtime receipt protocol fingerprint mismatch")
    if parent_twin.get("manifest_sha256") != preflight["manifest"]["sha256"]:
        raise RuntimeError("parent runtime receipt manifest mismatch")
    if parent_twin.get("environment") != environment_record():
        raise RuntimeError("candidate runtime gate must execute on the parent plain host/runtime")

    parent_e0_payload = torch.load(parent_e0_path, map_location="cpu", weights_only=False)
    parent_e0_core = full_state_hash(_e0_core(parent_e0_payload))
    if parent_e0_core != parent_twin.get("e0_core_sha256"):
        raise RuntimeError("parent e0 differs from its runtime-twin receipt")

    candidate = _spec(candidate_id, evidence["implementation"])
    candidate_e0 = create_e0(
        output_root=output_root, train_view=train_view, manifest_path=manifest_path,
        spec=candidate, gpu=gpu,
    )
    candidate_e0_core = full_state_hash(_e0_core(candidate_e0))
    e0_exact = candidate_e0_core == parent_e0_core
    if not e0_exact:
        raise RuntimeError("candidate-code e0 scientific core differs from parent paper e0")

    twin_root = output_root / "candidate_runtime_gate" / candidate_id / "current_plain"
    train_lane(
        lane_id="plain", output_root=twin_root, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu, resume=(
            twin_root / "lanes" / "plain" / "full_state_latest.pt"
        ).is_file(), engineering_stop_after_updates=2000, gate_context=True,
    )
    current_plain_step = scientific_core_hash(
        twin_root / "lanes" / "plain" / "full_state_latest.pt"
    )
    step_exact = current_plain_step == parent_twin.get("step_core_sha256")
    if not step_exact:
        raise RuntimeError("candidate-code native plain 2000-step transition differs from parent")

    zero_steps = 10
    plain_zero = _transition_after_updates(
        spec=lane_spec("plain"), e0=candidate_e0, output_root=output_root,
        train_view=train_view, manifest_path=manifest_path, gpu=gpu,
        updates=zero_steps,
    )
    candidate_zero = _transition_after_updates(
        spec=_spec(candidate_id, evidence["implementation"], zero=True),
        e0=candidate_e0, output_root=output_root, train_view=train_view,
        manifest_path=manifest_path, gpu=gpu, updates=zero_steps,
    )
    zero_exact = plain_zero == candidate_zero
    if not zero_exact:
        raise RuntimeError("candidate zero-intervention transition differs from native plain")

    policy = load_protocol()["resource_policy"]
    total = int(policy["resume_gate_updates"])
    split = int(policy["resume_split_updates"])
    resume_root = output_root / "candidate_runtime_gate" / candidate_id / "resume"
    continuous = resume_root / "continuous"
    resumed = resume_root / "resumed"
    continuous_latest = continuous / "lanes" / candidate_id / "full_state_latest.pt"
    resumed_latest = resumed / "lanes" / candidate_id / "full_state_latest.pt"
    train_spec(
        spec=candidate, output_root=continuous, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu,
        resume=continuous_latest.is_file(),
        engineering_stop_after_updates=total, gate_context=True,
        authorization_kind="candidate",
    )
    train_spec(
        spec=candidate, output_root=resumed, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu,
        resume=resumed_latest.is_file(),
        engineering_stop_after_updates=split, gate_context=True,
        authorization_kind="candidate",
    )
    train_spec(
        spec=candidate, output_root=resumed, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu, resume=True,
        engineering_stop_after_updates=total, gate_context=True,
        authorization_kind="candidate",
    )
    continuous_checkpoint = continuous_latest
    resumed_checkpoint = resumed_latest
    continuous_hash = scientific_core_hash(continuous_checkpoint)
    resumed_hash = scientific_core_hash(resumed_checkpoint)
    resume_exact = continuous_hash == resumed_hash
    if not resume_exact:
        raise RuntimeError("candidate continuous and resumed transitions differ")

    payload = torch.load(continuous_checkpoint, map_location="cpu", weights_only=False)
    model, primary, secondary, rows = prepare_lane(
        output_root=continuous, train_view=train_view, manifest_path=manifest_path,
        spec=candidate, gpu=gpu,
    )
    load_full_state(
        continuous_checkpoint, model=model, spec=candidate, primary=primary,
        secondary=secondary, expected_metadata=payload["metadata"],
    )
    repeat = run_evaluation_repeat_gate(
        output_root=output_root / "candidate_runtime_gate" / candidate_id,
        model=model, spec=candidate, rows=rows, data_root=data_root,
        protocol_hash=evaluation_bundle_fingerprint(),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    evaluation_exact = repeat.get("status") == "PASS"

    result = {
        "schema": RUNTIME_GATE_SCHEMA,
        "status": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "candidate_git_commit": candidate_commit,
        "candidate_protocol_fingerprint": current_fingerprint,
        "parent_scientific_git_commit": str(parent_scientific_git_commit),
        "parent_protocol_fingerprint": str(parent_protocol_fingerprint),
        "manifest_sha256": preflight["manifest"]["sha256"],
        "parent_runtime_receipt": str(parent_twin_path),
        "parent_runtime_receipt_sha256": file_sha256(parent_twin_path),
        "parent_e0": str(parent_e0_path),
        "parent_e0_sha256": file_sha256(parent_e0_path),
        "environment": environment_record(),
        "e0_scientific_core_exact": e0_exact,
        "parent_e0_scientific_core_sha256": parent_e0_core,
        "candidate_e0_scientific_core_sha256": candidate_e0_core,
        "plain_2000_transition_exact": step_exact,
        "parent_plain_2000_transition_sha256": parent_twin["step_core_sha256"],
        "candidate_plain_2000_transition_sha256": current_plain_step,
        "zero_intervention_identity_exact": zero_exact,
        "plain_zero_transition_sha256": plain_zero,
        "candidate_zero_transition_sha256": candidate_zero,
        "candidate_resume_exact": resume_exact,
        "candidate_resume_continuous_sha256": continuous_hash,
        "candidate_resume_split_sha256": resumed_hash,
        "candidate_evaluation_repeat_exact": evaluation_exact,
        "evaluation_bundle_fingerprint": evaluation_bundle_fingerprint(),
        "evaluation_first_sha256": repeat["first_result_sha256"],
        "evaluation_second_sha256": repeat["second_result_sha256"],
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    destination = (
        output_root / "candidate_runtime_gate" / candidate_id
        / "CANDIDATE_RUNTIME_GATE.json"
    )
    write_json(destination, result)
    return result


def _candidate_lock_path(output_root: Path, candidate_id: str) -> Path:
    return (
        Path(output_root).resolve() / "candidate_locks" / safe_candidate_id(candidate_id)
        / "CANDIDATE_LOCK.json"
    )


def load_candidate_spec(output_root: Path, candidate_id: str) -> tuple[LaneSpec, dict[str, Any]]:
    path = _candidate_lock_path(output_root, candidate_id)
    if not path.is_file():
        raise RuntimeError(f"candidate evidence lock is missing: {path}")
    lock = _read_json(path)
    if (
        lock.get("schema") != LOCK_SCHEMA
        or lock.get("status") != "PASS_FULL_DATA_CANDIDATE_LOCK"
        or lock.get("candidate_id") != candidate_id
        or lock.get("full_data_authorized") is not False
        or lock.get("paired_metric_control") is not False
        or lock.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate evidence lock is invalid")
    implementation_path = Path(lock["small25"]["implementation"])
    if (
        not implementation_path.is_file()
        or file_sha256(implementation_path)
        != lock["small25"]["implementation_sha256"]
    ):
        raise RuntimeError("candidate implementation changed after evidence lock")
    implementation = _read_json(implementation_path)
    if implementation.get("model") != lock.get("model"):
        raise RuntimeError("candidate lock model differs from frozen implementation")
    if implementation.get("method") != lock.get("method"):
        raise RuntimeError("candidate lock method differs from frozen implementation")
    return _spec(candidate_id, implementation), lock


def authorize_candidate(output_root: Path, candidate_id: str) -> dict[str, Any]:
    candidate_id = safe_candidate_id(candidate_id)
    candidate_commit = _require_clean_checkout()
    spec, lock = load_candidate_spec(output_root, candidate_id)
    lock_path = _candidate_lock_path(output_root, candidate_id)
    runtime_path = Path(lock["candidate_runtime_gate"]["path"])
    if (
        not runtime_path.is_file()
        or file_sha256(runtime_path) != lock["candidate_runtime_gate"]["sha256"]
    ):
        raise RuntimeError("candidate runtime gate changed after evidence lock")
    runtime = _read_json(runtime_path)
    current_fingerprint = protocol_fingerprint()
    if runtime.get("candidate_git_commit") != candidate_commit:
        raise RuntimeError("candidate Git commit changed after runtime gate")
    if runtime.get("candidate_protocol_fingerprint") != current_fingerprint:
        raise RuntimeError("candidate protocol fingerprint changed after runtime gate")
    preflight_path = Path(output_root).resolve() / "gates" / "PREFLIGHT.json"
    if not preflight_path.is_file():
        raise RuntimeError("candidate paper preflight is missing")
    preflight = _read_json(preflight_path)
    if (
        preflight.get("status") != "PASS"
        or preflight.get("node_role") != "training"
        or preflight.get("protocol_fingerprint") != current_fingerprint
        or preflight.get("manifest", {}).get("content_hashes_verified") is not True
    ):
        raise RuntimeError("candidate paper preflight is invalid or stale")
    result = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "PASS_FULL_DATA_CANDIDATE_AUTHORIZATION",
        "candidate_id": candidate_id,
        "lane": spec.to_dict(),
        "algorithm_fingerprint": lock["algorithm_fingerprint"],
        "candidate_git_commit": candidate_commit,
        "candidate_protocol_fingerprint": current_fingerprint,
        "candidate_lock_sha256": file_sha256(lock_path),
        "candidate_runtime_gate_sha256": file_sha256(runtime_path),
        "issued_unix_time": time.time(),
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    path = (
        Path(output_root).resolve() / "gates"
        / f"CANDIDATE_AUTHORIZATION_{candidate_id}.json"
    )
    write_json(path, result)
    return result


def require_candidate_authorization(output_root: Path, candidate_id: str) -> dict[str, Any]:
    candidate_id = safe_candidate_id(candidate_id)
    path = (
        Path(output_root).resolve() / "gates"
        / f"CANDIDATE_AUTHORIZATION_{candidate_id}.json"
    )
    if not path.is_file():
        raise RuntimeError(f"candidate long training blocked; missing {path}")
    payload = _read_json(path)
    spec, lock = load_candidate_spec(output_root, candidate_id)
    lock_path = _candidate_lock_path(output_root, candidate_id)
    required = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "PASS_FULL_DATA_CANDIDATE_AUTHORIZATION",
        "candidate_id": candidate_id,
        "lane": spec.to_dict(),
        "algorithm_fingerprint": lock["algorithm_fingerprint"],
        "candidate_git_commit": git_commit(),
        "candidate_protocol_fingerprint": protocol_fingerprint(),
        "candidate_lock_sha256": file_sha256(lock_path),
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    for key, expected in required.items():
        if payload.get(key) != expected:
            raise RuntimeError(f"candidate authorization is invalid or stale for {key}")
    return payload


def train_candidate(
    *, output_root: Path, candidate_id: str, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int, resume: bool,
    engineering_stop_after_updates: int | None = None,
) -> dict[str, Any]:
    spec, _ = load_candidate_spec(output_root, safe_candidate_id(candidate_id))
    return train_spec(
        spec=spec, output_root=output_root, train_view=train_view,
        data_root=data_root, manifest_path=manifest_path, gpu=gpu, resume=resume,
        engineering_stop_after_updates=engineering_stop_after_updates,
        gate_context=False, authorization_kind="candidate",
    )
