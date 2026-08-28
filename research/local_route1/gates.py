"""Fail-closed CPU/GPU engineering gates for the route-1 runner."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from .anchors import create_shared_e0, prepare_probe
from .evaluate import evaluate_model, select_discovery70
from .interfaces import CounterfactualAuditor, StateObservation
from .lineage import HISTORICAL_DT_SEMANTIC_HASHES
from .protocol import (
    ROOT,
    dt_lambda_for_physical_epoch,
    file_sha256,
    load_protocol,
    probe_spec,
    protocol_fingerprint,
    semantic_source_sha256,
    validate_protocol,
)
from .runtime import (
    capture_full_state,
    capture_rng,
    full_state_hash,
    load_full_state,
    model_state,
    read_manifest,
    restore_rng,
    save_full_state,
    write_json,
)


def _gate(name: str, fn) -> dict:
    try:
        detail = fn()
        return {"name": name, "status": "PASS", "detail": detail}
    except Exception as error:
        return {"name": name, "status": "FAIL", "detail": f"{type(error).__name__}: {error}"}


def run_cpu_gates(*, manifest_path: Path, train_view: Path, data_root: Path) -> dict:
    protocol = load_protocol()
    rows = read_manifest(manifest_path)
    gates = []
    gates.append(_gate("protocol_contract", lambda: (
        "valid" if not validate_protocol(protocol)
        else (_ for _ in ()).throw(RuntimeError(validate_protocol(protocol)))
    )))
    gates.append(_gate("manifest_hash", lambda: (
        file_sha256(manifest_path)
        if file_sha256(manifest_path) == protocol["manifest"]["sha256"]
        else (_ for _ in ()).throw(RuntimeError("manifest hash mismatch"))
    )))
    gates.append(_gate("discovery70_only", lambda: len(select_discovery70(rows))))

    def data_gate():
        if not train_view.is_dir():
            raise FileNotFoundError(train_view)
        for subdir in ("trainA", "trainB"):
            if not (train_view / subdir).is_dir():
                raise FileNotFoundError(train_view / subdir)
        missing = []
        for row in select_discovery70(rows):
            for key in ("input_relpath", "target_relpath"):
                path = data_root / row[key]
                if not path.is_file():
                    missing.append(str(path))
                    break
            if missing:
                break
        if missing:
            raise FileNotFoundError(missing[0])
        return {"train_view": str(train_view), "discovery_images": 420}

    gates.append(_gate("local_data_available", data_gate))

    def dt_hash_gate():
        actual = {
            "__init__.py": semantic_source_sha256(ROOT / "src/models/dtcov/__init__.py"),
            "dtcovmatch.py": semantic_source_sha256(ROOT / "src/models/dtcov/dtcovmatch.py"),
        }
        for name, digest in actual.items():
            if digest != HISTORICAL_DT_SEMANTIC_HASHES[name]:
                raise RuntimeError(f"authoritative DT core hash changed: {name}")
        return actual

    gates.append(_gate("dt_authoritative_core_hash", dt_hash_gate))

    def schedules():
        values = {epoch: dt_lambda_for_physical_epoch(epoch, protocol) for epoch in (1, 20, 21, 25, 35, 44, 45, 46, 200)}
        if values[20] != 0.0 or values[21] <= 0.0 or values[44] <= 0.0 or values[45] != 0.0 or values[200] != 0.0:
            raise RuntimeError(values)
        hj = probe_spec("hj", protocol).method
        if hj["hj_start_epoch"] != 5 or hj["hj_search_start_step"] != -1 or hj["hj_search_duration_steps"] != 0:
            raise RuntimeError("HJ is not physical-e5 through e200")
        return {"dt": values, "hj": "inactive e1-e4; active e5-e200"}

    gates.append(_gate("physical_epoch_schedules", schedules))

    def observable_lock():
        good = StateObservation(step=1, physical_epoch=1.0, bridge={"rollout_velocity": 0.1})
        good.validate_target_blind()
        bad = StateObservation(step=1, physical_epoch=1.0, method_internal={"paired_psnr": 1.0})
        try:
            bad.validate_target_blind()
        except ValueError:
            return "paired fields rejected"
        raise RuntimeError("paired field was accepted")

    gates.append(_gate("target_blind_observation_schema", observable_lock))

    def counterfactual_lock():
        parent = {"x": torch.tensor([1.0]), "nested": {"v": [1, 2]}}
        _, parent_hash = CounterfactualAuditor().run(
            parent, lambda branch: branch["x"].add_(3.0).tolist()
        )
        if full_state_hash(parent) != parent_hash:
            raise RuntimeError("parent state changed")
        return parent_hash

    gates.append(_gate("counterfactual_parent_isolation", counterfactual_lock))
    gates.append(_gate("protocol_fingerprint", lambda: protocol_fingerprint(manifest_path)))
    status = "PASS" if all(row["status"] == "PASS" for row in gates) else "FAIL"
    return {"schema": "local-route1-cpu-gate-v1", "status": status, "gates": gates, "confirmation20_opened": False}


def _empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_steps(model, primary, secondary, *, start: int, end: int, target: int = 30000) -> None:
    for zero_step in range(int(start), int(end)):
        epoch = 1 + zero_step // 150
        model.set_train_epoch(epoch)
        model.set_search_step(zero_step, target)
        model.set_input(primary.next(), secondary.next())
        model.optimize_parameters()
        if (zero_step + 1) % 150 == 0:
            model.update_learning_rate()


def _scientific_snapshot(model, primary, secondary, *, include_method: bool) -> dict:
    state = model_state(model)
    if not include_method:
        state.pop("method", None)
    return {
        "model": state,
        "rng": capture_rng(),
        "samplers": {"primary": primary.state_dict(), "secondary": secondary.state_dict()},
    }


def run_gpu_gates(
    *, output_root: Path, manifest_path: Path, train_view: Path,
    data_root: Path, gpu: int,
) -> dict:
    if not torch.cuda.is_available() and gpu >= 0:
        raise RuntimeError("CUDA is required for the registered local GPU gates")
    gate_root = output_root / "gates" / "scratch"
    protocol_hash = protocol_fingerprint(manifest_path)
    e0 = create_shared_e0(
        output_root=gate_root, train_view=train_view,
        manifest_path=manifest_path, gpu=gpu,
    )
    results = []

    def run_one(probe_id: str, *, disable_hnek: bool = False, include_method: bool = False):
        spec = probe_spec(probe_id)
        model, primary, secondary, _ = prepare_probe(
            spec=spec, output_root=gate_root, train_view=train_view,
            manifest_path=manifest_path, gpu=gpu, e0=e0,
        )
        if disable_hnek:
            from models.hnek.hnek_search import set_hnek_search_active

            set_hnek_search_active(model, False)
        _run_steps(model, primary, secondary, start=0, end=1)
        snapshot = _scientific_snapshot(model, primary, secondary, include_method=include_method)
        digest = full_state_hash(snapshot)
        del model, primary, secondary, snapshot
        _empty_cuda_cache()
        return digest

    def twin_gate():
        first = run_one("plain")
        second = run_one("plain")
        if first != second:
            raise RuntimeError(f"plain twin mismatch: {first} != {second}")
        return first

    results.append(_gate("plain_twin_one_step", twin_gate))

    plain_one_step = None
    def plain_reference():
        nonlocal plain_one_step
        plain_one_step = run_one("plain")
        return plain_one_step

    results.append(_gate("plain_reference_one_step", plain_reference))

    def inactive_gate(probe_id: str):
        digest = run_one(probe_id)
        if digest != plain_one_step:
            raise RuntimeError(f"inactive {probe_id} differs from plain")
        return digest

    results.append(_gate("inactive_hj_equals_plain", lambda: inactive_gate("hj")))
    results.append(_gate("inactive_dt_equals_plain", lambda: inactive_gate("dt")))
    results.append(_gate("hnek_disable_equals_plain", lambda: (
        (lambda digest: digest if digest == plain_one_step else (_ for _ in ()).throw(RuntimeError("disabled HNEK differs from plain")))(run_one("hnek", disable_hnek=True))
    )))

    def resume_gate():
        spec = probe_spec("plain")
        model, primary, secondary, _ = prepare_probe(
            spec=spec, output_root=gate_root, train_view=train_view,
            manifest_path=manifest_path, gpu=gpu, e0=e0,
        )
        _run_steps(model, primary, secondary, start=0, end=300)
        continuous = full_state_hash(_scientific_snapshot(model, primary, secondary, include_method=True))
        del model, primary, secondary
        _empty_cuda_cache()

        model, primary, secondary, _ = prepare_probe(
            spec=spec, output_root=gate_root, train_view=train_view,
            manifest_path=manifest_path, gpu=gpu, e0=e0,
        )
        _run_steps(model, primary, secondary, start=0, end=150)
        checkpoint = gate_root / "resume_gate_e1.pt"
        metadata = {"gate": "resume", "protocol_fingerprint": protocol_hash}
        save_full_state(
            checkpoint, model=model, spec=spec, step=150, target_steps=30000,
            primary=primary, secondary=secondary, metadata=metadata,
        )
        del model, primary, secondary
        _empty_cuda_cache()

        model, primary, secondary, _ = prepare_probe(
            spec=spec, output_root=gate_root, train_view=train_view,
            manifest_path=manifest_path, gpu=gpu, e0=e0,
        )
        payload = load_full_state(
            checkpoint, model=model, spec=spec, primary=primary, secondary=secondary,
            expected_metadata=metadata,
        )
        _run_steps(model, primary, secondary, start=int(payload["step"]), end=300)
        resumed = full_state_hash(_scientific_snapshot(model, primary, secondary, include_method=True))
        del model, primary, secondary, payload
        _empty_cuda_cache()
        if continuous != resumed:
            raise RuntimeError(f"resume mismatch: {continuous} != {resumed}")
        return continuous

    results.append(_gate("two_epoch_continuous_equals_one_plus_resume", resume_gate))

    def evaluation_gate():
        spec = probe_spec("plain")
        model, primary, secondary, rows = prepare_probe(
            spec=spec, output_root=gate_root, train_view=train_view,
            manifest_path=manifest_path, gpu=gpu, e0=e0,
        )
        before = full_state_hash(_scientific_snapshot(model, primary, secondary, include_method=True))
        first = evaluate_model(
            model, rows=rows, data_root=data_root,
            protocol_hash=protocol_hash, include_lpips=False,
        )
        middle = full_state_hash(_scientific_snapshot(model, primary, secondary, include_method=True))
        second = evaluate_model(
            model, rows=rows, data_root=data_root,
            protocol_hash=protocol_hash, include_lpips=False,
        )
        after = full_state_hash(_scientific_snapshot(model, primary, secondary, include_method=True))
        del model, primary, secondary
        _empty_cuda_cache()
        if first != second:
            raise RuntimeError("repeated per-image evaluation differs")
        if before != middle or before != after:
            raise RuntimeError("evaluation polluted training state")
        return {"state_sha256": before, "images": len(first["images"]), "evaluation_input_sha256": first["evaluation_input_sha256"]}

    results.append(_gate("evaluation_repeat_and_state_isolation", evaluation_gate))
    status = "PASS" if all(row["status"] == "PASS" for row in results) else "FAIL"
    report = {
        "schema": "local-route1-gpu-gate-v1",
        "status": status,
        "protocol_fingerprint": protocol_hash,
        "gates": results,
        "confirmation20_opened": False,
    }
    write_json(output_root / "gates" / "GPU_GATE.json", report)
    return report
