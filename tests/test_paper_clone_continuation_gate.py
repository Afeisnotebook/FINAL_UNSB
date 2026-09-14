import argparse
import hashlib
import json
from pathlib import Path

import pytest

from operations import paper_aio_clone_continuation_gate as gate


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _arguments(tmp_path: Path) -> argparse.Namespace:
    checkpoint = tmp_path / "source" / "lanes" / "plain" / "full_state_latest.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = hashlib.sha256(b"checkpoint").hexdigest()
    identity = {
        "updates": 2000,
        "e0_core_sha256": "e0",
        "step_core_sha256": "step",
        "protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
    }
    host = _write(tmp_path / "host.json", {
        "schema": gate.HOST_SCHEMA,
        "status": "NEW_PHYSICAL_GPU_CANDIDATE",
        "classification": {
            "requested_label": "5090B_CLONE",
            "long_training_launch_allowed_by_identity_gate": True,
        },
        "identity": {"gpu_uuid": "GPU-new"},
    })
    source_twin = _write(tmp_path / "source_twin.json", {
        "schema": gate.RUNTIME_SCHEMA,
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": "5090B_MATCHED_PLAIN",
        **identity,
    })
    replacement_twin = _write(tmp_path / "replacement_twin.json", {
        "schema": gate.RUNTIME_SCHEMA,
        "status": "PASS_EXACT_RUNTIME_COHORT",
        "host_label": "5090B_CLONE",
        "exact_runtime_equivalence": True,
        "differences": {},
        **identity,
    })
    sidecar = _write(checkpoint.with_suffix(checkpoint.suffix + ".json"), {
        "schema": gate.SIDECAR_SCHEMA,
        "lane_id": "plain",
        "step": 992148,
        "target_steps": 1710600,
        "full_state_sha256": checkpoint_sha,
        "metadata": {
            "git_commit": "commit",
            "protocol_fingerprint": "protocol",
            "manifest_sha256": "manifest",
        },
    })
    repo = tmp_path / "repo"
    repo.mkdir()
    return argparse.Namespace(
        host_identity_receipt=host,
        source_runtime_twin=source_twin,
        replacement_runtime_twin=replacement_twin,
        source_output=tmp_path / "source",
        source_checkpoint=checkpoint,
        source_sidecar=sidecar,
        training_repo=repo,
        python=Path("python"),
        manifest=Path("manifest"),
        data_root=Path("data"),
        train_view=Path("view"),
        probe_root=tmp_path / "probes",
        output=tmp_path / "receipt.json",
        lane="plain",
        gpu=0,
        probe_updates=1,
        source_host_label="5090B_MATCHED_PLAIN",
        source_gpu_uuid="GPU-old",
        replacement_host_label="5090B_CLONE",
        training_git_commit="commit",
        protocol_fingerprint="protocol",
    )


def test_static_gate_accepts_new_gpu_with_exact_runtime_and_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path)
    monkeypatch.setattr(
        gate, "_git", lambda _repo, *command: "commit" if command[-1] == "HEAD" else ""
    )
    result = gate.validate_static_inputs(args)
    assert result["source_step"] == 992148
    assert result["runtime_identity"]["step_core_sha256"] == "step"


def test_static_gate_accepts_registered_replacement_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path)
    host = json.loads(args.host_identity_receipt.read_text())
    host["status"] = "REGISTERED_HOST_MATCH"
    host["classification"]["outcome"] = "REGISTERED_HOST_MATCH"
    host["classification"]["registered_label"] = "5090B_CLONE"
    _write(args.host_identity_receipt, host)
    monkeypatch.setattr(
        gate, "_git", lambda _repo, *command: "commit" if command[-1] == "HEAD" else ""
    )

    result = gate.validate_static_inputs(args)

    assert result["host"]["status"] == "REGISTERED_HOST_MATCH"


def test_static_gate_rejects_runtime_core_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path)
    replacement = json.loads(args.replacement_runtime_twin.read_text())
    replacement["step_core_sha256"] = "different"
    _write(args.replacement_runtime_twin, replacement)
    monkeypatch.setattr(
        gate, "_git", lambda _repo, *command: "commit" if command[-1] == "HEAD" else ""
    )
    with pytest.raises(RuntimeError, match="runtime cores differ"):
        gate.validate_static_inputs(args)


def test_run_requires_two_exact_resume_branches_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path)
    source_sha = gate._sha256(args.source_checkpoint)
    monkeypatch.setattr(gate, "validate_static_inputs", lambda _args: {
        "host": {"identity": {"gpu_uuid": "GPU-new"}},
        "runtime_identity": {
            "updates": 2000,
            "e0_core_sha256": "e0",
            "step_core_sha256": "step",
            "protocol_fingerprint": "protocol",
            "manifest_sha256": "manifest",
        },
        "source_checkpoint_sha256": source_sha,
        "source_step": 992148,
        "target_steps": 1710600,
    })

    def probe(_args, *, destination: Path, stop_step: int) -> dict:
        return {
            "checkpoint_sha256": "same",
            "scientific_state_sha256": "same-science",
            "step": stop_step,
        }

    monkeypatch.setattr(gate, "_run_probe", probe)
    result = gate.run(args)
    assert result["status"] == "PASS_ENGINEERING_CLONE_CONTINUATION_GATE"
    assert result["resume_probe_branches_exact"] is True
    assert result["source_checkpoint_preserved"] is True
    assert result["same_physical_host_claimed"] is False
    assert result["paired_metric_control"] is False
    assert result["confirmation20_opened"] is False


def test_run_rejects_divergent_resume_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _arguments(tmp_path)
    source_sha = gate._sha256(args.source_checkpoint)
    monkeypatch.setattr(gate, "validate_static_inputs", lambda _args: {
        "host": {"identity": {"gpu_uuid": "GPU-new"}},
        "runtime_identity": {},
        "source_checkpoint_sha256": source_sha,
        "source_step": 992148,
        "target_steps": 1710600,
    })
    calls = iter(("first", "second"))
    monkeypatch.setattr(gate, "_run_probe", lambda *_args, **_kwargs: {
        "checkpoint_sha256": next(calls),
        "scientific_state_sha256": "same-science",
    })
    with pytest.raises(RuntimeError, match="resume branches differ"):
        gate.run(args)
