from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from operations import paper_aio_local_export_push as push


def _write(path: Path, value: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)
    return push.file_sha256(path)


def test_local_export_rows_accept_exact_source_bound_set(tmp_path: Path) -> None:
    run = tmp_path / "run"
    exports = run / "exports"
    commit = "c" * 40
    fingerprint = "f" * 64
    manifest = "m" * 64
    source_rows = []
    for epoch in push.EPOCHS:
        checkpoint = run / "lanes" / "dclgan" / "milestones" / f"e{epoch:03d}.pt"
        sidecar = Path(str(checkpoint) + ".json")
        checkpoint_hash = _write(checkpoint, f"checkpoint-{epoch}".encode())
        sidecar_hash = _write(sidecar, f"sidecar-{epoch}".encode())
        receipt = {
            "schema": "final-unsb-paper-dclgan-checkpoint-export-v1",
            "status": "ACCEPTED_SOURCE_BOUND_DCLGAN_CHECKPOINT",
            "lane_id": "dclgan",
            "epoch": epoch,
            "updates": epoch * 8553,
            "source_host_label": "LOCAL_GTX1660",
            "source_checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "source_sidecar": str(sidecar.resolve()),
            "sidecar_sha256": sidecar_hash,
            "scientific_state_sha256": "s" * 64,
            "training_git_commit": commit,
            "training_protocol_fingerprint": fingerprint,
            "manifest_sha256": manifest,
            "upstream_commit": "u" * 40,
            "performance_values_read": False,
            "checkpoint_copy_performed": False,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        }
        receipt_path = exports / "dclgan" / f"e{epoch:03d}.export.json"
        receipt_hash = _write(receipt_path, json.dumps(receipt).encode())
        source_rows.append({
            "epoch": epoch, "receipt": str(receipt_path),
            "receipt_sha256": receipt_hash,
        })
    export_set = {
        "schema": "final-unsb-paper-dclgan-source-export-set-v1",
        "status": "COMPLETE_SOURCE_BOUND_EXPORT_SET",
        "lane_id": "dclgan",
        "source_host_label": "LOCAL_GTX1660",
        "epochs": list(push.EPOCHS),
        "exports": source_rows,
        "performance_values_read": False,
        "checkpoint_copy_performed": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    _write(exports / "dclgan" / "EXPORT_SET.json", json.dumps(export_set).encode())
    rows = push.local_export_rows({
        "source_run_root": str(run), "export_root": str(exports),
        "lane_id": "dclgan", "source_host_label": "LOCAL_GTX1660",
        "required_training_git_commit": commit,
        "required_training_protocol_fingerprint": fingerprint,
        "required_manifest_sha256": manifest,
    })
    assert [row["epoch"] for row in rows] == list(push.EPOCHS)
    assert all(row["checkpoint"].is_file() for row in rows)


def test_contract_never_persists_password(tmp_path: Path, monkeypatch) -> None:
    repo = tmp_path / "repo"
    run = tmp_path / "run"
    exports = run / "exports"
    repo.mkdir(); exports.mkdir(parents=True)
    monkeypatch.setattr(push, "_repo_identity", lambda repo, commit: None)
    monkeypatch.setattr(push, "__file__", str(repo / "push.py"))
    (repo / "push.py").write_text("frozen", encoding="utf-8")
    args = SimpleNamespace(
        repo=repo, required_control_git_commit="a" * 40,
        source_run_root=run, export_root=exports,
        state_root=run / "operations" / "push",
        source_host_label="LOCAL_GTX1660", lane_id="dclgan",
        relay_id="local_dclgan", required_training_git_commit="c" * 40,
        required_training_protocol_fingerprint="f" * 64,
        required_manifest_sha256="m" * 64,
        destination_host="host", destination_port=22, destination_user="user",
        expected_host_key_sha256="SHA256:pinned",
        destination_root="/safe/imports",
        password_env="FINAL_UNSB_DCLGAN_PUSH_PASSWORD",
        poll_seconds=60, timeout_hours=480,
    )
    contract = push.make_contract(args)
    assert contract["password_persisted"] is False
    assert "secret" not in json.dumps(contract)
    assert contract["destination_root"] == "/safe/imports"


def test_remote_root_and_local_containment_are_fail_closed(tmp_path: Path) -> None:
    assert push.remote_root("/safe/imports") == "/safe/imports"
    for unsafe in ("relative", "/safe/../escape"):
        try:
            push.remote_root(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError("unsafe remote root accepted")
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    try:
        push.inside_local(outside, root, "test")
    except RuntimeError:
        pass
    else:
        raise AssertionError("outside local source accepted")


def test_import_payload_uses_remote_paths_only() -> None:
    lane_root = PurePosixPath("/imports") / "sources" / "LOCAL_GTX1660" / "dclgan"
    assert str(lane_root / "e200.pt") == "/imports/sources/LOCAL_GTX1660/dclgan/e200.pt"
