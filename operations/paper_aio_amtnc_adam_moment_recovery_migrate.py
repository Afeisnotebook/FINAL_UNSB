"""Create a source-bound AM-TNC continuation after Adam moment overflow.

The source checkpoint is immutable and finite.  Migration changes provenance
metadata only, then publishes a fresh output root for the precision-capacity
guard.  No paired metric or confirmation sample is addressable here.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from research.local_route1.runtime import (
    assert_finite,
    atomic_torch_save,
    full_state_hash,
    write_json,
)
from research.paper_aio.protocol import (
    E0_SCHEMA,
    FULL_STATE_SCHEMA,
    file_sha256,
    load_protocol,
    portable_source_sha256,
    protocol_fingerprint,
)


RECEIPT_SCHEMA = "final-unsb-paper-amtnc-adam-moment-recovery-migration-v1"
AUTHORIZATION_SCHEMA = (
    "final-unsb-paper-amtnc-adam-moment-recovery-authorization-v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def dynamics_only(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "metadata"}


def dynamics_only_hash(payload: dict[str, Any]) -> str:
    return full_state_hash(dynamics_only(payload))


def migrate_payloads(
    parent_e0: dict[str, Any], parent_checkpoint: dict[str, Any], *,
    new_git_commit: str, new_protocol_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    migrated_e0 = copy.deepcopy(parent_e0)
    migrated_e0["metadata"]["git_commit"] = new_git_commit
    migrated_e0["metadata"]["protocol_fingerprint"] = new_protocol_fingerprint
    migrated_e0_hash = full_state_hash(migrated_e0)

    migrated_checkpoint = copy.deepcopy(parent_checkpoint)
    migrated_checkpoint["metadata"]["git_commit"] = new_git_commit
    migrated_checkpoint["metadata"][
        "protocol_fingerprint"
    ] = new_protocol_fingerprint
    migrated_checkpoint["metadata"][
        "e0_scientific_state_sha256"
    ] = migrated_e0_hash
    return migrated_e0, migrated_checkpoint


def _validate_sidecar(
    checkpoint: Path, payload: dict[str, Any], *, lane_id: str,
) -> dict[str, Any]:
    sidecar = _read_json(Path(str(checkpoint) + ".json"))
    if (
        sidecar.get("schema") != FULL_STATE_SCHEMA
        or sidecar.get("lane_id") != lane_id
        or int(sidecar.get("step", -1)) != int(payload.get("step", -2))
        or sidecar.get("full_state_sha256") != file_sha256(checkpoint)
        or sidecar.get("scientific_state_sha256") != full_state_hash(payload)
        or sidecar.get("metadata") != payload.get("metadata")
    ):
        raise RuntimeError(f"checkpoint sidecar does not bind {checkpoint}")
    return sidecar


def _save_e0(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    atomic_torch_save(path, payload)
    sidecar = {
        "schema": E0_SCHEMA,
        "metadata": payload["metadata"],
        "checkpoint_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    atomic_torch_save(path, payload)
    sidecar = {
        "schema": FULL_STATE_SCHEMA,
        "lane_id": "amtnc",
        "step": int(payload["step"]),
        "physical_epoch_completed": int(payload["physical_epoch_completed"]),
        "target_steps": int(payload["target_steps"]),
        "full_state_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": payload["metadata"],
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def _milestone(value: str) -> tuple[int, Path]:
    try:
        raw_epoch, raw_path = value.split("=", 1)
        epoch = int(raw_epoch)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "milestone must be EPOCH=/absolute/checkpoint.pt"
        ) from error
    path = Path(raw_path)
    if epoch <= 0 or not path.is_absolute():
        raise argparse.ArgumentTypeError(
            "milestone must be EPOCH=/absolute/checkpoint.pt"
        )
    return epoch, path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--parent-e0", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--incident-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--start-epoch", type=int, required=True)
    parser.add_argument("--required-parent-e0-sha256", required=True)
    parser.add_argument("--required-parent-checkpoint-sha256", required=True)
    parser.add_argument("--required-parent-scientific-state-sha256", required=True)
    parser.add_argument("--required-incident-receipt-sha256", required=True)
    parser.add_argument("--required-parent-git-commit", required=True)
    parser.add_argument("--required-parent-protocol-fingerprint", required=True)
    parser.add_argument("--required-new-git-commit", required=True)
    parser.add_argument("--required-new-protocol-fingerprint", required=True)
    parser.add_argument("--required-manifest-sha256", required=True)
    parser.add_argument("--required-amtnc-source-sha256", required=True)
    parser.add_argument("--required-runtime-source-sha256", required=True)
    parser.add_argument("--milestone", action="append", type=_milestone, default=[])
    return parser.parse_args()


def main() -> int:
    args = arguments()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"fresh migration output required: {output}")
    if (
        _git(repo, "rev-parse", "HEAD") != args.required_new_git_commit
        or _git(repo, "status", "--porcelain")
    ):
        raise RuntimeError("recovery checkout is not clean at required commit")
    if file_sha256(args.manifest) != args.required_manifest_sha256:
        raise RuntimeError("manifest identity changed")
    current_fingerprint = protocol_fingerprint(args.manifest.resolve())
    if current_fingerprint != args.required_new_protocol_fingerprint:
        raise RuntimeError("recovery protocol fingerprint changed")
    if portable_source_sha256(
        repo / "src" / "models" / "route1" / "amtnc.py"
    ) != args.required_amtnc_source_sha256:
        raise RuntimeError("AM-TNC recovery source identity changed")
    if portable_source_sha256(
        repo / "research" / "local_route1" / "runtime.py"
    ) != args.required_runtime_source_sha256:
        raise RuntimeError("full-state recovery source identity changed")

    e0_path = args.parent_e0.resolve()
    checkpoint_path = args.parent_checkpoint.resolve()
    if file_sha256(e0_path) != args.required_parent_e0_sha256:
        raise RuntimeError("parent e0 hash changed")
    if file_sha256(checkpoint_path) != args.required_parent_checkpoint_sha256:
        raise RuntimeError("parent checkpoint hash changed")
    parent_e0 = torch.load(e0_path, map_location="cpu", weights_only=False)
    parent = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if (
        parent_e0.get("schema") != E0_SCHEMA
        or parent.get("schema") != FULL_STATE_SCHEMA
        or parent.get("lane", {}).get("id") != "amtnc"
        or int(parent.get("physical_epoch_completed", -1)) != args.start_epoch
        or full_state_hash(parent) != args.required_parent_scientific_state_sha256
    ):
        raise RuntimeError("parent payload identity changed")
    assert_finite(parent["model"])
    _validate_sidecar(checkpoint_path, parent, lane_id="amtnc")
    for payload in (parent_e0, parent):
        metadata = payload.get("metadata") or {}
        if (
            metadata.get("git_commit") != args.required_parent_git_commit
            or metadata.get("protocol_fingerprint")
            != args.required_parent_protocol_fingerprint
            or metadata.get("manifest_sha256") != args.required_manifest_sha256
        ):
            raise RuntimeError("parent source lineage changed")

    incident_path = args.incident_receipt.resolve()
    if file_sha256(incident_path) != args.required_incident_receipt_sha256:
        raise RuntimeError("incident receipt hash changed")
    incident = _read_json(incident_path)
    if (
        incident.get("status") != "REPEATED_ADAM_SECOND_MOMENT_FLOAT32_OVERFLOW"
        or int(incident.get("repeat_count", 0)) < 2
        or incident.get("last_finite_checkpoint_sha256")
        != args.required_parent_checkpoint_sha256
        or incident.get("last_finite_scientific_state_sha256")
        != args.required_parent_scientific_state_sha256
        or incident.get("exception_path")
        != "state.optimizers[0].state.2.exp_avg_sq"
        or incident.get("performance_values_read") is not False
        or incident.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("incident receipt is not the accepted repeated event")

    migrated_e0, migrated = migrate_payloads(
        parent_e0, parent,
        new_git_commit=args.required_new_git_commit,
        new_protocol_fingerprint=args.required_new_protocol_fingerprint,
    )
    parent_e0_dynamics = dynamics_only_hash(parent_e0)
    parent_dynamics = dynamics_only_hash(parent)
    if (
        dynamics_only_hash(migrated_e0) != parent_e0_dynamics
        or dynamics_only_hash(migrated) != parent_dynamics
    ):
        raise RuntimeError("migration changed transition-defining state")

    destination_e0 = output / "shared_e0" / "unsb_common" / "e0.pt"
    destination_latest = output / "lanes" / "amtnc" / "full_state_latest.pt"
    e0_sidecar = _save_e0(destination_e0, migrated_e0)
    latest_sidecar = _save_checkpoint(destination_latest, migrated)
    milestones = []
    for epoch, source in args.milestone:
        source = source.resolve()
        payload = torch.load(source, map_location="cpu", weights_only=False)
        _validate_sidecar(source, payload, lane_id="amtnc")
        if (
            int(payload.get("physical_epoch_completed", -1)) != epoch
            or payload.get("metadata") != parent.get("metadata")
        ):
            raise RuntimeError(f"milestone e{epoch} is outside parent lineage")
        _, migrated_milestone = migrate_payloads(
            parent_e0, payload,
            new_git_commit=args.required_new_git_commit,
            new_protocol_fingerprint=args.required_new_protocol_fingerprint,
        )
        if dynamics_only_hash(migrated_milestone) != dynamics_only_hash(payload):
            raise RuntimeError(f"milestone e{epoch} dynamics changed")
        destination = (
            output / "lanes" / "amtnc" / "milestones" / f"e{epoch:03d}.pt"
        )
        sidecar = _save_checkpoint(destination, migrated_milestone)
        milestones.append({
            "epoch": epoch,
            "source_sha256": file_sha256(source),
            "destination_sha256": sidecar["full_state_sha256"],
            "dynamics_only_sha256": dynamics_only_hash(payload),
        })

    write_json(output / "PAPER_PROTOCOL.json", {
        **load_protocol(),
        "protocol_fingerprint": args.required_new_protocol_fingerprint,
        "confirmation20_opened": False,
    })
    authorization = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": "PASS",
        "lane_id": "amtnc",
        "protocol_fingerprint": args.required_new_protocol_fingerprint,
        "parent_git_commit": args.required_parent_git_commit,
        "parent_protocol_fingerprint": args.required_parent_protocol_fingerprint,
        "parent_checkpoint_sha256": args.required_parent_checkpoint_sha256,
        "incident_receipt_sha256": args.required_incident_receipt_sha256,
        "authorization_scope": "finite_parent_to_fixed_e200_precision_capacity_continuation",
        "dynamics_state_changed_by_migration": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    authorization_path = output / "gates" / "LANE_AUTHORIZATION_amtnc.json"
    write_json(authorization_path, authorization)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_SOURCE_BOUND_ADAM_MOMENT_PRECISION_CONTINUATION_READY",
        "output": str(output),
        "start_data_epoch": args.start_epoch,
        "start_step": int(migrated["step"]),
        "target_step": int(migrated["target_steps"]),
        "parent_git_commit": args.required_parent_git_commit,
        "new_git_commit": args.required_new_git_commit,
        "parent_protocol_fingerprint": args.required_parent_protocol_fingerprint,
        "new_protocol_fingerprint": args.required_new_protocol_fingerprint,
        "parent_e0_sha256": args.required_parent_e0_sha256,
        "parent_e0_dynamics_only_sha256": parent_e0_dynamics,
        "migrated_e0_sha256": e0_sidecar["checkpoint_sha256"],
        "parent_checkpoint_sha256": args.required_parent_checkpoint_sha256,
        "parent_scientific_state_sha256": (
            args.required_parent_scientific_state_sha256
        ),
        "parent_dynamics_only_sha256": parent_dynamics,
        "migrated_checkpoint_sha256": latest_sidecar["full_state_sha256"],
        "migrated_scientific_state_sha256": latest_sidecar[
            "scientific_state_sha256"
        ],
        "migrated_dynamics_only_sha256": dynamics_only_hash(migrated),
        "milestones": milestones,
        "authorization_sha256": file_sha256(authorization_path),
        "parent_files_modified": False,
        "transition_defining_state_changed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    receipt_path = (
        output / "operations" / "AMTNC_ADAM_MOMENT_RECOVERY_MIGRATION.json"
    )
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
