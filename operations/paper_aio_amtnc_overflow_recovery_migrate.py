"""Create a source-bound AM-TNC continuation without mutating the failed run.

This operation is intentionally narrower than a general checkpoint converter.
It accepts the protected e178 state only after the float32-overflow localizer and
the overflow-safe counterfactual replay have both passed.  It changes provenance
metadata for the patched source tree, while proving that every transition-
defining value (networks, optimizers, schedulers, method state, RNG and samplers)
is unchanged.

The destination is publish-last: a fresh output root is populated and the
migration receipt is written only after every checkpoint and sidecar hash has
been verified.  The original run is read-only throughout.
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
    atomic_torch_save,
    full_state_hash,
    write_json,
)
from research.paper_aio.protocol import (
    E0_SCHEMA,
    FULL_STATE_SCHEMA,
    file_sha256,
    git_commit,
    load_protocol,
    portable_source_sha256,
    protocol_fingerprint,
)


RECEIPT_SCHEMA = "final-unsb-paper-amtnc-overflow-recovery-migration-v1"
AUTHORIZATION_SCHEMA = (
    "final-unsb-paper-amtnc-overflow-recovery-authorization-v1"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def dynamics_only(payload: dict[str, Any]) -> dict[str, Any]:
    """Return all checkpoint content except non-transition provenance metadata."""
    return {key: value for key, value in payload.items() if key != "metadata"}


def dynamics_only_hash(payload: dict[str, Any]) -> str:
    return full_state_hash(dynamics_only(payload))


def migrate_payloads(
    parent_e0: dict[str, Any], parent_checkpoint: dict[str, Any], *,
    new_git_commit: str, new_protocol_fingerprint: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy payloads and update only the three source-lineage metadata fields."""
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


def validate_replay(
    receipt: dict[str, Any], *, checkpoint_sha256: str,
    scientific_state_sha256: str,
) -> None:
    first = receipt.get("first_precision_fallback") or {}
    required = {
        "status": "OVERFLOW_SAFE_REPLAY_COMPLETE",
        "completed_replay_updates": 6250,
        "post_replay_state_finite": True,
        "source_checkpoint_unchanged": True,
        "scientific_lane_modified": False,
        "performance_values_read": False,
        "confirmation20_opened": False,
    }
    for key, expected in required.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"overflow-safe replay failed acceptance for {key}")
    if (
        int(receipt.get("precision_fallback_count", -1)) < 1
        or int(first.get("replay_update_offset", -1)) != 6245
        or first.get("player") != "GF"
        or receipt.get("source_checkpoint_sha256_before") != checkpoint_sha256
        or receipt.get("source_checkpoint_sha256_after") != checkpoint_sha256
        or receipt.get("source_scientific_state_sha256")
        != scientific_state_sha256
    ):
        raise RuntimeError("overflow-safe replay identity or localized event changed")


def validate_localization(
    receipt: dict[str, Any], *, checkpoint_sha256: str,
    scientific_state_sha256: str,
) -> None:
    failure = receipt.get("failure") or {}
    geometry = failure.get("geometry") or {}
    if (
        receipt.get("status") != "FAILURE_LOCALIZED"
        or receipt.get("source_checkpoint_unchanged") is not True
        or receipt.get("scientific_lane_modified") is not False
        or receipt.get("performance_values_read") is not False
        or receipt.get("confirmation20_opened") is not False
        or receipt.get("source_checkpoint_sha256_before") != checkpoint_sha256
        or receipt.get("source_checkpoint_sha256_after") != checkpoint_sha256
        or receipt.get("source_scientific_state_sha256")
        != scientific_state_sha256
        or failure.get("player") != "GF"
        or int(failure.get("replay_update_offset", -1)) != 6245
        or geometry.get("category_counts", {}).get(
            "FLOAT32_GEOMETRY_PRODUCT_OVERFLOW"
        ) != 3
    ):
        raise RuntimeError("nonfinite localization receipt is not the accepted event")


def _validate_sidecar(
    checkpoint: Path, sidecar_path: Path, payload: dict[str, Any], *, lane_id: str,
) -> dict[str, Any]:
    sidecar = _read_json(sidecar_path)
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


def _save_checkpoint(
    path: Path, payload: dict[str, Any], *, lane_id: str,
) -> dict[str, Any]:
    atomic_torch_save(path, payload)
    sidecar = {
        "schema": FULL_STATE_SCHEMA,
        "lane_id": lane_id,
        "step": int(payload["step"]),
        "physical_epoch_completed": int(payload["physical_epoch_completed"]),
        "target_steps": int(payload["target_steps"]),
        "full_state_sha256": file_sha256(path),
        "scientific_state_sha256": full_state_hash(payload),
        "metadata": payload["metadata"],
    }
    write_json(Path(str(path) + ".json"), sidecar)
    return sidecar


def _parse_milestone(value: str) -> tuple[int, Path]:
    try:
        raw_epoch, raw_path = value.split("=", 1)
        epoch = int(raw_epoch)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError(
            "milestone must be EPOCH=/absolute/checkpoint.pt"
        ) from error
    path = Path(raw_path)
    if epoch <= 0 or not path.is_absolute():
        raise argparse.ArgumentTypeError(
            "milestone must use a positive epoch and absolute path"
        )
    return epoch, path


def arguments() -> argparse.Namespace:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--parent-e0", type=Path, required=True)
    value.add_argument("--parent-checkpoint", type=Path, required=True)
    value.add_argument("--parent-checkpoint-sidecar", type=Path, required=True)
    value.add_argument("--localization-receipt", type=Path, required=True)
    value.add_argument("--replay-receipt", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--required-parent-e0-sha256", required=True)
    value.add_argument("--required-parent-checkpoint-sha256", required=True)
    value.add_argument("--required-parent-scientific-state-sha256", required=True)
    value.add_argument("--required-localization-receipt-sha256", required=True)
    value.add_argument("--required-replay-receipt-sha256", required=True)
    value.add_argument("--required-parent-git-commit", required=True)
    value.add_argument("--required-parent-protocol-fingerprint", required=True)
    value.add_argument("--required-new-git-commit", required=True)
    value.add_argument("--required-new-protocol-fingerprint", required=True)
    value.add_argument("--required-manifest-sha256", required=True)
    value.add_argument("--required-amtnc-source-sha256", required=True)
    value.add_argument(
        "--milestone", action="append", type=_parse_milestone, default=[],
    )
    return value.parse_args()


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
        raise RuntimeError("patched training checkout is not clean at required commit")
    current_fingerprint = protocol_fingerprint(args.manifest.resolve())
    if current_fingerprint != args.required_new_protocol_fingerprint:
        raise RuntimeError("patched training protocol fingerprint changed")
    if file_sha256(args.manifest) != args.required_manifest_sha256:
        raise RuntimeError("manifest identity changed")
    amtnc_source = repo / "src" / "models" / "route1" / "amtnc.py"
    if portable_source_sha256(amtnc_source) != args.required_amtnc_source_sha256:
        raise RuntimeError("overflow-safe AM-TNC source identity changed")

    parent_e0_path = args.parent_e0.resolve()
    parent_checkpoint_path = args.parent_checkpoint.resolve()
    if file_sha256(parent_e0_path) != args.required_parent_e0_sha256:
        raise RuntimeError("parent e0 hash changed")
    if (
        file_sha256(parent_checkpoint_path)
        != args.required_parent_checkpoint_sha256
    ):
        raise RuntimeError("parent e178 checkpoint hash changed")
    parent_e0 = torch.load(parent_e0_path, map_location="cpu", weights_only=False)
    parent_checkpoint = torch.load(
        parent_checkpoint_path, map_location="cpu", weights_only=False,
    )
    if (
        parent_e0.get("schema") != E0_SCHEMA
        or parent_checkpoint.get("schema") != FULL_STATE_SCHEMA
        or parent_checkpoint.get("lane", {}).get("id") != "amtnc"
        or int(parent_checkpoint.get("step", -1)) != 1_522_434
        or int(parent_checkpoint.get("physical_epoch_completed", -1)) != 178
        or full_state_hash(parent_checkpoint)
        != args.required_parent_scientific_state_sha256
    ):
        raise RuntimeError("parent e0/e178 payload identity changed")
    _validate_sidecar(
        parent_checkpoint_path, args.parent_checkpoint_sidecar.resolve(),
        parent_checkpoint, lane_id="amtnc",
    )
    for payload in (parent_e0, parent_checkpoint):
        metadata = payload.get("metadata") or {}
        if (
            metadata.get("git_commit") != args.required_parent_git_commit
            or metadata.get("protocol_fingerprint")
            != args.required_parent_protocol_fingerprint
            or metadata.get("manifest_sha256") != args.required_manifest_sha256
        ):
            raise RuntimeError("parent source lineage changed")

    localization_path = args.localization_receipt.resolve()
    replay_path = args.replay_receipt.resolve()
    if (
        file_sha256(localization_path)
        != args.required_localization_receipt_sha256
        or file_sha256(replay_path) != args.required_replay_receipt_sha256
    ):
        raise RuntimeError("localization or replay receipt hash changed")
    validate_localization(
        _read_json(localization_path),
        checkpoint_sha256=args.required_parent_checkpoint_sha256,
        scientific_state_sha256=args.required_parent_scientific_state_sha256,
    )
    validate_replay(
        _read_json(replay_path),
        checkpoint_sha256=args.required_parent_checkpoint_sha256,
        scientific_state_sha256=args.required_parent_scientific_state_sha256,
    )

    migrated_e0, migrated_latest = migrate_payloads(
        parent_e0, parent_checkpoint,
        new_git_commit=args.required_new_git_commit,
        new_protocol_fingerprint=args.required_new_protocol_fingerprint,
    )
    parent_e0_dynamics = dynamics_only_hash(parent_e0)
    parent_latest_dynamics = dynamics_only_hash(parent_checkpoint)
    if (
        dynamics_only_hash(migrated_e0) != parent_e0_dynamics
        or dynamics_only_hash(migrated_latest) != parent_latest_dynamics
    ):
        raise RuntimeError("migration changed transition-defining state")

    e0_path = output / "shared_e0" / "unsb_common" / "e0.pt"
    latest_path = output / "lanes" / "amtnc" / "full_state_latest.pt"
    e0_sidecar = _save_e0(e0_path, migrated_e0)
    latest_sidecar = _save_checkpoint(
        latest_path, migrated_latest, lane_id="amtnc",
    )
    milestones: list[dict[str, Any]] = []
    for epoch, source_path in args.milestone:
        source_path = source_path.resolve()
        source_payload = torch.load(
            source_path, map_location="cpu", weights_only=False,
        )
        _validate_sidecar(
            source_path, Path(str(source_path) + ".json"), source_payload,
            lane_id="amtnc",
        )
        if (
            int(source_payload.get("physical_epoch_completed", -1)) != epoch
            or source_payload.get("metadata") != parent_checkpoint.get("metadata")
        ):
            raise RuntimeError(f"milestone e{epoch} is outside the parent lineage")
        _, migrated_milestone = migrate_payloads(
            parent_e0, source_payload,
            new_git_commit=args.required_new_git_commit,
            new_protocol_fingerprint=args.required_new_protocol_fingerprint,
        )
        if dynamics_only_hash(migrated_milestone) != dynamics_only_hash(source_payload):
            raise RuntimeError(f"milestone e{epoch} dynamics changed")
        destination = (
            output / "lanes" / "amtnc" / "milestones" / f"e{epoch:03d}.pt"
        )
        sidecar = _save_checkpoint(destination, migrated_milestone, lane_id="amtnc")
        milestones.append({
            "epoch": epoch,
            "source": str(source_path),
            "source_sha256": file_sha256(source_path),
            "dynamics_only_sha256": dynamics_only_hash(source_payload),
            "destination": str(destination),
            "destination_sha256": sidecar["full_state_sha256"],
            "destination_scientific_state_sha256": sidecar[
                "scientific_state_sha256"
            ],
        })

    protocol = load_protocol()
    write_json(output / "PAPER_PROTOCOL.json", {
        **protocol,
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
        "localization_receipt_sha256": args.required_localization_receipt_sha256,
        "overflow_safe_replay_receipt_sha256": args.required_replay_receipt_sha256,
        "authorization_scope": "e178_to_fixed_e200_overflow_safe_continuation",
        "dynamics_state_changed_by_migration": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    authorization_path = output / "gates" / "LANE_AUTHORIZATION_amtnc.json"
    write_json(authorization_path, authorization)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_SOURCE_BOUND_OVERFLOW_SAFE_CONTINUATION_READY",
        "output": str(output),
        "parent_git_commit": args.required_parent_git_commit,
        "parent_protocol_fingerprint": args.required_parent_protocol_fingerprint,
        "new_git_commit": args.required_new_git_commit,
        "new_protocol_fingerprint": args.required_new_protocol_fingerprint,
        "manifest_sha256": args.required_manifest_sha256,
        "amtnc_source_sha256": args.required_amtnc_source_sha256,
        "parent_e0_sha256": args.required_parent_e0_sha256,
        "parent_e0_dynamics_only_sha256": parent_e0_dynamics,
        "migrated_e0_sha256": e0_sidecar["checkpoint_sha256"],
        "migrated_e0_scientific_state_sha256": e0_sidecar[
            "scientific_state_sha256"
        ],
        "parent_checkpoint_sha256": args.required_parent_checkpoint_sha256,
        "parent_scientific_state_sha256": (
            args.required_parent_scientific_state_sha256
        ),
        "parent_dynamics_only_sha256": parent_latest_dynamics,
        "migrated_checkpoint_sha256": latest_sidecar["full_state_sha256"],
        "migrated_scientific_state_sha256": latest_sidecar[
            "scientific_state_sha256"
        ],
        "migrated_dynamics_only_sha256": dynamics_only_hash(migrated_latest),
        "start_step": int(migrated_latest["step"]),
        "start_data_epoch": int(migrated_latest["physical_epoch_completed"]),
        "target_step": int(migrated_latest["target_steps"]),
        "milestones": milestones,
        "authorization": str(authorization_path),
        "authorization_sha256": file_sha256(authorization_path),
        "parent_files_modified": False,
        "transition_defining_state_changed": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    receipt_path = output / "operations" / "AMTNC_OVERFLOW_RECOVERY_MIGRATION.json"
    write_json(receipt_path, receipt)
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
