"""Adjudicate the frozen winner's long-horizon mechanism ablations.

The three inputs are source-bound terminal receipts produced after independent
small25 e200 runs from the common e0 state.  Paired metrics are read only after
all three trajectories are complete.  This tool never trains, chooses a
checkpoint, or changes the frozen winner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from operations.local_route1_cross_version_adjudicate import (
    SCHEMA as CROSS_SCHEMA,
    _rank_key,
    _validate_receipt,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import full_state_hash, write_json
from research.local_route1.single_seed_development import (
    validate_single_seed_development_freeze,
)


SCHEMA = "final-unsb-route1-winner-ablation-adjudication-v1"
ROLES = ("proposal_only", "observable_only", "projected_or_full")
POSITIVE_CROSS_STATUS = "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"
SINGLE_SEED_CHALLENGE_STATUS = (
    "ABLATION_CHALLENGER_READY_FOR_SINGLE_SEED_DEVELOPMENT_SELECTION"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_posthoc(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not prove confirmation20 remained closed")
    for key in (
        "paired_controller_access",
        "paired_metrics_used_for_training_or_control",
    ):
        if key in payload and payload[key] is not False:
            raise RuntimeError(f"{label} violates posthoc-only paired-metric policy: {key}")


def _observable_identity(output_root: Path, candidate_id: str) -> dict[str, Any]:
    candidate_root = output_root / "candidates" / candidate_id
    candidate_state = candidate_root / "full_state_latest.pt"
    plain_state = output_root / "anchors" / "plain" / "full_state_latest.pt"
    candidate_state_path = candidate_root / "full_state_latest.pt.json"
    plain_state_path = output_root / "anchors" / "plain" / "full_state_latest.pt.json"
    candidate_metric_path = candidate_root / "metrics" / "e200.json"
    plain_metric_path = output_root / "anchors" / "plain" / "metrics" / "e200.json"
    for path in (
        candidate_state, plain_state, candidate_state_path, plain_state_path,
        candidate_metric_path, plain_metric_path,
    ):
        if not path.is_file():
            raise RuntimeError(f"observable-only identity artifact missing: {path}")
    candidate_sidecar = _read_json(candidate_state_path)
    plain_sidecar = _read_json(plain_state_path)
    if int(candidate_sidecar.get("physical_epoch_completed", -1)) != 200:
        raise RuntimeError("observable-only state is not e200")
    if int(plain_sidecar.get("physical_epoch_completed", -1)) != 200:
        raise RuntimeError("plain identity authority is not e200")
    candidate_payload = torch.load(candidate_state, map_location="cpu", weights_only=False)
    plain_payload = torch.load(plain_state, map_location="cpu", weights_only=False)
    if candidate_sidecar.get("scientific_state_sha256") != full_state_hash(candidate_payload):
        raise RuntimeError("observable-only e200 checkpoint/sidecar integrity failed")
    if plain_sidecar.get("scientific_state_sha256") != full_state_hash(plain_payload):
        raise RuntimeError("plain e200 checkpoint/sidecar integrity failed")

    # Candidate id, git commit and protocol metadata intentionally differ.  The
    # dynamics authority is every state component that can change the next
    # optimizer update.  Observable-only is permitted to write diagnostics to
    # separate evidence files, never into this recoverable training state.
    def dynamics(payload: dict[str, Any]) -> dict[str, Any]:
        model = dict(payload["model"])
        method = dict(model.get("method", {}))
        # This is the only permitted exclusion.  The observer is recoverable
        # for diagnostic continuity but its source-bound implementation must
        # prove that it cannot enter forward outputs, gradients, RNG or sampler
        # transitions.  Every other method/controller field remains compared.
        method.pop("route1_observer", None)
        model["method"] = method
        return {
            "step": payload["step"],
            "physical_epoch_completed": payload["physical_epoch_completed"],
            "target_steps": payload["target_steps"],
            "model": model,
            "rng": payload["rng"],
            "samplers": payload["samplers"],
        }

    candidate_dynamics = full_state_hash(dynamics(candidate_payload))
    plain_dynamics = full_state_hash(dynamics(plain_payload))
    if candidate_dynamics != plain_dynamics:
        raise RuntimeError("observable-only e200 dynamics state differs from plain")

    candidate_metric = _read_json(candidate_metric_path)
    plain_metric = _read_json(plain_metric_path)
    ignored = {"probe_id"}
    candidate_metric = {key: value for key, value in candidate_metric.items() if key not in ignored}
    plain_metric = {key: value for key, value in plain_metric.items() if key not in ignored}
    if candidate_metric != plain_metric:
        raise RuntimeError("observable-only e200 evaluation differs from plain")
    return {
        "status": "EXACT_PLAIN_E200_DYNAMICS_IDENTITY",
        "candidate_full_state_sha256": file_sha256(candidate_state),
        "plain_full_state_sha256": file_sha256(plain_state),
        "candidate_dynamics_state_sha256": candidate_dynamics,
        "plain_dynamics_state_sha256": plain_dynamics,
        "candidate_state_sidecar_sha256": file_sha256(candidate_state_path),
        "plain_state_sidecar_sha256": file_sha256(plain_state_path),
        "candidate_metric_sha256": file_sha256(candidate_metric_path),
        "plain_metric_sha256": file_sha256(plain_metric_path),
    }


def adjudicate(
    *, output_root: Path, cross_adjudication_path: Path,
    proposal_receipt_path: Path, observable_receipt_path: Path,
    full_receipt_path: Path, output_path: Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    cross_adjudication_path = Path(cross_adjudication_path).resolve()
    cross = _read_json(cross_adjudication_path)
    _validate_posthoc(cross, label="cross-version e200 adjudication")
    if cross.get("schema") != CROSS_SCHEMA or cross.get("status") != POSITIVE_CROSS_STATUS:
        raise RuntimeError("winner ablations require a positive frozen seed2026 cross-version winner")

    receipts = {
        "proposal_only": _validate_receipt(Path(proposal_receipt_path)),
        "observable_only": _validate_receipt(Path(observable_receipt_path)),
        "projected_or_full": _validate_receipt(Path(full_receipt_path)),
    }
    candidate_ids = [str(value["candidate_id"]) for value in receipts.values()]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise RuntimeError("winner ablation roles must use three distinct candidate ids")

    full = receipts["projected_or_full"]
    if full["candidate_id"] != cross.get("selected_candidate_id"):
        raise RuntimeError("projected/full receipt is not the frozen cross-version winner")
    if full["algorithm_fingerprint"] != cross.get("selected_algorithm_fingerprint"):
        raise RuntimeError("projected/full algorithm fingerprint changed")

    baseline_fields = (
        "base_e0_scientific_state_sha256",
        "base_protocol_fingerprint",
        "manifest_sha256",
        "plain_e200_verification_sha256",
    )
    for field in baseline_fields:
        if len({str(receipt.get(field)) for receipt in receipts.values()}) != 1:
            raise RuntimeError(f"winner ablations differ on authoritative baseline: {field}")

    identity = _observable_identity(
        output_root, str(receipts["observable_only"]["candidate_id"]),
    )
    proposal = receipts["proposal_only"]
    proposal_challenges_full = _rank_key(proposal) < _rank_key(full)
    single_seed_policy = None
    freeze_path = output_root / "operations" / "SINGLE_SEED_DEVELOPMENT_FREEZE.json"
    if freeze_path.is_file():
        single_seed_policy = validate_single_seed_development_freeze(output_root)
    result = {
        "schema": SCHEMA,
        "status": (
            (
                SINGLE_SEED_CHALLENGE_STATUS
                if single_seed_policy is not None else
                "ABLATION_CHALLENGER_REQUIRES_FROZEN_SEED_VALIDATION"
            )
            if proposal_challenges_full else
            "COMPLETE_NO_SELECTION_CHANGE"
        ),
        "selected_candidate_id": full["candidate_id"],
        "selected_algorithm_fingerprint": full["algorithm_fingerprint"],
        "source_cross_version_adjudication_sha256": file_sha256(
            cross_adjudication_path
        ),
        "roles": {
            role: {
                "candidate_id": receipt["candidate_id"],
                "algorithm_fingerprint": receipt["algorithm_fingerprint"],
                "candidate_fingerprint": receipt["candidate_fingerprint"],
                "training_git_commit": receipt["training_git_commit"],
                "trajectory_status": receipt["trajectory_status"],
                "trajectory_sha256": receipt["trajectory_sha256"],
                "receipt_path": str(Path(path).resolve()),
                "receipt_sha256": file_sha256(Path(path).resolve()),
                "ranking_fields": receipt["ranking_fields"],
            }
            for role, receipt, path in (
                ("proposal_only", receipts["proposal_only"], proposal_receipt_path),
                ("observable_only", receipts["observable_only"], observable_receipt_path),
                ("projected_or_full", receipts["projected_or_full"], full_receipt_path),
            )
        },
        "observable_only_identity": identity,
        "proposal_only_out_ranks_full": proposal_challenges_full,
        "selection_changed": False,
        "selection_change_blocked_pending_seed_validation": (
            proposal_challenges_full and single_seed_policy is None
        ),
        "selection_ready_under_single_seed_development_policy": (
            proposal_challenges_full and single_seed_policy is not None
        ),
        "single_seed_development_freeze_sha256": (
            None if single_seed_policy is None else file_sha256(freeze_path)
        ),
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_path).resolve(), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cross-adjudication", type=Path, required=True)
    parser.add_argument("--proposal-receipt", type=Path, required=True)
    parser.add_argument("--observable-receipt", type=Path, required=True)
    parser.add_argument("--full-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adjudicate(
        output_root=args.output_root,
        cross_adjudication_path=args.cross_adjudication,
        proposal_receipt_path=args.proposal_receipt,
        observable_receipt_path=args.observable_receipt,
        full_receipt_path=args.full_receipt,
        output_path=args.output,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
