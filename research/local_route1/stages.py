"""Evidence readiness, derivation-card and candidate freeze stage controls."""

from __future__ import annotations

import json
from pathlib import Path

from .protocol import load_protocol
from .runtime import write_json


def prepare_audit_queue(output_root: Path) -> dict:
    """Select causal-audit states without pretending pending audits are evidence."""
    protocol = load_protocol()
    fixed = [20, 100, 150, 200]
    trajectory_path = output_root / "evidence" / "ANCHOR_TRAJECTORIES.json"
    if not trajectory_path.is_file():
        result = {
            "schema": "local-route1-audit-queue-v1",
            "status": "BLOCKED_ANCHORS_INCOMPLETE",
            "fixed_epochs": fixed,
            "jobs": [],
            "confirmation20_opened": False,
        }
        write_json(output_root / "audit" / "AUDIT_QUEUE.json", result)
        return result
    evidence = json.loads(trajectory_path.read_text(encoding="utf-8"))
    jobs = []
    missing = []
    for summary in evidence["summaries"]:
        probe = summary["probe_id"]
        trajectory = summary["trajectory"]
        dynamic = []
        for previous, current in zip(trajectory, trajectory[1:]):
            if previous["macro_psnr_delta"] * current["macro_psnr_delta"] < 0:
                dynamic.extend([previous["epoch"], current["epoch"]])
        if trajectory:
            peak = max(trajectory, key=lambda row: row["macro_psnr_delta"])["epoch"]
            final = trajectory[-1]["epoch"]
            dynamic.extend([peak, final])
        for epoch in sorted(set(fixed + dynamic)):
            method = output_root / "anchors" / probe / "milestones" / f"e{epoch:03d}.pt"
            plain = output_root / "anchors" / "plain" / "milestones" / f"e{epoch:03d}.pt"
            # Dynamic epochs that are not registered milestones require a later
            # deterministic replay from the nearest prior full state.
            if not method.is_file() or not plain.is_file():
                missing.append({"probe": probe, "epoch": epoch, "needs_replay": True})
                continue
            jobs.append({
                "probe": probe,
                "data_epoch": epoch,
                "updates": epoch * 150,
                "plain_state": str(plain),
                "method_state": str(method),
                "operators": ["u0(S_plain)", "ui(S_plain)", "u0(S_method)", "ui(S_method)"],
                "branch_horizons_updates": [1, 8, 32, 200],
                "branch_regimes": [
                    "continuous_intervention",
                    "one_step_pulse_then_native@8/32",
                    "eight_step_pulse_then_native@200",
                ],
                "pulse_interpretation": "causal propagation diagnostic only; never an exit/handoff candidate",
                "sampling_variance": {
                    "replicates": 8,
                    "axes": ["independent_unpaired_batch", "latent_time_bridge_rng"],
                },
                "paired_label_timing": "only after every virtual branch is frozen",
                "status": "PENDING_EXECUTION",
            })
    status = "READY" if jobs and not any(item["epoch"] in fixed for item in missing) else "NEEDS_REPLAY_OR_CHECKPOINTS"
    result = {
        "schema": "local-route1-audit-queue-v1",
        "status": status,
        "jobs": jobs,
        "missing_or_replay": missing,
        "observable_schema": [
            "native/correction gradient cosine", "next-independent-batch consensus",
            "block/time/domain correction sign", "GAN/SB/NCE directional derivatives",
            "endpoint dispersion", "bridge KDD", "rollout velocity",
            "D/G/E balance", "Adam moment-gradient angle", "sampling variance",
            "actual correction-field batch/latent/time covariance",
        ],
        "forbidden_observables": ["paired PSNR", "SSIM", "LPIPS", "discovery target", "confirmation target"],
        "confirmation20_opened": False,
    }
    write_json(output_root / "audit" / "AUDIT_QUEUE.json", result)
    return result


def derive_from_completed_atlas(output_root: Path) -> dict:
    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    atlas_path = output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    if not matrix_path.is_file() or not atlas_path.is_file():
        result = {
            "schema": "local-route1-derive-stage-v1",
            "status": "BLOCKED_CAUSAL_ATLAS_INCOMPLETE",
            "reason": "Algorithm generation is intentionally forbidden before the long causal matrix exists; this prevents a return to validating prewritten ideas.",
            "candidate_count": 0,
            "confirmation20_opened": False,
        }
        write_json(output_root / "derive" / "DERIVE_STATE.json", result)
        return result
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
        raise RuntimeError("causal matrix exists but is not complete")
    mechanisms = list(matrix.get("ranked_failure_mechanisms", []))[:3]
    cards = []
    route = {
        "correction_sign_reversal": "future_batch_consensus_or_one_sided_constraint",
        "correct_direction_unstable_magnitude": "adam_metric_trust_region",
        "sampling_variance": "unbiased_stratified_or_antithetic_estimator",
        "coordinate_horizon_imbalance": "identity_adaptive_coordinate",
        "rollout_distribution_speed": "bridge_gap_constrained_adaptive_teacher",
    }
    for index, mechanism in enumerate(mechanisms, 1):
        kind = mechanism["failure_type"]
        cards.append({
            "candidate_id": f"G1-{index:02d}-{kind.upper().replace('_', '-')}",
            "generation": 1,
            "parent_evidence": mechanism,
            "construction_route": route.get(kind, "REQUIRES_MANUAL_MATHEMATICAL_DERIVATION"),
            "required_before_implementation": [
                "explicit UNSB object and update equation",
                "identity/self-null or unbiased condition",
                "proof that paired targets are inaccessible",
                "single falsifying counterexample experiment",
                "compute/recovery-state accounting",
            ],
            "status": "DERIVATION_REQUIRED_NOT_IMPLEMENTED",
        })
    result = {
        "schema": "local-route1-derive-stage-v1",
        "status": "DERIVATION_CARDS_REQUIRED",
        "maximum_generation1_candidates": 3,
        "cards": cards,
        "confirmation20_opened": False,
    }
    write_json(output_root / "derive" / "DERIVATION_QUEUE.json", result)
    return result


def validate_candidate_ready(output_root: Path, candidate_id: str) -> dict:
    card = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation = output_root / "derive" / "implementations" / f"{candidate_id}.json"
    missing = [str(path) for path in (card, implementation) if not path.is_file()]
    if missing:
        return {
            "schema": "local-route1-candidate-stage-v1",
            "status": "BLOCKED_DERIVATION_OR_IMPLEMENTATION_MISSING",
            "candidate_id": candidate_id,
            "missing": missing,
            "reason": "A candidate may not be long-trained from a name alone.",
            "confirmation20_opened": False,
        }
    card_payload = json.loads(card.read_text(encoding="utf-8"))
    required = [
        "parent_evidence", "unsb_object", "formula", "identity_or_unbiased_condition",
        "target_inaccessibility_proof", "falsifying_experiment", "compute_cost",
    ]
    absent = [key for key in required if not card_payload.get(key)]
    if absent:
        raise RuntimeError(f"incomplete derivation card: {absent}")
    return {
        "schema": "local-route1-candidate-stage-v1",
        "status": "READY_FOR_REGISTERED_RUNNER_INTEGRATION",
        "candidate_id": candidate_id,
        "card": str(card),
        "implementation": str(implementation),
        "confirmation20_opened": False,
    }
