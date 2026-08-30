"""Evidence readiness, derivation-card and candidate freeze stage controls."""

from __future__ import annotations

import json
from pathlib import Path

from .candidates import load_candidate_registration, validate_candidate_id
from .protocol import ROOT, file_sha256, load_protocol
from .runtime import write_json


def _dynamic_audit_epoch_reasons(trajectory: list[dict]) -> dict[int, list[str]]:
    """Select the registered states that bracket the first reversal and worst drawdown.

    The paired trajectory is used only after every anchor has finished.  It chooses
    states to *label* counterfactual evidence; none of these values is visible to a
    training proposal or controller.
    """
    ordered = sorted(trajectory, key=lambda row: int(row["epoch"]))
    reasons: dict[int, list[str]] = {}

    def add(epoch: int, reason: str) -> None:
        bucket = reasons.setdefault(int(epoch), [])
        if reason not in bucket:
            bucket.append(reason)

    for previous, current in zip(ordered, ordered[1:]):
        if float(previous["macro_psnr_delta"]) * float(current["macro_psnr_delta"]) < 0.0:
            add(int(previous["epoch"]), "first_sign_reversal_left")
            add(int(current["epoch"]), "first_sign_reversal_right")
            break

    if not ordered:
        return reasons
    maximum = max(ordered, key=lambda row: float(row["macro_psnr_delta"]))
    add(int(maximum["epoch"]), "maximum_benefit")

    running_peak = ordered[0]
    worst_pair: tuple[dict, dict] | None = None
    worst_drawdown = 0.0
    for current in ordered[1:]:
        drawdown = float(running_peak["macro_psnr_delta"]) - float(
            current["macro_psnr_delta"]
        )
        if drawdown > worst_drawdown:
            worst_drawdown = drawdown
            worst_pair = (running_peak, current)
        if float(current["macro_psnr_delta"]) > float(
            running_peak["macro_psnr_delta"]
        ):
            running_peak = current
    if worst_pair is not None and worst_drawdown > 0.0:
        add(int(worst_pair[0]["epoch"]), "maximum_drawdown_peak")
        add(int(worst_pair[1]["epoch"]), "maximum_drawdown_trough")
    return reasons


def prepare_audit_queue(output_root: Path) -> dict:
    """Select causal-audit states without pretending pending audits are evidence."""
    protocol = load_protocol()
    # e175 guarantees a late 200-update label that remains inside the frozen
    # training domain.  e200 itself is a local terminal vector-field audit:
    # the registered scheduler has already reduced every optimizer LR to zero.
    fixed = [20, 100, 150, 175, 200]
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
    calibration_path = output_root / "evidence" / "PROXY_CALIBRATION.json"
    calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.is_file() else {}
    )
    proxy_status = calibration.get("status")
    jobs = []
    missing = []
    excluded_probes = []
    for summary in evidence["summaries"]:
        probe = summary["probe_id"]
        if (
            probe == "dt"
            and proxy_status == "NOT_CALIBRATED_PAUSE"
            and not bool(summary.get("complete_e200"))
        ):
            excluded_probes.append({
                "probe": "dt",
                "reason": "excluded_by_proxy_gate",
                "proxy_status": proxy_status,
                "scientific_interpretation": (
                    "DT was deliberately not run; this is not a missing checkpoint or "
                    "mechanism falsification."
                ),
            })
            continue
        trajectory = summary["trajectory"]
        dynamic_reasons = _dynamic_audit_epoch_reasons(trajectory)
        selection_reasons = {
            int(epoch): ["fixed_long_horizon_state"] for epoch in fixed
        }
        for epoch, reasons in dynamic_reasons.items():
            bucket = selection_reasons.setdefault(int(epoch), [])
            bucket.extend(reason for reason in reasons if reason not in bucket)
        for epoch in sorted(selection_reasons):
            method = output_root / "anchors" / probe / "milestones" / f"e{epoch:03d}.pt"
            plain = output_root / "anchors" / "plain" / "milestones" / f"e{epoch:03d}.pt"
            # Dynamic epochs that are not registered milestones require a later
            # deterministic replay from the nearest prior full state.
            if not method.is_file() or not plain.is_file():
                missing.append({
                    "probe": probe, "epoch": epoch, "needs_replay": True,
                    "selection_reasons": selection_reasons[epoch],
                })
                continue
            terminal = epoch >= int(protocol["local_view"]["target_epochs"])
            branch_horizons = [1, 8, 32] if terminal else [1, 8, 32, 200]
            jobs.append({
                "probe": probe,
                "data_epoch": epoch,
                "updates": epoch * 150,
                "selection_reasons": selection_reasons[epoch],
                "plain_state": str(plain),
                "method_state": str(method),
                "operators": ["u0(S_plain)", "ui(S_plain)", "u0(S_method)", "ui(S_method)"],
                "branch_horizons_updates": branch_horizons,
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
                "branch_semantics": (
                    "terminal_base_lr_vector_field_no_future_label"
                    if terminal else "registered_training_continuation"
                ),
                "status": "PENDING_EXECUTION",
            })
    status = "READY" if jobs and not any(item["epoch"] in fixed for item in missing) else "NEEDS_REPLAY_OR_CHECKPOINTS"
    result = {
        "schema": "local-route1-audit-queue-v1",
        "status": status,
        "proxy_status_at_queue_freeze": proxy_status,
        "excluded_probes": excluded_probes,
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
    ranked = list(matrix.get("ranked_failure_mechanisms", []))
    historical_index_path = ROOT / "evidence" / "LONG_HORIZON_EVIDENCE_INDEX.jsonl"
    mechanism_map_path = ROOT / "evidence" / "lineage" / "MECHANISM_OBJECT_MAP.json"
    reuse_boundary_path = ROOT / "evidence" / "lineage" / "SEARCH005_REUSE_BOUNDARY.json"
    historical_priors = []
    if historical_index_path.is_file():
        historical_priors = [
            json.loads(line) for line in historical_index_path.read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]
    mechanisms = [
        row for row in ranked if row.get("candidate_generation_eligible") is True
    ][:3]
    cards = []
    route = {
        "correction_sign_reversal": "future_batch_consensus_or_one_sided_constraint",
        "correct_direction_unstable_magnitude": "adam_metric_trust_region",
        "sampling_variance": "unbiased_stratified_or_antithetic_estimator",
        "coordinate_horizon_imbalance": "identity_adaptive_coordinate",
        "rollout_distribution_speed": "bridge_gap_constrained_adaptive_teacher",
        "state_feedback_missing": "state_conditional_self_null_intervention",
        "state_independent_late_bias": "current_state_rate_or_curvature_reformulation",
        "endpoint_dispersion_instability": "endpoint_law_preserving_variance_or_constraint",
        "game_balance_instability": "state_conditional_game_metric_constraint",
    }
    for index, mechanism in enumerate(mechanisms, 1):
        kind = mechanism["failure_type"]
        cards.append({
            "candidate_id": f"G1-{index:02d}-{kind.upper().replace('_', '-')}",
            "generation": 1,
            "parent_evidence": mechanism,
            "construction_route": route.get(kind, "REQUIRES_MANUAL_MATHEMATICAL_DERIVATION"),
            "required_before_implementation": [
                "lineage evidence and prior-equivalence audit against tested constructions",
                "explicit UNSB object and update equation",
                "identity/self-null or unbiased condition",
                "objective/estimator/coordinate/endpoint-law change declaration",
                "proof that paired targets are inaccessible",
                "expected applicable state and target-blind trigger semantics",
                "single falsifying counterexample experiment",
                "compute, memory and recovery-state accounting",
                "proposal-only, observable-only and projected/full ablations",
            ],
            "status": "DERIVATION_REQUIRED_NOT_IMPLEMENTED",
        })
    matrix_sha256 = file_sha256(matrix_path)
    atlas_sha256 = file_sha256(atlas_path)
    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger_identity = {
        "causal_matrix_sha256": matrix_sha256,
        "reversal_atlas_sha256": atlas_sha256,
        "historical_evidence_index_sha256": file_sha256(historical_index_path),
        "mechanism_object_map_sha256": file_sha256(mechanism_map_path),
        "reuse_boundary_sha256": file_sha256(reuse_boundary_path),
    }
    if ledger_path.is_file():
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
            raise RuntimeError("hypothesis ledger schema mismatch")
        if ledger.get("evidence_identity") != ledger_identity:
            raise RuntimeError(
                "hypothesis ledger is stale for the completed causal evidence"
            )
    else:
        ledger = {
            "schema": "final-unsb-route1-hypothesis-ledger-v1",
            "status": "ACTIVE_DERIVATION",
            "evidence_identity": ledger_identity,
            "generation_policy": {
                "maximum_generation1_candidates": 3,
                "maximum_revisions_per_mechanism": 1,
                "maximum_components_per_composition": 2,
                "fixed_window_or_hyperparameter_grid_forbidden": True,
            },
            "records": [
                {
                    "candidate_id": card["candidate_id"],
                    "generation": 1,
                    "parent_candidate_id": None,
                    "parent_evidence": card["parent_evidence"],
                    "construction_route": card["construction_route"],
                    "status": "DERIVATION_REQUIRED",
                    "revision_count": 0,
                    "experiments": [],
                    "paired_controller_access": False,
                    "confirmation20_opened": False,
                }
                for card in cards
            ],
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        write_json(ledger_path, ledger)
    result = {
        "schema": "local-route1-derive-stage-v1",
        "status": (
            "DERIVATION_CARDS_REQUIRED"
            if cards else "NO_ELIGIBLE_DRIVER_OR_UNBIASED_ROUTE"
        ),
        "maximum_generation1_candidates": 3,
        "ranked_failure_mechanisms": ranked,
        "target_blind_signal_screen": matrix.get("target_blind_signal_screen"),
        "historical_evidence_index": {
            "path": str(historical_index_path.resolve()),
            "sha256": (
                file_sha256(historical_index_path)
                if historical_index_path.is_file() else None
            ),
            "rows": historical_priors,
            "role": (
                "corroborating priors and negative boundaries only; current long causal "
                "matrix remains the authority for signal-driven candidate generation"
            ),
        },
        "historical_mechanism_context": {
            "mechanism_object_map": {
                "path": str(mechanism_map_path.resolve()),
                "sha256": file_sha256(mechanism_map_path),
            },
            "search005_reuse_boundary": {
                "path": str(reuse_boundary_path.resolve()),
                "sha256": file_sha256(reuse_boundary_path),
            },
            "required_card_bindings": {
                "historical_evidence_index_sha256": file_sha256(historical_index_path),
                "mechanism_object_map_sha256": file_sha256(mechanism_map_path),
                "reuse_boundary_sha256": file_sha256(reuse_boundary_path),
            },
            "rule": (
                "Every candidate must state why its operator is materially distinct "
                "from equivalent tested DT/HJ/HNEK, PCOA/NPOOA, LBST, PTQ, DCUM, "
                "AEB, TA or KCK constructions; renaming an old implementation is not "
                "new algorithm discovery."
            ),
        },
        "cards": cards,
        "hypothesis_ledger": {
            "path": str(ledger_path.resolve()),
            "sha256": file_sha256(ledger_path),
            "status": ledger["status"],
            "records": len(ledger["records"]),
            "rule": (
                "The ledger is append-only scientific history. Candidate closure or "
                "one allowed revision must record the causal reason and experiment; "
                "paired development scores cannot rewrite the parent hypothesis."
            ),
        },
        "guard": (
            "No paired-fitted controller may be generated. A failure mechanism without an eligible target-blind driver may only proceed through an independently justified unbiased construction."
        ),
        "confirmation20_opened": False,
    }
    write_json(output_root / "derive" / "DERIVATION_QUEUE.json", result)
    return result


def validate_candidate_ready(output_root: Path, candidate_id: str) -> dict:
    candidate_id = validate_candidate_id(candidate_id)
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
    registration = load_candidate_registration(output_root, candidate_id, require_gate=False)
    gate_path = output_root / "derive" / "gates" / f"{candidate_id}.json"
    if not gate_path.is_file():
        return {
            **registration.to_dict(),
            "schema": "local-route1-candidate-stage-v1",
            "status": "READY_FOR_CANDIDATE_GATES",
            "required_gate": str(gate_path),
            "reason": (
                "The evidence/card/code chain is frozen, but mathematical, identity, "
                "resume, cross-state, target-blind and micro engineering gates must pass "
                "before a matched e200 run."
            ),
            "confirmation20_opened": False,
        }
    registration = load_candidate_registration(output_root, candidate_id, require_gate=True)
    return {
        **registration.to_dict(),
        "schema": "local-route1-candidate-stage-v1",
        "status": "READY_FOR_MATCHED_E200",
        "confirmation20_opened": False,
    }
