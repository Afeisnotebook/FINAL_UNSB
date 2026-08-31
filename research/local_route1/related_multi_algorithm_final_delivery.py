"""Publish the terminal route-1 result as an algorithm set, not a sole winner.

The existing complete-frontier delivery remains an immutable compatibility
artifact.  This supplement is the scientific authority after the related
native/HNEK/HJ conditional-estimator family has completed.  It uses only
same-host 4090 deltas for action ordering and carries 5090 trajectories as
host-separated runtime evidence.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from research.local_route1.complete_frontier import (
    SCHEMA as FRONTIER_SCHEMA,
    STATUS as FRONTIER_STATUS,
)
from research.local_route1.complete_frontier_final_delivery import (
    POINTER as COMPLETE_POINTER,
    POINTER_SCHEMA as COMPLETE_POINTER_SCHEMA,
    PUBLISHED_FILES as COMPLETE_PUBLISHED_FILES,
    _candidate_domain_trajectory,
    _read_json,
    _selected_source,
)
from research.local_route1.frontier_final_delivery import _executor_contract
from research.local_route1.related_algorithm_adjudication import (
    COMBINED_SCHEMA,
    HOST_SCHEMA,
    _terminal_row,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


ALGORITHM_SET_SCHEMA = "final-unsb-route1-related-algorithm-set-v1"
RESULTS_SCHEMA = "final-unsb-route1-related-multi-algorithm-results-v1"
ACTION_SCHEMA = "final-unsb-route1-related-action-priority-v1"
CANDIDATE_SCHEMA = "final-unsb-route1-related-action-candidate-v1"
ALTERNATES_SCHEMA = "final-unsb-route1-related-action-alternates-v1"
POINTER_SCHEMA = "final-unsb-route1-related-multi-algorithm-final-pointer-v1"
POINTER = "RELATED_MULTI_ALGORITHM_FINAL_POINTER.json"
FINAL_SUBDIR = Path("final") / "related_multi_algorithm"
PUBLISHED_FILES = (
    "ALGORITHM_SET.json",
    "ACTION_PRIORITY.json",
    "CANDIDATE.json",
    "ALTERNATES.json",
    "RELATED_RESULTS.json",
    "RELATED_FINAL_REPORT.md",
)

RELATED_4090 = "RELATED_4090_HOST_ADJUDICATION.json"
RELATED_5090 = "RELATED_5090_HOST_ADJUDICATION.json"
RELATED_COMBINED = "RELATED_MULTI_HOST_ADJUDICATION.json"

PROPOSAL = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
HPCGR = "G3-01B-PHYSICAL-HORIZON-CONDITIONAL-GF-RESAMPLING"
HJCGR = "G3-02-HJ-CONDITIONAL-GF-RESAMPLING"
AMTNC = "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS"
PCRSMG_FULL = "G1-02B-PLAYER-CONDITIONAL-RSMG"
PCNR = "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING"
HJPCNR = "ABL-G3-02-HJCGR-SINGLE-VIEW"
HJPCNR_RECEIPT = "HJPCNR_GAIN_SOURCE_E200_RECEIPT.json"


def _boundary(value: dict[str, Any], label: str) -> None:
    fixed = {
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if key in value and value.get(key) != expected:
            raise RuntimeError(f"{label} changed scientific boundary: {key}")
    if value.get("paired_metrics_used_for_formula_or_training_control") not in (
        None, False,
    ):
        raise RuntimeError(f"{label} used paired metrics for formula/control")


def _complete_delivery(output_root: Path) -> tuple[dict[str, Any], Path]:
    operations = output_root / "operations"
    final = output_root / "final"
    path = operations / COMPLETE_POINTER
    value = _read_json(path)
    if (
        value.get("schema") != COMPLETE_POINTER_SCHEMA
        or value.get("status") != "COMPLETE_FRONTIER_FINAL_DELIVERY_COMPLETE"
    ):
        raise RuntimeError("complete-frontier compatibility delivery is not terminal")
    _boundary(value, "complete-frontier delivery")
    hashes = value.get("final_file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(COMPLETE_PUBLISHED_FILES):
        raise RuntimeError("complete-frontier published file set changed")
    for name, expected in hashes.items():
        if file_sha256(final / name) != expected:
            raise RuntimeError(f"complete-frontier final file changed: {name}")
    return value, path


def _host(value: dict[str, Any], *, label: str) -> dict[str, Any]:
    if (
        value.get("schema") != HOST_SCHEMA
        or value.get("status") != "RELATED_HOST_E200_ADJUDICATION_COMPLETE"
        or value.get("host_label") != label
        or value.get("action_priority_is_not_scientific_exclusivity") is not True
    ):
        raise RuntimeError(f"related host adjudication is not terminal: {label}")
    _boundary(value, f"related host {label}")
    rows = value.get("ranking")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"related host ranking is empty: {label}")
    for row in rows:
        snapshot = row.get("trajectory_snapshot")
        if (
            not isinstance(snapshot, dict)
            or snapshot.get("candidate_id") != row.get("candidate_id")
            or snapshot.get("confirmation20_opened") is not False
            or snapshot.get("paired_metrics_used_for_training_or_gate") is not False
        ):
            raise RuntimeError(f"related host trajectory is not portable: {label}")
    return value


def _related_inputs(
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Path]]:
    operations = output_root / "operations"
    paths = {
        "remote4090": operations / RELATED_4090,
        "remote5090": operations / RELATED_5090,
        "combined": operations / RELATED_COMBINED,
    }
    host4090 = _host(_read_json(paths["remote4090"]), label="remote4090")
    host5090 = _host(_read_json(paths["remote5090"]), label="remote5090")
    combined = _read_json(paths["combined"])
    if combined.get("schema") != COMBINED_SCHEMA:
        raise RuntimeError("related combined adjudication schema changed")
    _boundary(combined, "related combined adjudication")
    if combined.get("cross_runtime_is_not_cross_seed") is not True:
        raise RuntimeError("related combined adjudication conflates runtime and seed")
    bindings = {
        row.get("host_label"): row for row in combined.get("host_adjudications", [])
        if isinstance(row, dict)
    }
    for label, path in (("remote4090", paths["remote4090"]),
                        ("remote5090", paths["remote5090"])):
        if bindings.get(label, {}).get("sha256") != file_sha256(path):
            raise RuntimeError(f"related combined host binding changed: {label}")
    for algorithm in combined.get("algorithms", []):
        for result in algorithm.get("host_results", []):
            snapshot = result.get("trajectory_snapshot")
            if not isinstance(snapshot, dict):
                raise RuntimeError("combined related result lost its trajectory")
    return host4090, host5090, combined, paths


def _base_frontier(output_root: Path) -> tuple[dict[str, Any], Path]:
    path = output_root / "operations" / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
    value = _read_json(path)
    if (
        value.get("schema") != FRONTIER_SCHEMA
        or value.get("status") != FRONTIER_STATUS
        or value.get("canonical_candidate_is_action_priority_only") is not True
    ):
        raise RuntimeError("base 4090 frontier is not terminal")
    _boundary(value, "base 4090 frontier")
    return value, path


def _related_row(row: dict[str, Any]) -> dict[str, Any]:
    classification = {
        "strict_sustained_local_signal": "strict_sustained",
        "positive_but_fragile": "positive_but_fragile",
        "closed_current_operator_on_this_host": "closed_current_operator",
    }.get(str(row.get("classification")))
    if classification is None:
        raise RuntimeError("unknown related host classification")
    snapshot = row["trajectory_snapshot"]
    return {
        "candidate_id": row["candidate_id"],
        "source_role": "related_multi_algorithm_4090",
        "classification": classification,
        "classification_checks": row["strict_checks"],
        "trajectory_status": snapshot.get("status"),
        "algorithm_fingerprint": row["algorithm_fingerprint"],
        "candidate_fingerprint": row["candidate_fingerprint"],
        "training_git_commit": row["training_git_commit"],
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": row[
                "late_three_mean_macro_psnr_delta"
            ],
            "e200_macro_psnr_delta": row["e200_macro_psnr_delta"],
            "late_points_with_four_of_six_positive_domains": row[
                "late_points_with_four_of_six_positive_domains"
            ],
            "late_average_worst_domain_delta": row[
                "late_average_worst_domain_delta"
            ],
            "candidate_best_to_terminal_three_point_rolling_drawdown": row[
                "rolling_drawdown_db"
            ],
            "late_mean_macro_ssim_delta": row["late_mean_macro_ssim_delta"],
            "late_mean_macro_lpips_delta": row["late_mean_macro_lpips_delta"],
        },
        "receipt_path": row["terminal_receipt_path"],
        "receipt_sha256": row["terminal_receipt_sha256"],
        "trajectory_path": row["trajectory_path"],
        "trajectory_sha256": row["trajectory_sha256"],
        "median_epoch_wall_seconds": row["median_epoch_wall_seconds"],
    }


def _rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    strict = 2 if row["classification"] == "strict_sustained" else (
        1 if row["classification"] == "positive_but_fragile" else 0
    )
    fields = row["ranking_fields"]
    return (
        strict,
        float(fields["late_three_mean_macro_psnr_delta"]),
        float(fields["e200_macro_psnr_delta"]),
        int(fields["late_points_with_four_of_six_positive_domains"]),
        float(fields["late_average_worst_domain_delta"]),
        -float(row.get("median_epoch_wall_seconds", 0.0)),
    )


def _composite_4090(
    output_root: Path, base: dict[str, Any], related: dict[str, Any],
) -> list[dict[str, Any]]:
    authority = base["same_host_authority"]
    related_authorities = {
        (
            row["base_e0_scientific_state_sha256"],
            row["base_protocol_fingerprint"],
            row["manifest_sha256"],
        )
        for row in related["ranking"]
    }
    expected = {
        (
            authority["base_e0_scientific_state_sha256"],
            authority["base_protocol_fingerprint"],
            authority["manifest_sha256"],
        )
    }
    if related_authorities != expected:
        raise RuntimeError("related 4090 candidates do not share base frontier authority")
    rows = {str(row["candidate_id"]): dict(row) for row in base["ranking"]}
    for row in related["ranking"]:
        rows[str(row["candidate_id"])] = _related_row(row)
    # HJ-PCNR was launched after the original related-host adjudication as the
    # completed HJCGR gain-source control.  Once it has its own source-bound
    # common-e0 e200 receipt, classify it with the exact same host-local rules
    # and keep it in the scientific algorithm set.  Treating a passing,
    # cheaper one-view operator as a footnote would recreate the single-winner
    # objective drift this delivery is designed to prevent.
    hjpcnr_host = _terminal_row(output_root, HJPCNR, "remote4090")
    hjpcnr_authority = {
        (
            hjpcnr_host["base_e0_scientific_state_sha256"],
            hjpcnr_host["base_protocol_fingerprint"],
            hjpcnr_host["manifest_sha256"],
        )
    }
    if hjpcnr_authority != expected:
        raise RuntimeError("HJ-PCNR does not share base frontier authority")
    hjpcnr_row = _related_row(hjpcnr_host)
    hjpcnr_row["source_role"] = "posthoc_hjcgr_gain_source_control_4090"
    hjpcnr_row["posthoc_development_control"] = True
    rows[HJPCNR] = hjpcnr_row
    ranking = sorted(rows.values(), key=_rank_key, reverse=True)
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def _mechanism_gain_source_decomposition(
    output_root: Path, ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    """Separate parent-field gain from the matched estimator composition gain.

    These are differences between complete common-e0 trajectories, not an
    additive causal attribution inside a single nonlinear training path.
    Keeping that boundary explicit lets the final report say whether the shared
    conditional estimator improves native, HNEK, and HJ fields without
    pretending that their PSNR deltas form a linear model.
    """
    control_path = output_root / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
    control = _read_json(control_path)
    roles = control.get("roles")
    observable = roles.get("observable_only") if isinstance(roles, dict) else None
    proposal_control = roles.get("proposal_only") if isinstance(roles, dict) else None
    full_control = roles.get("projected_or_full") if isinstance(roles, dict) else None
    identity = control.get("observable_only_identity")
    if (
        control.get("schema")
        != "final-unsb-route1-winner-ablation-adjudication-v1"
        or control.get("status")
        != "ABLATION_CHALLENGER_READY_FOR_SINGLE_SEED_DEVELOPMENT_SELECTION"
        or set(roles or {}) != {"proposal_only", "observable_only", "projected_or_full"}
        or roles.get("proposal_only", {}).get("candidate_id") != PROPOSAL
        or not isinstance(proposal_control, dict)
        or not isinstance(full_control, dict)
        or full_control.get("candidate_id") != PCRSMG_FULL
        or not isinstance(observable, dict)
        or not isinstance(identity, dict)
        or identity.get("status") != "EXACT_PLAIN_E200_DYNAMICS_IDENTITY"
        or identity.get("matched_plain_delta_exact_zero") is not True
        or observable.get("ranking_fields", {}).get(
            "late_three_mean_macro_psnr_delta"
        ) != 0.0
        or observable.get("ranking_fields", {}).get("e200_macro_psnr_delta") != 0.0
        or control.get("paired_metrics_used_for_training_or_control") is not False
        or control.get("paired_controller_access") is not False
        or control.get("confirmation20_opened") is not False
        or control.get("proposal_only_strict_gate_pass") is not True
        or control.get("projected_or_full_strict_gate_pass") is not False
        or control.get("proposal_only_out_ranks_full") is not True
    ):
        raise RuntimeError("gain-source compute-only control is not exact plain")
    compute_only_control = {
        "schema": "final-unsb-route1-related-compute-only-control-v1",
        "status": "EXTRA_VIEW_OBSERVATION_IS_EXACT_PLAIN_E200_DYNAMICS",
        "candidate_id": observable["candidate_id"],
        "source_path": control_path.relative_to(output_root).as_posix(),
        "source_sha256": file_sha256(control_path),
        "candidate_dynamics_state_sha256": identity[
            "candidate_dynamics_state_sha256"
        ],
        "plain_dynamics_state_sha256": identity["plain_dynamics_state_sha256"],
        "dynamics_state_exact_plain": (
            identity["candidate_dynamics_state_sha256"]
            == identity["plain_dynamics_state_sha256"]
        ),
        "late_three_mean_macro_psnr_delta": 0.0,
        "e200_macro_psnr_delta": 0.0,
        "interpretation": (
            "Extra stochastic-view evaluation without committing the replicated "
            "estimator cannot explain the gain. The retained family changes the "
            "committed estimator covariance/finite-step coupling and remains "
            "compute-sensitive; native-budget equivalence is not claimed."
        ),
        "rules_out_wall_clock_or_observer_side_effect_only": True,
        "does_not_claim_native_compute_budget_equivalence": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if compute_only_control["dynamics_state_exact_plain"] is not True:
        raise RuntimeError("gain-source compute-only dynamics hashes differ")
    proposal_fields = proposal_control.get("ranking_fields")
    full_fields = full_control.get("ranking_fields")
    if not isinstance(proposal_fields, dict) or not isinstance(full_fields, dict):
        raise RuntimeError("gain-source player-scope control lacks e200 fields")
    proposal_late = float(proposal_fields["late_three_mean_macro_psnr_delta"])
    proposal_e200 = float(proposal_fields["e200_macro_psnr_delta"])
    full_late = float(full_fields["late_three_mean_macro_psnr_delta"])
    full_e200 = float(full_fields["e200_macro_psnr_delta"])
    if not (proposal_late > 0.0 and proposal_e200 > 0.0 and full_e200 <= 0.0):
        raise RuntimeError("gain-source player-scope control changed conclusion")
    player_scope_control = {
        "schema": "final-unsb-route1-related-player-scope-control-v1",
        "status": "GF_ONLY_REPLICATION_SUSTAINS_E200_WHILE_FULL_PLAYER_REPLICATION_DOES_NOT",
        "source_path": control_path.relative_to(output_root).as_posix(),
        "source_sha256": file_sha256(control_path),
        "gf_only": {
            "candidate_id": PROPOSAL,
            "late_three_mean_macro_psnr_delta": proposal_late,
            "e200_macro_psnr_delta": proposal_e200,
            "strict_gate_pass": True,
        },
        "all_players": {
            "candidate_id": PCRSMG_FULL,
            "late_three_mean_macro_psnr_delta": full_late,
            "e200_macro_psnr_delta": full_e200,
            "strict_gate_pass": False,
        },
        "gf_only_minus_all_players": {
            "late_three_macro_psnr_delta": proposal_late - full_late,
            "e200_macro_psnr_delta": proposal_e200 - full_e200,
        },
        "interpretation": (
            "Variance reduction is not monotonically beneficial across the sequential "
            "UNSB game. Retaining native D/E stochasticity and reducing only the "
            "post-D/E joint G/F conditional variance preserves the terminal margin; "
            "replicating D/E as well raises the late mean but loses e200 benefit."
        ),
        "linked_related_family_candidate_ids": [PROPOSAL, HPCGR, HJCGR],
        "does_not_claim_additive_single_path_causality": True,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }

    ranked = {str(row.get("candidate_id")): row for row in ranking}
    pcnr = ranked.get(PCNR)
    if not isinstance(pcnr, dict):
        raise RuntimeError("gain-source decomposition lacks the PCNR resampling control")
    pcnr_fields = pcnr.get("ranking_fields")
    if not isinstance(pcnr_fields, dict):
        raise RuntimeError("gain-source PCNR resampling control lacks e200 fields")
    pcnr_late = float(pcnr_fields["late_three_mean_macro_psnr_delta"])
    pcnr_e200 = float(pcnr_fields["e200_macro_psnr_delta"])
    if not (pcnr_late <= 0.0 and pcnr_e200 <= 0.0):
        raise RuntimeError("gain-source PCNR resampling-only conclusion changed")
    base_frontier_path = (
        output_root / "operations" / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
    )
    conditional_resampling_control = {
        "schema": "final-unsb-route1-related-conditional-resampling-control-v1",
        "status": (
            "FRESH_POST_DE_RESAMPLING_ALONE_FAILS_WHILE_TWO_VIEW_GF_MEAN_PASSES"
        ),
        "source_path": base_frontier_path.relative_to(output_root).as_posix(),
        "source_sha256": file_sha256(base_frontier_path),
        "resampling_only": {
            "candidate_id": PCNR,
            "operator": "native one-view D/E plus one fresh post-D/E G/F view",
            "late_three_mean_macro_psnr_delta": pcnr_late,
            "e200_macro_psnr_delta": pcnr_e200,
            "strict_gate_pass": False,
        },
        "resampling_plus_two_view_mean": {
            "candidate_id": PROPOSAL,
            "operator": (
                "native one-view D/E plus two fresh post-D/E G/F views "
                "averaged pre-Adam"
            ),
            "late_three_mean_macro_psnr_delta": proposal_late,
            "e200_macro_psnr_delta": proposal_e200,
            "strict_gate_pass": True,
        },
        "two_view_mean_increment_over_resampling_only": {
            "late_three_macro_psnr_delta": proposal_late - pcnr_late,
            "e200_macro_psnr_delta": proposal_e200 - pcnr_e200,
        },
        "interpretation": (
            "Within the tested native-field operators, merely breaking reuse across "
            "the D/E-to-G/F player boundary is insufficient. The strict e200 pass "
            "appears only when the fresh post-D/E G/F field is estimated by the "
            "two-view mean. This supports selective within-batch G/F conditional-"
            "variance reduction rather than resampling alone."
        ),
        "only_tested_operator_scope": True,
        "does_not_claim_global_necessity": True,
        "does_not_claim_additive_single_path_causality": True,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }

    matrix_path = output_root / "audit" / "LONG_CAUSAL_MATRIX.json"
    matrix = _read_json(matrix_path)
    if matrix.get("status") != "COMPLETE_CAUSAL_AUDIT":
        raise RuntimeError("gain-source variance-axis alignment requires causal audit")
    variance_summaries = {
        str(row.get("probe")): row
        for row in matrix.get("sampling_variance_summaries", [])
        if isinstance(row, dict)
    }

    def audited_axis(probe: str, axis: str) -> dict[str, Any]:
        summary = variance_summaries.get(probe)
        axes = summary.get("axes") if isinstance(summary, dict) else None
        value = axes.get(axis) if isinstance(axes, dict) else None
        if not isinstance(value, dict):
            raise RuntimeError(f"missing variance axis: {probe}/{axis}")
        return {
            "rows": int(value["rows"]),
            "variance_dominated_rows": int(value["variance_dominated_rows"]),
            "mean_variance_fraction": float(value["mean_variance_fraction"]),
        }

    hj_within = audited_axis("hj", "latent_time_bridge_rng")
    hnek_within = audited_axis("hnek", "latent_time_bridge_rng")
    hnek_batch = audited_axis("hnek", "independent_unpaired_batch")
    if not (
        hj_within["rows"] > 0
        and hj_within["variance_dominated_rows"] == hj_within["rows"]
        and hnek_within["rows"] > 0
        and hnek_within["variance_dominated_rows"] == 0
    ):
        raise RuntimeError("gain-source variance-axis alignment changed")
    variance_axis_alignment = {
        "schema": "final-unsb-route1-related-variance-axis-alignment-v1",
        "operator_axis": "within_batch_latent_time_bridge_and_feature_sampling",
        "causal_matrix_path": matrix_path.relative_to(output_root).as_posix(),
        "causal_matrix_sha256": file_sha256(matrix_path),
        "members": {
            PROPOSAL: {
                "alignment": "empirically_aligned_by_completed_factorial_controls",
                "evidence": (
                    "PCNR one-view resampling fails, two-view G/F mean passes, and "
                    "all-player replication loses its terminal margin"
                ),
            },
            HJCGR: {
                "alignment": "directly_aligned_with_parent_audited_variance_axis",
                "latent_time_bridge_rng": hj_within,
            },
            HPCGR: {
                "alignment": "compositional_transfer_hypothesis_not_direct_axis_repair",
                "latent_time_bridge_rng": hnek_within,
                "independent_unpaired_batch": hnek_batch,
                "unaddressed_parent_axis": "independent_unpaired_batch",
                "interpretation": (
                    "HPCGR tests whether the empirically successful selective G/F "
                    "estimator transfers to the independently useful HNEK field; it "
                    "does not claim to reduce HNEK's audited across-batch variance."
                ),
            },
        },
        "shared_theorem_does_not_imply_shared_failure_mode": True,
        "hpcgr_viability_must_be_decided_by_complete_e200_trajectory": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "confirmation20_opened": False,
    }

    anchor_path = output_root / "evidence" / "ANCHOR_TRAJECTORIES.json"
    anchor = _read_json(anchor_path)
    if anchor.get("schema") != "local-route1-anchor-summary-v1":
        raise RuntimeError("gain-source decomposition requires canonical anchors")
    summaries = {
        str(row.get("probe_id")): row for row in anchor.get("summaries", [])
        if isinstance(row, dict)
    }
    def parent_metrics(probe: str | None) -> dict[str, Any]:
        if probe is None:
            return {
                "parent_id": "plain",
                "late_three_mean_macro_psnr_delta": 0.0,
                "e200_macro_psnr_delta": 0.0,
            }
        summary = summaries.get(probe)
        if not isinstance(summary, dict) or summary.get("complete_e200") is not True:
            raise RuntimeError(f"gain-source parent is incomplete: {probe}")
        e200 = next(
            (
                row for row in summary.get("trajectory", [])
                if int(row.get("epoch", -1)) == 200
            ),
            None,
        )
        if not isinstance(e200, dict):
            raise RuntimeError(f"gain-source parent lacks e200: {probe}")
        return {
            "parent_id": probe,
            "late_three_mean_macro_psnr_delta": float(
                summary["late_three_mean_macro_psnr_delta"]
            ),
            "e200_macro_psnr_delta": float(e200["macro_psnr_delta"]),
        }

    hjpcnr_receipt_path = output_root / "operations" / HJPCNR_RECEIPT
    hjpcnr_receipt = _read_json(hjpcnr_receipt_path)
    hjpcnr_trajectory_path = (
        output_root / "candidates" / HJPCNR / "CANDIDATE_TRAJECTORY.json"
    )
    hjpcnr_trajectory = _read_json(hjpcnr_trajectory_path)
    hjpcnr_fields = hjpcnr_receipt.get("ranking_fields")
    if not (
        hjpcnr_receipt.get("status")
        == "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT"
        and hjpcnr_receipt.get("candidate_id") == HJPCNR
        and hjpcnr_receipt.get("training_git_commit")
        == hjpcnr_receipt.get("verification_git_commit")
        and isinstance(hjpcnr_fields, dict)
        and hjpcnr_receipt.get("trajectory_sha256")
        == file_sha256(hjpcnr_trajectory_path)
        and hjpcnr_trajectory.get("candidate_id") == HJPCNR
        and hjpcnr_trajectory.get("paired_metrics_used_for_training_or_gate") is False
        and hjpcnr_trajectory.get("confirmation20_opened") is False
        and hjpcnr_receipt.get("paired_metrics_used_for_training_or_control") is False
        and hjpcnr_receipt.get("confirmation20_opened") is False
    ):
        raise RuntimeError("HJ-PCNR gain-source receipt is incomplete or unbound")
    hj_parent = parent_metrics("hj")
    hjcgr_row = ranked.get(HJCGR)
    if not isinstance(hjcgr_row, dict):
        raise RuntimeError("HJ-PCNR attribution requires the HJCGR host result")
    hjcgr_fields = hjcgr_row.get("ranking_fields")
    if not isinstance(hjcgr_fields, dict):
        raise RuntimeError("HJ-PCNR attribution lacks HJCGR ranking fields")
    single_late = float(hjpcnr_fields["late_three_mean_macro_psnr_delta"])
    single_e200 = float(hjpcnr_fields["e200_macro_psnr_delta"])
    two_late = float(hjcgr_fields["late_three_mean_macro_psnr_delta"])
    two_e200 = float(hjcgr_fields["e200_macro_psnr_delta"])
    hj_specific_factorial_control = {
        "schema": "final-unsb-route1-hj-specific-resampling-variance-control-v1",
        "status": "COMPLETE_E200_HJ_ONE_VS_TWO_VIEW_FACTORIAL_CONTROL",
        "source_path": hjpcnr_receipt_path.relative_to(output_root).as_posix(),
        "source_sha256": file_sha256(hjpcnr_receipt_path),
        "trajectory_path": hjpcnr_trajectory_path.relative_to(output_root).as_posix(),
        "trajectory_sha256": file_sha256(hjpcnr_trajectory_path),
        "continuous_hj_parent": hj_parent,
        "one_fresh_view": {
            "candidate_id": HJPCNR,
            "operator": "native D/E then one fresh post-D/E HJ G/F view",
            "late_three_mean_macro_psnr_delta": single_late,
            "e200_macro_psnr_delta": single_e200,
        },
        "two_fresh_view_mean": {
            "candidate_id": HJCGR,
            "operator": "native D/E then two fresh HJ G/F views averaged pre-Adam",
            "late_three_mean_macro_psnr_delta": two_late,
            "e200_macro_psnr_delta": two_e200,
        },
        "one_view_increment_over_hj": {
            "late_three_macro_psnr_delta": single_late - float(
                hj_parent["late_three_mean_macro_psnr_delta"]
            ),
            "e200_macro_psnr_delta": single_e200 - float(
                hj_parent["e200_macro_psnr_delta"]
            ),
        },
        "two_view_mean_increment_over_one_view": {
            "late_three_macro_psnr_delta": two_late - single_late,
            "e200_macro_psnr_delta": two_e200 - single_e200,
        },
        "interpretation_rule": (
            "HJ-PCNR minus HJ isolates post-D/E resampling under the HJ objective; "
            "HJCGR minus HJ-PCNR isolates adding the two-view pre-Adam mean under "
            "the same resampling order. These are differences between complete "
            "nonlinear common-e0 trajectories, not additive single-path effects."
        ),
        "trajectory_snapshot": hjpcnr_trajectory,
        "paired_parent_result_used_only_to_authorize_completed_parent_ablation": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }

    specifications = (
        (PROPOSAL, None, "native_UNSB_field"),
        (HPCGR, "hnek", "HNEK_physical_horizon_bridge_game"),
        (HJCGR, "hj", "HJ_structure_projected_PatchNCE_objective"),
    )
    members = []
    for candidate_id, parent_probe, base_object in specifications:
        candidate = ranked.get(candidate_id)
        if not isinstance(candidate, dict):
            raise RuntimeError(f"gain-source candidate is missing: {candidate_id}")
        fields = candidate["ranking_fields"]
        parent = parent_metrics(parent_probe)
        child_late = float(fields["late_three_mean_macro_psnr_delta"])
        child_e200 = float(fields["e200_macro_psnr_delta"])
        late_increment = child_late - float(
            parent["late_three_mean_macro_psnr_delta"]
        )
        e200_increment = child_e200 - float(parent["e200_macro_psnr_delta"])
        if late_increment > 0.0 and e200_increment > 0.0:
            interpretation = "shared_estimator_improves_parent_field"
        elif child_late > 0.0 and child_e200 > 0.0:
            interpretation = "parent_gain_survives_but_estimator_does_not_improve_parent"
        else:
            interpretation = "composition_not_long_horizon_positive"
        members.append({
            "candidate_id": candidate_id,
            "base_object": base_object,
            "parent": parent,
            "composed": {
                "late_three_mean_macro_psnr_delta": child_late,
                "e200_macro_psnr_delta": child_e200,
            },
            "matched_compositional_increment_over_parent": {
                "late_three_macro_psnr_delta": late_increment,
                "e200_macro_psnr_delta": e200_increment,
            },
            "interpretation": interpretation,
        })
    supported = [
        row["candidate_id"] for row in members
        if row["interpretation"] == "shared_estimator_improves_parent_field"
    ]
    return {
        "schema": "final-unsb-route1-related-gain-source-decomposition-v1",
        "status": (
            "SHARED_ESTIMATOR_IMPROVES_MULTIPLE_PARENT_FIELDS"
            if len(supported) >= 2 else
            "SHARED_ESTIMATOR_IMPROVES_ONE_PARENT_FIELD"
            if len(supported) == 1 else
            "NO_POSITIVE_MATCHED_COMPOSITIONAL_INCREMENT"
        ),
        "members": members,
        "shared_estimator_positive_increment_candidate_ids": supported,
        "shared_estimator_positive_increment_count": len(supported),
        "conditional_mean_theorem": (
            "For finite-covariance conditionally iid views evaluated at one "
            "fixed post-D/E parent state, one fixed official unpaired batch "
            "(and one fixed parent controller state), "
            "the two-view mean preserves the parent conditional expected G/F "
            "gradient and halves its conditional covariance."
        ),
        "stochastic_variance_scope": {
            "conditioning_includes_official_unpaired_batch": True,
            "reduced_components": (
                "within-batch native G/F view randomness, including latent, bridge "
                "time, bridge noise and PatchNCE feature sampling"
            ),
            "not_reduced_components": (
                "official A/B sample identity, domain draw and other across-batch "
                "data-sampling variance"
            ),
            "iid_requirement": (
                "the two G/F views are conditionally iid with finite covariance and "
                "are evaluated before the single G/F optimizer commit"
            ),
        },
        "hj_state_transition_boundary": (
            "HJCGR starts both replicas from one HJ controller state, advances "
            "integer physical counters once, and mean-reduces floating diagnostics."
        ),
        "matched_increment_is_not_additive_causal_attribution": True,
        "compute_only_control": compute_only_control,
        "player_scope_control": player_scope_control,
        "conditional_resampling_control": conditional_resampling_control,
        "hj_specific_factorial_control": hj_specific_factorial_control,
        "variance_axis_alignment": variance_axis_alignment,
        "optimizer_nonlinearity_boundary": (
            "Unbiasedness is for the pre-Adam stochastic gradient estimator at a "
            "fixed realized parent state. Adam is applied once after averaging; no "
            "claim is made that expected finite Adam displacement or sample path "
            "equals the one-view parent."
        ),
        "cross_host_metrics_merged": False,
        "anchor_summary_sha256": file_sha256(anchor_path),
        "hjpcnr_gain_source_receipt_sha256": file_sha256(hjpcnr_receipt_path),
    }


def _member(
    output_root: Path, row: dict[str, Any], *, disposition: str,
) -> dict[str, Any]:
    (
        receipt, receipt_path, trajectory, trajectory_path, card, card_path,
        implementation,
    ) = _selected_source(output_root, row)
    candidate_id = str(row["candidate_id"])
    implementation_path = (
        output_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    executor_path, executor = _executor_contract(output_root, receipt)
    reproduction: dict[str, Any] = {
        "seed2026_e200": (
            "PYTHONPATH=<REPO> python -m "
            "operations.local_route1_candidate_executor "
            f"--contract <RUN_ROOT>/{executor_path.relative_to(output_root).as_posix()}"
        ),
        "executor_contract": {
            "path": executor_path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(executor_path),
            "candidate_git_commit": executor["candidate_git_commit"],
            "algorithm_fingerprint": executor["algorithm_fingerprint"],
            "candidate_fingerprint": executor["candidate_fingerprint"],
        },
        "deferred_seed_validation": [2027, 2028],
    }
    return {
        "candidate_id": candidate_id,
        "disposition": disposition,
        "classification": row["classification"],
        "source_role": row["source_role"],
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "candidate_fingerprint": receipt["candidate_fingerprint"],
        "training_git_commit": receipt["training_git_commit"],
        "ranking_fields": row["ranking_fields"],
        "trajectory": trajectory,
        "absolute_relative_domain_trajectory": _candidate_domain_trajectory(
            output_root, candidate_id,
        ),
        "mathematics": {
            "name": card.get("name", candidate_id),
            "unsb_object": card.get("unsb_object"),
            "formula": card.get("formula"),
            "identity_or_unbiased_condition": card.get(
                "identity_or_unbiased_condition"
            ),
            "unbiased_proof": card.get("unbiased_proof"),
            "target_inaccessibility_proof": card.get(
                "target_inaccessibility_proof"
            ),
        },
        "algorithm_hyperparameters": card.get("algorithm_hyperparameters"),
        "executable_configuration": {
            "model": implementation.get("model"),
            "method": implementation.get("method"),
        },
        "source_files": implementation.get("source_files", []),
        "complexity": {
            "compute_cost": card.get("compute_cost"),
            "memory_cost": card.get("memory_cost"),
            "recovery_state_cost": card.get("recovery_state_cost"),
        },
        "risk": {
            "expected_applicable_state": card.get("expected_applicable_state"),
            "falsifying_experiment": card.get("falsifying_experiment"),
            "single_seed_only": True,
            "cross_seed_stability_claimed": False,
            "full_10000_image_200_epoch_behavior_untested": True,
            "confirmation20_generalization_untested": True,
            "posthoc_gain_source_development_control": candidate_id == HJPCNR,
        },
        "reproduction": reproduction,
        "source_bound": {
            "terminal_receipt": {
                "path": receipt_path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(receipt_path),
            },
            "trajectory": {
                "path": trajectory_path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(trajectory_path),
            },
            "derivation_card": {
                "path": card_path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(card_path),
            },
            "implementation": {
                "path": implementation_path.relative_to(output_root).as_posix(),
                "sha256": file_sha256(implementation_path),
                "model": implementation.get("model"),
                "method": implementation.get("method"),
            },
        },
    }


def _report(
    algorithm_set: dict[str, Any], action: dict[str, Any],
    alternates: dict[str, Any],
) -> str:
    lines = [
        "# FINAL UNSB 路线一：多算法科学交付",
        "",
        f"- 下一步行动优先级：`{action['candidate_id']}`。这不是科学排他性冠军。",
        "- 严格可行算法（全部保留）：" + (
            "、".join(
                f"`{candidate_id}`"
                for candidate_id in algorithm_set["strict_viable_candidate_ids"]
            ) or "无"
        ) + "。",
        "- 正向但脆弱算法（全部保留）：" + (
            "、".join(
                f"`{candidate_id}`"
                for candidate_id in algorithm_set[
                    "positive_but_fragile_candidate_ids"
                ]
            ) or "无"
        ) + "。",
        "- 兼容性交付仍提供唯一行动候选；该身份不删除算法集合中的其他可行机制。",
        "- 所有排序仅使用4090同宿主、共同e0、small25、seed2026、真实e200结果。",
        "- 5090结果只作为独立运行时证据；没有把跨宿主差值平均成多seed结论。",
        "",
        "## 数学谱系",
        "",
        "- 相关族共享 post-D/E 条件独立双视图 G/F 均值；它保持各自父场的条件期望并降低协方差。",
        "- 三个父对象分别是原生UNSB场、HNEK physical-horizon bridge game、HJ结构投影PatchNCE目标。",
        "- AM-TNC是独立的Adam度量切向估计机制，不因相关族成立而被删除。",
        "- HJ-PCNR虽由已完成HJCGR结果触发，但若通过相同e200护栏，会作为低计算量开发算法进入总榜，而不只作为因果脚注；它不是confirmation结果。",
        "",
        "## 收益来源分解",
        "",
    ]
    for row in algorithm_set["mechanism_gain_source_decomposition"]["members"]:
        increment = row["matched_compositional_increment_over_parent"]
        lines.append(
            f"- `{row['candidate_id']}` 相对 `{row['parent']['parent_id']}`："
            f"late-three增量 `{increment['late_three_macro_psnr_delta']:+.6f}` dB，"
            f"e200增量 `{increment['e200_macro_psnr_delta']:+.6f}` dB；"
            f"裁决 `{row['interpretation']}`。"
        )
    hj_factorial = algorithm_set["mechanism_gain_source_decomposition"][
        "hj_specific_factorial_control"
    ]
    hj_one = hj_factorial["one_view_increment_over_hj"]
    hj_two = hj_factorial["two_view_mean_increment_over_one_view"]
    lines.extend([
        "- 上述增量来自共同e0的两条完整非线性轨迹之差，不解释为单轨迹内可加因果贡献。",
        f"- HJ专属一视图对照：单个fresh G/F view相对HJ为late-three `{hj_one['late_three_macro_psnr_delta']:+.6f}` dB、e200 `{hj_one['e200_macro_psnr_delta']:+.6f}` dB；二视图均值再相对一视图增加 `{hj_two['late_three_macro_psnr_delta']:+.6f}` / `{hj_two['e200_macro_psnr_delta']:+.6f}` dB。这把跨玩家重采样与条件方差缩减分开。",
        "- compute-only控制：额外视图仅观察、不提交复制估计器时，e200 dynamics与plain精确一致且delta为0；这排除观察计算/墙钟副作用，但不声称原生算力预算等价。",
        "- player-scope控制：仅G/F复制在e200保持正收益，而D/E/G/F全复制虽有更高late-three均值却在e200回到非正；长期收益不是全局降方差的单调结果。",
        "- resampling控制：仅在D/E后重新抽一个G/F view的PCNR在late-three与e200均非正；加入同batch双视图均值后才严格通过。当前证据支持的是选择性G/F条件方差缩减，而不是重新采样本身。",
        "- 方差边界：双视图共享同一官方unpaired batch；减小的是给定batch后的latent/time/bridge/PatchNCE条件方差，不减小跨batch或跨域采样方差。",
        "- 证据轴对齐：HJ的latent/time/bridge轴直接由方差主导；HPCGR则是把已验证的G/F估计器迁移到独立有效的HNEK父场，并不预称修复HNEK主要的跨batch方差。其成败由完整e200轨迹决定。",
        "- 无偏性只针对固定父状态下的pre-Adam梯度估计器；不声明有限步Adam位移或随机样本路径与父算法相同。",
        "",
        "## 结论边界",
        "",
        "- 当前是单seed开发裁决，不声称跨seed稳定。",
        "- confirmation20仍封存；paired指标从未用于公式、训练控制、退出或checkpoint选择。",
        "- `ALGORITHM_SET.json`保存每条可行/脆弱/关闭实现的公式、逐域轨迹和来源哈希。",
        "- `CANDIDATE.json`是当前行动入口；`ALTERNATES.json`保存两个证据排序递补。",
        "",
    ])
    lines.insert(8, "- 两个递补：" + "、".join(
        f"`{row['candidate_id']}`（{row['disposition']}）"
        for row in alternates["alternates"]
    ) + "。")
    return "\n".join(lines)


def materialize_related_multi_algorithm_final_delivery(
    output_root: Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    destination = output_root / FINAL_SUBDIR
    pointer_path = operations / POINTER
    if pointer_path.is_file():
        pointer = _read_json(pointer_path)
        if pointer.get("schema") != POINTER_SCHEMA:
            raise RuntimeError("related final pointer schema changed")
        _boundary(pointer, "related final pointer")
        for name, expected in pointer.get("final_file_sha256", {}).items():
            if file_sha256(destination / name) != expected:
                raise RuntimeError(f"related final file changed: {name}")
        return pointer

    complete_pointer, complete_pointer_path = _complete_delivery(output_root)
    base, base_path = _base_frontier(output_root)
    host4090, host5090, combined, related_paths = _related_inputs(output_root)
    ranking = _composite_4090(output_root, base, host4090)
    gain_source = _mechanism_gain_source_decomposition(output_root, ranking)
    selected = ranking[0]
    selected_id = str(selected["candidate_id"])

    members = []
    for row in ranking:
        if row["classification"] == "strict_sustained":
            disposition = "strict_viable_algorithm"
        elif row["classification"] == "positive_but_fragile":
            disposition = "positive_but_fragile_algorithm"
        else:
            disposition = "closed_current_operator_on_current_protocol"
        members.append(_member(
            output_root,
            row,
            disposition=disposition,
        ))

    strict_ids = [
        row["candidate_id"] for row in members
        if row["disposition"] == "strict_viable_algorithm"
    ]
    fragile_ids = [
        row["candidate_id"] for row in members
        if row["disposition"] == "positive_but_fragile_algorithm"
    ]
    algorithm_set = {
        "schema": ALGORITHM_SET_SCHEMA,
        "status": (
            "MULTIPLE_VIABLE_ALGORITHMS"
            if len(strict_ids) >= 2 else
            "ONE_VIABLE_ALGORITHM_WITH_RELATED_FRONTIER"
            if strict_ids else
            "NO_STRICT_ALGORITHM_RELATED_FRONTIER_PRESERVED"
        ),
        "action_priority_candidate_id": selected_id,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "strict_viable_candidate_ids": strict_ids,
        "positive_but_fragile_candidate_ids": fragile_ids,
        "same_host_4090_ranking": ranking,
        "members": members,
        "related_conditional_estimator_family": {
            "shared_operator": "post-D/E conditionally iid two-view G/F mean",
            "unbiased_mathematical_object": "pre-Adam joint G/F stochastic gradient estimator",
            "conditioning_scope": (
                "fixed realized post-D/E model, optimizer, official unpaired batch "
                "and any parent-controller state"
            ),
            "stochastic_variance_scope": {
                "conditioning_includes_official_unpaired_batch": True,
                "reduced_components": (
                    "within-batch latent, bridge-time, bridge-noise and PatchNCE "
                    "feature-sampling variance in the joint G/F estimator"
                ),
                "not_reduced_components": (
                    "official A/B identity, domain and other across-batch sampling variance"
                ),
            },
            "conditional_expectation_property": (
                "E[(g_1+g_2)/2 | post-D/E state]=E[g_native_parent | state]"
            ),
            "conditional_covariance_property": (
                "Cov[(g_1+g_2)/2 | state]=Cov[g_native_parent | state]/2"
            ),
            "adam_boundary": (
                "Adam is applied once after the mean; expected finite Adam displacement "
                "and exact parent sample-path equality are not claimed"
            ),
            "finite_step_coupling_change_is_intended": True,
            "native_de_stochasticity_retained": True,
            "members": [
                {"candidate_id": PROPOSAL, "base_object": "native UNSB field"},
                {"candidate_id": HPCGR, "base_object": "HNEK physical-horizon bridge game"},
                {"candidate_id": HJCGR, "base_object": "HJ structure-projected PatchNCE objective"},
            ],
            "gain_source_controls": [
                {
                    "candidate_id": HJPCNR,
                    "operator": "one fresh post-D/E HJ G/F view",
                    "ranked_if_same_e200_guardrails_pass": True,
                    "posthoc_development_control": True,
                    "confirmation_result": False,
                },
            ],
            "membership_is_not_assumed_viability": True,
            "shared_theorem_does_not_imply_shared_parent_failure_mode": True,
        },
        "independent_mechanism_members": [
            {"candidate_id": AMTNC, "mechanism": "Adam-metric tangential estimator"},
        ],
        "mechanism_gain_source_decomposition": gain_source,
        "cross_runtime_related_evidence": combined,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    selected_member = next(
        row for row in members if row["candidate_id"] == selected_id
    )
    action = {
        "schema": ACTION_SCHEMA,
        "status": "CURRENT_NEXT_ACTION_PRIORITY",
        "candidate_id": selected_id,
        "classification": selected_member["classification"],
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_set_status": algorithm_set["status"],
        "ranking_fields": selected_member["ranking_fields"],
        "mathematics": selected_member["mathematics"],
        "complexity": selected_member["complexity"],
        "risk": selected_member["risk"],
        "reproduction": selected_member["reproduction"],
        "source_bound": selected_member["source_bound"],
        "selection_seeds": [2026],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    alternate_members = [
        row for row in members if row["candidate_id"] != selected_id
    ][:2]
    if len(alternate_members) != 2:
        raise RuntimeError("related final delivery requires exactly two ranked alternates")
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "status": "CURRENT_ACTION_PRIORITY_FROM_MULTI_ALGORITHM_SET",
        "candidate_id": selected_id,
        "classification": selected_member["classification"],
        "disposition": selected_member["disposition"],
        "canonical_candidate_is_action_priority_only": True,
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "algorithm": selected_member,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    alternates = {
        "schema": ALTERNATES_SCHEMA,
        "status": "TWO_EVIDENCE_RANKED_ALTERNATES",
        "action_priority_candidate_id": selected_id,
        "alternates": [
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "classification": row["classification"],
                "disposition": row["disposition"],
                "algorithm": row,
            }
            for rank, row in enumerate(alternate_members, start=2)
        ],
        "action_priority_is_not_scientific_exclusivity": True,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "RELATED_MULTI_ALGORITHM_E200_COMPLETE",
        "action_priority_candidate_id": selected_id,
        "algorithm_set_status": algorithm_set["status"],
        "complete_frontier_compatibility_pointer": complete_pointer,
        "base_4090_frontier": base,
        "related_4090_host_adjudication": host4090,
        "related_5090_host_adjudication": host5090,
        "related_multi_host_adjudication": combined,
        "composite_same_host_4090_ranking": ranking,
        "mechanism_gain_source_decomposition": gain_source,
        "action_candidate": candidate,
        "ranked_alternates": alternates,
        "cross_host_deltas_merged": False,
        "cross_runtime_is_not_cross_seed": True,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }

    staging = destination / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    write_json(staging / "ALGORITHM_SET.json", algorithm_set)
    write_json(staging / "ACTION_PRIORITY.json", action)
    write_json(staging / "CANDIDATE.json", candidate)
    write_json(staging / "ALTERNATES.json", alternates)
    write_json(staging / "RELATED_RESULTS.json", results)
    (staging / "RELATED_FINAL_REPORT.md").write_text(
        _report(algorithm_set, action, alternates), encoding="utf-8",
    )
    hashes = {name: file_sha256(staging / name) for name in PUBLISHED_FILES}
    destination.mkdir(parents=True, exist_ok=True)
    for name in PUBLISHED_FILES:
        os.replace(staging / name, destination / name)
    if any(file_sha256(destination / name) != expected
           for name, expected in hashes.items()):
        raise RuntimeError("related final files changed during publication")

    pointer = {
        "schema": POINTER_SCHEMA,
        "status": "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_COMPLETE",
        "action_priority_candidate_id": selected_id,
        "algorithm_set_status": algorithm_set["status"],
        "strict_viable_candidate_count": len(strict_ids),
        "positive_but_fragile_candidate_count": len(fragile_ids),
        "final_subdir": FINAL_SUBDIR.as_posix(),
        "final_file_sha256": hashes,
        "complete_frontier_pointer_sha256": file_sha256(complete_pointer_path),
        "base_4090_frontier_sha256": file_sha256(base_path),
        "related_4090_host_adjudication_sha256": file_sha256(
            related_paths["remote4090"]
        ),
        "related_5090_host_adjudication_sha256": file_sha256(
            related_paths["remote5090"]
        ),
        "related_multi_host_adjudication_sha256": file_sha256(
            related_paths["combined"]
        ),
        "hjpcnr_gain_source_receipt_sha256": file_sha256(
            operations / HJPCNR_RECEIPT
        ),
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(pointer_path, pointer)
    return pointer
