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
from research.local_route1.related_algorithm_adjudication import (
    COMBINED_SCHEMA,
    HOST_SCHEMA,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


ALGORITHM_SET_SCHEMA = "final-unsb-route1-related-algorithm-set-v1"
RESULTS_SCHEMA = "final-unsb-route1-related-multi-algorithm-results-v1"
ACTION_SCHEMA = "final-unsb-route1-related-action-priority-v1"
POINTER_SCHEMA = "final-unsb-route1-related-multi-algorithm-final-pointer-v1"
POINTER = "RELATED_MULTI_ALGORITHM_FINAL_POINTER.json"
FINAL_SUBDIR = Path("final") / "related_multi_algorithm"
PUBLISHED_FILES = (
    "ALGORITHM_SET.json",
    "ACTION_PRIORITY.json",
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
    base: dict[str, Any], related: dict[str, Any],
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
    anchor_path = output_root / "evidence" / "ANCHOR_TRAJECTORIES.json"
    anchor = _read_json(anchor_path)
    if anchor.get("schema") != "local-route1-anchor-summary-v1":
        raise RuntimeError("gain-source decomposition requires canonical anchors")
    summaries = {
        str(row.get("probe_id")): row for row in anchor.get("summaries", [])
        if isinstance(row, dict)
    }
    ranked = {str(row.get("candidate_id")): row for row in ranking}

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
            "fixed post-D/E parent state (and one fixed parent controller state), "
            "the two-view mean preserves the parent conditional expected G/F "
            "gradient and halves its conditional covariance."
        ),
        "hj_state_transition_boundary": (
            "HJCGR starts both replicas from one HJ controller state, advances "
            "integer physical counters once, and mean-reduces floating diagnostics."
        ),
        "matched_increment_is_not_additive_causal_attribution": True,
        "cross_host_metrics_merged": False,
        "anchor_summary_sha256": file_sha256(anchor_path),
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
        "complexity": {
            "compute_cost": card.get("compute_cost"),
            "memory_cost": card.get("memory_cost"),
            "recovery_state_cost": card.get("recovery_state_cost"),
        },
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


def _report(algorithm_set: dict[str, Any], action: dict[str, Any]) -> str:
    lines = [
        "# FINAL UNSB 路线一：多算法科学交付",
        "",
        f"- 下一步行动优先级：`{action['candidate_id']}`。这不是科学排他性冠军。",
        f"- 严格可行算法：`{len(algorithm_set['strict_viable_candidate_ids'])}` 条。",
        f"- 正向但脆弱算法：`{len(algorithm_set['positive_but_fragile_candidate_ids'])}` 条。",
        "- 所有排序仅使用4090同宿主、共同e0、small25、seed2026、真实e200结果。",
        "- 5090结果只作为独立运行时证据；没有把跨宿主差值平均成多seed结论。",
        "",
        "## 数学谱系",
        "",
        "- 相关族共享 post-D/E 条件独立双视图 G/F 均值；它保持各自父场的条件期望并降低协方差。",
        "- 三个父对象分别是原生UNSB场、HNEK physical-horizon bridge game、HJ结构投影PatchNCE目标。",
        "- AM-TNC是独立的Adam度量切向估计机制，不因相关族成立而被删除。",
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
    lines.extend([
        "- 上述增量来自共同e0的两条完整非线性轨迹之差，不解释为单轨迹内可加因果贡献。",
        "",
        "## 结论边界",
        "",
        "- 当前是单seed开发裁决，不声称跨seed稳定。",
        "- confirmation20仍封存；paired指标从未用于公式、训练控制、退出或checkpoint选择。",
        "- `ALGORITHM_SET.json`保存每条可行/脆弱/关闭实现的公式、逐域轨迹和来源哈希。",
        "",
    ])
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
    ranking = _composite_4090(base, host4090)
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
        members.append(_member(output_root, row, disposition=disposition))

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
            "conditional_expectation_property": (
                "E[(g_1+g_2)/2 | post-D/E state]=E[g_native_parent | state]"
            ),
            "conditional_covariance_property": (
                "Cov[(g_1+g_2)/2 | state]=Cov[g_native_parent | state]/2"
            ),
            "members": [
                {"candidate_id": PROPOSAL, "base_object": "native UNSB field"},
                {"candidate_id": HPCGR, "base_object": "HNEK physical-horizon bridge game"},
                {"candidate_id": HJCGR, "base_object": "HJ structure-projected PatchNCE objective"},
            ],
            "membership_is_not_assumed_viability": True,
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
        "source_bound": selected_member["source_bound"],
        "selection_seeds": [2026],
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
    write_json(staging / "RELATED_RESULTS.json", results)
    (staging / "RELATED_FINAL_REPORT.md").write_text(
        _report(algorithm_set, action), encoding="utf-8",
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
