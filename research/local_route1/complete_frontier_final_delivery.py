"""Materialize the terminal multi-candidate route-1 delivery.

Canonical selection is made only from the complete same-host 4090 frontier.
The complete 5090 repaired/ablation frontier is preserved as host-separated
mechanistic evidence and never contributes a cross-host delta to that rank.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import _validate_receipt
from research.local_route1.complete_frontier import SCHEMA as FRONTIER_SCHEMA, STATUS as FRONTIER_STATUS
from research.local_route1.final_delivery import (
    _candidate_domain_trajectory,
    _median_epoch_seconds,
)
from research.local_route1.frontier_advancement import STRICT
from research.local_route1.frontier_final_delivery import _executor_contract
from research.local_route1.portable_extended_frontier import (
    validate_portable_extended_frontier,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


CANDIDATE_SCHEMA = "final-unsb-route1-complete-frontier-candidate-v1"
RESULTS_SCHEMA = "final-unsb-route1-complete-frontier-results-v1"
ALTERNATES_SCHEMA = "final-unsb-route1-complete-frontier-alternates-v1"
RESEARCH_FRONTIER_SCHEMA = "final-unsb-route1-complete-research-frontier-v1"
POINTER_SCHEMA = "final-unsb-route1-complete-frontier-delivery-pointer-v1"
POINTER = "COMPLETE_FRONTIER_FINAL_DELIVERY_POINTER.json"
LEGACY_FINAL_FILES = (
    "CANDIDATE.json", "RESULTS.json", "ALTERNATES.json", "FINAL_ROUTE1_REPORT.md",
)
PUBLISHED_FILES = (*LEGACY_FINAL_FILES, "RESEARCH_FRONTIER.json")

PCRSMG_PROPOSAL = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
RFAMMCRB = "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER"
RFMCRB = "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER"
G3_ADAM = "G3-02-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-ADAM-BARRIER"
G3_EUCLIDEAN = "G3-03-CONDITIONAL-SAMPLING-RESIDUAL-FEASIBLE-EUCLIDEAN-BARRIER"
G3_BARRIER = {G3_ADAM: RFAMMCRB, G3_EUCLIDEAN: RFMCRB}
HISTORICAL_PROBES = {"dt", "hj", "hnek"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _posthoc_boundary(value: dict[str, Any], *, label: str) -> None:
    if value.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not prove confirmation20 remained closed")
    for key in ("paired_controller_access", "paired_metrics_used_for_formula_or_training_control"):
        if key in value and value[key] is not False:
            raise RuntimeError(f"{label} violates target-blind scope: {key}")
    if value.get("cross_host_deltas_merged") not in (None, False):
        raise RuntimeError(f"{label} merged cross-host deltas")


def _complete_frontier(output_root: Path) -> tuple[dict[str, Any], Path]:
    path = output_root / "operations" / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
    value = _read_json(path)
    if (
        value.get("schema") != FRONTIER_SCHEMA
        or value.get("status") != FRONTIER_STATUS
        or value.get("canonical_candidate_is_action_priority_only") is not True
        or value.get("algorithm_discovery_collapsed_to_single_candidate") is not False
    ):
        raise RuntimeError("complete 4090 frontier is not terminal")
    _posthoc_boundary(value, label="complete 4090 frontier")
    ranking = value.get("ranking")
    if not isinstance(ranking, list) or len(ranking) < 3:
        raise RuntimeError("complete 4090 frontier lacks two alternates")
    ids = [str(row.get("candidate_id", "")) for row in ranking if isinstance(row, dict)]
    if len(ids) != len(ranking) or len(ids) != len(set(ids)):
        raise RuntimeError("complete 4090 frontier candidate ids are invalid")
    if value.get("action_priority_candidate_id") != ids[0]:
        raise RuntimeError("complete 4090 frontier action priority differs from rank one")
    return value, path


def _portable_frontier(output_root: Path) -> tuple[dict[str, Any], Path]:
    path = output_root / "operations" / "PORTABLE_EXTENDED_REPAIRED_FRONTIER_5090.json"
    value = validate_portable_extended_frontier(_read_json(path))
    _posthoc_boundary(value, label="portable 5090 extended frontier")
    return value, path


def _selected_source(
    output_root: Path, row: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    candidate_id = str(row["candidate_id"])
    receipt_path = Path(str(row["receipt_path"])).resolve()
    if (
        not receipt_path.is_file()
        or not receipt_path.is_relative_to(output_root)
        or file_sha256(receipt_path) != row.get("receipt_sha256")
    ):
        raise RuntimeError("selected complete-frontier receipt changed")
    receipt = _validate_receipt(receipt_path)
    if receipt.get("candidate_id") != candidate_id:
        raise RuntimeError("selected receipt candidate identity changed")
    trajectory_path = Path(str(receipt["trajectory_path"])).resolve()
    if (
        not trajectory_path.is_file()
        or not trajectory_path.is_relative_to(output_root)
        or file_sha256(trajectory_path) != receipt.get("trajectory_sha256")
    ):
        raise RuntimeError("selected complete-frontier trajectory changed")
    trajectory = _read_json(trajectory_path)
    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = output_root / "derive" / "implementations" / f"{candidate_id}.json"
    if (
        file_sha256(card_path) != receipt.get("derivation_card_sha256")
        or file_sha256(implementation_path) != receipt.get("implementation_sha256")
    ):
        raise RuntimeError("selected derivation or implementation changed")
    card = _read_json(card_path)
    implementation = _read_json(implementation_path)
    if card.get("candidate_id") != candidate_id or implementation.get("candidate_id") != candidate_id:
        raise RuntimeError("selected source candidate identity changed")
    return receipt, receipt_path, trajectory, trajectory_path, card, card_path, implementation


def _portable_parent_ablation(portable: dict[str, Any], parent_id: str) -> dict[str, Any]:
    matches = [
        row for row in portable["extended_adjudication"].get("parent_ablation_results", [])
        if isinstance(row, dict) and row.get("parent_candidate_id") == parent_id
    ]
    if len(matches) != 1:
        raise RuntimeError(f"portable parent ablation evidence missing: {parent_id}")
    return matches[0]


def _row_by_id(frontier: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    matches = [row for row in frontier["ranking"] if row.get("candidate_id") == candidate_id]
    if len(matches) != 1:
        raise RuntimeError(f"complete 4090 component receipt missing: {candidate_id}")
    return matches[0]


def _mechanism_evidence(
    output_root: Path, selected_id: str, frontier: dict[str, Any], portable: dict[str, Any],
) -> dict[str, Any]:
    if selected_id == PCRSMG_PROPOSAL:
        path = output_root / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
        value = _read_json(path)
        _posthoc_boundary(value, label="PC-RSMG winner ablation")
        roles = value.get("roles")
        if not isinstance(roles, dict) or set(roles) != {
            "proposal_only", "observable_only", "projected_or_full",
        }:
            raise RuntimeError("PC-RSMG winner ablation is incomplete")
        return {
            "kind": "same_host_three_role_ablation",
            "evidence_host": "remote4090",
            "same_host_as_selection": True,
            "path": path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(path),
            "roles": roles,
        }
    if selected_id in (RFAMMCRB, RFMCRB):
        return {
            "kind": "source_host_three_role_ablation",
            "evidence_host": "remote5090",
            "same_host_as_selection": False,
            "used_for_4090_candidate_ranking": False,
            "roles": _portable_parent_ablation(portable, selected_id),
        }
    if selected_id in G3_BARRIER:
        barrier_id = G3_BARRIER[selected_id]
        return {
            "kind": "same_host_component_factorial_plus_source_parent_ablation",
            "evidence_host": ["remote4090", "remote5090"],
            "used_for_cross_host_delta_ranking": False,
            "components": {
                "plain": {"role": "zero_component_matched_baseline"},
                "conditional_sampling_only": _row_by_id(frontier, PCRSMG_PROPOSAL),
                "residual_feasible_barrier_only": _row_by_id(frontier, barrier_id),
                "combined_full": _row_by_id(frontier, selected_id),
            },
            "barrier_three_role_source_ablation": _portable_parent_ablation(
                portable, barrier_id,
            ),
        }
    raise RuntimeError(f"selected candidate lacks a frozen mechanism-evidence route: {selected_id}")


def _archive_pre_complete(final: Path, operations: Path) -> dict[str, str]:
    archive = final / "pre_complete_frontier_delivery"
    manifest_path = archive / "ARCHIVE_MANIFEST.json"
    if manifest_path.is_file():
        manifest = _read_json(manifest_path)
        for name, expected in manifest.get("file_sha256", {}).items():
            if file_sha256(archive / name) != expected:
                raise RuntimeError(f"pre-complete frontier archive changed: {name}")
        return dict(manifest["file_sha256"])
    archive.mkdir(parents=True, exist_ok=True)
    sources = {name: final / name for name in LEGACY_FINAL_FILES}
    old_pointer = operations / "ROUTE1_FINAL_DELIVERY_POINTER.json"
    if old_pointer.is_file():
        sources["ROUTE1_FINAL_DELIVERY_POINTER.json"] = old_pointer
    if any(not path.is_file() for path in sources.values()):
        raise RuntimeError("pre-complete frontier delivery is incomplete")
    hashes = {}
    for name, source in sources.items():
        destination = archive / name
        shutil.copyfile(source, destination)
        hashes[name] = file_sha256(destination)
    write_json(manifest_path, {
        "schema": "final-unsb-route1-pre-complete-frontier-archive-v1",
        "status": "IMMUTABLE_PRE_COMPLETE_FRONTIER_DELIVERY_ARCHIVED",
        "file_sha256": hashes,
        "confirmation20_opened": False,
    })
    return hashes


def _historical_evidence(output_root: Path) -> dict[str, Any]:
    """Bind the probe, causal and hypothesis evidence behind the frontier."""

    paths = {
        "anchor_trajectories": output_root / "evidence" / "ANCHOR_TRAJECTORIES.json",
        "proxy_calibration": output_root / "evidence" / "PROXY_CALIBRATION.json",
        "long_causal_matrix": output_root / "audit" / "LONG_CAUSAL_MATRIX.json",
        "long_reversal_atlas": output_root / "audit" / "LONG_REVERSAL_ATLAS.jsonl",
        "sampling_variance_atlas": (
            output_root / "audit" / "SAMPLING_VARIANCE_ATLAS.jsonl"
        ),
        "hypothesis_ledger": output_root / "derive" / "HYPOTHESIS_LEDGER.json",
    }
    if any(not path.is_file() or not path.resolve().is_relative_to(output_root)
           for path in paths.values()):
        raise RuntimeError("complete frontier historical evidence is incomplete")
    anchors = _read_json(paths["anchor_trajectories"])
    proxy = _read_json(paths["proxy_calibration"])
    matrix = _read_json(paths["long_causal_matrix"])
    ledger = _read_json(paths["hypothesis_ledger"])
    summaries = anchors.get("summaries")
    if (
        anchors.get("schema") != "local-route1-anchor-summary-v1"
        or anchors.get("time_unit") != "data_epoch"
        or anchors.get("confirmation20_opened") is not False
        or not isinstance(summaries, list)
        or {str(row.get("probe_id", "")) for row in summaries} != HISTORICAL_PROBES
        or any(row.get("complete_e200") is not True for row in summaries)
    ):
        raise RuntimeError("DT/HJ/HNEK complete-e200 anchor evidence changed")
    if (
        proxy.get("schema") != "local-route1-proxy-calibration-v1"
        or proxy.get("status") != "CALIBRATED"
        or not set(proxy.get("passing_probes", [])).intersection({"hj", "hnek"})
        or proxy.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("complete frontier proxy is not calibrated")
    expected = int(matrix.get("expected_rows", -1))
    expected_variance = int(matrix.get("expected_sampling_variance_rows", -1))
    atlas_lines = sum(
        1 for line in paths["long_reversal_atlas"].read_text(
            encoding="utf-8",
        ).splitlines() if line.strip()
    )
    variance_atlas_lines = sum(
        1 for line in paths["sampling_variance_atlas"].read_text(
            encoding="utf-8",
        ).splitlines() if line.strip()
    )
    if (
        matrix.get("schema") != "final-unsb-local-route1-causal-matrix-v1"
        or matrix.get("status") != "COMPLETE_CAUSAL_AUDIT"
        or matrix.get("missing_rows") != []
        or matrix.get("missing_sampling_variance_rows") != []
        or matrix.get("rows") != expected
        or matrix.get("sampling_variance_rows") != expected_variance
        or atlas_lines != expected
        or variance_atlas_lines != expected_variance
        or matrix.get("paired_labels_joined_only_after_branches") is not True
        or matrix.get("paired_metrics_accessed_by_controller") is not False
        or matrix.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("complete frontier causal matrix/atlas is incomplete")
    records = ledger.get("records")
    if (
        ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1"
        or ledger.get("paired_controller_access") is not False
        or ledger.get("confirmation20_opened") is not False
        or not isinstance(records, list)
        or not records
    ):
        raise RuntimeError("complete frontier hypothesis ledger is incomplete")
    record_summary = [{
        key: row.get(key) for key in (
            "candidate_id", "generation", "parent_candidate_id",
            "construction_route", "status", "revision_count",
            "engineering_replacement",
        ) if key in row
    } for row in records]
    artifact_refs = {
        key: {
            "path": path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(path),
        }
        for key, path in paths.items()
    }
    return {
        "status": "COMPLETE_LONG_HORIZON_PROBE_CAUSAL_AND_DERIVATION_EVIDENCE",
        "dt_hj_hnek_anchor_trajectories": anchors,
        "proxy_calibration": proxy,
        "long_causal_matrix_summary": {
            "schema": matrix["schema"],
            "status": matrix["status"],
            "analysis_identity": matrix.get("analysis_identity"),
            "reversal_rows": expected,
            "sampling_variance_rows": expected_variance,
            "probe_summaries": matrix.get("probe_summaries"),
            "ranked_failure_mechanisms": matrix.get("ranked_failure_mechanisms"),
            "target_blind_signal_screen": matrix.get("target_blind_signal_screen"),
            "paired_labels_joined_only_after_branches": True,
            "paired_metrics_accessed_by_controller": False,
        },
        "hypothesis_ledger_summary": record_summary,
        "artifact_refs": artifact_refs,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _research_frontier(
    frontier: dict[str, Any], portable: dict[str, Any], selected_id: str,
) -> dict[str, Any]:
    """Preserve every complete mechanism-bearing branch beyond action ranking."""

    preserved_4090 = set(frontier.get("evidence_preserved_candidate_ids", []))
    rows_4090 = []
    for row in frontier["ranking"]:
        candidate_id = str(row["candidate_id"])
        if candidate_id == selected_id:
            disposition = "action_priority"
        elif row.get("classification") == STRICT:
            disposition = "co_leading_strict_frontier"
        elif candidate_id in preserved_4090:
            disposition = "mechanism_bearing_frontier_reserve"
        else:
            disposition = "closed_current_implementation_on_current_protocol"
        rows_4090.append({**row, "frontier_disposition": disposition})

    source_5090 = portable["extended_adjudication"]
    preserved_5090 = set(source_5090.get("evidence_preserved_candidate_ids", []))
    rows_5090 = []
    for row in source_5090["ranking"]:
        candidate_id = str(row["candidate_id"])
        rows_5090.append({
            **row,
            "frontier_disposition": (
                "source_host_mechanism_frontier"
                if candidate_id in preserved_5090
                else "closed_current_implementation_on_source_protocol"
            ),
            "eligible_for_cross_host_delta_ranking": False,
        })

    observable_ids = source_5090.get(
        "observable_only_candidate_ids_excluded_from_ranking", [],
    )
    return {
        "schema": RESEARCH_FRONTIER_SCHEMA,
        "status": "COMPLETE_MULTI_CANDIDATE_ROUTE1_RESEARCH_FRONTIER",
        "action_priority_candidate_id": selected_id,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "frontier_policy": (
            "Retain and advance all complete candidates with independent mechanistic "
            "evidence; use rank one only as the next-action default. Small single-seed "
            "differences do not constitute mechanism falsification."
        ),
        "remote4090_same_host_frontier": rows_4090,
        "remote4090_advanceable_candidate_ids": [
            row["candidate_id"] for row in rows_4090
            if row["frontier_disposition"] != (
                "closed_current_implementation_on_current_protocol"
            )
        ],
        "remote5090_source_host_frontier": rows_5090,
        "remote5090_mechanism_bearing_candidate_ids": [
            row["candidate_id"] for row in rows_5090
            if row["frontier_disposition"] == "source_host_mechanism_frontier"
        ],
        "remote5090_observable_negative_control_ids": observable_ids,
        "remote5090_parent_ablation_results": source_5090.get(
            "parent_ablation_results", [],
        ),
        "next_stage_policy": {
            "action_priority": "default next-scale validation entry",
            "co_leading_or_mechanism_bearing": (
                "retain as independent algorithm directions; advance when compute is "
                "available or when the primary's scale behavior falsifies its mechanism"
            ),
            "negative_control": "retain for causal interpretation, never rank as a method",
            "closed_current_implementation": (
                "does not falsify the parent idea without stronger causal evidence"
            ),
        },
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def _report(candidate: dict[str, Any], alternates: dict[str, Any], results: dict[str, Any]) -> str:
    fields = candidate["seed2026_e200_result"]
    lines = [
        "# FINAL UNSB 路线一完整候选前沿",
        "",
        f"- 行动主项：`{candidate['candidate_id']}`（{candidate['name']}）",
        f"- 长程分类：`{candidate['classification']}`",
        f"- late-three宏PSNR delta：`{fields.get('late_three_mean_macro_psnr_delta')}`",
        f"- e200宏PSNR delta：`{fields.get('e200_macro_psnr_delta')}`",
        "- 裁决单位：small25、batch1、seed2026、共同e0、真实200 data epochs。",
        "- 唯一主项只是下一步行动优先级，不删除其他完整算法证据。",
        "- `RESEARCH_FRONTIER.json`保存全部值得继续思考、修订或扩尺度验证的分支。",
        "",
        "## 两个递补",
        "",
    ]
    for row in alternates["alternates"]:
        lines.append(
            f"- `{row['candidate_id']}`：`{row['classification']}`；{row['reason_not_selected']}"
        )
    lines.extend([
        "",
        "## 证据边界",
        "",
        "- 4090完整前沿只在该宿主内matched排名。",
        "- 5090完整full/proposal/observable前沿作为宿主分离的机理证据，不合并delta。",
        "- paired指标只在完整e200后排名；未用于公式、训练控制、退出或checkpoint选择。",
        "- DT/HJ/HNEK长期锚点、474/140因果图谱与完整假设谱系均写入主结果，而非只留在旧归档。",
        "- seed2027/2028、一万张全量数据、confirmation20和论文级外推尚未验证。",
        "",
        "## 完整轨迹与复现",
        "",
        "逐域candidate/plain绝对值和delta、全部候选排名、推导、复杂度、风险与源码指纹见 "
        "`CANDIDATE.json`和`RESULTS.json`。",
        "",
    ])
    return "\n".join(lines)


def materialize_complete_frontier_final_delivery(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    final = output_root / "final"
    pointer_path = operations / POINTER
    if pointer_path.is_file():
        pointer = _read_json(pointer_path)
        if pointer.get("schema") != POINTER_SCHEMA:
            raise RuntimeError("complete frontier final pointer schema changed")
        fixed = {
            "canonical_candidate_is_action_priority_only": True,
            "algorithm_discovery_collapsed_to_single_candidate": False,
            "cross_host_deltas_merged": False,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "cross_seed_stability_claimed": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        for key, expected in fixed.items():
            if pointer.get(key) != expected:
                raise RuntimeError(f"complete frontier final pointer changed: {key}")
        for name, expected in pointer.get("final_file_sha256", {}).items():
            if file_sha256(final / name) != expected:
                raise RuntimeError(f"complete frontier final file changed: {name}")
        input_paths = {
            "complete_4090_frontier_sha256": (
                operations / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
            ),
            "portable_5090_frontier_sha256": (
                operations / "PORTABLE_EXTENDED_REPAIRED_FRONTIER_5090.json"
            ),
        }
        for key, path in input_paths.items():
            if file_sha256(path) != pointer.get(key):
                raise RuntimeError(f"complete frontier final input changed: {path.name}")
        return pointer

    frontier, frontier_path = _complete_frontier(output_root)
    portable, portable_path = _portable_frontier(output_root)
    selected_row = frontier["ranking"][0]
    selected_id = str(selected_row["candidate_id"])
    (
        receipt, receipt_path, trajectory, trajectory_path, card, card_path,
        implementation,
    ) = _selected_source(output_root, selected_row)
    implementation_path = output_root / "derive" / "implementations" / f"{selected_id}.json"
    domain_trajectory = _candidate_domain_trajectory(output_root, selected_id)
    mechanism_evidence = _mechanism_evidence(
        output_root, selected_id, frontier, portable,
    )
    executor_path, executor = _executor_contract(output_root, receipt)
    historical_evidence = _historical_evidence(output_root)
    archived = _archive_pre_complete(final, operations)

    alternate_rows = []
    for row in frontier["ranking"][1:3]:
        alternate_rows.append({
            "candidate_id": row["candidate_id"],
            "classification": row["classification"],
            "trajectory_status": row["trajectory_status"],
            "source_role": row["source_role"],
            "ranking_fields": row["ranking_fields"],
            "reason_not_selected": "lower complete same-host 4090 e200 rank than the action priority",
        })
    if len(alternate_rows) != 2:
        raise RuntimeError("complete frontier final delivery requires two alternates")
    alternates = {
        "schema": ALTERNATES_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected_id,
        "alternates": [
            {"rank": rank, **row} for rank, row in enumerate(alternate_rows, start=2)
        ],
        "complete_4090_frontier_ranking": frontier["ranking"],
        "complete_5090_frontier_ranking": portable["extended_adjudication"]["ranking"],
        "old_probe_reserved_slot": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    }
    research_frontier = _research_frontier(frontier, portable, selected_id)
    research_frontier["historical_probe_causal_and_derivation_evidence"] = (
        historical_evidence
    )
    conclusion_boundaries = {
        "scientific_conclusion": (
            "The selected operator is the current action priority under the complete "
            "same-host 4090 seed2026 small25 e200 frontier."
        ),
        "engineering_failures": (
            "Implementation-invalid or incomplete trajectories are retained as diagnostics "
            "and excluded from scientific ranking."
        ),
        "proxy_distortion": (
            "Local 1660 proxy non-calibration does not overwrite the calibrated 4090 frontier."
        ),
        "untested_hypotheses": [
            "seed2027/2028 stability",
            "full 10000-image 200-epoch behavior",
            "confirmation20 generalization",
            "paper-scale compute-matched external replication",
        ],
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "status": "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE",
        "candidate_id": selected_id,
        "name": card.get("name", selected_id),
        "classification": selected_row["classification"],
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "seed2026_candidate_fingerprint": receipt["candidate_fingerprint"],
        "training_git_commit": receipt["training_git_commit"],
        "candidate_training_core_fingerprint": receipt[
            "candidate_training_core_fingerprint"
        ],
        "selected_fixed_checkpoint": {
            "data_epoch": 200, "updates": 30000, "best_checkpoint_selection": False,
        },
        "source_bound_terminal_receipt": {
            "path": receipt_path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(receipt_path),
        },
        "source_derivation_card": {
            "path": card_path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(card_path),
        },
        "source_implementation": {
            "path": implementation_path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(implementation_path),
        },
        "mathematics": {
            "unsb_object": card.get("unsb_object"),
            "formula": card.get("formula"),
            "identity_or_unbiased_condition": card.get("identity_or_unbiased_condition"),
            "objective_change": card.get("objective_change"),
            "estimator_change": card.get("estimator_change"),
            "coordinate_change": card.get("coordinate_change"),
            "endpoint_law_change": card.get("endpoint_law_change"),
            "target_inaccessibility_proof": card.get("target_inaccessibility_proof"),
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
        },
        "conclusion_boundaries": conclusion_boundaries,
        "seed2026_e200_result": {
            "trajectory_status": receipt["trajectory_status"],
            **receipt["ranking_fields"],
        },
        "trajectory": trajectory,
        "absolute_relative_domain_trajectory": domain_trajectory,
        "mechanism_evidence": mechanism_evidence,
        "median_epoch_wall_seconds": _median_epoch_seconds(
            output_root / "candidates" / selected_id,
        ),
        "reproduction": {
            "seed2026_e200": (
                "PYTHONPATH=<REPO> python -m operations.local_route1_candidate_executor "
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
        },
        "target_data_epochs": 200,
        "target_updates": 30000,
        "training_batch_size": 1,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected_id,
        "classification": selected_row["classification"],
        "complete_4090_frontier": frontier,
        "complete_5090_extended_frontier": portable["extended_adjudication"],
        "portable_5090_candidate_evidence": portable["candidate_evidence"],
        "selected_trajectory": trajectory,
        "selected_absolute_relative_domain_trajectory": domain_trajectory,
        "selected_mechanism_evidence": mechanism_evidence,
        "historical_probe_causal_and_derivation_evidence": historical_evidence,
        "conclusion_boundaries": conclusion_boundaries,
        "pre_complete_frontier_delivery": {"archived_file_sha256": archived},
        "host_separated_complete_frontier": {
            "remote4090": frontier,
            "remote5090": portable["extended_adjudication"],
            "cross_host_deltas_merged": False,
        },
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_e200_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    }

    staging = final / "complete_frontier_staging"
    staging.mkdir(parents=True, exist_ok=True)
    write_json(staging / "CANDIDATE.json", candidate)
    write_json(staging / "RESULTS.json", results)
    write_json(staging / "ALTERNATES.json", alternates)
    write_json(staging / "RESEARCH_FRONTIER.json", research_frontier)
    (staging / "FINAL_ROUTE1_REPORT.md").write_text(
        _report(candidate, alternates, results), encoding="utf-8",
    )
    staged_hashes = {name: file_sha256(staging / name) for name in PUBLISHED_FILES}
    for name in PUBLISHED_FILES:
        os.replace(staging / name, final / name)
    if any(file_sha256(final / name) != expected for name, expected in staged_hashes.items()):
        raise RuntimeError("complete frontier final files changed during atomic publication")
    pointer = {
        "schema": POINTER_SCHEMA,
        "status": "COMPLETE_FRONTIER_FINAL_DELIVERY_COMPLETE",
        "selected_candidate_id": selected_id,
        "final_file_sha256": staged_hashes,
        "complete_4090_frontier_sha256": file_sha256(frontier_path),
        "portable_5090_frontier_sha256": file_sha256(portable_path),
        "pre_complete_frontier_files_archived": True,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "research_frontier_unique_candidate_count": len({
            row["candidate_id"]
            for key in (
                "remote4090_same_host_frontier",
                "remote5090_source_host_frontier",
            )
            for row in research_frontier[key]
        }),
        "research_frontier_host_scoped_row_count": (
            len(research_frontier["remote4090_same_host_frontier"])
            + len(research_frontier["remote5090_source_host_frontier"])
        ),
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(pointer_path, pointer)
    return pointer
