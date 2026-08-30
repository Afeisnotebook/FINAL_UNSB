"""Materialize the post-frontier route-1 delivery without cross-host ranking."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import (
    _rank_key,
    _validate_receipt,
    adjudicate,
)
from operations.local_route1_frontier_cross_host_successor import (
    NO_REPLAY,
    REPLAY_READY,
    validate_remote_decision,
)
from research.local_route1.generation1_adjudication import POSITIVE_STATUS
from research.local_route1.final_delivery import (
    _candidate_domain_trajectory,
    _median_epoch_seconds,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


CANDIDATE_SCHEMA = "final-unsb-route1-frontier-final-candidate-v1"
RESULTS_SCHEMA = "final-unsb-route1-frontier-final-results-v1"
ALTERNATES_SCHEMA = "final-unsb-route1-frontier-final-alternates-v1"
POINTER_SCHEMA = "final-unsb-route1-frontier-final-delivery-pointer-v1"
FINAL_SELECTION = "ROUTE1_FRONTIER_FINAL_SELECTION.json"
POINTER = "ROUTE1_FINAL_DELIVERY_POINTER.json"
WINNER_ABLATION_RESULT = "FRONTIER_WINNER_ABLATION_RESULT.json"
BASE_FILES = (
    "CANDIDATE.json",
    "RESULTS.json",
    "ALTERNATES.json",
    "FINAL_ROUTE1_REPORT.md",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _posthoc_closed(value: dict[str, Any], *, label: str) -> None:
    if value.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not prove confirmation20 remained closed")
    for key in (
        "paired_metrics_used_for_training_or_control", "paired_controller_access",
    ):
        if key in value and value[key] is not False:
            raise RuntimeError(f"{label} violates target-blind training: {key}")


def _existing_delivery(output_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    final = output_root / "final"
    paths = [final / name for name in BASE_FILES]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("pre-frontier final delivery is incomplete")
    candidate = _read_json(final / "CANDIDATE.json")
    results = _read_json(final / "RESULTS.json")
    alternates = _read_json(final / "ALTERNATES.json")
    for label, value in (
        ("pre-frontier candidate", candidate),
        ("pre-frontier results", results),
        ("pre-frontier alternates", alternates),
    ):
        _posthoc_closed(value, label=label)
    candidate_id = str(candidate.get("candidate_id", ""))
    if not candidate_id or results.get("selected_candidate_id") != candidate_id:
        raise RuntimeError("pre-frontier selected candidate identity is inconsistent")
    return candidate, results, alternates


def _archive_base(output_root: Path) -> dict[str, str]:
    final = output_root / "final"
    archive = final / "pre_frontier_delivery"
    archive.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for name in BASE_FILES:
        source = final / name
        destination = archive / name
        if destination.is_file():
            if destination.read_bytes() != source.read_bytes():
                raise RuntimeError(f"pre-frontier archive changed: {destination}")
        else:
            shutil.copyfile(source, destination)
        hashes[name] = file_sha256(destination)
    return hashes


def _same_host_selection(
    output_root: Path, base_candidate_id: str, cross_result: dict[str, Any],
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    operations = output_root / "operations"
    base_receipt_path = operations / "terminal_receipts" / f"{base_candidate_id}.json"
    base_receipt = _validate_receipt(base_receipt_path)
    status = str(cross_result.get("status", ""))
    selection_path = operations / FINAL_SELECTION
    if status == "COMPLETE_REMOTE_FRONTIER_NEGATIVE_4090_REPLAY_SKIPPED":
        result = {
            "schema": "final-unsb-route1-frontier-final-selection-v1",
            "status": "PRE_FRONTIER_4090_PRIMARY_RETAINED_REMOTE_FRONTIER_NEGATIVE",
            "selected_candidate_id": base_candidate_id,
            "selected_algorithm_fingerprint": base_receipt["algorithm_fingerprint"],
            "selected_candidate_fingerprint": base_receipt["candidate_fingerprint"],
            "selection_role": "same_host_4090_primary_retained",
            "same_host_4090_receipts_compared": [base_candidate_id],
            "remote_5090_used_for_replay_routing_only": True,
            "cross_host_deltas_merged": False,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "cross_seed_stability_claimed": False,
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        write_json(selection_path, result)
        return result, base_receipt_path, base_receipt
    if status != "COMPLETE_ONE_FRONTIER_4090_REPLAY_ADJUDICATION_REQUIRED":
        raise RuntimeError("frontier cross-host result is not terminal")
    replay_path = Path(str(cross_result.get("receipt_path", ""))).resolve()
    if (
        not replay_path.is_file()
        or file_sha256(replay_path) != cross_result.get("receipt_sha256")
    ):
        raise RuntimeError("frontier 4090 replay receipt changed")
    replay_receipt = _validate_receipt(replay_path)
    ranked = adjudicate([base_receipt_path, replay_path], selection_path)
    ranked.update({
        "selection_scope": "post_frontier_same_host_4090_seed2026_e200",
        "remote_5090_used_for_replay_routing_only": True,
        "cross_host_deltas_merged": False,
        "additional_seed_replication_deferred": [2027, 2028],
        "cross_seed_stability_claimed": False,
    })
    write_json(selection_path, ranked)
    selected_id = str(ranked["selected_candidate_id"])
    if selected_id == base_candidate_id:
        return ranked, base_receipt_path, base_receipt
    if selected_id != replay_receipt["candidate_id"]:
        raise RuntimeError("same-host frontier selection returned an unknown candidate")
    return ranked, replay_path, replay_receipt


def _source_files(
    output_root: Path, candidate_id: str, receipt: dict[str, Any],
) -> tuple[Path, Path, dict, dict]:
    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    if not card_path.is_file() or not implementation_path.is_file():
        raise RuntimeError(f"selected candidate source records are missing: {candidate_id}")
    card = _read_json(card_path)
    implementation = _read_json(implementation_path)
    if card.get("candidate_id") != candidate_id or implementation.get(
        "candidate_id"
    ) != candidate_id:
        raise RuntimeError("selected candidate source identity mismatch")
    if receipt.get("derivation_card_sha256") != file_sha256(card_path):
        raise RuntimeError("selected derivation card changed after terminal receipt")
    if receipt.get("implementation_sha256") != file_sha256(implementation_path):
        raise RuntimeError("selected implementation changed after terminal receipt")
    return card_path, implementation_path, card, implementation


def _winner_specific_selection(
    output_root: Path,
) -> tuple[dict[str, Any], Path, dict[str, Any], Path, dict[str, Any], Path, dict[str, Any]]:
    operations = output_root / "operations"
    result_path = operations / WINNER_ABLATION_RESULT
    if not result_path.is_file():
        raise RuntimeError("frontier selected algorithm ablation result is missing")
    result = _read_json(result_path)
    _posthoc_closed(result, label="frontier winner ablation result")
    if result.get("status") not in (
        "REUSED_PRE_FRONTIER_SELECTED_WINNER_ABLATIONS",
        "FRONTIER_SELECTED_ALGORITHM_ABLATIONS_COMPLETE",
    ):
        raise RuntimeError("frontier selected algorithm ablations are not complete")

    def bound_path(key: str, sha_key: str) -> Path:
        path = Path(str(result.get(key, ""))).resolve()
        if not path.is_file() or not path.is_relative_to(output_root):
            raise RuntimeError(f"frontier winner evidence escaped run root: {key}")
        if file_sha256(path) != result.get(sha_key):
            raise RuntimeError(f"frontier winner evidence changed: {key}")
        return path

    selection_path = bound_path(
        "post_ablation_selection_path", "post_ablation_selection_sha256",
    )
    selection = _read_json(selection_path)
    _posthoc_closed(selection, label="frontier post-ablation selection")
    receipt_path = bound_path("selected_receipt_path", "selected_receipt_sha256")
    receipt = _validate_receipt(receipt_path)
    ablation_path = bound_path(
        "winner_ablation_adjudication_path", "winner_ablation_adjudication_sha256",
    )
    ablation = _read_json(ablation_path)
    _posthoc_closed(ablation, label="selected algorithm winner ablation adjudication")
    selected_id = str(result.get("selected_candidate_id", ""))
    if (
        not selected_id
        or selection.get("selected_candidate_id") != selected_id
        or receipt.get("candidate_id") != selected_id
        or result.get("selected_algorithm_fingerprint") not in (
            None, receipt.get("algorithm_fingerprint"),
        )
    ):
        raise RuntimeError("frontier post-ablation selection identity mismatch")
    evidence = result.get("winner_ablation_evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("frontier winner ablation result lacks role evidence")
    roles = ablation.get("roles")
    if not isinstance(roles, dict) or set(roles) != {
        "proposal_only", "observable_only", "projected_or_full",
    }:
        raise RuntimeError("selected algorithm ablation lacks the three required roles")
    role_ids = {str(row.get("candidate_id", "")) for row in roles.values()}
    if selected_id not in role_ids:
        raise RuntimeError("selected algorithm is not represented by its ablation roles")
    return result, result_path, selection, selection_path, receipt, receipt_path, ablation


def _executor_contract(output_root: Path, receipt: dict[str, Any]) -> tuple[Path, dict]:
    candidate_id = str(receipt["candidate_id"])
    matches = []
    for path in (output_root / "operations").glob("CANDIDATE_EXECUTOR_CONTRACT_*.json"):
        try:
            value = _read_json(path)
        except Exception:
            continue
        if (
            value.get("candidate_id") == candidate_id
            and value.get("candidate_git_commit") == receipt.get("training_git_commit")
            and value.get("algorithm_fingerprint") == receipt.get("algorithm_fingerprint")
            and value.get("candidate_fingerprint") == receipt.get("candidate_fingerprint")
        ):
            required = {
                "schema": "final-unsb-route1-candidate-executor-contract-v1",
                "manifest_sha256": receipt.get("manifest_sha256"),
                "target_data_epochs": 200,
                "paired_metric_early_stop": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            }
            mismatches = {
                key: {"expected": expected, "actual": value.get(key)}
                for key, expected in required.items()
                if value.get(key) != expected
            }
            if mismatches:
                raise RuntimeError(
                    f"selected candidate executor contract mismatch: {path}: {mismatches}"
                )
            matches.append((path, value))
    if len(matches) != 1:
        raise RuntimeError(
            f"selected candidate requires one exact executor contract; found {len(matches)}"
        )
    return matches[0]


def _alternate_rows(
    *, selected_id: str, same_host_selection: dict[str, Any],
    base_alternates: dict[str, Any], remote_adjudication: dict[str, Any],
) -> list[dict[str, Any]]:
    values = []
    seen = {selected_id}
    for row in same_host_selection.get("ranking", []):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id and candidate_id not in seen:
            values.append({
                "candidate_id": candidate_id,
                "role": "same_host_4090_direct_competitor",
                "rank_scope": "authoritative_same_host_4090",
                "trajectory_status": row.get("trajectory_status"),
                "reason_not_selected": "lower complete same-host seed2026 e200 rank",
            })
            seen.add(candidate_id)
    for row in base_alternates.get("alternates", []):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id and candidate_id not in seen:
            values.append({
                "candidate_id": candidate_id,
                "role": row.get("role", "pre_frontier_tested_alternate"),
                "rank_scope": "pre_frontier_authoritative_4090",
                "trajectory_status": row.get("trajectory_status"),
                "reason_not_selected": row.get(
                    "reason_not_selected", "lower pre-frontier same-host rank"
                ),
            })
            seen.add(candidate_id)
    for row in remote_adjudication.get("ranking", []):
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id and candidate_id not in seen:
            values.append({
                "candidate_id": candidate_id,
                "role": "tested_5090_host_local_frontier",
                "rank_scope": "5090_host_local_only_not_cross_host_comparable",
                "trajectory_status": row.get("trajectory_status"),
                "reason_not_selected": (
                    "not promoted by a winning same-host 4090 replay; cross-host delta not merged"
                ),
            })
            seen.add(candidate_id)
    return values[:2]


def _selected_ablation_results(
    output_root: Path, ablation: dict[str, Any],
) -> dict[str, Any]:
    roles = ablation["roles"]
    values = {}
    for role in ("proposal_only", "observable_only", "projected_or_full"):
        row = roles[role]
        candidate_id = str(row.get("candidate_id", ""))
        receipt_path = Path(str(row.get("receipt_path", ""))).resolve()
        if not receipt_path.is_file() or not receipt_path.is_relative_to(output_root):
            raise RuntimeError(f"selected algorithm ablation receipt escaped run root: {role}")
        if file_sha256(receipt_path) != row.get("receipt_sha256"):
            raise RuntimeError(f"selected algorithm ablation receipt changed: {role}")
        receipt = _validate_receipt(receipt_path)
        if (
            receipt.get("candidate_id") != candidate_id
            or receipt.get("algorithm_fingerprint") != row.get(
                "algorithm_fingerprint"
            )
        ):
            raise RuntimeError(f"selected algorithm ablation identity changed: {role}")
        trajectory_path = (
            output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        )
        if file_sha256(trajectory_path) != receipt.get("trajectory_sha256"):
            raise RuntimeError(f"selected algorithm ablation trajectory changed: {role}")
        values[role] = {
            "candidate_id": candidate_id,
            "trajectory_status": receipt["trajectory_status"],
            "ranking_fields": receipt["ranking_fields"],
            "receipt_path": receipt_path.relative_to(output_root).as_posix(),
            "receipt_sha256": file_sha256(receipt_path),
            "trajectory": _read_json(trajectory_path),
            "absolute_relative_domain_trajectory": _candidate_domain_trajectory(
                output_root, candidate_id,
            ),
        }
    return values


def _report(
    candidate: dict[str, Any], alternates: dict[str, Any], results: dict[str, Any],
) -> str:
    metrics = candidate["seed2026_e200_result"]
    boundaries = candidate["conclusion_boundaries"]
    lines = [
        "# FINAL UNSB 路线一候选（frontier-complete）",
        "",
        f"- 主候选：`{candidate['candidate_id']}`（{candidate['name']}）",
        f"- 分类：`{candidate['classification']}`",
        f"- late-three宏PSNR delta：`{metrics.get('late_three_mean_macro_psnr_delta')}`",
        f"- e200宏PSNR delta：`{metrics.get('e200_macro_psnr_delta')}`",
        "- 科学单位：small25、batch1、seed2026、真实200 data epochs（30000 updates）。",
        "- seed2027/2028延期；不得声称已证明跨seed稳定。",
        "- confirmation20、全量数据、路线二、退出窗口与paired控制均未启用。",
        "- 5090结果只在5090宿主内排序；只有完成4090复跑的候选才进入主候选竞争。",
        "",
        "## 科学结论",
        "",
        f"- 当前证据支持：{boundaries['scientific_conclusion']['supported']}",
        f"- 当前证据不支持：{boundaries['scientific_conclusion']['not_supported']}",
        "- paired指标只在完整e200轨迹冻结后用于排序，未进入训练、控制或退出。",
        "",
        "## 工程失败与科学结果的边界",
        "",
        f"- {boundaries['engineering_failures']['rule']}",
        "- 历史RSMG player-state语义错误属于engineering-invalid，不进入算法优劣排名。",
        "",
        "## 代理失真边界",
        "",
        f"- {boundaries['proxy_distortion']['rule']}",
        "- 本地1660轨迹只作补充；canonical排序使用4090同宿主matched证据。",
        "",
        "## 尚未验证",
        "",
    ]
    for item in boundaries["untested_hypotheses"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## 两个递补",
        "",
    ])
    for row in alternates["alternates"]:
        lines.append(
            f"- `{row['candidate_id']}`：{row['role']}；{row['reason_not_selected']}"
        )
    lines.extend([
        "",
        "## 宿主分离的完整候选前沿",
        "",
    ])
    host_frontier = results["host_separated_complete_frontier"]
    for host_key, title in (
        ("remote4090_pre_frontier", "4090已完成候选"),
        ("remote5090_frontier", "5090前沿"),
    ):
        lines.append(f"### {title}")
        lines.append("")
        ranking = host_frontier[host_key].get("ranking", [])
        if not ranking:
            lines.append("- 无可列出的完整排名行。")
        for row in ranking:
            fields = row.get("ranking_fields", {})
            lines.append(
                f"- `{row.get('candidate_id')}`：`{row.get('trajectory_status')}`；"
                f"late-three `{fields.get('late_three_mean_macro_psnr_delta')}`；"
                f"e200 `{fields.get('e200_macro_psnr_delta')}`。"
            )
        lines.append("")
    lines.extend([
        "逐域candidate/plain绝对值与delta、完整公式、复杂度、风险、源码指纹和可执行复现"
        "合同见同目录CANDIDATE.json与RESULTS.json。",
        "",
    ])
    return "\n".join(lines)


def materialize_frontier_final_delivery(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    operations = output_root / "operations"
    final = output_root / "final"
    pointer_path = operations / POINTER
    if pointer_path.is_file():
        pointer = _read_json(pointer_path)
        if pointer.get("schema") != POINTER_SCHEMA:
            raise RuntimeError("frontier final delivery pointer schema mismatch")
        for name, expected in pointer.get("final_file_sha256", {}).items():
            if file_sha256(final / name) != expected:
                raise RuntimeError(f"frontier final delivery changed after freeze: {name}")
        return pointer

    base_candidate, base_results, base_alternates = _existing_delivery(output_root)
    cross_result_path = operations / "FRONTIER_CROSS_HOST_RESULT.json"
    envelope_path = operations / "FRONTIER_5090_TERMINAL_ENVELOPE.json"
    if not cross_result_path.is_file() or not envelope_path.is_file():
        raise RuntimeError("frontier terminal/cross-host evidence is incomplete")
    cross_result = _read_json(cross_result_path)
    envelope = _read_json(envelope_path)
    _posthoc_closed(cross_result, label="frontier cross-host result")
    _posthoc_closed(envelope, label="remote frontier envelope")
    decision = envelope.get("decision", {})
    remote_adjudication = envelope.get("adjudication", {})
    if not isinstance(decision, dict) or not isinstance(remote_adjudication, dict):
        raise RuntimeError("remote frontier envelope is malformed")
    validate_remote_decision(decision, remote_adjudication)
    if decision.get("status") == NO_REPLAY and cross_result.get("status") != (
        "COMPLETE_REMOTE_FRONTIER_NEGATIVE_4090_REPLAY_SKIPPED"
    ):
        raise RuntimeError("remote no-replay decision/cross-host result mismatch")
    if decision.get("status") == REPLAY_READY and cross_result.get("status") != (
        "COMPLETE_ONE_FRONTIER_4090_REPLAY_ADJUDICATION_REQUIRED"
    ):
        raise RuntimeError("remote replay decision/cross-host result mismatch")

    (
        winner_ablation_result, winner_ablation_result_path,
        selection, selection_path, receipt, receipt_path, winner_ablation,
    ) = _winner_specific_selection(output_root)
    selected_id = str(receipt["candidate_id"])
    card_path, implementation_path, card, implementation = _source_files(
        output_root, selected_id, receipt,
    )
    trajectory_path = output_root / "candidates" / selected_id / "CANDIDATE_TRAJECTORY.json"
    trajectory = _read_json(trajectory_path)
    if file_sha256(trajectory_path) != receipt.get("trajectory_sha256"):
        raise RuntimeError("selected trajectory changed after terminal receipt")
    absolute_relative_domain_trajectory = _candidate_domain_trajectory(
        output_root, selected_id,
    )
    selected_ablation_results = _selected_ablation_results(
        output_root, winner_ablation,
    )
    executor_path, executor = _executor_contract(output_root, receipt)
    archived = _archive_base(output_root)

    alternate_rows = _alternate_rows(
        selected_id=selected_id,
        same_host_selection=selection,
        base_alternates=base_alternates,
        remote_adjudication=remote_adjudication,
    )
    if len(alternate_rows) != 2:
        raise RuntimeError("frontier final delivery requires exactly two evidence-backed alternates")
    alternates = {
        "schema": ALTERNATES_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected_id,
        "alternates": [
            {"rank": rank, **row}
            for rank, row in enumerate(alternate_rows, start=2)
        ],
        "complete_5090_frontier_ranking": remote_adjudication.get("ranking", []),
        "cross_host_deltas_merged": False,
        "old_probe_reserved_slot": False,
        "confirmation20_opened": False,
    }
    classification = (
        "single_seed_strict_local_signal"
        if receipt.get("trajectory_status") == POSITIVE_STATUS else
        "weak_fallback_single_seed_development"
    )
    conclusion_boundaries = {
        "scientific_conclusion": {
            "supported": (
                "该算法在共同e0、small25、batch1、seed2026的4090同宿主matched e200"
                "协议中通过了完整长期门。"
                if receipt.get("trajectory_status") == POSITIVE_STATUS else
                "该算法是完整seed2026/e200证据下的当前最优fallback，但未通过全部长期门。"
            ),
            "not_supported": (
                "跨seed稳定性、全量一万张数据收益、confirmation20泛化或论文级最终结论。"
            ),
            "best_checkpoint_selection": False,
            "paired_training_control": False,
        },
        "engineering_failures": {
            "rule": (
                "只有source-bound完整e200 receipt进入排名；NaN、身份不一致、未完成轨迹和"
                "历史player-state语义错误均归为工程失败，不可冒充算法负结果。"
            ),
            "known_excluded_incident": "G1-02-SAMPLING-VARIANCE player-state semantic mismatch",
            "failed_or_incomplete_trajectory_ranked": False,
        },
        "proxy_distortion": {
            "rule": (
                "small25是算法发现代理；本地1660 proxy不校准不覆盖已校准的4090同宿主"
                "证据，也不允许把small25结论外推为全量数据结论。"
            ),
            "canonical_ranking_host": "remote4090",
            "cross_host_delta_merge": False,
        },
        "untested_hypotheses": [
            "seed2027/2028尚未运行，单seed结果可能不稳定。",
            "一万张全量训练视图及其真实200 data epochs尚未运行。",
            "confirmation20仍封存，未用于选择或泛化结论。",
            "4090与5090运行时的数值delta没有合并，跨硬件可迁移性仍未证明。",
            "论文级计算量匹配、完整消融图表和外部数据泛化仍属于后续工作。",
        ],
    }
    host_separated_complete_frontier = {
        "remote4090_pre_frontier": {
            "selected_candidate_id": base_results.get("selected_candidate_id"),
            "ranking": base_results.get(
                "ranking", base_results.get("generation1_ranking", []),
            ),
            "candidate_results": base_results.get("candidate_results", {}),
        },
        "remote4090_post_frontier": selection,
        "remote5090_frontier": remote_adjudication,
        "cross_host_deltas_merged": False,
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "status": "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE",
        "classification": classification,
        "candidate_id": selected_id,
        "name": card.get("name", selected_id),
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "seed2026_candidate_fingerprint": receipt["candidate_fingerprint"],
        "training_git_commit": receipt["training_git_commit"],
        "candidate_training_core_fingerprint": receipt[
            "candidate_training_core_fingerprint"
        ],
        "selected_fixed_checkpoint": {
            "data_epoch": 200,
            "updates": 30000,
            "best_checkpoint_selection": False,
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
        "risks": conclusion_boundaries["untested_hypotheses"],
        "conclusion_boundaries": conclusion_boundaries,
        "seed2026_e200_result": {
            "trajectory_status": receipt["trajectory_status"],
            **receipt["ranking_fields"],
        },
        "trajectory": trajectory,
        "absolute_relative_domain_trajectory": absolute_relative_domain_trajectory,
        "median_epoch_wall_seconds": _median_epoch_seconds(
            output_root / "candidates" / selected_id,
        ),
        "frontier_selection": {
            "path": selection_path.relative_to(output_root).as_posix(),
            "sha256": file_sha256(selection_path),
            "remote_5090_host_local_adjudication": remote_adjudication,
            "cross_host_deltas_merged": False,
        },
        "ablation_evidence": {
            "frontier_winner_ablation_result": winner_ablation_result,
            "winner_ablation_adjudication": winner_ablation,
            "selected_algorithm_roles": winner_ablation_result[
                "winner_ablation_evidence"
            ],
            "experimental_results": selected_ablation_results,
        },
        "reproduction": {
            "seed2026_e200": (
                "PYTHONPATH=<REPO> python -m operations.local_route1_candidate_executor "
                "--contract "
                f"<RUN_ROOT>/{executor_path.relative_to(output_root).as_posix()}"
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
        "training_batch_size": 1,
        "target_data_epochs": 200,
        "target_updates": 30000,
        "confirmation20_opened": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
    }
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected_id,
        "classification": classification,
        "conclusion_boundaries": conclusion_boundaries,
        "same_host_4090_final_selection": selection,
        "selected_trajectory": trajectory,
        "selected_absolute_relative_domain_trajectory": (
            absolute_relative_domain_trajectory
        ),
        "selected_algorithm_winner_ablation_result": winner_ablation_result,
        "selected_algorithm_winner_ablation_adjudication": winner_ablation,
        "selected_algorithm_winner_ablation_results": selected_ablation_results,
        "pre_frontier_delivery": {
            "selected_candidate_id": base_candidate["candidate_id"],
            "archived_file_sha256": archived,
        },
        "remote_5090_frontier": remote_adjudication,
        "host_separated_complete_frontier": host_separated_complete_frontier,
        "frontier_cross_host_result": cross_result,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_only_after_complete_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(final / "CANDIDATE.json", candidate)
    write_json(final / "RESULTS.json", results)
    write_json(final / "ALTERNATES.json", alternates)
    (final / "FINAL_ROUTE1_REPORT.md").write_text(
        _report(candidate, alternates, results), encoding="utf-8",
    )
    pointer = {
        "schema": POINTER_SCHEMA,
        "status": "FRONTIER_FINAL_DELIVERY_COMPLETE",
        "selected_candidate_id": selected_id,
        "final_file_sha256": {
            name: file_sha256(final / name) for name in BASE_FILES
        },
        "selection_sha256": file_sha256(selection_path),
        "winner_ablation_result_sha256": file_sha256(winner_ablation_result_path),
        "cross_host_result_sha256": file_sha256(cross_result_path),
        "remote_terminal_envelope_sha256": file_sha256(envelope_path),
        "pre_frontier_files_archived": True,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(pointer_path, pointer)
    return pointer
