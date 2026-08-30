"""Materialize the final route-1 delivery across frozen candidate code versions.

Candidate code is never imported here.  Source-bound e200 receipts, the frozen
cross-version ranking, completed seed adjudication, and completed long-horizon
winner ablations are the only authorities allowed to cross version boundaries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import (
    SCHEMA as CROSS_SCHEMA,
    _validate_receipt,
)
from operations.local_route1_winner_ablation_adjudicate import (
    SCHEMA as ABLATION_SCHEMA,
)
from research.local_route1.final_delivery import (
    COMPLETE_MULTI_SEED,
    _candidate_domain_trajectory,
    _classification,
    _median_epoch_seconds,
    _seed_domain_trajectory,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json
from research.local_route1.seed_validation import MULTI_SEED_ADJUDICATION_SCHEMA


SCHEMA = "final-unsb-route1-cross-version-candidate-delivery-v1"
RESULTS_SCHEMA = "final-unsb-route1-cross-version-final-results-v1"
ALTERNATES_SCHEMA = "final-unsb-route1-cross-version-final-alternates-v1"
POSITIVE_CROSS_STATUS = "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _posthoc(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not prove confirmation20 remained closed")
    for key in (
        "paired_controller_access",
        "paired_metrics_used_for_training_or_control",
        "paired_metric_changed_algorithm",
    ):
        if key in payload and payload[key] is not False:
            raise RuntimeError(f"{label} violates posthoc-only paired-metric policy: {key}")


def _receipt_path(output_root: Path, candidate_id: str) -> Path:
    return output_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"


def _load_cross_receipts(output_root: Path, cross: dict[str, Any]) -> dict[str, dict[str, Any]]:
    receipts: dict[str, dict[str, Any]] = {}
    for row in cross["ranking"]:
        candidate_id = str(row["candidate_id"])
        receipt = _validate_receipt(_receipt_path(output_root, candidate_id))
        for key in (
            "algorithm_fingerprint", "candidate_fingerprint", "training_git_commit",
            "candidate_training_core_fingerprint", "trajectory_sha256",
        ):
            if receipt.get(key) != row.get(key):
                raise RuntimeError(f"cross-version ranking/receipt mismatch for {candidate_id}: {key}")
        trajectory_path = output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        if receipt["trajectory_sha256"] != file_sha256(trajectory_path):
            raise RuntimeError(f"ranked trajectory changed after receipt: {candidate_id}")
        receipts[candidate_id] = receipt
    return receipts


def _source_bound_method(
    output_root: Path, receipt: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    candidate_id = str(receipt["candidate_id"])
    card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    if not card_path.is_file() or not implementation_path.is_file():
        raise RuntimeError("selected source-bound card or implementation is missing")
    if file_sha256(card_path) != receipt.get("derivation_card_sha256"):
        raise RuntimeError("selected derivation card changed after terminal receipt")
    if file_sha256(implementation_path) != receipt.get("implementation_sha256"):
        raise RuntimeError("selected implementation changed after terminal receipt")
    return (
        _read_json(card_path), _read_json(implementation_path), card_path,
        implementation_path,
    )


def _multi_seed(output_root: Path, winner: str, algorithm: str) -> dict[str, Any]:
    path = output_root / "candidates" / winner / "MULTI_SEED_ADJUDICATION.json"
    if not path.is_file():
        raise RuntimeError("cross-version winner lacks completed frozen seed adjudication")
    value = _read_json(path)
    _posthoc(value, label="multi-seed adjudication")
    if (
        value.get("schema") != MULTI_SEED_ADJUDICATION_SCHEMA
        or value.get("status") not in COMPLETE_MULTI_SEED
        or value.get("candidate_id") != winner
        or value.get("algorithm_fingerprint") != algorithm
        or value.get("included_seeds") not in ([2026, 2027], [2026, 2027, 2028])
    ):
        raise RuntimeError("cross-version winner seed adjudication is incomplete or stale")
    return value


def _ablation(output_root: Path, cross_path: Path, winner: str, algorithm: str) -> dict[str, Any]:
    path = output_root / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
    if not path.is_file():
        raise RuntimeError("final delivery requires winner proposal/observable/full e200 ablations")
    value = _read_json(path)
    _posthoc(value, label="winner ablation adjudication")
    if (
        value.get("schema") != ABLATION_SCHEMA
        or value.get("status") != "COMPLETE_NO_SELECTION_CHANGE"
        or value.get("selected_candidate_id") != winner
        or value.get("selected_algorithm_fingerprint") != algorithm
        or value.get("source_cross_version_adjudication_sha256") != file_sha256(cross_path)
        or value.get("proposal_only_out_ranks_full") is not False
        or value.get("selection_changed") is not False
    ):
        raise RuntimeError("winner ablation adjudication is incomplete or challenges selection")
    roles = value.get("roles")
    if not isinstance(roles, dict) or set(roles) != {
        "proposal_only", "observable_only", "projected_or_full",
    }:
        raise RuntimeError("winner ablation adjudication lacks the three required roles")
    if value.get("observable_only_identity", {}).get("status") != (
        "EXACT_PLAIN_E200_SCIENTIFIC_IDENTITY"
    ):
        raise RuntimeError("observable-only ablation is not exact plain identity")
    for role, row in roles.items():
        receipt_path = Path(row["receipt_path"])
        receipt = _validate_receipt(receipt_path)
        if (
            receipt["candidate_id"] != row.get("candidate_id")
            or receipt["algorithm_fingerprint"] != row.get("algorithm_fingerprint")
            or file_sha256(receipt_path) != row.get("receipt_sha256")
        ):
            raise RuntimeError(f"winner ablation receipt changed: {role}")
    return value


def _report(path: Path, candidate: dict[str, Any], alternates: dict[str, Any]) -> None:
    lines = [
        "# FINAL_UNSB 路线一跨版本最终裁决",
        "",
        f"- 唯一候选：`{candidate['candidate_id']}`（{candidate['name']}）",
        f"- 分类：`{candidate['classification']}`",
        "- 科学单位：small25、batch1、真实200 data epochs；固定e200，不选择最佳checkpoint",
        "- paired指标只在完整轨迹后裁决；confirmation20仍封存",
        "- 候选各自在原始训练代码身份内验收，仅source-bound receipt跨版本排名",
        "",
        "## 消融",
        "",
        "proposal-only、observable-only、projected/full均已从共同e0完成e200。"
        "observable-only必须与plain保持e200科学状态精确一致；若proposal-only胜过full，"
        "本交付会拒绝生成并要求重新做冻结seed裁决。",
        "",
        "## 备选",
        "",
    ]
    for row in alternates["alternates"]:
        lines.append(f"- `{row['candidate_id']}`：{row['role']}；{row['reason_not_selected']}")
    lines += [
        "",
        "结论只覆盖当前small25长期代理，不自动外推到一万张全量数据。复现身份、"
        "逐域轨迹、复杂度和风险见同目录CANDIDATE.json与RESULTS.json。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def materialize_cross_version_final_delivery(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    cross_path = output_root / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    if not cross_path.is_file():
        raise RuntimeError("cross-version final delivery is blocked until e200 adjudication")
    cross = _read_json(cross_path)
    _posthoc(cross, label="cross-version e200 adjudication")
    if cross.get("schema") != CROSS_SCHEMA:
        raise RuntimeError("cross-version e200 adjudication schema mismatch")
    if cross.get("status") != POSITIVE_CROSS_STATUS:
        raise RuntimeError(
            "negative cross-version outcome requires the allowed causal-revision adjudication "
            "before final delivery"
        )
    ranking = cross.get("ranking")
    if not isinstance(ranking, list) or len(ranking) < 2:
        raise RuntimeError("cross-version final delivery requires both complete candidates")
    receipts = _load_cross_receipts(output_root, cross)
    winner = str(cross["selected_candidate_id"])
    if winner not in receipts:
        raise RuntimeError("selected cross-version winner has no accepted receipt")
    selected_receipt = receipts[winner]
    algorithm = str(selected_receipt["algorithm_fingerprint"])
    if algorithm != cross.get("selected_algorithm_fingerprint"):
        raise RuntimeError("selected cross-version algorithm fingerprint changed")
    multi_seed = _multi_seed(output_root, winner, algorithm)
    ablation = _ablation(output_root, cross_path, winner, algorithm)
    card, implementation, card_path, implementation_path = _source_bound_method(
        output_root, selected_receipt,
    )

    candidate_results = {}
    for row in ranking:
        candidate_id = str(row["candidate_id"])
        candidate_results[candidate_id] = {
            "rank": int(row["rank"]),
            "source_bound_receipt_sha256": file_sha256(
                _receipt_path(output_root, candidate_id)
            ),
            "summary": _read_json(
                output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
            ),
            "absolute_relative_domain_trajectory": _candidate_domain_trajectory(
                output_root, candidate_id,
            ),
            "median_epoch_wall_seconds": _median_epoch_seconds(
                output_root / "candidates" / candidate_id
            ),
        }
    ablation_results = {}
    for role, row in ablation["roles"].items():
        candidate_id = str(row["candidate_id"])
        ablation_results[role] = {
            "candidate_id": candidate_id,
            "summary": _read_json(
                output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
            ),
            "absolute_relative_domain_trajectory": _candidate_domain_trajectory(
                output_root, candidate_id,
            ),
            "receipt_sha256": row["receipt_sha256"],
        }
    seed_results = {}
    for seed in multi_seed["included_seeds"]:
        if int(seed) == 2026:
            continue
        seed_root = output_root / "seed_validation" / f"seed{int(seed)}"
        seed_results[str(int(seed))] = {
            "summary": _read_json(seed_root / "SEED_VALIDATION_SUMMARY.json"),
            "absolute_relative_domain_trajectory": _seed_domain_trajectory(
                output_root, winner, int(seed),
            ),
        }

    final_root = output_root / "final"
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": winner,
        "cross_version_adjudication_sha256": file_sha256(cross_path),
        "ranking": ranking,
        "candidate_results": candidate_results,
        "seed_results": seed_results,
        "winner_ablation_adjudication": ablation,
        "winner_ablation_results": ablation_results,
        "paired_metrics_used_only_after_complete_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    write_json(final_root / "RESULTS.json", results)

    runner_up = next(row for row in ranking if row["candidate_id"] != winner)
    proposal = ablation["roles"]["proposal_only"]
    alternates = {
        "schema": ALTERNATES_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": winner,
        "alternates": [
            {
                "rank": 2,
                "candidate_id": runner_up["candidate_id"],
                "role": "tested_generation1_alternate",
                "trajectory_status": runner_up["trajectory_status"],
                "reason_not_selected": "lower frozen post-e200 cross-version rank",
            },
            {
                "rank": 3,
                "candidate_id": proposal["candidate_id"],
                "role": "tested_long_horizon_proposal_only_alternate",
                "trajectory_status": proposal["trajectory_status"],
                "reason_not_selected": "did not outrank the frozen full operator at e200",
            },
        ],
        "old_probe_reserved_slot": False,
        "confirmation20_opened": False,
    }
    write_json(final_root / "ALTERNATES.json", alternates)

    classification = _classification(multi_seed)
    compute_sensitive = winner == "G1-02B-PLAYER-CONDITIONAL-RSMG"
    report_path = final_root / "FINAL_ROUTE1_REPORT.md"
    candidate = {
        "schema": SCHEMA,
        "status": "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE",
        "classification": classification,
        "candidate_id": winner,
        "name": card["name"],
        "algorithm_fingerprint": algorithm,
        "seed2026_candidate_fingerprint": selected_receipt["candidate_fingerprint"],
        "training_git_commit": selected_receipt["training_git_commit"],
        "candidate_training_core_fingerprint": selected_receipt[
            "candidate_training_core_fingerprint"
        ],
        "source_bound_terminal_receipt_sha256": file_sha256(
            _receipt_path(output_root, winner)
        ),
        "selected_fixed_checkpoint": {
            "data_epoch": 200, "best_checkpoint_selection": False,
        },
        "formula": card["formula"],
        "unsb_object": card["unsb_object"],
        "identity_or_unbiased_condition": card["identity_or_unbiased_condition"],
        "target_inaccessibility_proof": card["target_inaccessibility_proof"],
        "algorithm_hyperparameters": card["algorithm_hyperparameters"],
        "executable_configuration": {
            "model": implementation["model"], "method": implementation["method"],
        },
        "complexity": {
            "compute": card["compute_cost"],
            "memory": card["memory_cost"],
            "recovery_state": card["recovery_state_cost"],
            "compute_sensitive_signal": compute_sensitive,
        },
        "winner_ablation_adjudication_sha256": file_sha256(
            output_root / "operations" / "WINNER_ABLATION_ADJUDICATION.json"
        ),
        "risks": [
            "small25 proxy evidence is not a full-dataset conclusion",
            "confirmation20 remains sealed",
            "4090 and 5090 runtime trajectories are independent and are not numerically merged",
            *(["positive signal may depend on the fixed replicated-compute budget"] if compute_sensitive else []),
        ],
        "derivation_card_path": card_path.relative_to(output_root).as_posix(),
        "derivation_card_sha256": file_sha256(card_path),
        "implementation_path": implementation_path.relative_to(output_root).as_posix(),
        "implementation_sha256": file_sha256(implementation_path),
        "source_files": implementation["source_files"],
        "reproduction_commands": {
            "seed2026_e200": (
                "python operations/local_route1_candidate_executor.py --contract "
                f"<RUN_ROOT>/operations/CANDIDATE_EXECUTOR_CONTRACT_{winner}.json"
            ),
            "seed_validation": (
                "python operations/local_route1_seed_executor.py --contract "
                f"<RUN_ROOT>/operations/SEED_EXECUTOR_CONTRACT_{winner}_s<SEED>.json"
            ),
            "source_identity": (
                f"checkout training_git_commit {selected_receipt['training_git_commit']} "
                "for the full candidate; do not load it under a sibling training core"
            ),
        },
        "training_batch_size": 1,
        "target_data_epochs": 200,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _report(report_path, candidate, alternates)
    candidate.update({
        "results_path": "final/RESULTS.json",
        "results_sha256": file_sha256(final_root / "RESULTS.json"),
        "alternates_path": "final/ALTERNATES.json",
        "alternates_sha256": file_sha256(final_root / "ALTERNATES.json"),
        "final_report_path": "final/FINAL_ROUTE1_REPORT.md",
        "final_report_sha256": file_sha256(report_path),
    })
    write_json(final_root / "CANDIDATE.json", candidate)
    return candidate
