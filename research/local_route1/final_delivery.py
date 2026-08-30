"""Fail-closed final route-1 candidate and evidence materialization.

This module is downstream of complete e200, optional frozen seed validation,
and (only when necessary) the one allowed causal-revision adjudication.  It
cannot train, select a checkpoint, read confirmation20, or modify an algorithm.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from .candidates import load_candidate_registration, validate_candidate_id
from .evaluate import compare_to_plain
from .protocol import file_sha256
from .runtime import write_json
from .seed_validation import MULTI_SEED_ADJUDICATION_SCHEMA


SCHEMA = "final-unsb-route1-candidate-delivery-v1"
RESULTS_SCHEMA = "final-unsb-route1-final-results-v1"
ALTERNATES_SCHEMA = "final-unsb-route1-final-alternates-v1"
REVISION_OUTCOME_SCHEMA = "final-unsb-route1-final-revision-outcome-v1"
POSITIVE_ADJUDICATION = "SEED2026_WINNER_READY_FOR_FROZEN_SEED2027"
NEGATIVE_ADJUDICATION = "NO_SEED2026_NUMERIC_GATE_PASS_CAUSAL_DEFECT_ADJUDICATION_REQUIRED"
COMPLETE_MULTI_SEED = {"ROUTE1_SUSTAINED_LOCAL", "MULTI_SEED_NOT_SUSTAINED"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _assert_posthoc_integrity(payload: dict[str, Any], *, label: str) -> None:
    if payload.get("confirmation20_opened") is not False:
        raise RuntimeError(f"{label} does not prove confirmation20 remained closed")
    for key in (
        "paired_controller_access", "paired_metric_changed_algorithm",
        "paired_metrics_used_for_training_or_control",
        "paired_metrics_used_for_training_or_gate",
    ):
        if key in payload and payload[key] is not False:
            raise RuntimeError(f"{label} violates posthoc-only paired-metric policy: {key}")


def _median_epoch_seconds(candidate_root: Path) -> float | None:
    path = candidate_root / "TRAIN_TRACE.jsonl"
    if not path.is_file():
        return None
    values = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line).get("epoch_wall_seconds")
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number > 0.0:
            values.append(number)
    return None if not values else float(statistics.median(values))


def _metric_domain_row(candidate: dict, plain: dict, *, epoch: int) -> dict:
    delta = compare_to_plain(candidate, plain, epoch=epoch)
    domains = {}
    for domain in sorted(plain["domains"]):
        method_row = candidate["domains"][domain]
        plain_row = plain["domains"][domain]
        domains[domain] = {
            "candidate": {
                "psnr": method_row["psnr"], "ssim": method_row["ssim"],
                "lpips": method_row["lpips"],
            },
            "plain": {
                "psnr": plain_row["psnr"], "ssim": plain_row["ssim"],
                "lpips": plain_row["lpips"],
            },
            "delta": delta["domain_delta"][domain],
        }
    return {
        "data_epoch": int(epoch), "updates": int(delta["updates"]),
        "candidate_macro": {
            "psnr": candidate["macro_psnr"], "ssim": candidate["macro_ssim"],
            "lpips": candidate["macro_lpips"],
        },
        "plain_macro": {
            "psnr": plain["macro_psnr"], "ssim": plain["macro_ssim"],
            "lpips": plain["macro_lpips"],
        },
        "macro_delta": {
            "psnr": delta["macro_psnr_delta"],
            "ssim": delta["macro_ssim_delta"],
            "lpips": delta["macro_lpips_delta"],
        },
        "positive_domains": delta["positive_domains"],
        "worst_domain_delta": delta["worst_domain_delta"],
        "domains": domains,
    }


def _candidate_domain_trajectory(output_root: Path, candidate_id: str) -> list[dict]:
    trajectory_path = output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
    trajectory = _read_json(trajectory_path)
    _assert_posthoc_integrity(trajectory, label=f"{candidate_id} trajectory")
    rows = []
    for summary in trajectory.get("trajectory", []):
        epoch = int(summary["epoch"])
        candidate_path = output_root / "candidates" / candidate_id / "metrics" / f"e{epoch:03d}.json"
        plain_path = output_root / "anchors" / "plain" / "metrics" / f"e{epoch:03d}.json"
        if not candidate_path.is_file() or not plain_path.is_file():
            raise RuntimeError(f"raw matched metric missing for {candidate_id} e{epoch}")
        rows.append(_metric_domain_row(
            _read_json(candidate_path), _read_json(plain_path), epoch=epoch,
        ))
    if not rows or rows[-1]["data_epoch"] != 200:
        raise RuntimeError(f"{candidate_id} has no complete e200 domain trajectory")
    return rows


def _seed_domain_trajectory(
    output_root: Path, candidate_id: str, seed: int,
) -> list[dict]:
    seed_root = output_root / "seed_validation" / f"seed{int(seed)}"
    summary_path = seed_root / "SEED_VALIDATION_SUMMARY.json"
    summary = _read_json(summary_path)
    _assert_posthoc_integrity(summary, label=f"seed{seed} summary")
    rows = []
    for item in summary.get("trajectory", []):
        epoch = int(item["epoch"])
        candidate_path = seed_root / "candidate" / "metrics" / f"e{epoch:03d}.json"
        plain_path = seed_root / "plain" / "metrics" / f"e{epoch:03d}.json"
        if not candidate_path.is_file() or not plain_path.is_file():
            raise RuntimeError(f"raw matched seed{seed} metric missing at e{epoch}")
        rows.append(_metric_domain_row(
            _read_json(candidate_path), _read_json(plain_path), epoch=epoch,
        ))
    if not rows or rows[-1]["data_epoch"] != 200:
        raise RuntimeError(f"seed{seed} has no complete e200 domain trajectory")
    return rows


def _revision_outcome(output_root: Path, adjudication: dict) -> tuple[str, dict]:
    path = output_root / "operations" / "FINAL_CAUSAL_REVISION_OUTCOME.json"
    if not path.is_file():
        raise RuntimeError(
            "negative Generation-1 results require explicit target-blind causal-revision adjudication"
        )
    outcome = _read_json(path)
    required = {
        "schema": REVISION_OUTCOME_SCHEMA,
        "source_generation1_adjudication_sha256": file_sha256(
            output_root / "operations" / "GENERATION1_E200_ADJUDICATION.json"
        ),
        "paired_metric_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    for key, value in required.items():
        if outcome.get(key) != value:
            raise RuntimeError(f"invalid final causal-revision outcome field: {key}")
    status = outcome.get("status")
    if status not in (
        "NO_REVISION_APPLICABLE_FINAL_FALLBACK",
        "REVISION_E200_COMPLETE_FINAL_ADJUDICATION",
    ):
        raise RuntimeError("causal-revision outcome is not final")
    selected = validate_candidate_id(str(outcome.get("selected_candidate_id", "")))
    if status == "NO_REVISION_APPLICABLE_FINAL_FALLBACK":
        if selected != adjudication.get("selected_candidate_id"):
            raise RuntimeError("revision outcome changed the frozen Generation-1 fallback")
        records = outcome.get("candidate_defect_adjudications")
        if not isinstance(records, list) or len(records) < 2:
            raise RuntimeError("revision inapplicability requires both candidate defect adjudications")
        if any(row.get("revision_applicable") is not False for row in records):
            raise RuntimeError("revision marked inapplicable while a candidate remains eligible")
    else:
        if int(outcome.get("revision_count", -1)) != 1:
            raise RuntimeError("final revised candidate must be the single allowed revision")
        parent = validate_candidate_id(str(outcome.get("parent_candidate_id", "")))
        if parent not in {row["candidate_id"] for row in adjudication["ranking"]}:
            raise RuntimeError("final revision parent is not a frozen Generation-1 candidate")
        trajectory_path = output_root / "candidates" / selected / "CANDIDATE_TRAJECTORY.json"
        if not trajectory_path.is_file():
            raise RuntimeError("final revised candidate trajectory is missing")
        if outcome.get("selected_revision_trajectory_sha256") != file_sha256(trajectory_path):
            raise RuntimeError("final revised candidate trajectory changed")
        trajectory = _read_json(trajectory_path)
        if trajectory.get("status") not in (
            "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION",
            "LONG_HORIZON_NEGATIVE_CURRENT_IMPLEMENTATION",
        ):
            raise RuntimeError("final revised candidate has no complete e200 adjudication")
        if not any(int(row.get("epoch", -1)) == 200 for row in trajectory.get("trajectory", [])):
            raise RuntimeError("final revised candidate has no e200 trajectory row")
    return selected, outcome


def _classification(multi_seed: dict | None) -> str:
    if multi_seed is None:
        return "weak_fallback_after_causal_revision_adjudication"
    return (
        "route1_sustained_local"
        if multi_seed["status"] == "ROUTE1_SUSTAINED_LOCAL"
        else "positive_seed2026_but_not_multi_seed_sustained"
    )


def _portable_commands(candidate_id: str) -> dict[str, str]:
    return {
        "seed2026_e200": (
            "python operations/local_route1_candidate_executor.py --contract "
            f"<RUN_ROOT>/operations/CANDIDATE_EXECUTOR_CONTRACT_{candidate_id}.json"
        ),
        "seed_validation": (
            "python operations/local_route1_seed_executor.py --contract "
            f"<RUN_ROOT>/operations/SEED_EXECUTOR_CONTRACT_{candidate_id}_s<SEED>.json"
        ),
        "final_adjudication": (
            "python -m research.local_route1.run --stage seed_validate "
            f"--validation-action aggregate --candidate-id {candidate_id} --output <RUN_ROOT>"
        ),
        "exact_checkpoint_policy": "fixed e200 only; no best-checkpoint selection",
    }


def _write_final_report(
    path: Path, *, selected_id: str, name: str, classification: str,
    trajectory: dict, revision: dict, alternates: dict, compute_sensitive: bool,
) -> None:
    lines = [
        "# FINAL_UNSB 路线一最终裁决",
        "",
        f"- 当前唯一候选：`{selected_id}`（{name}）",
        f"- 科学分类：`{classification}`",
        "- 固定裁决点：small25、batch1、seed2026、e150/e175/e200，最终 checkpoint 固定 e200",
        "- paired 指标只在完整轨迹后裁决；confirmation20 仍封存",
        "",
        "## 结果摘要",
        "",
        f"- late-three 宏 PSNR delta：`{trajectory.get('late_three_mean_macro_psnr_delta')}`",
        f"- e200 宏 PSNR delta：`{trajectory.get('e200_macro_psnr_delta')}`",
        f"- e200 状态：`{trajectory.get('status')}`",
        f"- 因果修订裁决：`{revision.get('status')}`",
        f"- 计算敏感信号：`{str(bool(compute_sensitive)).lower()}`",
        "",
        "## 结论边界",
        "",
        "- 科学结论只覆盖冻结的 small25 长程代理，不自动外推到一万张全量数据或4090论文终局。",
        "- 工程事故与科学结果分离：持久执行、恢复和身份门有独立证据，旧启动前事故不计入轨迹。",
        "- proxy 已由同宿主 HJ/HNEK 长程正对照校准，但旧探针没有候选保留名额。",
        "- 未测试假设包括 full-data 泛化和 confirmation20；这两项没有被当前文件暗示为已验证。",
        "",
        "## 备选与关闭方向",
        "",
    ]
    for row in alternates["alternates"]:
        lines.append(
            f"- `{row['candidate_id']}`：{row['role']}；{row['reason_not_selected']}"
        )
    lines += [
        "",
        "## 可复现入口",
        "",
        "精确算法、指纹、复杂度、风险和命令见 `CANDIDATE.json`；逐域绝对/相对轨迹见 "
        "`RESULTS.json`。不得用最佳 checkpoint、跨宿主 delta 或 confirmation20 改写本裁决。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)


def materialize_final_delivery(output_root: Path) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    adjudication_path = output_root / "operations" / "GENERATION1_E200_ADJUDICATION.json"
    if not adjudication_path.is_file():
        raise RuntimeError("final delivery is blocked until Generation-1 e200 adjudication")
    adjudication = _read_json(adjudication_path)
    _assert_posthoc_integrity(adjudication, label="Generation-1 adjudication")
    if adjudication.get("status") not in (POSITIVE_ADJUDICATION, NEGATIVE_ADJUDICATION):
        raise RuntimeError("final delivery refuses incomplete Generation-1 adjudication")
    ranking = adjudication.get("ranking")
    if not isinstance(ranking, list) or len(ranking) < 2:
        raise RuntimeError("final delivery requires the complete two-candidate ranking")

    multi_seed = None
    revision = None
    if adjudication["status"] == POSITIVE_ADJUDICATION:
        selected_id = validate_candidate_id(str(adjudication["selected_candidate_id"]))
        if adjudication.get("winner_frozen_for_seed2027") is not True:
            raise RuntimeError("positive winner was not immutably frozen for seed validation")
        multi_path = output_root / "candidates" / selected_id / "MULTI_SEED_ADJUDICATION.json"
        if not multi_path.is_file():
            raise RuntimeError("positive seed2026 winner requires complete frozen seed adjudication")
        multi_seed = _read_json(multi_path)
        if multi_seed.get("schema") != MULTI_SEED_ADJUDICATION_SCHEMA:
            raise RuntimeError("multi-seed adjudication schema mismatch")
        if multi_seed.get("status") not in COMPLETE_MULTI_SEED:
            raise RuntimeError("multi-seed adjudication is not complete")
        if multi_seed.get("candidate_id") != selected_id:
            raise RuntimeError("multi-seed adjudication belongs to another candidate")
        if multi_seed.get("included_seeds") not in ([2026, 2027], [2026, 2027, 2028]):
            raise RuntimeError("multi-seed adjudication lacks the required frozen seed sequence")
        _assert_posthoc_integrity(multi_seed, label="multi-seed adjudication")
        revision = {"status": "INAPPLICABLE_POSITIVE_SEED2026_NUMERIC_WINNER"}
    else:
        selected_id, revision = _revision_outcome(output_root, adjudication)

    selected_trajectory_path = (
        output_root / "candidates" / selected_id / "CANDIDATE_TRAJECTORY.json"
    )
    selected_trajectory = _read_json(selected_trajectory_path)
    if (
        adjudication["status"] == NEGATIVE_ADJUDICATION
        and revision.get("status") == "REVISION_E200_COMPLETE_FINAL_ADJUDICATION"
        and selected_trajectory.get("status")
        == "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION"
    ):
        multi_path = output_root / "candidates" / selected_id / "MULTI_SEED_ADJUDICATION.json"
        if not multi_path.is_file():
            raise RuntimeError("positive revised candidate requires complete frozen seed adjudication")
        multi_seed = _read_json(multi_path)
        if (
            multi_seed.get("schema") != MULTI_SEED_ADJUDICATION_SCHEMA
            or multi_seed.get("status") not in COMPLETE_MULTI_SEED
            or multi_seed.get("candidate_id") != selected_id
            or multi_seed.get("included_seeds") not in ([2026, 2027], [2026, 2027, 2028])
        ):
            raise RuntimeError("positive revised candidate multi-seed adjudication is incomplete")
        _assert_posthoc_integrity(multi_seed, label="revised multi-seed adjudication")

    registration = load_candidate_registration(output_root, selected_id, require_gate=True)
    trajectory_path = selected_trajectory_path
    trajectory = selected_trajectory
    if trajectory.get("algorithm_fingerprint") != registration.algorithm_fingerprint:
        raise RuntimeError("selected trajectory algorithm fingerprint changed")
    if multi_seed is not None and multi_seed.get(
        "algorithm_fingerprint"
    ) != registration.algorithm_fingerprint:
        raise RuntimeError("multi-seed algorithm fingerprint changed")
    card = _read_json(registration.card_path)
    implementation = _read_json(registration.implementation_path)
    delivery_ranking = list(ranking)
    if selected_id not in {row["candidate_id"] for row in ranking}:
        delivery_ranking = [{
            "rank": 1,
            "candidate_id": selected_id,
            "algorithm_fingerprint": registration.algorithm_fingerprint,
            "candidate_fingerprint": registration.candidate_fingerprint,
            "trajectory_sha256": file_sha256(trajectory_path),
            "trajectory_status": trajectory["status"],
            "late_three_mean_macro_psnr_delta": trajectory.get(
                "late_three_mean_macro_psnr_delta"
            ),
            "e200_macro_psnr_delta": trajectory.get("e200_macro_psnr_delta"),
        }] + [{**row, "rank": index} for index, row in enumerate(ranking, start=2)]
    candidate_results = {}
    for ranked in delivery_ranking:
        candidate_id = validate_candidate_id(str(ranked["candidate_id"]))
        candidate_path = output_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json"
        if ranked.get("trajectory_sha256") != file_sha256(candidate_path):
            raise RuntimeError(f"ranked trajectory changed after adjudication: {candidate_id}")
        candidate_results[candidate_id] = {
            "rank": ranked["rank"],
            "trajectory_sha256": file_sha256(candidate_path),
            "summary": _read_json(candidate_path),
            "absolute_relative_domain_trajectory": _candidate_domain_trajectory(
                output_root, candidate_id,
            ),
            "median_epoch_wall_seconds": _median_epoch_seconds(
                output_root / "candidates" / candidate_id
            ),
        }
    seed_results = {}
    if multi_seed is not None:
        for seed in multi_seed.get("included_seeds", []):
            if int(seed) == 2026:
                continue
            seed_results[str(int(seed))] = {
                "summary": _read_json(
                    output_root / "seed_validation" / f"seed{int(seed)}"
                    / "SEED_VALIDATION_SUMMARY.json"
                ),
                "absolute_relative_domain_trajectory": _seed_domain_trajectory(
                    output_root, selected_id, int(seed),
                ),
            }
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected_id,
        "generation1_adjudication_sha256": file_sha256(adjudication_path),
        "ranking": delivery_ranking,
        "generation1_ranking": ranking,
        "candidate_results": candidate_results,
        "seed_results": seed_results,
        "multi_seed_adjudication": multi_seed,
        "paired_metrics_used_only_after_complete_trajectories": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    final_root = output_root / "final"
    write_json(final_root / "RESULTS.json", results)

    other = [row for row in delivery_ranking if row["candidate_id"] != selected_id]
    alternate_rows = [
        {
            "rank": index + 2,
            "candidate_id": row["candidate_id"],
            "role": "tested_generation1_alternate",
            "trajectory_status": row["trajectory_status"],
            "reason_not_selected": "lower frozen post-e200 rank under the registered policy",
        }
        for index, row in enumerate(other[:2])
    ]
    if len(alternate_rows) < 2:
        alternate_rows.append({
            "rank": 3,
            "candidate_id": "DEFERRED-DT-STATE-FEEDBACK",
            "role": "closed_unfrozen_direction",
            "trajectory_status": "NOT_IMPLEMENTED_EVIDENCE_INSUFFICIENT",
            "reason_not_selected": (
                "Only four DT-specific covariance-mismatch driver records supported the most direct "
                "operator, and its one-sided projection template risked equivalence with the already "
                "negative SEARCH-005 constraint family; no third method was forced to fill a quota."
            ),
        })
    alternates = {
        "schema": ALTERNATES_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected_id,
        "alternates": alternate_rows,
        "old_probe_reserved_slot": False,
        "confirmation20_opened": False,
    }
    write_json(final_root / "ALTERNATES.json", alternates)

    classification = _classification(multi_seed)
    compute_sensitive = selected_id == "G1-02-SAMPLING-VARIANCE"
    report_path = final_root / "FINAL_ROUTE1_REPORT.md"
    _write_final_report(
        report_path, selected_id=selected_id, name=card["name"],
        classification=classification, trajectory=trajectory,
        revision=revision, alternates=alternates,
        compute_sensitive=compute_sensitive,
    )
    candidate = {
        "schema": SCHEMA,
        "status": "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE",
        "classification": classification,
        "candidate_id": selected_id,
        "name": card["name"],
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "seed2026_candidate_fingerprint": registration.candidate_fingerprint,
        "selected_fixed_checkpoint": {"data_epoch": 200, "best_checkpoint_selection": False},
        "formula": card["formula"],
        "unsb_object": card["unsb_object"],
        "identity_or_unbiased_condition": card["identity_or_unbiased_condition"],
        "target_inaccessibility_proof": card["target_inaccessibility_proof"],
        "objective_change": card["objective_change"],
        "estimator_change": card["estimator_change"],
        "coordinate_change": card["coordinate_change"],
        "endpoint_law_change": card["endpoint_law_change"],
        "algorithm_hyperparameters": card["algorithm_hyperparameters"],
        "ablation_definitions": card["ablation_definitions"],
        "parent_evidence": card["parent_evidence"],
        "causal_matrix_sha256": card["causal_matrix_sha256"],
        "reversal_atlas_sha256": card["reversal_atlas_sha256"],
        "executable_configuration": {
            "model": implementation["model"],
            "method": implementation["method"],
        },
        "complexity": {
            "compute": card["compute_cost"],
            "memory": card["memory_cost"],
            "recovery_state": card["recovery_state_cost"],
            "compute_sensitive_signal": compute_sensitive,
        },
        "risks": [
            "small25 proxy evidence is not a full-dataset conclusion",
            "confirmation20 remains sealed",
            "4090 and 5090 runtime trajectories must not be numerically merged",
            *(["positive signal may reflect the fixed two-replica compute budget"] if compute_sensitive else []),
        ],
        "revision_adjudication": revision,
        "results_path": "final/RESULTS.json",
        "results_sha256": file_sha256(final_root / "RESULTS.json"),
        "alternates_path": "final/ALTERNATES.json",
        "alternates_sha256": file_sha256(final_root / "ALTERNATES.json"),
        "final_report_path": "final/FINAL_ROUTE1_REPORT.md",
        "final_report_sha256": file_sha256(report_path),
        "derivation_card_path": str(registration.card_path),
        "derivation_card_sha256": file_sha256(registration.card_path),
        "implementation_path": str(registration.implementation_path),
        "implementation_sha256": file_sha256(registration.implementation_path),
        "source_files": implementation["source_files"],
        "reproduction_commands": _portable_commands(selected_id),
        "training_batch_size": 1,
        "target_data_epochs": 200,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(final_root / "CANDIDATE.json", candidate)
    return candidate
