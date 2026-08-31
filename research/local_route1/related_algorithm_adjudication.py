"""Host-local and cross-host adjudication for the related algorithm family.

Numeric deltas are ranked only against the matched plain trajectory on the
same host.  The combined artifact preserves multiple algorithms and never
averages 4090 and 5090 deltas into a pseudo-seed estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


HOST_SCHEMA = "final-unsb-route1-related-host-adjudication-v1"
COMBINED_SCHEMA = "final-unsb-route1-related-multi-host-adjudication-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _classification(trajectory: dict[str, Any]) -> tuple[str, dict[str, bool]]:
    checks = {
        "late_three_positive": float(
            trajectory.get("late_three_mean_macro_psnr_delta", -1e9)
        ) > 0.0,
        "e200_positive": float(trajectory.get("e200_macro_psnr_delta", -1e9)) > 0.0,
        "late_domain_coverage": int(
            trajectory.get("late_points_with_four_of_six_positive_domains", 0)
        ) >= 2,
        "worst_domain_guardrail": float(
            trajectory.get("late_average_worst_domain_delta", -1e9)
        ) > -1.0,
        "ssim_guardrail": float(
            trajectory.get("late_mean_macro_ssim_delta", -1e9)
        ) >= 0.0,
        "lpips_guardrail": float(
            trajectory.get("late_mean_macro_lpips_delta", 1e9)
        ) <= 0.0,
        "rolling_drawdown_guardrail": float(
            trajectory.get(
                "candidate_best_to_terminal_three_point_rolling_drawdown", 1e9
            )
        ) <= float(trajectory.get("maximum_allowed_rolling_drawdown_db", 0.3)),
        "not_plain_collapse": (
            trajectory.get("plain_collapse_adjudication", {}).get("status")
            == "PASS_NOT_PLAIN_COLLAPSE"
        ),
    }
    if all(checks.values()):
        return "strict_sustained_local_signal", checks
    if checks["late_three_positive"] and checks["e200_positive"]:
        return "positive_but_fragile", checks
    return "closed_current_operator_on_this_host", checks


def _terminal_row(run_root: Path, candidate_id: str, host_label: str) -> dict[str, Any]:
    receipt_path = (
        run_root / "operations" / "terminal_receipts" / f"{candidate_id}.json"
    )
    if not receipt_path.is_file():
        raise RuntimeError(f"missing terminal receipt: {candidate_id}")
    receipt = _read_json(receipt_path)
    trajectory_path = Path(str(receipt.get("trajectory_path", ""))).resolve()
    try:
        trajectory_path.relative_to(run_root.resolve())
    except ValueError as error:
        raise RuntimeError(f"trajectory escapes run root: {candidate_id}") from error
    card_path = run_root / "derive" / "cards" / f"{candidate_id}.json"
    implementation_path = run_root / "derive" / "implementations" / f"{candidate_id}.json"
    if not all(path.is_file() for path in (trajectory_path, card_path, implementation_path)):
        raise RuntimeError(f"terminal source chain is incomplete: {candidate_id}")
    trajectory = _read_json(trajectory_path)
    fixed = {
        "candidate_id": candidate_id,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "trajectory_sha256": file_sha256(trajectory_path),
        "derivation_card_sha256": file_sha256(card_path),
        "implementation_sha256": file_sha256(implementation_path),
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"terminal receipt mismatch {candidate_id}: {key}")
    if (
        trajectory.get("candidate_id") != candidate_id
        or trajectory.get("confirmation20_opened") is not False
        or trajectory.get("paired_metrics_used_for_training_or_gate") is not False
    ):
        raise RuntimeError(f"trajectory control boundary changed: {candidate_id}")
    classification, checks = _classification(trajectory)
    return {
        "candidate_id": candidate_id,
        "host_label": host_label,
        "classification": classification,
        "strict_checks": checks,
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "candidate_fingerprint": receipt["candidate_fingerprint"],
        "training_git_commit": receipt["training_git_commit"],
        "base_e0_scientific_state_sha256": receipt[
            "base_e0_scientific_state_sha256"
        ],
        "base_protocol_fingerprint": receipt["base_protocol_fingerprint"],
        "manifest_sha256": receipt["manifest_sha256"],
        "late_three_mean_macro_psnr_delta": float(
            trajectory["late_three_mean_macro_psnr_delta"]
        ),
        "e200_macro_psnr_delta": float(trajectory["e200_macro_psnr_delta"]),
        "late_points_with_four_of_six_positive_domains": int(
            trajectory["late_points_with_four_of_six_positive_domains"]
        ),
        "late_average_worst_domain_delta": float(
            trajectory["late_average_worst_domain_delta"]
        ),
        "late_mean_macro_ssim_delta": float(
            trajectory["late_mean_macro_ssim_delta"]
        ),
        "late_mean_macro_lpips_delta": float(
            trajectory["late_mean_macro_lpips_delta"]
        ),
        "rolling_drawdown_db": float(
            trajectory["candidate_best_to_terminal_three_point_rolling_drawdown"]
        ),
        "median_epoch_wall_seconds": float(receipt.get("median_epoch_wall_seconds", 0.0)),
        "terminal_receipt_path": str(receipt_path),
        "terminal_receipt_sha256": file_sha256(receipt_path),
        "trajectory_path": str(trajectory_path),
        "trajectory_sha256": file_sha256(trajectory_path),
    }


def adjudicate_related_host(
    run_root: Path, *, host_label: str, candidate_ids: list[str], output_path: Path,
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    if not candidate_ids or len(candidate_ids) != len(set(candidate_ids)):
        raise RuntimeError("host adjudication requires unique candidate ids")
    rows = [_terminal_row(run_root, candidate_id, host_label) for candidate_id in candidate_ids]
    protocol_ids = {
        (row["base_e0_scientific_state_sha256"], row["base_protocol_fingerprint"], row["manifest_sha256"])
        for row in rows
    }
    if len(protocol_ids) != 1:
        raise RuntimeError("host-local related candidates do not share e0/protocol/manifest")
    order = {
        "strict_sustained_local_signal": 2,
        "positive_but_fragile": 1,
        "closed_current_operator_on_this_host": 0,
    }
    rows.sort(key=lambda row: (
        order[row["classification"]],
        row["late_three_mean_macro_psnr_delta"],
        row["e200_macro_psnr_delta"],
        row["late_points_with_four_of_six_positive_domains"],
        row["late_average_worst_domain_delta"],
        -row["median_epoch_wall_seconds"],
    ), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["host_local_rank"] = rank
    strict = [
        row["candidate_id"] for row in rows
        if row["classification"] == "strict_sustained_local_signal"
    ]
    fragile = [
        row["candidate_id"] for row in rows
        if row["classification"] == "positive_but_fragile"
    ]
    result = {
        "schema": HOST_SCHEMA,
        "status": "RELATED_HOST_E200_ADJUDICATION_COMPLETE",
        "host_label": host_label,
        "run_root": str(run_root),
        "candidate_ids": list(candidate_ids),
        "ranking": rows,
        "strict_sustained_candidate_ids": strict,
        "positive_but_fragile_candidate_ids": fragile,
        "action_priority_candidate_id": rows[0]["candidate_id"],
        "action_priority_is_not_scientific_exclusivity": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "selection_seeds": [2026],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_path).resolve(), result)
    return result


def combine_related_hosts(
    host_paths: list[Path], *, output_path: Path,
) -> dict[str, Any]:
    if len(host_paths) < 2:
        raise RuntimeError("combined related adjudication requires at least two hosts")
    hosts = []
    seen_labels = set()
    for path in host_paths:
        path = Path(path).resolve()
        value = _read_json(path)
        if value.get("schema") != HOST_SCHEMA or value.get("status") != (
            "RELATED_HOST_E200_ADJUDICATION_COMPLETE"
        ):
            raise RuntimeError(f"invalid host adjudication: {path}")
        label = value.get("host_label")
        if not label or label in seen_labels:
            raise RuntimeError("combined adjudication host labels are not unique")
        seen_labels.add(label)
        hosts.append({"path": str(path), "sha256": file_sha256(path), "value": value})

    algorithms: dict[str, dict[str, Any]] = {}
    for host in hosts:
        label = host["value"]["host_label"]
        for row in host["value"]["ranking"]:
            algorithm = algorithms.setdefault(row["algorithm_fingerprint"], {
                "algorithm_fingerprint": row["algorithm_fingerprint"],
                "candidate_ids": set(),
                "host_results": [],
            })
            algorithm["candidate_ids"].add(row["candidate_id"])
            algorithm["host_results"].append({
                "host_label": label,
                "candidate_id": row["candidate_id"],
                "classification": row["classification"],
                "host_local_rank": row["host_local_rank"],
                "late_three_mean_macro_psnr_delta": row[
                    "late_three_mean_macro_psnr_delta"
                ],
                "e200_macro_psnr_delta": row["e200_macro_psnr_delta"],
                "terminal_receipt_sha256": row["terminal_receipt_sha256"],
            })
    algorithm_rows = []
    for algorithm in algorithms.values():
        results = algorithm["host_results"]
        strict_hosts = [
            row["host_label"] for row in results
            if row["classification"] == "strict_sustained_local_signal"
        ]
        fragile_hosts = [
            row["host_label"] for row in results
            if row["classification"] == "positive_but_fragile"
        ]
        algorithm_rows.append({
            **algorithm,
            "candidate_ids": sorted(algorithm["candidate_ids"]),
            "strict_positive_hosts": strict_hosts,
            "fragile_positive_hosts": fragile_hosts,
            "viable_on_at_least_one_host": bool(strict_hosts),
            "cross_runtime_positive": len(set(strict_hosts)) >= 2,
            "cross_seed_stability_claimed": False,
        })
    algorithm_rows.sort(key=lambda row: (
        len(row["strict_positive_hosts"]),
        len(row["fragile_positive_hosts"]),
    ), reverse=True)
    viable = [
        row["algorithm_fingerprint"] for row in algorithm_rows
        if row["viable_on_at_least_one_host"]
    ]
    result = {
        "schema": COMBINED_SCHEMA,
        "status": (
            "MULTIPLE_VIABLE_ALGORITHMS"
            if len(viable) >= 2 else
            "ONE_VIABLE_ALGORITHM_WITH_RELATED_FRONTIER"
            if len(viable) == 1 else
            "NO_STRICT_ALGORITHM_RELATED_FRONTIER_PRESERVED"
        ),
        "host_adjudications": [
            {
                "host_label": host["value"]["host_label"],
                "path": host["path"],
                "sha256": host["sha256"],
                "action_priority_candidate_id": host["value"][
                    "action_priority_candidate_id"
                ],
            }
            for host in hosts
        ],
        "algorithms": algorithm_rows,
        "viable_algorithm_fingerprints": viable,
        "viable_algorithm_count": len(viable),
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "cross_host_deltas_merged": False,
        "cross_runtime_is_not_cross_seed": True,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    write_json(Path(output_path).resolve(), result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    host = subparsers.add_parser("host")
    host.add_argument("--run-root", type=Path, required=True)
    host.add_argument("--host-label", required=True)
    host.add_argument("--candidate-id", action="append", required=True)
    host.add_argument("--output", type=Path, required=True)
    combine = subparsers.add_parser("combine")
    combine.add_argument("--host-adjudication", type=Path, action="append", required=True)
    combine.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "host":
        result = adjudicate_related_host(
            args.run_root,
            host_label=args.host_label,
            candidate_ids=args.candidate_id,
            output_path=args.output,
        )
    else:
        result = combine_related_hosts(
            args.host_adjudication, output_path=args.output,
        )
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

