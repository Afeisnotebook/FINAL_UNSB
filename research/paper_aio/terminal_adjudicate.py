"""Posthoc lead-lag adjudication for terminal bridge pathology.

All target-blind audits are validated before any paired discovery metric is
opened.  Fixed thresholds are protocol decisions, not fitted exit rules.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from operations.paper_aio_local_terminal_audit_successor import (
    AUDIT_EPOCHS,
    RECEIPT_SCHEMA as AUDIT_RECEIPT_SCHEMA,
    _audit_result,
)
from research.local_route1.runtime import write_json

from .protocol import (
    EVALUATION_SCHEMA,
    EXPECTED_MANIFEST_SHA256,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    file_sha256,
)
from .unified import UNIFIED_RECEIPT_SCHEMA


SCHEMA = "final-unsb-paper-terminal-pathology-adjudication-v1"
BINDING_SCHEMA = "final-unsb-paper-terminal-pathology-metric-bindings-v1"
PROBES = {
    "4090A_plain": {"lane_id": "plain", "source_host_label": "4090A"},
    "5090C_proposal": {"lane_id": "proposal", "source_host_label": "5090C"},
    "4090A_amtnc": {"lane_id": "amtnc", "source_host_label": "4090A"},
    "5090A_stcgr": {
        "lane_id": "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
        "source_host_label": "5090A",
    },
}
SPECTRAL_COLLAPSE_RATIO = 0.90
AMPLIFICATION_RATIO = 1.10
FUTURE_DECLINE_DB = -0.05
MIN_SUPPORT_METHODS = 2
MIN_SUPPORT_DOMAINS = 3


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _safe_ratio(after: float, before: float) -> float:
    return float(after) / max(abs(float(before)), 1e-12)


def _audit_summary(audit: dict[str, Any]) -> dict[str, dict[str, Any]]:
    summaries = {}
    for record in audit["records"]:
        by_time = {int(row["time_index"]): row for row in record["steps"]}
        terminal = by_time[4]
        late = [by_time[3], by_time[4]]
        summaries[str(record["domain"])] = {
            "stem": str(record["stem"]),
            "terminal_increment_trace": float(terminal["increment_spectrum"]["trace"]),
            "terminal_increment_effective_rank": float(
                terminal["increment_spectrum"]["effective_rank"]
            ),
            "late_rollout_jacobian_max": max(
                float(row["rollout_jacobian_top_singular_proxy"]) for row in late
            ),
            "late_finite_perturbation_gain_max": max(
                float(row["perturbation_gain_to_final_output"]) for row in late
            ),
            "nfe4_to_nfe5_output_rms_mean": float(
                record["nfe4_to_nfe5_output_rms_mean"]
            ),
        }
    return summaries


def _load_all_audits(audit_root: Path) -> tuple[dict, list[dict]]:
    """Validate the complete target-blind set before paired files are opened."""
    values = {}
    evidence = []
    for probe_id, probe in PROBES.items():
        values[probe_id] = {}
        for epoch in AUDIT_EPOCHS:
            root = Path(audit_root) / "probes" / probe_id / f"e{epoch:03d}"
            receipt_path = root / "AUDIT_RECEIPT.json"
            audit_path = root / "TERMINAL_AUDIT.jsonl"
            if not receipt_path.is_file() or not audit_path.is_file():
                raise RuntimeError(
                    f"target-blind terminal audit is incomplete: {probe_id} e{epoch}"
                )
            receipt = _read_json(receipt_path)
            if (
                receipt.get("schema") != AUDIT_RECEIPT_SCHEMA
                or receipt.get("status") != "PASS_FIXED_TARGET_BLIND_TERMINAL_AUDIT"
                or receipt.get("probe_id") != probe_id
                or receipt.get("host_label") != probe["source_host_label"]
                or receipt.get("lane_id") != probe["lane_id"]
                or int(receipt.get("epoch", -1)) != epoch
                or file_sha256(audit_path) != receipt.get("audit_sha256")
                or receipt.get("parent_state_and_rng_unchanged") is not True
                or receipt.get("performance_values_read") is not False
                or receipt.get("paired_metric_control") is not False
                or receipt.get("confirmation20_opened") is not False
            ):
                raise RuntimeError(
                    f"invalid target-blind audit receipt: {receipt_path}"
                )
            audit = _audit_result(audit_path)
            if audit.get("lane_id") != probe["lane_id"]:
                raise RuntimeError(f"target-blind audit lane mismatch: {audit_path}")
            values[probe_id][epoch] = _audit_summary(audit)
            evidence.append(
                {
                    "probe_id": probe_id,
                    "epoch": epoch,
                    "audit_receipt": str(receipt_path.resolve()),
                    "audit_receipt_sha256": file_sha256(receipt_path),
                    "audit_sha256": file_sha256(audit_path),
                }
            )
    return values, evidence


def _binding_cells(binding: dict[str, Any]) -> dict[str, dict[int, dict[str, Path]]]:
    if binding.get("schema") != BINDING_SCHEMA:
        raise RuntimeError("terminal pathology metric-binding schema mismatch")
    raw = binding.get("probes")
    if not isinstance(raw, dict) or set(raw) != set(PROBES):
        raise RuntimeError("terminal pathology metric bindings have the wrong probes")
    result = {}
    for probe_id, epochs in raw.items():
        if not isinstance(epochs, dict) or set(epochs) != {
            f"e{epoch:03d}" for epoch in AUDIT_EPOCHS
        }:
            raise RuntimeError(f"metric bindings have the wrong epochs: {probe_id}")
        result[probe_id] = {}
        for epoch in AUDIT_EPOCHS:
            row = epochs[f"e{epoch:03d}"]
            if not isinstance(row, dict):
                raise RuntimeError(f"invalid metric binding: {probe_id} e{epoch}")
            receipt = Path(str(row.get("receipt", "")))
            metric = Path(str(row.get("metric", "")))
            if not receipt.is_absolute() or not metric.is_absolute():
                raise RuntimeError(
                    f"metric binding paths must be absolute: {probe_id} e{epoch}"
                )
            result[probe_id][epoch] = {
                "receipt": receipt.resolve(),
                "metric": metric.resolve(),
            }
    return result


def _metric_images(
    metric: dict[str, Any], *, epoch: int
) -> dict[str, dict[tuple, dict]]:
    expected_count = 80 if epoch == 200 else 70
    expected_replicates = 5 if epoch == 200 else 1
    if (
        metric.get("schema") != EVALUATION_SCHEMA
        or metric.get("split") != "discovery"
        or int(metric.get("epoch", -1)) != epoch
        or int(metric.get("count_per_domain", -1)) != expected_count
        or int(metric.get("replicates", -1)) != expected_replicates
        or int(metric.get("primary_nfe", -1)) != 5
        or 5 not in metric.get("nfe_values", [])
        or metric.get("protocol_fingerprint") != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
        or metric.get("training_checkpoint_read_only") is not True
        or metric.get("cross_host_training_delta_merged") is not False
        or metric.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"terminal pathology metric protocol mismatch at e{epoch}")
    grouped: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for row in metric.get("images", []):
        if int(row.get("nfe", -1)) != 5 or int(row.get("replicate", -1)) != 0:
            continue
        key = (str(row.get("stem")), int(row.get("order", -1)))
        if key in grouped[str(row.get("domain"))]:
            raise RuntimeError("duplicate common-CRN metric image")
        if (
            not isinstance(row.get("psnr"), (int, float))
            or not math.isfinite(float(row["psnr"]))
            or not row.get("crn_bundle_sha256")
        ):
            raise RuntimeError("invalid terminal pathology image metric")
        grouped[str(row.get("domain"))][key] = row
    if len(grouped) != 6 or any(
        len(rows) != expected_count for rows in grouped.values()
    ):
        raise RuntimeError(f"terminal pathology e{epoch} lacks six complete domains")
    if any(
        {int(key[1]) for key in rows} != set(range(expected_count))
        for rows in grouped.values()
    ):
        raise RuntimeError(f"terminal pathology e{epoch} image order is incomplete")
    return dict(grouped)


def _load_metrics(
    cells: dict[str, dict[int, dict[str, Path]]],
) -> tuple[dict, list[dict]]:
    values = {}
    evidence = []
    environment = None
    evaluator = None
    for probe_id, epochs in cells.items():
        probe = PROBES[probe_id]
        values[probe_id] = {}
        for epoch, paths in epochs.items():
            receipt_path, metric_path = paths["receipt"], paths["metric"]
            if not receipt_path.is_file() or not metric_path.is_file():
                raise RuntimeError(
                    f"missing posthoc metric binding: {probe_id} e{epoch}"
                )
            receipt = _read_json(receipt_path)
            metric = _read_json(metric_path)
            if (
                receipt.get("schema") != UNIFIED_RECEIPT_SCHEMA
                or receipt.get("status") != "PASS_UNIFIED_READ_ONLY_EVALUATION"
                or receipt.get("lane_id") != probe["lane_id"]
                or receipt.get("source_host_label") != probe["source_host_label"]
                or int(receipt.get("epoch", -1)) != epoch
                or Path(str(receipt.get("metric", ""))).resolve() != metric_path
                or file_sha256(metric_path) != receipt.get("metric_sha256")
                or receipt.get("evaluation_schema") != EVALUATION_SCHEMA
                or receipt.get("evaluation_bundle_fingerprint")
                != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
                or receipt.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
                or receipt.get("training_checkpoint_read_only") is not True
                or receipt.get("paired_metric_control") is not False
                or receipt.get("cross_host_training_delta_merged") is not False
                or receipt.get("confirmation20_opened") is not False
            ):
                raise RuntimeError(f"invalid unified metric receipt: {receipt_path}")
            if (
                metric.get("lane_id") != probe["lane_id"]
                or metric.get("source_host_label") != probe["source_host_label"]
                or metric.get("manifest_sha256") != EXPECTED_MANIFEST_SHA256
                or metric.get("source_checkpoint_sha256")
                != receipt.get("source_checkpoint_sha256")
                or metric.get("training_protocol_fingerprint")
                != receipt.get("training_protocol_fingerprint")
                or metric.get("unified_environment")
                != receipt.get("unified_environment")
                or metric.get("unified_evaluator_protocol_fingerprint")
                != receipt.get("unified_evaluator_protocol_fingerprint")
            ):
                raise RuntimeError(f"unified metric lane mismatch: {metric_path}")
            images = _metric_images(metric, epoch=epoch)
            current_environment = receipt.get("unified_environment")
            current_evaluator = receipt.get("unified_evaluator_protocol_fingerprint")
            if environment is None:
                environment, evaluator = current_environment, current_evaluator
            elif current_environment != environment or current_evaluator != evaluator:
                raise RuntimeError(
                    "posthoc labels were not produced by one evaluator runtime"
                )
            values[probe_id][epoch] = images
            evidence.append(
                {
                    "probe_id": probe_id,
                    "epoch": epoch,
                    "metric_receipt": str(receipt_path),
                    "metric_receipt_sha256": file_sha256(receipt_path),
                    "metric_sha256": file_sha256(metric_path),
                }
            )
    return values, evidence


def _common70_domain_psnr(
    metrics: dict[int, dict[str, dict[tuple, dict]]], domain: str
) -> dict[int, float]:
    reference = metrics[150][domain]
    reference_keys = set(reference)
    if len(reference_keys) != 70 or set(metrics[100][domain]) != reference_keys:
        raise RuntimeError(f"e100/e150 discovery identities differ for {domain}")
    if not reference_keys.issubset(metrics[200][domain]):
        raise RuntimeError(f"e200 does not contain the common discovery70 for {domain}")
    for epoch in AUDIT_EPOCHS:
        for key in reference_keys:
            if metrics[epoch][domain][key].get("crn_bundle_sha256") != reference[
                key
            ].get("crn_bundle_sha256"):
                raise RuntimeError(f"CRN identity changed for {domain} {key} e{epoch}")
    return {
        epoch: float(
            np.mean(
                [
                    float(metrics[epoch][domain][key]["psnr"])
                    for key in sorted(reference_keys)
                ]
            )
        )
        for epoch in AUDIT_EPOCHS
    }


def adjudicate_terminal_pathology(
    *, audit_root: Path, metric_bindings: Path, destination: Path
) -> dict[str, Any]:
    audits, audit_evidence = _load_all_audits(Path(audit_root).resolve())
    # This is the intentional information barrier: paired files are not even
    # parsed until every target-blind audit and state/RNG receipt has passed.
    binding_path = Path(metric_bindings).resolve()
    binding = _read_json(binding_path)
    metrics, metric_evidence = _load_metrics(_binding_cells(binding))

    cells = []
    common_domains = None
    for probe_id, probe in PROBES.items():
        domains = set(audits[probe_id][100])
        if domains != set(audits[probe_id][150]) or domains != set(
            audits[probe_id][200]
        ):
            raise RuntimeError(f"audit domain identities changed: {probe_id}")
        if common_domains is None:
            common_domains = domains
        elif domains != common_domains:
            raise RuntimeError("terminal audit domains differ across probes")
        for domain in sorted(domains):
            early = audits[probe_id][100][domain]
            precursor = audits[probe_id][150][domain]
            terminal = audits[probe_id][200][domain]
            if not (early["stem"] == precursor["stem"] == terminal["stem"]):
                raise RuntimeError(
                    f"terminal audit sample identity changed: {probe_id} {domain}"
                )
            psnr = _common70_domain_psnr(metrics[probe_id], domain)
            ratios = {
                "terminal_increment_trace": _safe_ratio(
                    precursor["terminal_increment_trace"],
                    early["terminal_increment_trace"],
                ),
                "terminal_increment_effective_rank": _safe_ratio(
                    precursor["terminal_increment_effective_rank"],
                    early["terminal_increment_effective_rank"],
                ),
                "late_rollout_jacobian_max": _safe_ratio(
                    precursor["late_rollout_jacobian_max"],
                    early["late_rollout_jacobian_max"],
                ),
                "late_finite_perturbation_gain_max": _safe_ratio(
                    precursor["late_finite_perturbation_gain_max"],
                    early["late_finite_perturbation_gain_max"],
                ),
            }
            spectral = bool(
                ratios["terminal_increment_trace"] <= SPECTRAL_COLLAPSE_RATIO
                and ratios["terminal_increment_effective_rank"]
                <= SPECTRAL_COLLAPSE_RATIO
            )
            amplification = bool(
                ratios["late_rollout_jacobian_max"] >= AMPLIFICATION_RATIO
                and ratios["late_finite_perturbation_gain_max"] >= AMPLIFICATION_RATIO
            )
            future_delta = float(psnr[200] - psnr[150])
            future_decline = future_delta <= FUTURE_DECLINE_DB
            cells.append(
                {
                    "probe_id": probe_id,
                    "lane_id": probe["lane_id"],
                    "domain": domain,
                    "diagnostic_window": "e100_to_e150",
                    "future_label_window": "e150_to_e200_common_discovery70_replicate0_nfe5",
                    "diagnostic_ratios": ratios,
                    "spectral_collapse_precursor": spectral,
                    "perturbation_amplification_precursor": amplification,
                    "future_psnr_delta_db": future_delta,
                    "future_decline_label": future_decline,
                }
            )

    support = {}
    confirmed = []
    for mechanism, field in (
        ("spectral_collapse", "spectral_collapse_precursor"),
        ("perturbation_amplification", "perturbation_amplification_precursor"),
    ):
        rows = [row for row in cells if row[field] and row["future_decline_label"]]
        methods = sorted({row["probe_id"] for row in rows})
        domains = sorted({row["domain"] for row in rows})
        passed = (
            len(methods) >= MIN_SUPPORT_METHODS and len(domains) >= MIN_SUPPORT_DOMAINS
        )
        support[mechanism] = {
            "status": "PASS_SHARED_LEADING_SIGNAL"
            if passed
            else "INSUFFICIENT_SHARED_SUPPORT",
            "support_cell_count": len(rows),
            "support_methods": methods,
            "support_domains": domains,
        }
        if passed:
            confirmed.append(mechanism)

    result = {
        "schema": SCHEMA,
        "status": (
            "TERMINAL_PATHOLOGY_CONFIRMED_FOR_DERIVATION"
            if confirmed
            else "TERMINAL_PATHOLOGY_NOT_CONFIRMED_DO_NOT_ADD_MODULE"
        ),
        "terminal_pathology_confirmed": bool(confirmed),
        "confirmed_mechanisms": confirmed,
        "fixed_thresholds": {
            "spectral_collapse_ratio_max": SPECTRAL_COLLAPSE_RATIO,
            "amplification_ratio_min": AMPLIFICATION_RATIO,
            "future_psnr_decline_db_max": FUTURE_DECLINE_DB,
            "minimum_support_methods": MIN_SUPPORT_METHODS,
            "minimum_support_domains": MIN_SUPPORT_DOMAINS,
        },
        "lead_lag_design": {
            "target_blind_diagnostic_window": "e100_to_e150",
            "paired_future_label_window": "e150_to_e200",
            "paired_comparison_subset": "common_discovery70_replicate0_nfe5",
            "thresholds_fitted_to_results": False,
        },
        "mechanism_support": support,
        "cells": cells,
        "audit_evidence": audit_evidence,
        "metric_binding": str(binding_path),
        "metric_binding_sha256": file_sha256(binding_path),
        "metric_evidence": metric_evidence,
        "all_target_blind_audits_validated_before_paired_metric_read": True,
        "paired_labels_attached_posthoc": True,
        "training_control_authorized": False,
        "algorithm_or_module_automatically_started": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
        "next_action": (
            "write a derivation card; confirmation is not direct implementation authority"
            if confirmed
            else "do not add a terminal repair module"
        ),
    }
    destination = Path(destination).resolve()
    if destination.is_file():
        existing = _read_json(destination)
        if existing != result:
            raise RuntimeError(
                "terminal pathology adjudication already exists and differs"
            )
    else:
        write_json(destination, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--audit-root", type=Path, required=True)
    value.add_argument("--metric-bindings", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    result = adjudicate_terminal_pathology(
        audit_root=args.audit_root,
        metric_bindings=args.metric_bindings,
        destination=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
