"""Build incomplete-safe paper tables without best-checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.local_route1.runtime import write_json

from .protocol import REQUIRED_PAPER_TABLE, lane_spec, load_protocol


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(output_root: Path, lane: str, epoch: int) -> dict | None:
    path = Path(output_root) / "lanes" / lane / "metrics" / f"e{epoch:03d}.json"
    return _read(path) if path.is_file() else None


def _domain_delta(method: dict, plain: dict) -> dict:
    return {
        domain: {
            "psnr": method["domains"][domain]["psnr"] - plain["domains"][domain]["psnr"],
            "ssim": method["domains"][domain]["ssim"] - plain["domains"][domain]["ssim"],
            "lpips": (
                None if method["domains"][domain]["lpips"] is None or plain["domains"][domain]["lpips"] is None
                else method["domains"][domain]["lpips"] - plain["domains"][domain]["lpips"]
            ),
        }
        for domain in sorted(plain["domains"])
    }


def _crn_identity(metric: dict) -> list[tuple]:
    return [
        (
            row.get("domain"), row.get("stem"), int(row.get("order", -1)),
            int(row.get("replicate", -1)), int(row.get("nfe", -1)),
            row.get("crn_bundle_sha256"),
        )
        for row in metric.get("images", []) if isinstance(row, dict)
    ]


def _candidate_definitions(output_root: Path) -> list[dict]:
    definitions = []
    root = Path(output_root) / "candidate_locks"
    if not root.is_dir():
        return definitions
    cohort_path = Path(output_root) / "gates" / "UNIFIED_EVALUATION_COHORT.json"
    unified = False
    if cohort_path.is_file():
        cohort = _read(cohort_path)
        unified = (
            cohort.get("schema") == "final-unsb-paper-unified-evaluation-cohort-v1"
            and cohort.get("status") == "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT"
            and cohort.get("cross_host_training_delta_merged") is False
            and cohort.get("confirmation20_opened") is False
        )
    for path in sorted(root.glob("*/CANDIDATE_LOCK.json")):
        lock = _read(path)
        if (
            lock.get("schema") != "final-unsb-paper-candidate-lock-v1"
            or lock.get("status") != "PASS_FULL_DATA_CANDIDATE_LOCK"
            or lock.get("full_data_authorized") is not False
            or lock.get("paired_metric_control") is not False
            or lock.get("confirmation20_opened") is not False
        ):
            raise RuntimeError(f"invalid dynamic paper candidate lock: {path}")
        candidate_id = str(lock["candidate_id"])
        definitions.append({
            "lane_id": candidate_id,
            "role": "evidence-locked full-data paper candidate",
            "backend": "internal",
            "family": "unsb",
            "metrics_root": Path(output_root),
            "plain_root": (
                Path(output_root) if unified else
                Path(lock["parent_paper"]["parent_output"])
            ),
            "comparison_scope": (
                "one_container_unified_evaluation" if unified else
                "same_host_cross_code_runtime_gate"
            ),
            "candidate_lock": str(path.resolve()),
            "first_wave": False,
        })
    return definitions


def adjudicate(output_root: Path) -> dict:
    protocol = load_protocol()
    output_root = Path(output_root)
    definitions = [{
        "lane_id": "input", "role": "evaluation-only degraded input reference",
        "backend": "evaluation_only", "family": "input",
        "metrics_root": output_root, "plain_root": output_root,
        "comparison_scope": "one_container_unified_evaluation",
        "candidate_lock": None,
        "first_wave": True,
    }]
    definitions.extend([
        {
            "lane_id": row["id"], "role": row["role"],
            "backend": row["backend"], "family": lane_spec(row["id"], protocol).family,
            "metrics_root": output_root, "plain_root": output_root,
            "comparison_scope": (
                "same_runtime_output_root" if lane_spec(row["id"], protocol).family == "unsb"
                else "standalone_fixed_protocol"
            ),
            "candidate_lock": None,
            "first_wave": bool(row.get("first_wave", True)),
        }
        for row in protocol["lanes"]
    ])
    definitions.extend(_candidate_definitions(output_root))
    lanes = []
    for definition in definitions:
        lane = definition["lane_id"]
        metrics_root = Path(definition["metrics_root"])
        plain_root = Path(definition["plain_root"])
        terminal = _metric(metrics_root, lane, 200)
        supervisor_path = metrics_root / "gates" / f"SUPERVISOR_{lane}.json"
        supervisor = _read(supervisor_path) if supervisor_path.is_file() else {}
        authorization_path = (
            metrics_root / "gates" / f"CANDIDATE_AUTHORIZATION_{lane}.json"
        )
        if terminal is not None:
            status = "COMPLETE_E200"
        elif lane == "ddsb" and definition["backend"] == "external_locked":
            status = "REPRODUCTION_INCOMPLETE"
        elif str(supervisor.get("status", "")).startswith("BLOCKED"):
            status = "ENGINEERING_BLOCKED"
        elif definition["candidate_lock"] is not None:
            status = (
                "INCOMPLETE_AUTHORIZED" if authorization_path.is_file() else
                "EVIDENCE_LOCKED_NOT_AUTHORIZED"
            )
        elif definition["first_wave"] is False:
            status = "DEFERRED_NOT_FIRST_WAVE"
        else:
            status = "INCOMPLETE"
        entry = {
            "lane_id": lane,
            "role": definition["role"],
            "backend": definition["backend"],
            "comparison_scope": definition["comparison_scope"],
            "candidate_lock": definition["candidate_lock"],
            "status": status,
            "first_wave": bool(definition["first_wave"]),
            "supervisor_status": supervisor.get("status"),
            "terminal": None if terminal is None else {
                key: terminal[key] for key in ("macro_psnr", "macro_ssim", "macro_lpips")
            },
        }
        if terminal is not None:
            entry["terminal"]["stochasticity"] = terminal.get("stochasticity")
            entry["terminal"]["count_per_domain"] = terminal.get("count_per_domain")
            entry["terminal"]["replicates"] = terminal.get("replicates")
        if definition["family"] == "unsb" and lane != "plain":
            trajectory = []
            crn_exact = True
            for epoch in protocol["training"]["late_epochs"]:
                method = _metric(metrics_root, lane, epoch)
                plain = _metric(plain_root, "plain", epoch)
                if method is None or plain is None:
                    continue
                matched = (
                    method.get("protocol_fingerprint") == plain.get("protocol_fingerprint")
                    and method.get("evaluation_input_sha256")
                    == plain.get("evaluation_input_sha256")
                    and _crn_identity(method) == _crn_identity(plain)
                    and method.get("confirmation20_opened") is False
                    and plain.get("confirmation20_opened") is False
                )
                crn_exact = crn_exact and matched
                deltas = _domain_delta(method, plain)
                psnr = [value["psnr"] for value in deltas.values()]
                trajectory.append({
                    "epoch": epoch,
                    "macro_psnr_delta": method["macro_psnr"] - plain["macro_psnr"],
                    "macro_ssim_delta": method["macro_ssim"] - plain["macro_ssim"],
                    "macro_lpips_delta": (
                        None if method["macro_lpips"] is None or plain["macro_lpips"] is None
                        else method["macro_lpips"] - plain["macro_lpips"]
                    ),
                    "positive_domains": sum(value > 0 for value in psnr),
                    "worst_domain_delta": min(psnr),
                    "domain_delta": deltas,
                    "crn_exact": matched,
                    "candidate_macro_psnr": method["macro_psnr"],
                    "plain_macro_psnr": plain["macro_psnr"],
                })
            entry["late_trajectory"] = trajectory
            if len(trajectory) == 3:
                lpips = [row["macro_lpips_delta"] for row in trajectory]
                late_mean = float(np.mean([row["macro_psnr_delta"] for row in trajectory]))
                terminal_delta = float(trajectory[-1]["macro_psnr_delta"])
                candidate_change = float(
                    trajectory[-1]["candidate_macro_psnr"]
                    - trajectory[0]["candidate_macro_psnr"]
                )
                plain_change = float(
                    trajectory[-1]["plain_macro_psnr"]
                    - trajectory[0]["plain_macro_psnr"]
                )
                plain_collapse_guard = not (
                    candidate_change < -0.3 and plain_change < -0.3
                )
                accepted = (
                    crn_exact and late_mean > 0 and terminal_delta > 0
                    and sum(row["positive_domains"] >= 4 for row in trajectory) >= 2
                    and float(np.mean([row["worst_domain_delta"] for row in trajectory])) > -1.0
                    and float(np.mean([row["macro_ssim_delta"] for row in trajectory])) >= 0
                    and all(value is not None and value <= 0 for value in lpips)
                    and plain_collapse_guard
                )
                entry["scientific_gate"] = {
                    "status": "PASS" if accepted else "FAIL",
                    "late_three_macro_psnr_delta": late_mean,
                    "e200_macro_psnr_delta": terminal_delta,
                    "crn_exact_at_all_late_points": crn_exact,
                    "candidate_e150_to_e200_change_db": candidate_change,
                    "plain_e150_to_e200_change_db": plain_change,
                    "plain_collapse_guard": (
                        "PASS_NOT_PLAIN_COLLAPSE" if plain_collapse_guard else
                        "FAIL_RELATIVE_ADVANTAGE_COINCIDES_WITH_BOTH_COLLAPSING"
                    ),
                    "confirmation20_opened": False,
                }
        lanes.append(entry)
    cohort_path = output_root / "gates" / "UNIFIED_EVALUATION_COHORT.json"
    cohort = _read(cohort_path) if cohort_path.is_file() else {}
    unified_cohort_pass = (
        cohort.get("schema") == "final-unsb-paper-unified-evaluation-cohort-v1"
        and cohort.get("status") == "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT"
        and cohort.get("cross_host_training_delta_merged") is False
        and cohort.get("confirmation20_opened") is False
        and cohort.get("required_lanes") == list(REQUIRED_PAPER_TABLE)
    )
    complete_required = unified_cohort_pass and all(
        next(row for row in lanes if row["lane_id"] == lane)["status"] == "COMPLETE_E200"
        for lane in REQUIRED_PAPER_TABLE
    )
    result = {
        "schema": "final-unsb-paper-results-v1",
        "status": "FIRST_WAVE_COMPLETE" if complete_required else "FIRST_WAVE_INCOMPLETE",
        "primary_epoch": 200,
        "best_checkpoint_selection": False,
        "unified_evaluation_cohort_pass": unified_cohort_pass,
        "unified_evaluation_cohort": (
            None if not cohort_path.is_file() else str(cohort_path.resolve())
        ),
        "lanes": lanes,
        "cross_5090_delta_merged": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "PAPER_RESULTS.json", result)
    algorithm_set = {
        "schema": "final-unsb-paper-algorithm-set-v1",
        "status": (
            "FIRST_WAVE_EVIDENCE_READY_CANDIDATES_PENDING"
            if result["status"] == "FIRST_WAVE_COMPLETE" else "INCOMPLETE"
        ),
        "accepted_algorithms": [
            row["lane_id"] for row in lanes
            if row.get("scientific_gate", {}).get("status") == "PASS"
        ],
        "scientific_failures": [
            row["lane_id"] for row in lanes
            if row.get("scientific_gate", {}).get("status") == "FAIL"
        ],
        "reproduction_incomplete": [
            row["lane_id"] for row in lanes
            if row["status"] == "REPRODUCTION_INCOMPLETE"
        ],
        "engineering_blocked": [
            row["lane_id"] for row in lanes
            if row["status"] == "ENGINEERING_BLOCKED"
        ],
        "deferred_not_first_wave": [
            row["lane_id"] for row in lanes
            if row["status"] == "DEFERRED_NOT_FIRST_WAVE"
        ],
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "multiple_algorithms_allowed": True,
        "confirmation20_opened": False,
    }
    write_json(output_root / "ALGORITHM_SET.json", algorithm_set)
    return {"results": result, "algorithm_set": algorithm_set}
