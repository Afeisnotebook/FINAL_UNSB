"""Fail-closed audit of the retrieved terminal route-1 research delivery.

The terminal materializer already hashes every published artifact.  This
module adds the missing Goal-level check: it verifies that the retrieved
delivery actually proves the long-horizon scientific requirements instead of
merely being a self-consistent set of files.  It is posthoc-only and cannot
train, select a checkpoint, open confirmation data, or merge host deltas.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from operations.local_route1_complete_final_result_relay import (
    EXTRA_FILES,
    validate_local_delivery,
)
from research.local_route1.complete_frontier_final_delivery import (
    ALTERNATES_SCHEMA,
    CANDIDATE_SCHEMA,
    POINTER,
    RESEARCH_FRONTIER_SCHEMA,
    RESULTS_SCHEMA,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-goal-completion-audit-v1"
RELAY_MANIFEST_SCHEMA = "final-unsb-route1-complete-final-relay-manifest-v1"
EXPECTED_PROBES = {"dt", "hj", "hnek"}
LATE_EPOCHS = {150, 175, 200}
REQUIRED_MECHANISM_ROLES = {
    "proposal_only", "observable_only", "projected_or_full",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _substantive(value: Any) -> bool:
    if _nonempty(value):
        return True
    if isinstance(value, (dict, list, tuple, set)):
        return bool(value)
    return value is not None


def _posthoc_boundary(value: dict[str, Any], *, label: str) -> None:
    _require(
        value.get("confirmation20_opened") is False,
        f"{label} does not prove confirmation20 remained closed",
    )
    for key in (
        "paired_controller_access",
        "paired_metrics_used_for_formula_or_training_control",
        "paired_metrics_used_for_training_or_control",
    ):
        if key in value:
            _require(value[key] is False, f"{label} violates target blindness: {key}")
    if "cross_host_deltas_merged" in value:
        _require(
            value["cross_host_deltas_merged"] is False,
            f"{label} merged cross-host deltas",
        )


def _domain_trajectory(rows: Any, *, label: str) -> dict[str, Any]:
    _require(isinstance(rows, list) and bool(rows), f"{label} trajectory is empty")
    epochs = [int(row.get("data_epoch", -1)) for row in rows if isinstance(row, dict)]
    _require(len(epochs) == len(rows), f"{label} trajectory row is malformed")
    _require(epochs == sorted(set(epochs)), f"{label} epochs are not ordered and unique")
    _require(epochs[-1] == 200, f"{label} does not terminate at e200")
    _require(LATE_EPOCHS.issubset(epochs), f"{label} lacks the fixed late-three epochs")
    domain_sets: list[set[str]] = []
    for row in rows:
        epoch = int(row["data_epoch"])
        _require(
            int(row.get("updates", -1)) == epoch * 150,
            f"{label} e{epoch} does not use the frozen data-epoch/update conversion",
        )
        domains = row.get("domains")
        _require(
            isinstance(domains, dict) and len(domains) == 6,
            f"{label} e{epoch} does not contain six domains",
        )
        domain_sets.append(set(domains))
        for domain, values in domains.items():
            _require(isinstance(values, dict), f"{label} {domain} is malformed")
            _require(
                set(values) == {"candidate", "plain", "delta"},
                f"{label} {domain} lacks absolute/relative decomposition",
            )
            for role in ("candidate", "plain", "delta"):
                metrics = values[role]
                _require(
                    isinstance(metrics, dict)
                    and {"psnr", "ssim", "lpips"}.issubset(metrics),
                    f"{label} {domain}:{role} lacks PSNR/SSIM/LPIPS",
                )
        if epoch in LATE_EPOCHS:
            for macro_key in ("candidate_macro", "plain_macro", "macro_delta"):
                metrics = row.get(macro_key)
                _require(
                    isinstance(metrics, dict)
                    and all(metrics.get(name) is not None for name in ("psnr", "ssim", "lpips")),
                    f"{label} e{epoch} lacks a complete late {macro_key}",
                )
    _require(
        all(domains == domain_sets[0] for domains in domain_sets),
        f"{label} domain identities change across epochs",
    )
    return {
        "terminal_data_epoch": 200,
        "terminal_updates": 30000,
        "trajectory_epochs": epochs,
        "domains": sorted(domain_sets[0]),
    }


def _mechanism_evidence(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "selected candidate lacks mechanism evidence")
    kind = value.get("kind")
    if kind == "same_host_three_role_ablation":
        roles = value.get("roles")
        _require(
            isinstance(roles, dict) and set(roles) == REQUIRED_MECHANISM_ROLES,
            "selected same-host ablation does not contain all three roles",
        )
    elif kind == "source_host_three_role_ablation":
        parent = value.get("roles")
        roles = parent.get("roles") if isinstance(parent, dict) else None
        _require(
            isinstance(roles, dict) and set(roles) == REQUIRED_MECHANISM_ROLES,
            "selected source-host ablation does not contain all three roles",
        )
        _require(
            value.get("used_for_4090_candidate_ranking") is False,
            "source-host ablation was used as a cross-host numeric rank",
        )
    elif kind == "same_host_component_factorial_plus_source_parent_ablation":
        components = value.get("components")
        _require(
            isinstance(components, dict)
            and set(components) == {
                "plain", "conditional_sampling_only",
                "residual_feasible_barrier_only", "combined_full",
            },
            "selected synthesis lacks the registered four-component evidence",
        )
        _require(
            value.get("used_for_cross_host_delta_ranking") is False,
            "selected synthesis merged cross-host component deltas",
        )
    else:
        raise RuntimeError(f"unknown selected mechanism-evidence kind: {kind}")
    return {"kind": kind, "roles_complete": True}


def _candidate(value: dict[str, Any], selected_id: str) -> dict[str, Any]:
    _require(value.get("schema") == CANDIDATE_SCHEMA, "candidate schema changed")
    _require(value.get("status") == "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE", "candidate is not terminal")
    _require(value.get("candidate_id") == selected_id, "selected candidate identity differs")
    _require(value.get("canonical_candidate_is_action_priority_only") is True, "candidate became scientifically exclusive")
    _require(value.get("algorithm_discovery_collapsed_to_single_candidate") is False, "algorithm discovery collapsed to one candidate")
    _posthoc_boundary(value, label="candidate")
    checkpoint = value.get("selected_fixed_checkpoint")
    _require(
        checkpoint == {"data_epoch": 200, "updates": 30000, "best_checkpoint_selection": False},
        "candidate did not use the frozen e200 checkpoint",
    )
    _require(value.get("target_data_epochs") == 200, "candidate horizon is not e200")
    _require(value.get("target_updates") == 30000, "candidate update horizon is not 30000")
    _require(value.get("training_batch_size") == 1, "candidate batch size changed")
    mathematics = value.get("mathematics")
    _require(isinstance(mathematics, dict), "candidate lacks mathematics")
    for key in (
        "unsb_object", "formula", "identity_or_unbiased_condition",
        "target_inaccessibility_proof",
    ):
        _require(_nonempty(mathematics.get(key)), f"candidate mathematics lacks {key}")
    complexity = value.get("complexity")
    _require(isinstance(complexity, dict), "candidate lacks complexity evidence")
    for key in ("compute_cost", "memory_cost", "recovery_state_cost"):
        _require(_nonempty(complexity.get(key)), f"candidate complexity lacks {key}")
    risk = value.get("risk")
    _require(
        isinstance(risk, dict)
        and _substantive(risk.get("expected_applicable_state"))
        and _nonempty(risk.get("falsifying_experiment"))
        and risk.get("single_seed_only") is True
        and risk.get("cross_seed_stability_claimed") is False,
        "candidate risk/cross-seed boundary is incomplete",
    )
    source_files = value.get("source_files")
    _require(isinstance(source_files, list) and bool(source_files), "candidate lacks source files")
    reproduction = value.get("reproduction")
    _require(
        isinstance(reproduction, dict)
        and _nonempty(reproduction.get("seed2026_e200"))
        and reproduction.get("deferred_seed_validation") == [2027, 2028],
        "candidate reproduction/deferred-seed contract is incomplete",
    )
    trajectory = _domain_trajectory(
        value.get("absolute_relative_domain_trajectory"), label="selected candidate",
    )
    mechanism = _mechanism_evidence(value.get("mechanism_evidence"))
    return {
        "candidate_id": selected_id,
        "classification": value.get("classification"),
        "trajectory": trajectory,
        "mechanism_evidence": mechanism,
    }


def _historical_evidence(value: Any, frontier_ids: set[str]) -> dict[str, Any]:
    _require(isinstance(value, dict), "historical evidence is missing")
    _require(
        value.get("status") == "COMPLETE_LONG_HORIZON_PROBE_CAUSAL_AND_DERIVATION_EVIDENCE",
        "historical evidence is not terminal",
    )
    _posthoc_boundary(value, label="historical evidence")
    anchors = value.get("dt_hj_hnek_anchor_trajectories")
    summaries = anchors.get("summaries") if isinstance(anchors, dict) else None
    _require(
        isinstance(summaries, list)
        and {str(row.get("probe_id", "")) for row in summaries} == EXPECTED_PROBES
        and all(row.get("complete_e200") is True for row in summaries),
        "DT/HJ/HNEK complete-e200 evidence is missing",
    )
    proxy = value.get("proxy_calibration")
    _require(
        isinstance(proxy, dict)
        and proxy.get("status") == "CALIBRATED"
        and bool(set(proxy.get("passing_probes", [])).intersection({"hj", "hnek"})),
        "authoritative proxy is not calibrated",
    )
    matrix = value.get("long_causal_matrix_summary")
    _require(
        isinstance(matrix, dict)
        and matrix.get("status") == "COMPLETE_CAUSAL_AUDIT"
        and int(matrix.get("reversal_rows", 0)) == 474
        and int(matrix.get("sampling_variance_rows", 0)) == 140
        and matrix.get("paired_labels_joined_only_after_branches") is True
        and matrix.get("paired_metrics_accessed_by_controller") is False,
        "authoritative 474/140 long causal atlas is incomplete",
    )
    ledger = value.get("hypothesis_ledger_summary")
    ledger_ids = {
        str(row.get("candidate_id", "")) for row in ledger
        if isinstance(row, dict)
    } if isinstance(ledger, list) else set()
    _require(frontier_ids.issubset(ledger_ids), "frontier candidates are missing from the hypothesis ledger")
    return {
        "probes": sorted(EXPECTED_PROBES),
        "reversal_rows": 474,
        "sampling_variance_rows": 140,
        "ledger_candidate_count": len(ledger_ids),
    }


def audit_complete_delivery(delivery: Path) -> dict[str, Any]:
    """Validate a fully retrieved terminal delivery and return Goal proof."""

    delivery = Path(delivery).resolve()
    pointer = validate_local_delivery(delivery)
    relay = _read_json(delivery / "RELAY_MANIFEST.json")
    _require(
        relay.get("schema") == RELAY_MANIFEST_SCHEMA
        and relay.get("status") == "COMPLETE_EXACT_FINAL_DELIVERY_RETRIEVED",
        "terminal delivery was not retrieved by the durable exact relay",
    )
    _posthoc_boundary(relay, label="relay manifest")
    for name, expected in relay.get("file_sha256", {}).items():
        _require(file_sha256(delivery / name) == expected, f"relay manifest hash changed: {name}")

    candidate = _read_json(delivery / "CANDIDATE.json")
    alternates = _read_json(delivery / "ALTERNATES.json")
    results = _read_json(delivery / "RESULTS.json")
    research = _read_json(delivery / "RESEARCH_FRONTIER.json")
    report_path = delivery / "FINAL_ROUTE1_REPORT.md"
    selected_id = str(pointer["selected_candidate_id"])

    candidate_proof = _candidate(candidate, selected_id)
    _require(alternates.get("schema") == ALTERNATES_SCHEMA, "alternates schema changed")
    _require(alternates.get("status") == "COMPLETE", "alternates are not terminal")
    _posthoc_boundary(alternates, label="alternates")
    alternate_rows = alternates.get("alternates")
    _require(
        isinstance(alternate_rows, list)
        and len(alternate_rows) == 2
        and len({str(row.get("candidate_id", "")) for row in alternate_rows}) == 2
        and all(str(row.get("candidate_id", "")) != selected_id for row in alternate_rows),
        "delivery does not contain exactly two distinct alternates",
    )

    _require(results.get("schema") == RESULTS_SCHEMA, "results schema changed")
    _require(results.get("status") == "COMPLETE", "results are not terminal")
    _require(results.get("selected_candidate_id") == selected_id, "results selected identity differs")
    _posthoc_boundary(results, label="results")
    _require(results.get("selection_seeds") == [2026], "results selection seed changed")
    _require(results.get("deferred_seed_validation") == [2027, 2028], "results deferred seeds changed")
    _require(results.get("cross_seed_stability_claimed") is False, "results claim cross-seed stability")

    _require(research.get("schema") == RESEARCH_FRONTIER_SCHEMA, "research-frontier schema changed")
    _require(
        research.get("status") == "COMPLETE_MULTI_CANDIDATE_ROUTE1_RESEARCH_FRONTIER",
        "research frontier is not terminal",
    )
    _require(research.get("action_priority_candidate_id") == selected_id, "research-frontier priority differs")
    _require(research.get("canonical_candidate_is_action_priority_only") is True, "research frontier made priority exclusive")
    _require(research.get("algorithm_discovery_collapsed_to_single_candidate") is False, "research frontier collapsed to one algorithm")
    _posthoc_boundary(research, label="research frontier")
    rows_4090 = research.get("remote4090_same_host_frontier")
    rows_5090 = research.get("remote5090_source_host_frontier")
    _require(isinstance(rows_4090, list) and len(rows_4090) >= 3, "4090 complete frontier lacks primary plus two alternates")
    _require(isinstance(rows_5090, list) and bool(rows_5090), "5090 mechanism frontier is empty")
    ids_4090 = {str(row.get("candidate_id", "")) for row in rows_4090}
    _require(len(ids_4090) == len(rows_4090) and selected_id in ids_4090, "4090 frontier identities are invalid")
    evidence_4090 = research.get("remote4090_complete_candidate_evidence")
    evidence_5090 = research.get("remote5090_complete_candidate_evidence")
    _require(
        isinstance(evidence_4090, list)
        and {str(row.get("candidate_id", "")) for row in evidence_4090} == ids_4090,
        "4090 frontier lacks complete per-candidate evidence",
    )
    trajectory_proofs = {}
    for row in evidence_4090:
        candidate_id = str(row["candidate_id"])
        trajectory_proofs[f"remote4090:{candidate_id}"] = _domain_trajectory(
            row.get("absolute_relative_domain_trajectory"),
            label=f"remote4090 {candidate_id}",
        )
        for key in ("receipt", "trajectory", "derivation_card", "implementation"):
            _require(isinstance(row.get(key), dict), f"remote4090 {candidate_id} lacks {key}")
    _require(isinstance(evidence_5090, list) and bool(evidence_5090), "5090 frontier lacks complete evidence")
    evidence_5090_ids = {str(row.get("candidate_id", "")) for row in evidence_5090}
    _require(
        {str(row.get("candidate_id", "")) for row in rows_5090}.issubset(evidence_5090_ids),
        "5090 ranked frontier is missing from portable evidence",
    )
    for row in evidence_5090:
        candidate_id = str(row["candidate_id"])
        trajectory_proofs[f"remote5090:{candidate_id}"] = _domain_trajectory(
            row.get("absolute_relative_domain_trajectory"),
            label=f"remote5090 {candidate_id}",
        )
        for key in ("receipt", "trajectory", "derivation_card", "implementation"):
            _require(isinstance(row.get(key), dict), f"remote5090 {candidate_id} lacks {key}")
    historical = _historical_evidence(
        research.get("historical_probe_causal_and_derivation_evidence"), ids_4090,
    )

    report = report_path.read_text(encoding="utf-8")
    for phrase in (
        "证据边界", "工程失败", "proxy", "未验证", "RESEARCH_FRONTIER.json",
    ):
        _require(phrase in report, f"final report lacks required boundary section: {phrase}")

    requirements = [
        {"id": 1, "status": "PROVEN", "evidence": "durable exact relay manifest and immutable hashes"},
        {"id": 2, "status": "PROVEN", "evidence": "authoritative HJ/HNEK proxy calibration embedded"},
        {"id": 3, "status": "PROVEN", "evidence": "DT/HJ/HNEK complete-e200 trajectories embedded"},
        {"id": 4, "status": "PROVEN", "evidence": "474 reversal and 140 sampling-variance rows embedded"},
        {"id": 5, "status": "PROVEN", "evidence": "selected evidence-derived algorithm math, implementation and matched e200"},
        {"id": 6, "status": "PROVEN", "evidence": "hypothesis ledger and complete frontier preserve revision lineage"},
        {"id": 7, "status": "PROVEN", "evidence": "one action priority, two alternates and complete research frontier"},
        {"id": 8, "status": "PROVEN", "evidence": "six-domain absolute/relative trajectories, cost, risk and reproduction"},
        {"id": 9, "status": "PENDING_REPOSITORY_FINAL_COMMIT", "evidence": "terminal artifacts verified; compact final adjudication must still be committed and pushed"},
        {"id": 10, "status": "PROVEN", "evidence": "report separates scientific, engineering, proxy and untested boundaries"},
    ]
    return {
        "schema": SCHEMA,
        "status": "ROUTE1_TERMINAL_ARTIFACTS_PROVEN_FINAL_GIT_COMMIT_REQUIRED",
        "selected_candidate_id": selected_id,
        "candidate": candidate_proof,
        "alternates": [str(row["candidate_id"]) for row in alternate_rows],
        "research_frontier_unique_candidate_count": pointer[
            "research_frontier_unique_candidate_count"
        ],
        "research_frontier_host_scoped_row_count": pointer[
            "research_frontier_host_scoped_row_count"
        ],
        "trajectory_proofs": trajectory_proofs,
        "historical_evidence": historical,
        "requirements": requirements,
        "terminal_artifact_requirements_proven": True,
        "final_repository_commit_and_push_required": True,
        "completion_claim_allowed": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        "source_delivery_sha256": {
            name: file_sha256(delivery / name)
            for name in (
                POINTER, "RELAY_MANIFEST.json", "CANDIDATE.json", "ALTERNATES.json",
                "RESULTS.json", "RESEARCH_FRONTIER.json", "FINAL_ROUTE1_REPORT.md",
                *EXTRA_FILES,
            )
        },
    }


def materialize_goal_completion_audit(delivery: Path, output: Path) -> dict[str, Any]:
    result = audit_complete_delivery(delivery)
    write_json(Path(output).resolve(), result)
    return result
