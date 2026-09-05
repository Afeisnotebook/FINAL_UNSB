"""Produce the fixed full-data complexity table and multi-algorithm portfolio.

This is the final unattended paper-discovery handoff. It waits for every
predeclared e200 evaluation/disposition, profiles the fixed e200 checkpoints in
one 4090 environment, and combines results without ever changing a running
experiment. AM-TNC retains its 4090A plain comparison. Proposal and ST-CGR use
the plain source named by the frozen deployment contract, but only after the
post-hoc dispositions prove that exact runtime relation at every late epoch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from operations.paper_aio_algorithm_evaluation_successor import (
    COMPLETE_STATUS as ALGORITHM_COMPLETE_STATUS,
    validate_local_export_lane,
)
from operations.paper_aio_unified_evaluation_successor import (
    COMPLETE_STATUS as FIRST_WAVE_COMPLETE_STATUS,
    StateHeartbeat,
    _acquire_lock,
    _read_json,
    _write_json,
    import_lane_path,
    imports_ready,
    parse_lane_source,
    validate_import_lane,
)
from research.paper_aio.protocol import (
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    file_sha256,
    protocol_fingerprint,
)
from research.paper_aio.runtime_relation import runtime_pair_passed


CONTRACT_SCHEMA = "final-unsb-paper-final-delivery-successor-contract-v3"
STATE_SCHEMA = "final-unsb-paper-final-delivery-successor-state-v2"
PORTFOLIO_SCHEMA = "final-unsb-paper-full-data-algorithm-portfolio-v1"
COMPLEXITY_SCHEMA = "final-unsb-paper-complexity-profile-v1"
DISPOSITION_SCHEMA = "final-unsb-paper-algorithm-disposition-v1"
COMPLETE_STATUS = "COMPLETE_SUCCESSOR_E200_FULL_DATA_PAPER_DISCOVERY_DELIVERY"
STCGR_ID = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"
FIRST_WAVE_LANES = ("plain", "proposal", "cut", "cyclegan")
COMPLEXITY_LANES = ("plain", "proposal", "cut", "cyclegan", "amtnc", STCGR_ID)
FIRST_WAVE_STATE_SCHEMA = "final-unsb-paper-unified-evaluation-successor-state-v2"
ALGORITHM_STATE_SCHEMA = "final-unsb-paper-algorithm-evaluation-successor-state-v1"
BASELINE_PORTFOLIO_SCHEMA = "final-unsb-paper-baseline-portfolio-v1"


def validate_baseline_portfolio(value: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if the pre-result baseline/reporting partition drifted."""
    core = {
        str(row.get("id")): row
        for row in value.get("core_controlled_main_table", [])
        if isinstance(row, dict)
    }
    direct = {
        str(row.get("id")): row
        for row in value.get("direct_external_extensions", [])
        if isinstance(row, dict)
    }
    contextual = {
        str(row.get("id")): row
        for row in value.get("domain_specific_context", [])
        if isinstance(row, dict)
    }
    ceilings = {
        str(row.get("id")): row
        for row in value.get("paired_ceiling_block", [])
        if isinstance(row, dict)
    }
    hard = value.get("hard_reporting_rules") or {}
    valid = (
        value.get("schema") == BASELINE_PORTFOLIO_SCHEMA
        and value.get("status")
        == "CORE_MAIN_TABLE_RUNNING_EXTENSIONS_FAIL_CLOSED_OR_NONBLOCKING"
        and set(core) == {
            "input", "cyclegan", "cut", "plain_unsb", "proposal_only",
            "stcgr", "amtnc",
        }
        and set(direct) == {"ddsb", "dclgan", "negcut"}
        and direct["ddsb"].get("status") == "reproduction_incomplete_fail_closed"
        and direct["ddsb"].get("main_table_number_allowed") is False
        and direct["negcut"].get("main_table_number_allowed") is False
        and "dehazesb" in contextual
        and "never impute missing domains"
        in str(contextual["dehazesb"].get("reporting_rule", ""))
        and set(ceilings) == {"restorevar", "promptir"}
        and all(
            "not an unpaired competitor" in str(row.get("role", ""))
            and "no delta" in str(row.get("reporting_rule", ""))
            for row in ceilings.values()
        )
        and (value.get("priority_and_scheduling") or {}).get(
            "current_gpu_queue_changed"
        ) is False
        and hard.get("main_table_checkpoint") == "e200_only"
        and hard.get("best_checkpoint_selection") is False
        and hard.get("missing_baseline_number_fabricated") is False
        and hard.get("partial_domain_result_used_as_six_domain_macro") is False
        and hard.get("paired_method_called_unpaired_competitor") is False
        and hard.get("external_baseline_called_matched_without_runtime_identity")
        is False
        and hard.get("confirmation20_opened") is False
    )
    if not valid:
        raise RuntimeError("paper baseline reporting portfolio is incomplete or unsafe")
    return value


def _baseline_reporting_summary(value: dict[str, Any]) -> dict[str, Any]:
    value = validate_baseline_portfolio(value)
    row_aliases = {"plain_unsb": "plain", "proposal_only": "proposal"}
    default_labels = {
        "input": "Input",
        "plain": "Plain UNSB",
        "proposal": "Proposal-only",
        "stcgr": "ST-CGR",
        "amtnc": "AM-TNC",
        "hjcgr": "HJCGR",
        "ddsb": "DDSB",
        "dclgan": "DCLGAN",
        "negcut": "NEGCUT",
    }

    def metadata(row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        source_id = str(row["id"])
        row_id = row_aliases.get(source_id, source_id)
        role = str(row.get("role", "")).strip()
        label = str(row.get("paper_label") or default_labels.get(row_id, row_id)).strip()
        scope = str(
            row.get("comparison_rule") or row.get("reporting_rule") or role
        ).strip()
        if not label or not role or not scope:
            raise RuntimeError(f"baseline reporting metadata is incomplete: {source_id}")
        return row_id, {
            "source_id": source_id,
            "paper_label": label,
            "role": role,
            "reproduction_or_comparison_scope": scope,
            "status": row.get("training_status", row.get("status")),
        }

    rows = [
        *value["core_controlled_main_table"],
        *value["direct_external_extensions"],
    ]
    main_table_metadata = dict(metadata(row) for row in rows)
    return {
        "core_controlled_main_table": [
            row["id"] for row in value["core_controlled_main_table"]
        ],
        "direct_external_extensions": {
            row["id"]: row["status"]
            for row in value["direct_external_extensions"]
        },
        "domain_specific_context": {
            row["id"]: row["status"] for row in value["domain_specific_context"]
        },
        "paired_ceiling_block": {
            row["id"]: row["status"] for row in value["paired_ceiling_block"]
        },
        "main_table_metadata": main_table_metadata,
        "main_table_checkpoint": "e200_only",
        "partial_domain_macro_allowed": False,
        "paired_ceiling_as_unpaired_competitor_allowed": False,
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def completion_decision(path: Path, required_status: str) -> str:
    if not Path(path).is_file():
        return "WAIT"
    status = str(_read_json(path).get("status", ""))
    if status == required_status:
        return "READY"
    if status.startswith(("BLOCKED", "FAIL", "FATAL")):
        return "BLOCKED"
    return "WAIT"


def _artifact_binding(
    state: dict[str, Any], *, field: str, expected_path: Path,
) -> bool:
    path = Path(str(state.get(field, "")))
    expected_path = Path(expected_path).resolve()
    digest = state.get(f"{field}_sha256")
    return (
        path.resolve() == expected_path
        and expected_path.is_file()
        and isinstance(digest, str)
        and len(digest) == 64
        and file_sha256(expected_path) == digest
    )


def first_wave_completion_decision(path: Path, output: Path) -> str:
    """Require the completion state to bind all first-wave result artifacts.

    Only paths, hashes, fixed status and safety booleans are inspected here;
    performance payloads are not parsed and cannot affect release.
    """
    path = Path(path)
    if not path.is_file():
        return "WAIT"
    state = _read_json(path)
    status = str(state.get("status", ""))
    if status.startswith(("BLOCKED", "FAIL", "FATAL")):
        return "BLOCKED"
    if status != FIRST_WAVE_COMPLETE_STATUS:
        return "WAIT"
    output = Path(output).resolve()
    valid = (
        state.get("schema") == FIRST_WAVE_STATE_SCHEMA
        and state.get("cohort_status") == "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT"
        and state.get("paper_results_status") == "FIRST_WAVE_COMPLETE"
        and state.get("algorithm_set_status")
        == "FIRST_WAVE_EVIDENCE_READY_CANDIDATES_PENDING"
        and state.get("performance_values_in_control_state") is False
        and state.get("paired_metric_control") is False
        and state.get("best_checkpoint_selection") is False
        and state.get("confirmation20_opened") is False
        and _artifact_binding(
            state, field="cohort",
            expected_path=output / "gates" / "UNIFIED_EVALUATION_COHORT.json",
        )
        and _artifact_binding(
            state, field="paper_results",
            expected_path=output / "PAPER_RESULTS.json",
        )
        and _artifact_binding(
            state, field="algorithm_set",
            expected_path=output / "ALGORITHM_SET.json",
        )
    )
    return "READY" if valid else "BLOCKED"


def algorithm_completion_decision(
    path: Path, *, output: Path, method_lane: str,
) -> str:
    """Bind a completed algorithm successor to its immutable disposition."""
    path = Path(path)
    if not path.is_file():
        return "WAIT"
    state = _read_json(path)
    status = str(state.get("status", ""))
    if status.startswith(("BLOCKED", "FAIL", "FATAL")):
        return "BLOCKED"
    if status != ALGORITHM_COMPLETE_STATUS:
        return "WAIT"
    expected = Path(output).resolve() / "algorithm_dispositions" / f"{method_lane}.json"
    valid = (
        state.get("schema") == ALGORITHM_STATE_SCHEMA
        and state.get("method_lane") == method_lane
        and state.get("performance_values_in_control_state") is False
        and state.get("metric_used_for_training_or_scheduling") is False
        and state.get("paired_metric_control") is False
        and state.get("best_checkpoint_selection") is False
        and state.get("confirmation20_opened") is False
        and _artifact_binding(state, field="disposition", expected_path=expected)
    )
    return "READY" if valid else "BLOCKED"


def validate_disposition(path: Path, method_lane: str) -> dict[str, Any]:
    value = _read_json(path)
    entry = value.get("entry") or {}
    if (
        value.get("schema") != DISPOSITION_SCHEMA
        or value.get("status") != "COMPLETE_POSTHOC_ALGORITHM_DISPOSITION"
        or value.get("method_lane") != method_lane
        or value.get("primary_epoch") != 200
        or value.get("fixed_epochs") != [100, 125, 150, 175, 200]
        or entry.get("lane_id") != method_lane
        or entry.get("status") != "COMPLETE_E200"
        or (entry.get("scientific_gate") or {}).get("status") not in ("PASS", "FAIL")
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("cross_non_equivalent_runtime_delta") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"invalid terminal algorithm disposition: {method_lane}")
    receipts = value.get("evaluation_receipts")
    if not isinstance(receipts, list) or len(receipts) != 5:
        raise RuntimeError(f"incomplete algorithm evaluation receipts: {method_lane}")
    for row in receipts:
        receipt = Path(str(row.get("path", "")))
        if not receipt.is_file() or file_sha256(receipt) != row.get("sha256"):
            raise RuntimeError(f"algorithm evaluation receipt changed: {receipt}")
    return value


def validate_complexity_receipt(
    path: Path, *, lane_id: str, checkpoint_sha256: str,
    expected_protocol_fingerprint: str,
    candidate_authority: Path | None = None,
) -> dict[str, Any]:
    value = _read_json(path)
    expected_authority = (
        file_sha256(candidate_authority) if candidate_authority is not None else None
    )
    if (
        value.get("schema") != COMPLEXITY_SCHEMA
        or value.get("status") != "PASS_TARGET_BLIND_CHECKPOINT_READ_ONLY_PROFILE"
        or value.get("lane_id") != lane_id
        or value.get("checkpoint_sha256") != checkpoint_sha256
        or value.get("checkpoint_unchanged") is not True
        or value.get("protocol_fingerprint") != expected_protocol_fingerprint
        or value.get("evaluation_bundle_fingerprint")
        != FROZEN_EVALUATION_BUNDLE_FINGERPRINT
        or value.get("portable_candidate_authority_sha256") != expected_authority
        or value.get("source_input", {}).get("target_path_read") is not False
        or value.get("performance_metric_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError(f"invalid complexity receipt: {lane_id}")
    if value.get("flops", {}).get("reported") is not False:
        raise RuntimeError("unaudited FLOPs must not be claimed")
    return value


def _source_rows(contract: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    import_root = Path(contract["import_root"])
    rows = {
        lane: validate_import_lane(
            import_lane_path(import_root, lane, host), import_root=import_root,
            lane_id=lane, host_label=host,
        )
        for lane, host in contract["first_wave_lane_sources"].items()
    }
    stcgr_host = contract["stcgr_source_host"]
    rows[STCGR_ID] = validate_import_lane(
        import_lane_path(import_root, STCGR_ID, stcgr_host),
        import_root=import_root, lane_id=STCGR_ID, host_label=stcgr_host,
    )
    rows["amtnc"] = validate_local_export_lane(
        Path(contract["amtnc_export_root"]), lane_id="amtnc",
        source_host_label="4090A",
    )
    return rows


def _e200(rows: list[dict[str, Any]]) -> dict[str, Any]:
    matches = [row for row in rows if int(row["epoch"]) == 200]
    if len(matches) != 1:
        raise RuntimeError("fixed source lacks exactly one e200 checkpoint")
    return matches[0]


def _complexity_command(
    *, contract: dict[str, Any], lane_id: str, checkpoint: Path,
    destination: Path,
) -> list[str]:
    candidate = lane_id == STCGR_ID
    command = [
        contract["python"], "-m", "research.paper_aio.run",
        "--stage", "complexity",
        "--lane", "candidate" if candidate else lane_id,
        "--output", str(Path(contract["output_root"]) / "complexity_work" / lane_id),
        "--manifest", contract["manifest"],
        "--data-root", contract["data_root"],
        "--train-view", contract["train_view"],
        "--gpu", str(contract["gpu"]),
        "--checkpoint", str(checkpoint),
        "--receipt-output", str(destination),
    ]
    if candidate:
        command.extend([
            "--candidate-id", STCGR_ID,
            "--candidate-authority", contract["candidate_authority"],
        ])
    return command


def _complexity_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "lane_id": value["lane_id"],
        "checkpoint_sha256": value["checkpoint_sha256"],
        "environment": value["environment"],
        "parameters": value["parameters"],
        "inference": value["inference"],
        "training_step": value["training_step"],
        "flops": value["flops"],
        "checkpoint_unchanged": True,
        "target_path_read": False,
    }


def _validated_control_relation(
    entry: dict[str, Any], *, lane_id: str, method_source_host: str,
    plain_source_host: str, candidate_cross_code: bool = False,
) -> dict[str, Any]:
    """Bind a reported delta to one reviewed late-epoch runtime relation."""
    trajectory = entry.get("late_trajectory")
    if not isinstance(trajectory, list) or [
        int(row.get("epoch", -1)) for row in trajectory if isinstance(row, dict)
    ] != [150, 175, 200]:
        raise RuntimeError(f"{lane_id} lacks the fixed late-three trajectory")
    relations = []
    allowed_statuses = (
        {
            "PASS_SAME_HOST_CROSS_CODE_CANDIDATE_GATE",
            "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION",
        }
        if candidate_cross_code else
        {"PASS_SAME_SOURCE_RUNTIME", "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"}
    )
    for row in trajectory:
        relation = row.get("runtime_relation") or {}
        cross_host = method_source_host != plain_source_host
        cross_host_proof = (
            not cross_host
            or (
                int(relation.get("runtime_twin_updates", -1)) == 2000
                and all(
                    isinstance(relation.get(key), str)
                    and len(relation[key]) == 64
                    for key in ("e0_core_sha256", "step_core_sha256")
                )
                and relation.get("performance_values_read") is False
            )
        )
        if (
            row.get("crn_exact") is not True
            or not runtime_pair_passed(relation)
            or relation.get("status") not in allowed_statuses
            or relation.get("method_source_host_label") != method_source_host
            or relation.get("plain_source_host_label") != plain_source_host
            or not cross_host_proof
        ):
            raise RuntimeError(
                f"{lane_id} is not bound to the frozen matched plain relation"
            )
        relations.append(relation)
    identity = {
        key: relations[0].get(key)
        for key in (
            "status", "method_source_host_label", "plain_source_host_label",
            "runtime_twin_updates", "e0_core_sha256", "step_core_sha256",
        )
    }
    if any(
        {
            key: relation.get(key) for key in identity
        } != identity
        for relation in relations[1:]
    ):
        raise RuntimeError(f"{lane_id} runtime relation changed across late epochs")
    return identity


def build_portfolio(
    *, first_wave_results: dict[str, Any], amtnc_disposition: dict[str, Any],
    stcgr_disposition: dict[str, Any], complexity: dict[str, dict[str, Any]],
    source_hashes: dict[str, str], method_portfolio: dict[str, Any],
    baseline_portfolio: dict[str, Any],
    first_wave_lane_sources: dict[str, str], stcgr_source_host: str,
) -> dict[str, Any]:
    if (
        first_wave_results.get("schema") != "final-unsb-paper-results-v1"
        or first_wave_results.get("status") != "FIRST_WAVE_COMPLETE"
        or first_wave_results.get("best_checkpoint_selection") is not False
        or first_wave_results.get("paired_metric_control") is not False
        or first_wave_results.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("first-wave paper result is incomplete or unsafe")
    by_lane = {
        row["lane_id"]: row for row in first_wave_results.get("lanes", [])
        if isinstance(row, dict) and row.get("lane_id")
    }
    required = {"input", "plain", "proposal", "cut", "cyclegan", STCGR_ID}
    if not required.issubset(by_lane):
        raise RuntimeError("first-wave result lacks fixed lanes or ST-CGR")
    if set(first_wave_lane_sources) != set(FIRST_WAVE_LANES):
        raise RuntimeError("final portfolio has an incomplete first-wave source map")
    plain_source_host = first_wave_lane_sources["plain"]
    proposal_relation = _validated_control_relation(
        by_lane["proposal"], lane_id="proposal",
        method_source_host=first_wave_lane_sources["proposal"],
        plain_source_host=plain_source_host,
    )
    stcgr_relation = _validated_control_relation(
        stcgr_disposition["entry"], lane_id=STCGR_ID,
        method_source_host=stcgr_source_host,
        plain_source_host=plain_source_host,
        candidate_cross_code=True,
    )
    methods = {
        "proposal": {
            "algorithm_id": "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
            "matched_plain": f"{plain_source_host}/plain",
            "comparison_scope": by_lane["proposal"].get("comparison_scope"),
            "runtime_relation": proposal_relation,
            "result": by_lane["proposal"],
        },
        "amtnc": {
            "algorithm_id": "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS",
            "matched_plain": "4090A/plain",
            "comparison_scope": "same_source_runtime",
            "result": amtnc_disposition["entry"],
        },
        "stcgr": {
            "algorithm_id": STCGR_ID,
            "matched_plain": f"{plain_source_host}/plain",
            "comparison_scope": stcgr_disposition["entry"].get("comparison_scope"),
            "runtime_relation": stcgr_relation,
            "result": stcgr_disposition["entry"],
        },
    }
    accepted = [
        key for key, value in methods.items()
        if value["result"].get("scientific_gate", {}).get("status") == "PASS"
    ]
    failed = [
        key for key, value in methods.items()
        if value["result"].get("scientific_gate", {}).get("status") == "FAIL"
    ]
    external = {
        lane: by_lane[lane] for lane in ("input", "cut", "cyclegan")
    }
    deferred = {
        "hjcgr": method_portfolio["methods"]["hjcgr"],
        "ddsb": method_portfolio["controls_and_external"]["ddsb"],
    }
    return {
        "schema": PORTFOLIO_SCHEMA,
        "status": "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_AWAITING_CONFIRMATION_DECISION",
        "primary_epoch": 200,
        "sustained_epochs": [150, 175, 200],
        "methods": methods,
        "accepted_algorithms": accepted,
        "failed_current_implementation_and_protocol": failed,
        "external_baselines": external,
        "plain_control": by_lane["plain"],
        "frozen_source_hosts": {
            "first_wave": first_wave_lane_sources,
            STCGR_ID: stcgr_source_host,
            "amtnc": "4090A",
        },
        "complexity": {
            lane: _complexity_summary(complexity[lane])
            for lane in COMPLEXITY_LANES
        },
        "deferred_or_reproduction_incomplete": deferred,
        "baseline_reporting_tiers": _baseline_reporting_summary(
            baseline_portfolio
        ),
        "source_artifact_sha256": source_hashes,
        "multiple_algorithms_allowed": True,
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("final delivery checkout must be clean")
    if head != args.required_control_git_commit:
        raise RuntimeError("final delivery checkout moved")
    pairs = [parse_lane_source(value) for value in args.lane_source]
    lane_sources = dict(pairs)
    if len(lane_sources) != len(pairs) or set(lane_sources) != set(FIRST_WAVE_LANES):
        raise RuntimeError(
            "--lane-source must name plain, proposal, cut, and cyclegan exactly once"
        )
    _, stcgr_source_host = parse_lane_source(
        f"{STCGR_ID}={args.stcgr_source_host}"
    )
    baseline_portfolio = repo / "configs" / "PAPER_BASELINE_PORTFOLIO.json"
    validate_baseline_portfolio(_read_json(baseline_portfolio))
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(repo),
        "control_git_commit": head,
        "control_script": str(Path(__file__).resolve()),
        "control_script_sha256": file_sha256(Path(__file__).resolve()),
        "baseline_portfolio": str(baseline_portfolio.resolve()),
        "baseline_portfolio_sha256": file_sha256(baseline_portfolio),
        "paper_protocol_fingerprint": protocol_fingerprint(args.manifest),
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "python": str(args.python.resolve()),
        "output_root": str(args.output.resolve()),
        "first_wave_output": str(args.first_wave_output.resolve()),
        "amtnc_output": str(args.amtnc_output.resolve()),
        "import_root": str(args.import_root.resolve()),
        "amtnc_export_root": str(args.amtnc_export_root.resolve()),
        "candidate_authority": str(args.candidate_authority.resolve()),
        "first_wave_lane_sources": lane_sources,
        "stcgr_source_host": stcgr_source_host,
        "first_wave_state": str(args.first_wave_state.resolve()),
        "amtnc_state": str(args.amtnc_state.resolve()),
        "stcgr_state": str(args.stcgr_state.resolve()),
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "gpu_lock": str(args.gpu_lock.resolve()),
        "gpu": int(args.gpu),
        "complexity_lanes": list(COMPLEXITY_LANES),
        "primary_epoch": 200,
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "metric_values_available_to_scheduler": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _freeze(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if int(args.poll_seconds) < 30 or int(args.poll_seconds) > 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if float(args.timeout_hours) < 24:
        raise ValueError("timeout must be at least 24 hours")
    proposed = _contract(args)
    path = Path(proposed["output_root"]) / "operations" / "FINAL_DELIVERY_CONTRACT.json"
    if path.is_file():
        if _read_json(path) != proposed:
            raise RuntimeError("final delivery contract changed")
    else:
        _write_json(path, proposed)
    return path, proposed


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["control_repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"]
        or _git(repo, "status", "--porcelain")
        or file_sha256(Path(contract["control_script"]))
        != contract["control_script_sha256"]
        or protocol_fingerprint(Path(contract["manifest"]))
        != contract["paper_protocol_fingerprint"]
        or file_sha256(Path(contract["baseline_portfolio"]))
        != contract["baseline_portfolio_sha256"]
    ):
        raise RuntimeError("final delivery control identity changed")
    validate_baseline_portfolio(_read_json(Path(contract["baseline_portfolio"])))


def _dependencies(contract: dict[str, Any]) -> tuple[bool, dict[str, str]]:
    states = {
        "first_wave": first_wave_completion_decision(
            Path(contract["first_wave_state"]), Path(contract["first_wave_output"]),
        ),
        "amtnc": algorithm_completion_decision(
            Path(contract["amtnc_state"]), output=Path(contract["amtnc_output"]),
            method_lane="amtnc",
        ),
        "stcgr": algorithm_completion_decision(
            Path(contract["stcgr_state"]), output=Path(contract["first_wave_output"]),
            method_lane=STCGR_ID,
        ),
    }
    if "BLOCKED" in states.values():
        raise RuntimeError(f"final delivery dependency blocked: {states}")
    imported = imports_ready(
        Path(contract["import_root"]),
        {
            **contract["first_wave_lane_sources"],
            STCGR_ID: contract["stcgr_source_host"],
        },
    )
    amtnc = (
        Path(contract["amtnc_export_root"]) / "amtnc" / "EXPORT_SET.json"
    ).is_file()
    states["all_imports"] = "READY" if imported else "WAIT"
    states["amtnc_export"] = "READY" if amtnc else "WAIT"
    return all(value == "READY" for value in states.values()), states


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract_path, contract = _freeze(args)
    output = Path(contract["output_root"])
    state_path = output / "operations" / "FINAL_DELIVERY_STATE.json"
    lock_path = output / "operations" / "FINAL_DELIVERY.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    base = {
        "schema": STATE_SCHEMA,
        "pid": os.getpid(),
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "control_git_commit": contract["control_git_commit"],
        "metric_values_available_to_scheduler": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }
    heartbeat = StateHeartbeat(state_path, base, contract["poll_seconds"])
    started = time.time()
    with lock_path.open("a+", encoding="utf-8") as process_handle:
        if not _acquire_lock(process_handle, blocking=False):
            raise RuntimeError("final delivery successor is already running")
        heartbeat.start()
        try:
            while True:
                _verify_control(contract)
                ready, dependencies = _dependencies(contract)
                if ready:
                    break
                if time.time() - started > contract["timeout_hours"] * 3600:
                    raise TimeoutError("final delivery successor timed out")
                heartbeat.update(
                    status="WAITING_FOR_ALL_FIXED_E200_RESULTS",
                    dependencies=dependencies, complexity_completed=0,
                    performance_values_read=False,
                    metric_values_used_for_training_or_scheduling=False,
                )
                time.sleep(contract["poll_seconds"])

            rows = _source_rows(contract)
            complexity_root = output / "complexity"
            complexity_root.mkdir(parents=True, exist_ok=True)
            values: dict[str, dict[str, Any]] = {}
            gpu_lock = Path(contract["gpu_lock"])
            gpu_lock.parent.mkdir(parents=True, exist_ok=True)
            with gpu_lock.open("a+", encoding="utf-8") as gpu_handle:
                while not _acquire_lock(gpu_handle, blocking=False):
                    heartbeat.update(
                        status="WAITING_FOR_SHARED_EVALUATION_GPU",
                        complexity_completed=len(values),
                        performance_values_read=False,
                        metric_values_used_for_training_or_scheduling=False,
                    )
                    time.sleep(contract["poll_seconds"])
                for lane_id in COMPLEXITY_LANES:
                    source = _e200(rows[lane_id])
                    receipt = complexity_root / f"{lane_id}.json"
                    authority = (
                        Path(contract["candidate_authority"])
                        if lane_id == STCGR_ID else None
                    )
                    if receipt.is_file():
                        values[lane_id] = validate_complexity_receipt(
                            receipt, lane_id=lane_id,
                            checkpoint_sha256=source["checkpoint_sha256"],
                            expected_protocol_fingerprint=contract[
                                "paper_protocol_fingerprint"
                            ],
                            candidate_authority=authority,
                        )
                        continue
                    heartbeat.update(
                        status="PROFILING_FIXED_E200_COMPLEXITY",
                        current_lane=lane_id,
                        complexity_completed=len(values),
                        performance_values_read=False,
                        metric_values_used_for_training_or_scheduling=False,
                    )
                    log = output / "logs" / f"COMPLEXITY_{lane_id}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    with log.open("a", encoding="utf-8") as handle:
                        completed = subprocess.run(
                            _complexity_command(
                                contract=contract, lane_id=lane_id,
                                checkpoint=Path(source["checkpoint"]),
                                destination=receipt,
                            ),
                            cwd=contract["control_repo"], stdout=handle,
                            stderr=subprocess.STDOUT, check=False,
                        )
                    if completed.returncode != 0:
                        raise RuntimeError(f"complexity profiling failed: {lane_id}")
                    values[lane_id] = validate_complexity_receipt(
                        receipt, lane_id=lane_id,
                        checkpoint_sha256=source["checkpoint_sha256"],
                        expected_protocol_fingerprint=contract[
                            "paper_protocol_fingerprint"
                        ],
                        candidate_authority=authority,
                    )

            heartbeat.update(
                status="BUILDING_FIXED_MULTI_ALGORITHM_PORTFOLIO",
                current_lane=None, complexity_completed=len(values),
                performance_values_read=True,
                metric_values_used_for_training_or_scheduling=False,
            )
            still_ready, final_dependencies = _dependencies(contract)
            if not still_ready:
                raise RuntimeError(
                    "final delivery dependencies changed after complexity profiling: "
                    f"{final_dependencies}"
                )
            first_root = Path(contract["first_wave_output"])
            amtnc_root = Path(contract["amtnc_output"])
            first_results_path = first_root / "PAPER_RESULTS.json"
            first_algorithm_set_path = first_root / "ALGORITHM_SET.json"
            cohort_path = first_root / "gates" / "UNIFIED_EVALUATION_COHORT.json"
            amtnc_path = amtnc_root / "algorithm_dispositions" / "amtnc.json"
            stcgr_path = first_root / "algorithm_dispositions" / f"{STCGR_ID}.json"
            amtnc = validate_disposition(amtnc_path, "amtnc")
            stcgr = validate_disposition(stcgr_path, STCGR_ID)
            source_hashes = {
                "first_wave_results": file_sha256(first_results_path),
                "first_wave_algorithm_set": file_sha256(first_algorithm_set_path),
                "first_wave_cohort": file_sha256(cohort_path),
                "amtnc_disposition": file_sha256(amtnc_path),
                "stcgr_disposition": file_sha256(stcgr_path),
                "baseline_portfolio": contract["baseline_portfolio_sha256"],
            }
            method_portfolio = _read_json(
                Path(contract["control_repo"]) / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json"
            )
            baseline_portfolio = validate_baseline_portfolio(
                _read_json(Path(contract["baseline_portfolio"]))
            )
            result = build_portfolio(
                first_wave_results=_read_json(first_results_path),
                amtnc_disposition=amtnc, stcgr_disposition=stcgr,
                complexity=values, source_hashes=source_hashes,
                method_portfolio=method_portfolio,
                baseline_portfolio=baseline_portfolio,
                first_wave_lane_sources=contract["first_wave_lane_sources"],
                stcgr_source_host=contract["stcgr_source_host"],
            )
            result_path = output / "PAPER_ALGORITHM_PORTFOLIO.json"
            _write_json(result_path, result)
            final = {
                **base,
                "status": COMPLETE_STATUS,
                "complexity_completed": len(values),
                "portfolio": str(result_path.resolve()),
                "portfolio_sha256": file_sha256(result_path),
                "performance_values_read": True,
                "performance_values_in_control_state": False,
                "metric_values_used_for_training_or_scheduling": False,
                "paper_claims_frozen": False,
                "confirmation_authorized": False,
            }
            heartbeat.update(**final)
            return final
        except Exception as error:
            heartbeat.update(
                status="FAIL_CLOSED_REQUIRES_CODEX_AUDIT",
                error_type=type(error).__name__, error_message=str(error),
                metric_values_used_for_training_or_scheduling=False,
                paired_metric_control=False, confirmation20_opened=False,
            )
            raise
        finally:
            heartbeat.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--python", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--first-wave-output", type=Path, required=True)
    value.add_argument("--amtnc-output", type=Path, required=True)
    value.add_argument("--import-root", type=Path, required=True)
    value.add_argument("--amtnc-export-root", type=Path, required=True)
    value.add_argument("--candidate-authority", type=Path, required=True)
    value.add_argument("--lane-source", action="append", required=True)
    value.add_argument("--stcgr-source-host", default="5090A")
    value.add_argument("--first-wave-state", type=Path, required=True)
    value.add_argument("--amtnc-state", type=Path, required=True)
    value.add_argument("--stcgr-state", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
