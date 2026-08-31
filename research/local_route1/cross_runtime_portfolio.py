"""Portable 4090-to-5090 replay portfolio for two evidence-worthy mechanisms."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from research.local_route1.complete_frontier import (
    SCHEMA as COMPLETE_FRONTIER_SCHEMA,
    STATUS as COMPLETE_FRONTIER_STATUS,
)
from research.local_route1.frontier_advancement import NEAR, STRICT
from research.local_route1.portable_extended_frontier import (
    _canonical_json_sha256,
    _source_artifacts,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


PCRSMG_PROPOSAL_ID = "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY"
AMTNC_ID = "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS"
REPLAY_IDS = (PCRSMG_PROPOSAL_ID, AMTNC_ID)
SOURCE_PARENT_ID = "G1-02B-PLAYER-CONDITIONAL-RSMG"
SCHEMA = "final-unsb-route1-portable-4090-cross-runtime-portfolio-v1"
STATUS = "PORTABLE_4090_EVIDENCE_QUALIFIED_5090_REPLAY_PORTFOLIO"
RESULT_SCHEMA = "final-unsb-route1-cross-runtime-5090-portfolio-result-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_portable_cross_runtime_portfolio(
    value: dict[str, Any],
) -> dict[str, Any]:
    if value.get("schema") != SCHEMA or value.get("status") != STATUS:
        raise RuntimeError("cross-runtime portfolio schema/status mismatch")
    fixed = {
        "replay_candidate_ids": list(REPLAY_IDS),
        "maximum_parallel_replays": 2,
        "complete_source_e200_only": True,
        "checkpoint_transfer": False,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise RuntimeError(f"cross-runtime portfolio changed: {key}")
    frontier = value.get("source_complete_4090_frontier")
    if (
        not isinstance(frontier, dict)
        or frontier.get("schema") != COMPLETE_FRONTIER_SCHEMA
        or frontier.get("status") != COMPLETE_FRONTIER_STATUS
        or _canonical_json_sha256(frontier) != value.get("source_frontier_sha256")
    ):
        raise RuntimeError("cross-runtime source frontier changed")
    rows = {
        str(row.get("candidate_id", "")): row
        for row in frontier.get("ranking", []) if isinstance(row, dict)
    }
    if set(REPLAY_IDS) - set(rows):
        raise RuntimeError("cross-runtime source frontier lacks a replay candidate")
    if rows[PCRSMG_PROPOSAL_ID].get("classification") != STRICT:
        raise RuntimeError("PC-RSMG proposal-only is not strict on the source host")
    if rows[AMTNC_ID].get("classification") != NEAR:
        raise RuntimeError("AM-TNC is not the registered independent near-boundary source")
    preserved = set(frontier.get("evidence_preserved_candidate_ids", []))
    if not set(REPLAY_IDS).issubset(preserved):
        raise RuntimeError("cross-runtime replay candidate is not evidence-preserved")

    evidence = value.get("candidate_evidence")
    if not isinstance(evidence, list) or {
        str(row.get("candidate_id", "")) for row in evidence
        if isinstance(row, dict)
    } != set(REPLAY_IDS):
        raise RuntimeError("cross-runtime candidate evidence set changed")
    for item in evidence:
        candidate_id = str(item["candidate_id"])
        ranking = rows[candidate_id]
        for sha_key, payload_key in (
            ("receipt_sha256", "receipt"),
            ("trajectory_sha256", "trajectory"),
            ("derivation_card_sha256", "derivation_card"),
            ("implementation_sha256", "implementation"),
            ("source_ledger_record_sha256", "source_ledger_record"),
        ):
            payload = item.get(payload_key)
            if (
                not isinstance(payload, dict)
                or _canonical_json_sha256(payload) != item.get(sha_key)
            ):
                raise RuntimeError(
                    f"cross-runtime embedded artifact changed: {candidate_id}:{payload_key}"
                )
        receipt = item["receipt"]
        implementation = item["implementation"]
        if (
            receipt.get("candidate_id") != candidate_id
            or receipt.get("algorithm_fingerprint")
            != ranking.get("algorithm_fingerprint")
            or receipt.get("candidate_fingerprint")
            != ranking.get("candidate_fingerprint")
            or item.get("receipt_sha256") != ranking.get("receipt_sha256")
            or implementation.get("candidate_id") != candidate_id
            or implementation.get("training_target_access") != "unpaired_only"
            or implementation.get("paired_controller_access") is not False
            or item["source_ledger_record"].get("candidate_id") != candidate_id
            or item["source_ledger_record"].get("status") != "FROZEN_FOR_GATES"
        ):
            raise RuntimeError("cross-runtime candidate identity/boundary changed")
    dependencies = value.get("portable_dependencies")
    if not isinstance(dependencies, dict):
        raise RuntimeError("cross-runtime portfolio lacks portable dependencies")
    parent = dependencies.get("pcrsmg_parent_terminal_receipt")
    if (
        not isinstance(parent, dict)
        or _canonical_json_sha256(parent) != dependencies.get(
            "pcrsmg_parent_terminal_receipt_sha256"
        )
        or parent.get("candidate_id") != SOURCE_PARENT_ID
    ):
        raise RuntimeError("cross-runtime PC-RSMG parent provenance changed")
    proposal = next(
        row for row in evidence if row["candidate_id"] == PCRSMG_PROPOSAL_ID
    )
    if proposal["derivation_card"].get("parent_terminal_receipt_sha256") != (
        dependencies["pcrsmg_parent_terminal_receipt_sha256"]
    ):
        raise RuntimeError("proposal-only card no longer binds its parent receipt")
    return value


def export_cross_runtime_portfolio(
    output_root: Path, *, frontier_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    frontier_path = (
        output_root / "operations" / "COMPLETE_FRONTIER_4090_ADJUDICATION.json"
        if frontier_path is None else Path(frontier_path).resolve()
    )
    if not frontier_path.is_file() or not frontier_path.is_relative_to(output_root):
        raise RuntimeError("cross-runtime source frontier escaped the run root")
    frontier = _read_json(frontier_path)
    if (
        frontier.get("schema") != COMPLETE_FRONTIER_SCHEMA
        or frontier.get("status") != COMPLETE_FRONTIER_STATUS
        or frontier.get("canonical_candidate_is_action_priority_only") is not True
        or frontier.get("algorithm_discovery_collapsed_to_single_candidate") is not False
        or frontier.get("paired_controller_access") is not False
        or frontier.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("4090 complete frontier is not portable")
    ranking = {
        str(row.get("candidate_id", "")): row
        for row in frontier.get("ranking", []) if isinstance(row, dict)
    }
    if set(REPLAY_IDS) - set(ranking):
        raise RuntimeError("4090 complete frontier lacks the replay portfolio")
    ledger = _read_json(output_root / "derive" / "HYPOTHESIS_LEDGER.json")
    records = {
        str(row.get("candidate_id", "")): row
        for row in ledger.get("records", []) if isinstance(row, dict)
    }
    evidence = []
    for candidate_id in REPLAY_IDS:
        row = ranking[candidate_id]
        receipt_path = Path(str(row.get("receipt_path", ""))).resolve()
        if file_sha256(receipt_path) != row.get("receipt_sha256"):
            raise RuntimeError("cross-runtime source receipt changed")
        item = _source_artifacts(output_root, receipt_path)
        record = records.get(candidate_id)
        if not isinstance(record, dict) or record.get("status") != "FROZEN_FOR_GATES":
            raise RuntimeError("cross-runtime source ledger record is not frozen")
        item.update({
            "source_classification": row["classification"],
            "source_ledger_record": record,
            "source_ledger_record_sha256": _canonical_json_sha256(record),
        })
        evidence.append(item)
    parent_path = (
        output_root / "operations" / "terminal_receipts" / f"{SOURCE_PARENT_ID}.json"
    )
    parent = _read_json(parent_path)
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_frontier_sha256": file_sha256(frontier_path),
        "source_complete_4090_frontier": frontier,
        "candidate_evidence": evidence,
        "portable_dependencies": {
            "pcrsmg_parent_terminal_receipt": parent,
            "pcrsmg_parent_terminal_receipt_sha256": file_sha256(parent_path),
        },
        "replay_candidate_ids": list(REPLAY_IDS),
        "selection_reason": {
            PCRSMG_PROPOSAL_ID: (
                "strict source-host action priority and strongest sustained new operator"
            ),
            AMTNC_ID: (
                "mechanism-independent Adam geometry with positive late-three/e200 "
                "and one small LPIPS guardrail failure"
            ),
        },
        "excluded_current_operator_ids": [
            "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
            "F2-01-RESIDUAL-FEASIBLE-ADAM-METRIC-BARRIER",
            "F2-02-RESIDUAL-FEASIBLE-EUCLIDEAN-COVARIANCE-BARRIER",
        ],
        "maximum_parallel_replays": 2,
        "complete_source_e200_only": True,
        "checkpoint_transfer": False,
        "selection_seeds": [2026],
        "cross_host_deltas_merged": False,
        "paired_metrics_used_only_after_complete_e200_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    validate_portable_cross_runtime_portfolio(result)
    output_path = (
        output_root / "operations" / "PORTABLE_4090_TO_5090_REPLAY_PORTFOLIO.json"
        if output_path is None else Path(output_path).resolve()
    )
    if not output_path.is_relative_to(output_root):
        raise RuntimeError("cross-runtime portable output escaped the run root")
    write_json(output_path, result)
    return result


def _source_env(source_repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(source_repo), str(source_repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def register_cross_runtime_candidate(
    output_root: Path, *, authority_path: Path, candidate_id: str,
    source_repo: Path, python: Path,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    authority_path = Path(authority_path).resolve()
    source_repo = Path(source_repo).resolve()
    python = Path(python).resolve()
    authority = validate_portable_cross_runtime_portfolio(
        _read_json(authority_path)
    )
    if candidate_id not in REPLAY_IDS:
        raise RuntimeError("candidate is not in the cross-runtime replay portfolio")
    rows = {row["candidate_id"]: row for row in authority["candidate_evidence"]}
    row = rows[candidate_id]
    receipt = row["receipt"]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty or commit != receipt.get("training_git_commit"):
        raise RuntimeError("cross-runtime source worktree is not the training commit")
    for source in row["implementation"].get("source_files", []):
        relative = str(source.get("path", ""))
        path = (source_repo / relative).resolve()
        if (
            not relative or not path.is_relative_to(source_repo) or not path.is_file()
            or file_sha256(path) != source.get("sha256")
        ):
            raise RuntimeError(f"cross-runtime source file changed: {relative}")

    for folder, key, sha_key in (
        ("cards", "derivation_card", "derivation_card_sha256"),
        ("implementations", "implementation", "implementation_sha256"),
    ):
        path = output_root / "derive" / folder / f"{candidate_id}.json"
        if path.is_file():
            if file_sha256(path) != row[sha_key]:
                raise RuntimeError("destination cross-runtime artifact differs")
        else:
            write_json(path, row[key])
    if candidate_id == PCRSMG_PROPOSAL_ID:
        dependency = authority["portable_dependencies"]
        path = (
            output_root / "operations" / "terminal_receipts"
            / f"{SOURCE_PARENT_ID}.json"
        )
        payload = dependency["pcrsmg_parent_terminal_receipt"]
        expected = dependency["pcrsmg_parent_terminal_receipt_sha256"]
        if path.is_file():
            if file_sha256(path) != expected:
                raise RuntimeError("destination PC-RSMG parent receipt differs")
        else:
            write_json(path, payload)
            if file_sha256(path) != expected:
                raise RuntimeError("portable PC-RSMG parent receipt hash changed")

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    matches = [
        item for item in ledger.get("records", [])
        if isinstance(item, dict) and item.get("candidate_id") == candidate_id
    ]
    expected_record = copy.deepcopy(row["source_ledger_record"])
    for key in (
        "derivation_card_sha256", "implementation_sha256",
        "algorithm_fingerprint", "freeze_event",
    ):
        expected_record.pop(key, None)
    expected_record.update({
        "status": "DERIVATION_REQUIRED",
        "experiments": [],
        "source_bound_cross_host_replay": {
            "source_host_role": "4090_complete_e200_frontier",
            "destination_host_role": "5090_common_e0_replay",
            "portable_authority_sha256": file_sha256(authority_path),
            "source_receipt_sha256": row["receipt_sha256"],
            "source_trajectory_sha256": row["trajectory_sha256"],
            "source_training_git_commit": receipt["training_git_commit"],
            "formula_changed": False,
            "paired_metrics_used_only_for_resource_allocation": True,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    })
    if not matches:
        ledger.setdefault("records", []).append(expected_record)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("destination cross-runtime candidate id is duplicated")
    elif matches[0].get("status") != "FROZEN_FOR_GATES":
        if matches[0] != expected_record:
            raise RuntimeError("destination cross-runtime ledger slot differs")

    command = (
        "import json; from pathlib import Path; "
        "from research.local_route1.candidates import freeze_candidate_derivation; "
        f"r=freeze_candidate_derivation(Path({str(output_root)!r}), {candidate_id!r}); "
        "print(json.dumps(r.to_dict()))"
    )
    result = subprocess.run(
        [str(python), "-c", command], cwd=source_repo,
        env=_source_env(source_repo), capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "cross-runtime candidate freeze failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    registration = json.loads(result.stdout)
    if registration.get("algorithm_fingerprint") != receipt.get(
        "algorithm_fingerprint"
    ):
        raise RuntimeError("destination cross-runtime algorithm differs from source")
    return {
        "status": "CROSS_RUNTIME_ALGORITHM_FROZEN_FOR_DESTINATION_GATES",
        "candidate": registration,
        "source_classification": row["source_classification"],
        "source_authority_sha256": file_sha256(authority_path),
        "restart_from_destination_common_e0": True,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
