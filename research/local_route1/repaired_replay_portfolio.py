"""Portable, source-bound replay authority for repaired route-1 algorithms.

The 5090 host may discover more than one evidence-worthy repaired operator.
This module exports at most two complete-e200, strict/near repaired identities
and registers those *algorithms* on another host without pretending that the
destination host executed the implementation-invalid parent trajectory.

Paired metrics are used only after complete e200 to allocate replay compute.
They never alter a formula, an update, or an intervention schedule.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from operations.local_route1_cross_version_adjudicate import _validate_receipt
from operations.local_route1_freeze_rfammcrb_replacement import (
    CANDIDATE_ID as RFAMMCRB_ID,
    INCIDENT as RFAMMCRB_INCIDENT,
    PARENT_ID as RFAMMCRB_PARENT,
    SPEC as RFAMMCRB_SPEC,
)
from operations.local_route1_freeze_rfmcrb_replacement import (
    CANDIDATE_ID as RFMCRB_ID,
    INCIDENT as RFMCRB_INCIDENT,
    PARENT_ID as RFMCRB_PARENT,
    SPEC as RFMCRB_SPEC,
)
from research.local_route1.frontier_advancement import NEAR, STRICT
from research.local_route1.protocol import file_sha256
from research.local_route1.repaired_frontier_adjudication import (
    ACTIONABLE_STATUS,
    FALLBACK_STATUS,
    SCHEMA as ADJUDICATION_SCHEMA,
)
from research.local_route1.runtime import write_json


SCHEMA = "final-unsb-route1-repaired-replay-portfolio-v1"
STATUS = "PORTABLE_REPAIRED_E200_REPLAY_PORTFOLIO"
REPAIRED_IDS = (RFAMMCRB_ID, RFMCRB_ID)
MAXIMUM_REPLAYS = 2
_SPECS = {
    RFAMMCRB_ID: {
        "parent_id": RFAMMCRB_PARENT,
        "incident": RFAMMCRB_INCIDENT,
        "generation": 3,
        **RFAMMCRB_SPEC,
    },
    RFMCRB_ID: {
        "parent_id": RFMCRB_PARENT,
        "incident": RFMCRB_INCIDENT,
        "generation": 1,
        **RFMCRB_SPEC,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def validate_portable_authority(value: dict[str, Any]) -> dict[str, Any]:
    if value.get("schema") != SCHEMA or value.get("status") != STATUS:
        raise RuntimeError("repaired replay authority schema/status mismatch")
    fixed = {
        "complete_source_e200_only": True,
        "maximum_4090_replays": MAXIMUM_REPLAYS,
        "selection_seeds": [2026],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise RuntimeError(f"repaired replay authority changed: {key}")
    rows = value.get("replay_candidates")
    if not isinstance(rows, list) or len(rows) > MAXIMUM_REPLAYS:
        raise RuntimeError("repaired replay authority exceeds the two-candidate cap")
    ids = [str(row.get("candidate_id", "")) for row in rows if isinstance(row, dict)]
    if len(ids) != len(rows) or len(ids) != len(set(ids)):
        raise RuntimeError("repaired replay authority candidate identities are invalid")
    for row in rows:
        candidate_id = str(row.get("candidate_id", ""))
        if candidate_id not in REPAIRED_IDS:
            raise RuntimeError("repaired replay authority contains a non-repaired algorithm")
        if row.get("classification") not in (STRICT, NEAR):
            raise RuntimeError("4090 replay requires a strict or near complete-e200 source")
        for key in (
            "algorithm_fingerprint", "remote_candidate_fingerprint",
            "remote_training_git_commit", "remote_receipt_sha256",
            "remote_trajectory_sha256", "derivation_card_sha256",
            "implementation_sha256", "derivation_card", "implementation",
        ):
            if key in ("derivation_card", "implementation"):
                if not isinstance(row.get(key), dict):
                    raise RuntimeError(f"repaired replay row is missing {key}")
            elif not isinstance(row.get(key), str) or not row[key]:
                raise RuntimeError(f"repaired replay row is missing {key}")
        card = row["derivation_card"]
        implementation = row["implementation"]
        if (
            card.get("candidate_id") != candidate_id
            or implementation.get("candidate_id") != candidate_id
            or implementation.get("training_target_access") != "unpaired_only"
            or implementation.get("paired_controller_access") is not False
        ):
            raise RuntimeError("repaired replay embedded artifact identity changed")
        if row["derivation_card_sha256"] != _json_file_equivalent_sha256(card):
            raise RuntimeError("repaired replay derivation card changed")
        if row["implementation_sha256"] != _json_file_equivalent_sha256(
            implementation
        ):
            raise RuntimeError("repaired replay implementation changed")
    return value


def _json_file_equivalent_sha256(value: dict[str, Any]) -> str:
    """Hash JSON exactly as :func:`write_json` persists it."""
    import hashlib

    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _write_portable_json(path: Path, value: dict[str, Any]) -> None:
    """Persist canonical LF JSON so a Linux source hash survives any relay OS."""
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise RuntimeError(f"non-identical portable JSON exists: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def export_portable_authority(
    output_root: Path,
    *,
    adjudication_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    adjudication_path = (
        output_root / "operations" / "REPAIRED_FRONTIER_E200_ADJUDICATION.json"
        if adjudication_path is None else Path(adjudication_path).resolve()
    )
    if not adjudication_path.is_file() or not adjudication_path.is_relative_to(output_root):
        raise RuntimeError("repaired replay adjudication escaped the source run root")
    adjudication = _read_json(adjudication_path)
    if (
        adjudication.get("schema") != ADJUDICATION_SCHEMA
        or adjudication.get("status") not in (ACTIONABLE_STATUS, FALLBACK_STATUS)
        or adjudication.get("paired_metrics_used_for_formula_or_training_control") is not False
        or adjudication.get("paired_controller_access") is not False
        or adjudication.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("repaired replay source adjudication is not admissible")
    ranking = {
        str(row.get("candidate_id", "")): row
        for row in adjudication.get("ranking", [])
        if isinstance(row, dict)
    }
    requested = [
        str(candidate_id)
        for candidate_id in adjudication.get("recommended_4090_replay_queue", [])
        if str(candidate_id) in REPAIRED_IDS
    ][:MAXIMUM_REPLAYS]
    replay_rows = []
    for candidate_id in requested:
        row = ranking.get(candidate_id)
        if row is None or row.get("classification") not in (STRICT, NEAR):
            raise RuntimeError("repaired replay queue is inconsistent with complete ranking")
        receipt_path = Path(str(row.get("receipt_path", ""))).resolve()
        if (
            not receipt_path.is_file()
            or not receipt_path.is_relative_to(output_root)
            or file_sha256(receipt_path) != row.get("receipt_sha256")
        ):
            raise RuntimeError("repaired replay source receipt changed")
        receipt = _validate_receipt(receipt_path)
        if receipt.get("candidate_id") != candidate_id:
            raise RuntimeError("repaired replay receipt identity mismatch")
        implementation_path = (
            output_root / "derive" / "implementations" / f"{candidate_id}.json"
        )
        card_path = output_root / "derive" / "cards" / f"{candidate_id}.json"
        if (
            receipt.get("derivation_card_sha256") != file_sha256(card_path)
            or receipt.get("implementation_sha256") != file_sha256(implementation_path)
        ):
            raise RuntimeError("repaired replay receipt is not bound to source artifacts")
        implementation = _read_json(implementation_path)
        card = _read_json(card_path)
        replay_rows.append({
            "source_rank": int(row["rank"]),
            "candidate_id": candidate_id,
            "classification": row["classification"],
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "remote_candidate_fingerprint": receipt["candidate_fingerprint"],
            "remote_training_git_commit": receipt["training_git_commit"],
            "remote_receipt_sha256": file_sha256(receipt_path),
            "remote_trajectory_sha256": receipt["trajectory_sha256"],
            "derivation_card_sha256": receipt["derivation_card_sha256"],
            "implementation_sha256": receipt["implementation_sha256"],
            "derivation_card": card,
            "implementation": implementation,
            "source_ranking_fields": receipt["ranking_fields"],
        })
    result = {
        "schema": SCHEMA,
        "status": STATUS,
        "source_adjudication_sha256": file_sha256(adjudication_path),
        "source_same_host_authority": adjudication["same_host_authority"],
        "source_action_priority_candidate_id": adjudication[
            "action_priority_candidate_id"
        ],
        "replay_candidates": replay_rows,
        "complete_source_e200_only": True,
        "maximum_4090_replays": MAXIMUM_REPLAYS,
        "selection_seeds": [2026],
        "intermediate_metric_routing": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_only_after_complete_e200_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    validate_portable_authority(result)
    output_path = (
        output_root / "operations" / "REPAIRED_4090_REPLAY_PORTFOLIO.json"
        if output_path is None else Path(output_path).resolve()
    )
    write_json(output_path, result)
    return result


def register_portable_replay(
    output_root: Path, *, authority_path: Path, candidate_id: str,
    source_repo: Path, python: Path,
) -> dict[str, Any]:
    """Register one source-identical repaired algorithm on a new host.

    This does not import optimizer/model state.  The destination candidate gets
    a new candidate fingerprint and must restart from that host's common e0.
    """
    output_root = Path(output_root).resolve()
    authority_path = Path(authority_path).resolve()
    authority = validate_portable_authority(_read_json(authority_path))
    rows = {
        str(row["candidate_id"]): row for row in authority["replay_candidates"]
    }
    if candidate_id not in rows:
        raise RuntimeError("candidate is not authorized by the repaired replay portfolio")
    row = rows[candidate_id]
    spec = _SPECS[candidate_id]
    source_repo = Path(source_repo).resolve()
    python = Path(python).resolve()
    if not python.is_file():
        raise RuntimeError("repaired replay Python is missing")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty or commit != row["remote_training_git_commit"]:
        raise RuntimeError("repaired replay source worktree is not the training commit")
    card = row["derivation_card"]
    implementation = row["implementation"]
    for source in implementation.get("source_files", []):
        relative = str(source.get("path", ""))
        path = (source_repo / relative).resolve()
        if (
            not relative
            or not path.is_relative_to(source_repo)
            or not path.is_file()
            or file_sha256(path) != source.get("sha256")
        ):
            raise RuntimeError(f"repaired replay source file changed: {relative}")
    card_destination = output_root / "derive" / "cards" / f"{candidate_id}.json"
    _write_portable_json(card_destination, card)
    implementation_path = (
        output_root / "derive" / "implementations" / f"{candidate_id}.json"
    )
    _write_portable_json(implementation_path, implementation)
    if (
        file_sha256(card_destination) != row["derivation_card_sha256"]
        or file_sha256(implementation_path) != row["implementation_sha256"]
    ):
        raise RuntimeError("destination repaired artifacts differ from source e200 identity")

    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "final-unsb-route1-hypothesis-ledger-v1":
        raise RuntimeError("destination hypothesis ledger schema mismatch")
    matches = [
        record for record in ledger.get("records", [])
        if isinstance(record, dict) and record.get("candidate_id") == candidate_id
    ]
    incident_path = (source_repo / spec["incident"]).resolve()
    if not incident_path.is_file():
        raise RuntimeError("repaired replay training commit lacks its semantic incident")
    replacement_binding = {
        "parent_candidate_id": spec["parent_id"],
        "incident_path": spec["incident"],
        "incident_sha256": file_sha256(incident_path),
        "consumes_generation1_scientific_slot": False,
        "consumes_causal_revision": False,
        "restart_from_common_e0": True,
        "source_bound_cross_host_replica": True,
        "portable_authority_sha256": file_sha256(authority_path),
        "remote_algorithm_fingerprint": row["algorithm_fingerprint"],
        "remote_receipt_sha256": row["remote_receipt_sha256"],
        "remote_trajectory_sha256": row["remote_trajectory_sha256"],
    }
    expected = {
        "candidate_id": candidate_id,
        "generation": int(spec["generation"]),
        "parent_candidate_id": spec["parent_id"],
        "parent_evidence": card["parent_evidence"],
        "construction_route": "source_bound_cross_host_complete_e200_replay",
        "status": "DERIVATION_REQUIRED",
        "revision_count": 0,
        "engineering_replacement": replacement_binding,
        "experiments": [],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    if not matches:
        ledger.setdefault("records", []).append(expected)
        write_json(ledger_path, ledger)
    elif len(matches) != 1:
        raise RuntimeError("destination repaired replay candidate id is not unique")
    else:
        frozen = matches[0]
        if frozen.get("status") == "FROZEN_FOR_GATES":
            binding = frozen.get("engineering_replacement") or {}
            if any(binding.get(key) != value for key, value in replacement_binding.items()):
                raise RuntimeError("existing repaired replay freeze has different authority")
        elif frozen != expected:
            raise RuntimeError("existing repaired replay ledger record differs")

    env = dict(os.environ)
    prefix = os.pathsep.join((str(source_repo), str(source_repo / "src")))
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    command = (
        "import json; from pathlib import Path; "
        "from research.local_route1.candidates import freeze_candidate_derivation; "
        f"r=freeze_candidate_derivation(Path({str(output_root)!r}), {candidate_id!r}); "
        "print(json.dumps(r.to_dict()))"
    )
    result = subprocess.run(
        [str(python), "-c", command], cwd=source_repo, env=env,
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "source-bound repaired replay freeze failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    registration = json.loads(result.stdout)
    if registration.get("algorithm_fingerprint") != row["algorithm_fingerprint"]:
        raise RuntimeError("frozen destination algorithm differs from portable authority")
    return {
        "schema": "final-unsb-route1-repaired-cross-host-registration-v1",
        "status": "REPAIRED_ALGORITHM_FROZEN_FOR_DESTINATION_GATES",
        "candidate": registration,
        "portable_authority_path": str(authority_path),
        "portable_authority_sha256": file_sha256(authority_path),
        "remote_candidate_fingerprint_not_reused": True,
        "restart_from_destination_common_e0": True,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
