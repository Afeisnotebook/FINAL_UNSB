"""Source-bound replay registration for the evidence-backed PCNR alternate.

The repaired-frontier replay portfolio intentionally handled only the two
residual feasible barrier replacements.  PCNR is a different parent mechanism
and must not be silently discarded merely because it missed the strict e200
gate on its source host.  This module accepts PCNR only when the complete 5090
frontier names it as the action-priority, evidence-preserved alternate.

Paired metrics have already been frozen in the complete source trajectory and
are used only to allocate destination compute.  They cannot alter the PCNR
formula, its training state, or its update schedule.
"""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from operations.local_route1_candidate_terminal_receipt import (
    SCHEMA as RECEIPT_SCHEMA,
)
from research.local_route1.frontier_advancement import ALTERNATE
from research.local_route1.portable_extended_frontier import (
    validate_portable_extended_frontier,
)
from research.local_route1.protocol import file_sha256
from research.local_route1.runtime import write_json


CANDIDATE_ID = "F1-01-PLAYER-CONDITIONAL-NATIVE-RESAMPLING"
REGISTRATION_SCHEMA = "final-unsb-route1-pcnr-alternate-replay-registration-v1"
RESULT_SCHEMA = "final-unsb-route1-pcnr-alternate-4090-result-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    import hashlib

    payload = json.dumps(value, ensure_ascii=False, indent=2) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_pcnr_alternate(value: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen PCNR evidence or fail closed.

    The extended frontier's legacy ``recommended_4090_replay_queue`` contains
    strict candidates only.  That resource-allocation policy is deliberately
    not consulted here: the admissible alternate must instead be the complete
    frontier action priority *and* explicitly evidence-preserved.
    """

    value = validate_portable_extended_frontier(value)
    adjudication = value["extended_adjudication"]
    ranking = {
        str(row.get("candidate_id", "")): row
        for row in adjudication.get("ranking", [])
        if isinstance(row, dict)
    }
    row = ranking.get(CANDIDATE_ID)
    if row is None:
        raise RuntimeError("portable extended frontier has no PCNR ranking row")
    if adjudication.get("action_priority_candidate_id") != CANDIDATE_ID:
        raise RuntimeError("PCNR is not the source repaired-frontier action priority")
    if CANDIDATE_ID not in adjudication.get("evidence_preserved_candidate_ids", []):
        raise RuntimeError("PCNR is not evidence-preserved on the source host")
    if row.get("classification") != ALTERNATE:
        raise RuntimeError("PCNR replay requires an evidence-backed alternate source")

    evidence_rows = {
        str(item.get("candidate_id", "")): item
        for item in value.get("candidate_evidence", [])
        if isinstance(item, dict)
    }
    evidence = evidence_rows.get(CANDIDATE_ID)
    if evidence is None:
        raise RuntimeError("portable extended frontier lacks PCNR evidence")
    receipt = evidence.get("receipt")
    card = evidence.get("derivation_card")
    implementation = evidence.get("implementation")
    if not all(isinstance(item, dict) for item in (receipt, card, implementation)):
        raise RuntimeError("portable PCNR artifacts are malformed")
    if (
        receipt.get("schema") != RECEIPT_SCHEMA
        or receipt.get("candidate_id") != CANDIDATE_ID
        or receipt.get("algorithm_fingerprint") != row.get("algorithm_fingerprint")
        or receipt.get("candidate_fingerprint") != row.get("candidate_fingerprint")
        or receipt.get("trajectory_sha256") != row.get("trajectory_sha256")
        or evidence.get("receipt_sha256") != row.get("receipt_sha256")
        or receipt.get("derivation_card_sha256") != evidence.get(
            "derivation_card_sha256"
        )
        or receipt.get("implementation_sha256") != evidence.get(
            "implementation_sha256"
        )
    ):
        raise RuntimeError("portable PCNR receipt/ranking binding changed")
    if (
        _canonical_sha256(card) != evidence.get("derivation_card_sha256")
        or _canonical_sha256(implementation) != evidence.get("implementation_sha256")
        or implementation.get("model") != "route1_pcnr"
        or implementation.get("training_target_access") != "unpaired_only"
        or implementation.get("paired_controller_access") is not False
        or receipt.get("paired_metrics_used_for_training_or_control") is not False
        or receipt.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("portable PCNR algorithm or target-blind boundary changed")
    return {
        "ranking": row,
        "evidence": evidence,
        "receipt": receipt,
        "derivation_card": card,
        "implementation": implementation,
    }


def _source_env(source_repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(source_repo), str(source_repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def _source_registration(
    *, output_root: Path, source_repo: Path, python: Path,
    require_hypothesis_ledger: bool,
) -> dict[str, Any]:
    command = (
        "import json; from pathlib import Path; "
        "from research.local_route1.candidates import load_candidate_registration; "
        f"r=load_candidate_registration(Path({str(output_root)!r}), "
        f"{CANDIDATE_ID!r}, require_hypothesis_ledger="
        f"{require_hypothesis_ledger!r}); print(json.dumps(r.to_dict()))"
    )
    result = subprocess.run(
        [str(python), "-c", command], cwd=source_repo,
        env=_source_env(source_repo), capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            "source-bound PCNR registration validation failed:\n"
            f"{result.stdout}\n{result.stderr}"
        )
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise RuntimeError("PCNR source registration did not return an object")
    return value


def _without_source_hashes(value: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(value)
    for row in normalized.get("source_files", []):
        if isinstance(row, dict) and "sha256" in row:
            row["sha256"] = "<source-bound>"
    return normalized


def _write_once_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_bytes() != payload:
            raise RuntimeError(f"non-identical PCNR replay archive exists: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _assert_unexecuted(output_root: Path) -> None:
    forbidden = [
        output_root / "derive" / "gates" / f"{CANDIDATE_ID}.json",
        output_root / "candidates" / CANDIDATE_ID,
        output_root / "operations" / "terminal_receipts" / f"{CANDIDATE_ID}.json",
        output_root / "operations" / "terminal_receipts" / f"{CANDIDATE_ID}_4090.json",
    ]
    forbidden.extend(
        (output_root / "operations").glob(
            f"CANDIDATE_EXECUTOR_CONTRACT_{CANDIDATE_ID}*.json"
        )
    )
    existing = [str(path) for path in forbidden if path.exists()]
    if existing:
        raise RuntimeError(
            "cannot rebind a PCNR registration after destination execution: "
            f"{existing}"
        )


def register_pcnr_alternate(
    output_root: Path, *, authority_path: Path, source_repo: Path, python: Path,
) -> dict[str, Any]:
    """Bind the complete 5090 PCNR algorithm to a fresh 4090 e0 replay.

    One older, unexecuted destination registration may differ only in frozen
    source-file hashes.  It is archived before rebinding; any semantic change or
    any pre-existing execution artifact fails closed.
    """

    output_root = Path(output_root).resolve()
    authority_path = Path(authority_path).resolve()
    source_repo = Path(source_repo).resolve()
    python = Path(python).resolve()
    if not authority_path.is_file() or not python.is_file():
        raise RuntimeError("PCNR replay authority or Python is missing")
    selected = select_pcnr_alternate(_read_json(authority_path))
    receipt = selected["receipt"]
    card = selected["derivation_card"]
    implementation = selected["implementation"]

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=source_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=source_repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty or commit != receipt.get("training_git_commit"):
        raise RuntimeError("PCNR source worktree is not the source training commit")
    for source in implementation.get("source_files", []):
        relative = str(source.get("path", ""))
        path = (source_repo / relative).resolve()
        if (
            not relative
            or not path.is_relative_to(source_repo)
            or not path.is_file()
            or file_sha256(path) != source.get("sha256")
        ):
            raise RuntimeError(f"PCNR source file changed: {relative}")

    card_path = output_root / "derive" / "cards" / f"{CANDIDATE_ID}.json"
    implementation_path = (
        output_root / "derive" / "implementations" / f"{CANDIDATE_ID}.json"
    )
    ledger_path = output_root / "derive" / "HYPOTHESIS_LEDGER.json"
    if not ledger_path.is_file():
        raise RuntimeError("destination hypothesis ledger is missing")
    if card_path.is_file() and file_sha256(card_path) != selected["evidence"].get(
        "derivation_card_sha256"
    ):
        raise RuntimeError("destination PCNR derivation card differs from source")
    if not card_path.is_file():
        write_json(card_path, card)

    ledger = _read_json(ledger_path)
    matches = [
        row for row in ledger.get("records", [])
        if isinstance(row, dict) and row.get("candidate_id") == CANDIDATE_ID
    ]
    if len(matches) != 1 or matches[0].get("status") != "FROZEN_FOR_GATES":
        raise RuntimeError("destination PCNR ledger slot is not uniquely frozen")
    record = matches[0]
    if record.get("experiments") not in (None, []):
        raise RuntimeError("destination PCNR ledger already records an experiment")

    source_implementation_sha = str(selected["evidence"]["implementation_sha256"])
    current_implementation_sha = (
        file_sha256(implementation_path) if implementation_path.is_file() else None
    )
    binding_changed = (
        current_implementation_sha != source_implementation_sha
        or record.get("implementation_sha256") != source_implementation_sha
        or record.get("algorithm_fingerprint") != receipt.get("algorithm_fingerprint")
    )
    archive = (
        output_root / "operations" / "source_bound_rebindings" / CANDIDATE_ID
    )
    if binding_changed:
        _assert_unexecuted(output_root)
        if implementation_path.is_file():
            current = _read_json(implementation_path)
            if _without_source_hashes(current) != _without_source_hashes(implementation):
                raise RuntimeError(
                    "unexecuted destination PCNR registration differs semantically"
                )
            _write_once_bytes(
                archive / "PREVIOUS_IMPLEMENTATION.json",
                implementation_path.read_bytes(),
            )
        _write_once_bytes(
            archive / "PREVIOUS_LEDGER_RECORD.json",
            (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        )
        write_json(implementation_path, implementation)

        unbound = _source_registration(
            output_root=output_root, source_repo=source_repo, python=python,
            require_hypothesis_ledger=False,
        )
        if (
            unbound.get("algorithm_fingerprint")
            != receipt.get("algorithm_fingerprint")
            or unbound.get("candidate_training_core_fingerprint")
            != receipt.get("candidate_training_core_fingerprint")
        ):
            raise RuntimeError("destination PCNR algorithm differs from source receipt")
        record.update({
            "implementation_sha256": source_implementation_sha,
            "algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "source_bound_cross_host_replay": {
                "source_host_role": "5090_complete_e200_repaired_frontier",
                "destination_host_role": "4090_common_e0_replay",
                "portable_authority_sha256": file_sha256(authority_path),
                "source_receipt_sha256": selected["evidence"]["receipt_sha256"],
                "source_trajectory_sha256": selected["evidence"][
                    "trajectory_sha256"
                ],
                "source_training_git_commit": receipt["training_git_commit"],
                "previous_unexecuted_implementation_sha256": (
                    current_implementation_sha
                ),
                "formula_changed": False,
                "paired_metrics_used_only_for_resource_allocation": True,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        })
        write_json(ledger_path, ledger)
        write_json(archive / "REBINDING_DECISION.json", {
            "schema": "final-unsb-route1-pcnr-unexecuted-source-rebinding-v1",
            "candidate_id": CANDIDATE_ID,
            "previous_implementation_sha256": current_implementation_sha,
            "source_implementation_sha256": source_implementation_sha,
            "source_algorithm_fingerprint": receipt["algorithm_fingerprint"],
            "semantic_difference": "source_file_hash_only",
            "destination_execution_existed_before_rebinding": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })

    registration = _source_registration(
        output_root=output_root, source_repo=source_repo, python=python,
        require_hypothesis_ledger=True,
    )
    if registration.get("algorithm_fingerprint") != receipt.get(
        "algorithm_fingerprint"
    ):
        raise RuntimeError("registered destination PCNR fingerprint changed")
    return {
        "schema": REGISTRATION_SCHEMA,
        "status": "PCNR_EVIDENCE_BACKED_ALTERNATE_FROZEN_FOR_4090_GATES",
        "candidate": registration,
        "source_classification": selected["ranking"]["classification"],
        "source_authority_path": str(authority_path),
        "source_authority_sha256": file_sha256(authority_path),
        "source_receipt_sha256": selected["evidence"]["receipt_sha256"],
        "source_trajectory_sha256": selected["evidence"]["trajectory_sha256"],
        "restart_from_destination_common_e0": True,
        "paired_metrics_used_only_for_resource_allocation": True,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    }
