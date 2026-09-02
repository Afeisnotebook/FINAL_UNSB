"""Fail-closed bridge from a small25 route-1 result to a paper candidate.

This module is control-plane only.  It does not train a model, choose a
candidate from an intermediate metric, or authorize a full-data run by itself.
It binds six independent facts which must all be present before the separate
GPU runtime/equivalence gate may issue a full-data candidate authorization:

* a source-bound, complete small25 e200 receipt;
* a positive frozen trajectory and its no-plain-collapse adjudication;
* the exact derivation card and implementation manifest named by that receipt;
* a complete same-host paper plain trajectory;
* the frozen parent paper protocol/commit/e0 identity; and
* a candidate-specific cross-code runtime gate.

The last item is deliberately external to this pure validator.  A source
change alters the paper protocol fingerprint, so a candidate may never reuse
an old paper authorization merely because its network initialization appears
similar.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Any

from research.local_route1.runtime import write_json

from .protocol import FROZEN_EVALUATION_BUNDLE_FINGERPRINT, ROOT, file_sha256


LOCK_SCHEMA = "final-unsb-paper-candidate-lock-v1"
TERMINAL_RECEIPT_SCHEMA = "final-unsb-route1-candidate-terminal-receipt-v1"
TRAJECTORY_SCHEMA = "final-unsb-route1-candidate-trajectory-v1"
POSITIVE_TRAJECTORY_STATUS = "NUMERIC_GATE_PASS_PENDING_CAUSAL_ADJUDICATION"
RUNTIME_GATE_SCHEMA = "final-unsb-paper-candidate-runtime-gate-v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def safe_candidate_id(value: str) -> str:
    candidate_id = str(value)
    if not _SAFE_ID.fullmatch(candidate_id):
        raise ValueError(f"unsafe candidate id: {candidate_id!r}")
    return candidate_id


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"candidate {label} is not numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"candidate {label} is not finite")
    return number


def _validate_small25_evidence(
    *, candidate_id: str, receipt_path: Path, trajectory_path: Path,
    card_path: Path, implementation_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = _read_json(receipt_path)
    trajectory = _read_json(trajectory_path)
    card = _read_json(card_path)
    implementation = _read_json(implementation_path)

    required_receipt = {
        "schema": TERMINAL_RECEIPT_SCHEMA,
        "status": "ACCEPTED_SOURCE_BOUND_COMPLETE_E200_RECEIPT",
        "candidate_id": candidate_id,
        "trajectory_status": POSITIVE_TRAJECTORY_STATUS,
        "paired_metrics_used_only_after_complete_trajectory": True,
        "paired_metrics_used_for_training_or_control": False,
        "confirmation20_opened": False,
    }
    for key, expected in required_receipt.items():
        if receipt.get(key) != expected:
            raise RuntimeError(f"small25 terminal receipt mismatch for {key}")
    for key in (
        "algorithm_fingerprint", "candidate_fingerprint",
        "candidate_training_core_fingerprint", "base_e0_scientific_state_sha256",
        "base_protocol_fingerprint", "training_git_commit",
    ):
        value = receipt.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"small25 terminal receipt lacks {key}")

    if file_sha256(trajectory_path) != receipt.get("trajectory_sha256"):
        raise RuntimeError("candidate trajectory differs from its source-bound receipt")
    if file_sha256(card_path) != receipt.get("derivation_card_sha256"):
        raise RuntimeError("candidate derivation card differs from its source-bound receipt")
    if file_sha256(implementation_path) != receipt.get("implementation_sha256"):
        raise RuntimeError("candidate implementation differs from its source-bound receipt")
    if trajectory.get("schema") != TRAJECTORY_SCHEMA:
        raise RuntimeError("candidate trajectory schema mismatch")
    if trajectory.get("candidate_id") != candidate_id:
        raise RuntimeError("candidate trajectory id mismatch")
    if trajectory.get("status") != POSITIVE_TRAJECTORY_STATUS:
        raise RuntimeError("only a positive complete small25 trajectory may enter full-data gates")
    if trajectory.get("algorithm_fingerprint") != receipt["algorithm_fingerprint"]:
        raise RuntimeError("candidate trajectory algorithm fingerprint mismatch")
    if trajectory.get("paired_metrics_used_for_training_or_gate") is not False:
        raise RuntimeError("candidate trajectory used paired metrics for training or promotion")
    if trajectory.get("confirmation20_opened") is not False:
        raise RuntimeError("candidate trajectory opened confirmation20")

    # Re-evaluate the preregistered numeric boundary instead of trusting only a
    # status label copied from another checkout.
    if _finite(trajectory.get("late_three_mean_macro_psnr_delta"), label="late delta") <= 0:
        raise RuntimeError("candidate small25 late-three PSNR delta is not positive")
    if _finite(trajectory.get("e200_macro_psnr_delta"), label="e200 delta") <= 0:
        raise RuntimeError("candidate small25 e200 PSNR delta is not positive")
    if int(trajectory.get("late_points_with_four_of_six_positive_domains", -1)) < 2:
        raise RuntimeError("candidate lacks two late points with four positive domains")
    if _finite(trajectory.get("late_average_worst_domain_delta"), label="worst domain") <= -1.0:
        raise RuntimeError("candidate small25 worst-domain guard failed")
    if _finite(trajectory.get("late_mean_macro_ssim_delta"), label="SSIM delta") < 0:
        raise RuntimeError("candidate small25 SSIM guard failed")
    if _finite(trajectory.get("late_mean_macro_lpips_delta"), label="LPIPS delta") > 0:
        raise RuntimeError("candidate small25 LPIPS guard failed")
    if _finite(
        trajectory.get("candidate_best_to_terminal_three_point_rolling_drawdown"),
        label="terminal drawdown",
    ) > 0.3:
        raise RuntimeError("candidate small25 terminal drawdown guard failed")
    collapse = trajectory.get("plain_collapse_adjudication")
    if not isinstance(collapse, dict) or collapse.get("status") != "PASS_NOT_PLAIN_COLLAPSE":
        raise RuntimeError("candidate did not prove that its advantage is not plain collapse")

    if card.get("candidate_id") != candidate_id:
        raise RuntimeError("derivation card candidate id mismatch")
    if implementation.get("schema") != "final-unsb-route1-candidate-implementation-v1":
        raise RuntimeError("candidate implementation schema mismatch")
    if implementation.get("candidate_id") != candidate_id:
        raise RuntimeError("implementation candidate id mismatch")
    if implementation.get("status") != "FROZEN_FOR_GATES":
        raise RuntimeError("candidate implementation is not frozen for gates")
    if implementation.get("paired_controller_access") is not False:
        raise RuntimeError("candidate implementation does not deny paired controller access")
    if implementation.get("training_target_access") != "unpaired_only":
        raise RuntimeError("candidate implementation must be unpaired-only")
    if not isinstance(implementation.get("model"), str) or not implementation["model"]:
        raise RuntimeError("candidate implementation lacks a model entry")
    if not isinstance(implementation.get("method"), dict):
        raise RuntimeError("candidate implementation lacks a frozen method object")
    zero = implementation.get("zero_intervention")
    if not isinstance(zero, dict) or not zero:
        raise RuntimeError("candidate implementation lacks a zero-intervention method")
    state = implementation.get("state_contract")
    if not isinstance(state, dict) or any(
        state.get(key) is not True for key in (
            "full_state_restorable", "zero_intervention_identity_test",
            "parent_state_isolation_test",
        )
    ):
        raise RuntimeError("candidate implementation state contract is incomplete")
    source_files = implementation.get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise RuntimeError("candidate implementation lacks frozen source files")
    seen: set[str] = set()
    for row in source_files:
        if not isinstance(row, dict) or not row.get("path") or not row.get("sha256"):
            raise RuntimeError("candidate source row lacks path or sha256")
        source = (ROOT / str(row["path"])).resolve()
        try:
            relative = source.relative_to(ROOT.resolve()).as_posix()
        except ValueError as error:
            raise RuntimeError("candidate source escapes repository root") from error
        if relative in seen:
            raise RuntimeError(f"duplicate candidate source: {relative}")
        seen.add(relative)
        if not source.is_file() or file_sha256(source) != row["sha256"]:
            raise RuntimeError(f"candidate source identity mismatch: {relative}")
    return receipt, {
        "trajectory": trajectory,
        "card": card,
        "implementation": implementation,
    }


def _validate_parent_plain(
    *, parent_output: Path, required_commit: str, required_protocol_fingerprint: str,
) -> dict[str, Any]:
    preflight_path = parent_output / "gates" / "PREFLIGHT.json"
    authorization_path = parent_output / "gates" / "LANE_AUTHORIZATION_plain.json"
    run_state_path = parent_output / "lanes" / "plain" / "RUN_STATE.json"
    latest_sidecar_path = parent_output / "lanes" / "plain" / "full_state_latest.pt.json"
    e0_sidecar_path = parent_output / "shared_e0" / "unsb_common" / "e0.pt.json"
    required_paths = (
        preflight_path, authorization_path, run_state_path, latest_sidecar_path,
        e0_sidecar_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"same-host parent paper plain is incomplete: {missing}")
    preflight = _read_json(preflight_path)
    authorization = _read_json(authorization_path)
    run_state = _read_json(run_state_path)
    latest = _read_json(latest_sidecar_path)
    e0 = _read_json(e0_sidecar_path)
    if preflight.get("status") != "PASS" or preflight.get("node_role") != "training":
        raise RuntimeError("parent paper training preflight did not pass")
    if preflight.get("manifest", {}).get("content_hashes_verified") is not True:
        raise RuntimeError("parent paper full-data hashes were not verified")
    if authorization.get("status") != "PASS" or authorization.get("lane_id") != "plain":
        raise RuntimeError("parent paper plain authorization did not pass")
    if run_state.get("status") != "COMPLETE_E200":
        raise RuntimeError("parent paper plain has not completed e200")
    if int(run_state.get("final_updates", -1)) != 1_710_600:
        raise RuntimeError("parent paper plain did not complete 1710600 updates")
    if int(latest.get("step", -1)) != 1_710_600:
        raise RuntimeError("parent paper plain latest checkpoint is not e200")
    metadata = latest.get("metadata", {})
    for record, label in (
        (preflight, "preflight"), (authorization, "authorization"),
        (metadata, "plain checkpoint"), (e0.get("metadata", {}), "e0"),
    ):
        if record.get("protocol_fingerprint") != required_protocol_fingerprint:
            raise RuntimeError(f"parent paper {label} protocol fingerprint mismatch")
    for record, label in ((metadata, "plain checkpoint"), (e0.get("metadata", {}), "e0")):
        if record.get("git_commit") != required_commit:
            raise RuntimeError(f"parent paper {label} scientific commit mismatch")
    for record, label in ((preflight, "preflight"), (authorization, "authorization"),
                          (run_state, "run state"), (metadata, "plain checkpoint")):
        if record.get("confirmation20_opened") is not False:
            raise RuntimeError(f"parent paper {label} opened confirmation20")
    if metadata.get("paired_controller_access") is not False:
        raise RuntimeError("parent paper plain checkpoint reports paired controller access")
    return {
        "parent_output": str(parent_output.resolve()),
        "preflight_sha256": file_sha256(preflight_path),
        "plain_authorization_sha256": file_sha256(authorization_path),
        "plain_run_state_sha256": file_sha256(run_state_path),
        "plain_latest_sidecar_sha256": file_sha256(latest_sidecar_path),
        "plain_e0_sidecar_sha256": file_sha256(e0_sidecar_path),
        "plain_terminal_scientific_state_sha256": latest.get("scientific_state_sha256"),
        "parent_e0_scientific_state_sha256": e0.get("scientific_state_sha256"),
    }


def _validate_runtime_gate(
    *, path: Path, candidate_id: str, algorithm_fingerprint: str,
    parent_commit: str, parent_protocol_fingerprint: str,
) -> dict[str, Any]:
    gate = _read_json(path)
    required = {
        "schema": RUNTIME_GATE_SCHEMA,
        "status": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": algorithm_fingerprint,
        "parent_scientific_git_commit": parent_commit,
        "parent_protocol_fingerprint": parent_protocol_fingerprint,
        "e0_scientific_core_exact": True,
        "plain_2000_transition_exact": True,
        "zero_intervention_identity_exact": True,
        "candidate_resume_exact": True,
        "candidate_evaluation_repeat_exact": True,
        "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    for key, expected in required.items():
        if gate.get(key) != expected:
            raise RuntimeError(f"candidate runtime gate mismatch for {key}")
    for key in (
        "candidate_git_commit", "candidate_protocol_fingerprint",
        "parent_e0_scientific_core_sha256", "candidate_e0_scientific_core_sha256",
        "parent_plain_2000_transition_sha256", "candidate_plain_2000_transition_sha256",
        "plain_zero_transition_sha256", "candidate_zero_transition_sha256",
        "candidate_resume_continuous_sha256", "candidate_resume_split_sha256",
        "evaluation_first_sha256", "evaluation_second_sha256",
    ):
        if not isinstance(gate.get(key), str) or not gate[key]:
            raise RuntimeError(f"candidate runtime gate lacks {key}")
    return gate


def materialize_candidate_lock(
    *, output_root: Path, candidate_id: str, terminal_receipt: Path,
    trajectory: Path, derivation_card: Path, implementation: Path,
    runtime_gate: Path, parent_output: Path, parent_scientific_git_commit: str,
    parent_protocol_fingerprint: str,
) -> dict[str, Any]:
    """Materialize a PASS lock only after every independent prerequisite passes."""
    candidate_id = safe_candidate_id(candidate_id)
    paths = tuple(
        Path(value).resolve() for value in (
            terminal_receipt, trajectory, derivation_card, implementation, runtime_gate,
        )
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"candidate lock inputs are missing: {missing}")
    receipt, evidence = _validate_small25_evidence(
        candidate_id=candidate_id, receipt_path=paths[0], trajectory_path=paths[1],
        card_path=paths[2], implementation_path=paths[3],
    )
    parent = _validate_parent_plain(
        parent_output=Path(parent_output).resolve(),
        required_commit=str(parent_scientific_git_commit),
        required_protocol_fingerprint=str(parent_protocol_fingerprint),
    )
    gate = _validate_runtime_gate(
        path=paths[4], candidate_id=candidate_id,
        algorithm_fingerprint=receipt["algorithm_fingerprint"],
        parent_commit=str(parent_scientific_git_commit),
        parent_protocol_fingerprint=str(parent_protocol_fingerprint),
    )
    if gate["candidate_git_commit"] == str(parent_scientific_git_commit) and (
        gate["candidate_protocol_fingerprint"] == str(parent_protocol_fingerprint)
    ):
        raise RuntimeError("candidate lock did not expose a distinct source/protocol identity")

    result = {
        "schema": LOCK_SCHEMA,
        "status": "PASS_FULL_DATA_CANDIDATE_LOCK",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": receipt["algorithm_fingerprint"],
        "candidate_fingerprint": receipt["candidate_fingerprint"],
        "model": evidence["implementation"]["model"],
        "method": evidence["implementation"]["method"],
        "issued_unix_time": time.time(),
        "small25": {
            "terminal_receipt": str(paths[0]),
            "terminal_receipt_sha256": file_sha256(paths[0]),
            "trajectory": str(paths[1]),
            "trajectory_sha256": file_sha256(paths[1]),
            "derivation_card": str(paths[2]),
            "derivation_card_sha256": file_sha256(paths[2]),
            "implementation": str(paths[3]),
            "implementation_sha256": file_sha256(paths[3]),
            "trajectory_status": receipt["trajectory_status"],
            "ranking_fields": receipt.get("ranking_fields"),
            "plain_collapse_adjudication": evidence["trajectory"].get(
                "plain_collapse_adjudication"
            ),
        },
        "parent_paper": {
            "scientific_git_commit": str(parent_scientific_git_commit),
            "protocol_fingerprint": str(parent_protocol_fingerprint),
            **parent,
        },
        "candidate_runtime_gate": {
            "path": str(paths[4]),
            "sha256": file_sha256(paths[4]),
            "candidate_git_commit": gate["candidate_git_commit"],
            "candidate_protocol_fingerprint": gate["candidate_protocol_fingerprint"],
        },
        "full_data_authorized": False,
        "authorization_next_step": (
            "issue a separate candidate lane authorization bound to this lock; "
            "this evidence lock alone cannot start training"
        ),
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    destination = (
        Path(output_root).resolve() / "candidate_locks" / candidate_id
        / "CANDIDATE_LOCK.json"
    )
    write_json(destination, result)
    return result
