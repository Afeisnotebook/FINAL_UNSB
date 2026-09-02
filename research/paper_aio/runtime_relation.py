"""Fail-closed runtime matching for paper method-versus-plain comparisons.

Unified evaluation removes evaluator differences; it does not by itself prove
that checkpoints trained on different hosts are a matched experiment.  This
module keeps those two claims separate and validates only metric-blind runtime
identity metadata.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .protocol import ROOT, file_sha256


RELATIONS_PATH = ROOT / "configs" / "PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json"
PASS_STATUSES = {
    "PASS_SAME_SOURCE_RUNTIME",
    "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION",
    "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION",
    "PASS_SAME_HOST_CROSS_CODE_CANDIDATE_GATE",
    "PASS_LEGACY_LOCAL_TEST_METADATA",
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _contains_performance_field(value: Any) -> bool:
    forbidden = ("psnr", "ssim", "lpips", "fid", "kid", "ranking", "delta")
    if isinstance(value, dict):
        return any(
            any(token in str(key).lower() for token in forbidden)
            or _contains_performance_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_performance_field(item) for item in value)
    return False


def _identity(metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_host_label": metric.get("source_host_label"),
        "training_protocol_fingerprint": metric.get("training_protocol_fingerprint"),
        "manifest_sha256": metric.get("manifest_sha256"),
        "confirmation20_opened": metric.get("confirmation20_opened"),
    }


def normalized_environment(value: dict[str, Any]) -> dict[str, Any]:
    """Retain runtime-defining fields while excluding host/path identity."""
    return {
        key: value.get(key)
        for key in (
            "python", "torch", "torch_cuda", "cudnn", "gpu",
            "cublas_workspace_config", "tf32_matmul", "tf32_cudnn",
        )
    }


def relation_candidates(registry: dict[str, Any], lane_id: str) -> list[dict[str, Any]]:
    """Accept the original single relation and the new multi-control form."""
    raw = (registry.get("relations") or {}).get(lane_id)
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list) and all(isinstance(item, dict) for item in raw):
        return raw
    return []


def exact_runtime_relation_payload(
    *, lane_id: str, method_source_host_label: str,
    plain_source_host_label: str, method_runtime_receipt: Path,
    plain_runtime_receipt: Path, method_authorization_receipt: Path,
) -> dict[str, Any]:
    """Validate primary gate receipts and return a metric-blind relation."""
    method_path = Path(method_runtime_receipt).resolve()
    plain_path = Path(plain_runtime_receipt).resolve()
    authorization_path = Path(method_authorization_receipt).resolve()
    method = _read(method_path)
    plain = _read(plain_path)
    authorization = _read(authorization_path)
    if any(_contains_performance_field(value) for value in (method, plain, authorization)):
        raise RuntimeError("runtime relation primary receipts contain performance fields")
    runtime_schema = "final-unsb-paper-runtime-twin-receipt-v1"
    if (
        method.get("schema") != runtime_schema
        or plain.get("schema") != runtime_schema
        or method.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or plain.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or method.get("host_label") != method_source_host_label
        or plain.get("host_label") != plain_source_host_label
        or method.get("confirmation20_opened") is not False
        or plain.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("runtime receipts are not exact, sealed host identities")
    identity_keys = (
        "updates", "e0_core_sha256", "step_core_sha256",
        "protocol_fingerprint", "manifest_sha256",
    )
    differences = {
        key: {"method": method.get(key), "plain": plain.get(key)}
        for key in identity_keys if method.get(key) != plain.get(key)
    }
    method_environment = normalized_environment(method.get("environment") or {})
    plain_environment = normalized_environment(plain.get("environment") or {})
    if method_environment != plain_environment:
        differences["normalized_environment"] = {
            "method": method_environment, "plain": plain_environment,
        }
    if differences or int(method.get("updates", -1)) != 2000:
        raise RuntimeError("runtime receipts do not prove an exact 2000-update relation")
    method_hash = file_sha256(method_path)
    if (
        authorization.get("schema") != "final-unsb-paper-lane-authorization-v1"
        or authorization.get("status") != "PASS"
        or authorization.get("lane_id") != lane_id
        or authorization.get("protocol_fingerprint")
        != method.get("protocol_fingerprint")
        or (authorization.get("comparison") or {}).get("runtime_receipt_sha256")
        != method_hash
        or authorization.get("failures") != []
        or authorization.get("paired_metric_control") is not False
        or authorization.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("method authorization is not bound to its runtime receipt")
    return {
        "status": "PASS_EXACT_RUNTIME_RELATION",
        "method_lane": lane_id,
        "method_source_host_label": method_source_host_label,
        "plain_source_host_label": plain_source_host_label,
        "updates": 2000,
        "training_protocol_fingerprint": method["protocol_fingerprint"],
        "manifest_sha256": method["manifest_sha256"],
        "e0_core_sha256": method["e0_core_sha256"],
        "step_core_sha256": method["step_core_sha256"],
        "method_runtime_receipt_sha256": method_hash,
        "plain_runtime_receipt_sha256": file_sha256(plain_path),
        "method_authorization_receipt_sha256": file_sha256(authorization_path),
        "normalized_environment": method_environment,
        "differences": {},
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def exact_candidate_control_relation_payload(
    *, candidate_id: str, method_source_host_label: str,
    plain_source_host_label: str, candidate_runtime_gate: Path,
    candidate_authorization: Path, candidate_metadata_import: Path,
    candidate_authority: Path, plain_runtime_receipt: Path,
) -> dict[str, Any]:
    """Prove a cross-host control for a separately fingerprinted candidate.

    This is deliberately stronger than assuming transitivity.  The candidate
    gate must prove exact zero-intervention and native-plain transitions against
    its parent implementation, while the new plain receipt must reproduce that
    parent's e0, step core, protocol and normalized runtime exactly.
    """
    gate_path = Path(candidate_runtime_gate).resolve()
    authorization_path = Path(candidate_authorization).resolve()
    metadata_path = Path(candidate_metadata_import).resolve()
    authority_path = Path(candidate_authority).resolve()
    plain_path = Path(plain_runtime_receipt).resolve()
    gate = _read(gate_path)
    authorization = _read(authorization_path)
    metadata = _read(metadata_path)
    authority = _read(authority_path)
    plain = _read(plain_path)
    if any(
        _contains_performance_field(value)
        for value in (gate, authorization, metadata, authority, plain)
    ):
        raise RuntimeError("candidate control relation primary evidence is metric-contaminated")

    gate_sha = file_sha256(gate_path)
    authorization_sha = file_sha256(authorization_path)
    authority_sha = file_sha256(authority_path)
    if (
        metadata.get("schema") != "final-unsb-paper-candidate-metadata-import-v1"
        or metadata.get("status") != "COMPLETE_VERIFIED_CANDIDATE_METADATA_IMPORT"
        or metadata.get("candidate_id") != candidate_id
        or metadata.get("source_host_label") != method_source_host_label
        or metadata.get("runtime_gate_sha256") != gate_sha
        or metadata.get("authorization_sha256") != authorization_sha
        or metadata.get("authority_sha256") != authority_sha
        or metadata.get("frozen_prior_evidence_transferred") is not True
        or metadata.get("paired_performance_used_for_training_or_scheduling") is not False
        or metadata.get("paired_metric_control") is not False
        or metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate metadata import does not bind the control proofs")
    if (
        gate.get("schema") != "final-unsb-paper-candidate-runtime-gate-v1"
        or gate.get("status") != "PASS_CROSS_CODE_CANDIDATE_RUNTIME"
        or gate.get("candidate_id") != candidate_id
        or gate.get("e0_scientific_core_exact") is not True
        or gate.get("plain_2000_transition_exact") is not True
        or gate.get("zero_intervention_identity_exact") is not True
        or gate.get("candidate_resume_exact") is not True
        or gate.get("candidate_evaluation_repeat_exact") is not True
        or gate.get("parent_e0_scientific_core_sha256")
        != gate.get("candidate_e0_scientific_core_sha256")
        or gate.get("parent_plain_2000_transition_sha256")
        != gate.get("candidate_plain_2000_transition_sha256")
        or gate.get("paired_metric_control") is not False
        or gate.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate runtime gate does not prove cross-code identity")
    if (
        authorization.get("schema") != "final-unsb-paper-candidate-authorization-v1"
        or authorization.get("status") != "PASS_FULL_DATA_CANDIDATE_AUTHORIZATION"
        or authorization.get("candidate_id") != candidate_id
        or authorization.get("candidate_runtime_gate_sha256") != gate_sha
        or authorization.get("candidate_protocol_fingerprint")
        != gate.get("candidate_protocol_fingerprint")
        or authorization.get("parent_e200_required_before_matched_adjudication") is not True
        or authorization.get("paired_metric_control") is not False
        or authorization.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate authorization does not bind the runtime gate")
    source = authority.get("source_evidence") or {}
    training_identity = authority.get("training_identity") or {}
    if (
        authority.get("schema")
        != "final-unsb-paper-portable-candidate-evaluation-authority-v1"
        or authority.get("status") != "FROZEN_EVALUATION_ONLY_AUTHORITY"
        or authority.get("candidate_id") != candidate_id
        or source.get("runtime_gate_sha256") != gate_sha
        or source.get("authorization_sha256") != authorization_sha
        or training_identity.get("git_commit") != gate.get("candidate_git_commit")
        or training_identity.get("protocol_fingerprint")
        != gate.get("candidate_protocol_fingerprint")
        or authority.get("evaluation_only") is not True
        or authority.get("authorizes_training") is not False
        or authority.get("performance_metric_values_included") is not False
        or authority.get("paired_metric_control") is not False
        or authority.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("candidate authority does not bind the imported evidence")

    parent_environment = normalized_environment(gate.get("environment") or {})
    plain_environment = normalized_environment(plain.get("environment") or {})
    differences = {}
    comparisons = {
        "updates": (2000, plain.get("updates")),
        "e0_core_sha256": (
            gate.get("parent_e0_scientific_core_sha256"), plain.get("e0_core_sha256"),
        ),
        "step_core_sha256": (
            gate.get("parent_plain_2000_transition_sha256"), plain.get("step_core_sha256"),
        ),
        "training_protocol_fingerprint": (
            gate.get("parent_protocol_fingerprint"), plain.get("protocol_fingerprint"),
        ),
        "manifest_sha256": (gate.get("manifest_sha256"), plain.get("manifest_sha256")),
    }
    for key, (parent_value, plain_value) in comparisons.items():
        if parent_value != plain_value:
            differences[key] = {"candidate_parent": parent_value, "plain": plain_value}
    if parent_environment != plain_environment:
        differences["normalized_environment"] = {
            "candidate_parent": parent_environment,
            "plain": plain_environment,
        }
    if (
        plain.get("schema") != "final-unsb-paper-runtime-twin-receipt-v1"
        or plain.get("status") != "PASS_EXACT_RUNTIME_COHORT"
        or plain.get("host_label") != plain_source_host_label
        or plain.get("exact_runtime_equivalence") is not True
        or plain.get("differences") != {}
        or plain.get("confirmation20_opened") is not False
        or differences
    ):
        raise RuntimeError("new plain does not exactly reproduce the candidate parent runtime")

    return {
        "status": "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION",
        "method_lane": candidate_id,
        "candidate_id": candidate_id,
        "method_source_host_label": method_source_host_label,
        "plain_source_host_label": plain_source_host_label,
        "updates": 2000,
        "candidate_protocol_fingerprint": gate["candidate_protocol_fingerprint"],
        "plain_training_protocol_fingerprint": gate["parent_protocol_fingerprint"],
        "manifest_sha256": gate["manifest_sha256"],
        "e0_core_sha256": gate["parent_e0_scientific_core_sha256"],
        "step_core_sha256": gate["parent_plain_2000_transition_sha256"],
        "candidate_runtime_gate_sha256": gate_sha,
        "candidate_authorization_sha256": authorization_sha,
        "candidate_metadata_import_sha256": file_sha256(metadata_path),
        "candidate_authority_sha256": authority_sha,
        "candidate_parent_runtime_receipt_sha256": gate["parent_runtime_receipt_sha256"],
        "plain_runtime_receipt_sha256": file_sha256(plain_path),
        "normalized_environment": parent_environment,
        "proof_chain": {
            "candidate_to_parent": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
            "parent_to_plain": "PASS_EXACT_RUNTIME_COHORT",
        },
        "differences": {},
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def materialize_exact_runtime_relation(
    *, lane_id: str, method_source_host_label: str,
    plain_source_host_label: str, method_runtime_receipt: Path,
    plain_runtime_receipt: Path, method_authorization_receipt: Path,
    destination: Path,
) -> dict[str, Any]:
    """Build a metric-blind relation candidate from primary gate receipts."""
    result = exact_runtime_relation_payload(
        lane_id=lane_id,
        method_source_host_label=method_source_host_label,
        plain_source_host_label=plain_source_host_label,
        method_runtime_receipt=method_runtime_receipt,
        plain_runtime_receipt=plain_runtime_receipt,
        method_authorization_receipt=method_authorization_receipt,
    )
    destination = Path(destination).resolve()
    if destination.exists():
        raise RuntimeError(f"runtime relation candidate already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def runtime_pair_status(
    *, method: dict[str, Any], plain: dict[str, Any], lane_id: str,
    candidate_cross_code_gate: bool = False,
    allow_legacy_missing: bool = False,
    relations_path: Path = RELATIONS_PATH,
) -> dict[str, Any]:
    """Return metric-blind evidence that a method/plain comparison is legal."""
    left = _identity(method)
    right = _identity(plain)
    if left["confirmation20_opened"] is not False or right["confirmation20_opened"] is not False:
        return {"status": "FAIL_CONFIRMATION_NOT_SEALED"}
    method_host = left["source_host_label"]
    plain_host = right["source_host_label"]

    # Lightweight unit fixtures and pre-unified local diagnostics predate
    # source metadata.  They cannot produce a final paper cohort, which is
    # independently locked by unified.py, but remain useful for pure math tests.
    if method_host is None and plain_host is None and allow_legacy_missing:
        return {"status": "PASS_LEGACY_LOCAL_TEST_METADATA"}
    if not method_host or not plain_host:
        return {"status": "FAIL_INCOMPLETE_SOURCE_RUNTIME_IDENTITY"}

    same_manifest = (
        bool(left["manifest_sha256"])
        and left["manifest_sha256"] == right["manifest_sha256"]
    )
    if candidate_cross_code_gate and method_host == plain_host:
        passed = method_host == plain_host and same_manifest
        return {
            "status": (
                "PASS_SAME_HOST_CROSS_CODE_CANDIDATE_GATE" if passed
                else "FAIL_CANDIDATE_NOT_SAME_HOST_OR_MANIFEST"
            ),
            "method_source_host_label": method_host,
            "plain_source_host_label": plain_host,
        }
    if method_host == plain_host:
        passed = (
            same_manifest
            and bool(left["training_protocol_fingerprint"])
            and left["training_protocol_fingerprint"]
            == right["training_protocol_fingerprint"]
        )
        return {
            "status": (
                "PASS_SAME_SOURCE_RUNTIME" if passed
                else "FAIL_SAME_HOST_TRAINING_IDENTITY_MISMATCH"
            ),
            "method_source_host_label": method_host,
            "plain_source_host_label": plain_host,
        }

    registry = _read(relations_path)
    if (
        registry.get("schema") != "final-unsb-paper-matched-runtime-relations-v1"
        or registry.get("status") != "ACTIVE_METRIC_BLIND_RELATIONS"
        or _contains_performance_field(registry)
    ):
        raise RuntimeError("matched-runtime relation registry is invalid or metric-contaminated")
    candidates = relation_candidates(registry, lane_id)
    matching = [
        relation for relation in candidates
        if relation.get("method_source_host_label") == method_host
        and relation.get("plain_source_host_label") == plain_host
    ]
    if not matching:
        return {"status": "FAIL_MISSING_EXACT_CROSS_HOST_RUNTIME_RELATION"}
    if len(matching) != 1:
        return {"status": "FAIL_AMBIGUOUS_EXACT_CROSS_HOST_RUNTIME_RELATION"}
    relation = matching[0]
    if candidate_cross_code_gate:
        passed = (
            relation.get("status")
            == "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION"
            and relation.get("method_lane") == lane_id
            and relation.get("candidate_id") == lane_id
            and relation.get("method_source_host_label") == method_host
            and relation.get("plain_source_host_label") == plain_host
            and int(relation.get("updates", -1)) == 2000
            and relation.get("candidate_protocol_fingerprint")
            == left["training_protocol_fingerprint"]
            and relation.get("plain_training_protocol_fingerprint")
            == right["training_protocol_fingerprint"]
            and relation.get("manifest_sha256")
            == left["manifest_sha256"]
            == right["manifest_sha256"]
            and relation.get("proof_chain") == {
                "candidate_to_parent": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
                "parent_to_plain": "PASS_EXACT_RUNTIME_COHORT",
            }
            and all(
                isinstance(relation.get(key), str) and len(relation[key]) == 64
                for key in (
                    "e0_core_sha256",
                    "step_core_sha256",
                    "candidate_runtime_gate_sha256",
                    "candidate_authorization_sha256",
                    "candidate_metadata_import_sha256",
                    "candidate_authority_sha256",
                    "candidate_parent_runtime_receipt_sha256",
                    "plain_runtime_receipt_sha256",
                )
            )
            and relation.get("differences") == {}
            and relation.get("performance_values_read") is False
            and relation.get("paired_metric_control") is False
            and relation.get("confirmation20_opened") is False
        )
        return {
            "status": (
                "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION"
                if passed else "FAIL_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION_MISMATCH"
            ),
            "method_source_host_label": method_host,
            "plain_source_host_label": plain_host,
            "runtime_twin_updates": relation.get("updates"),
            "e0_core_sha256": relation.get("e0_core_sha256"),
            "step_core_sha256": relation.get("step_core_sha256"),
            "performance_values_read": False,
        }
    passed = (
        relation.get("status") == "PASS_EXACT_RUNTIME_RELATION"
        and relation.get("method_lane") == lane_id
        and relation.get("method_source_host_label") == method_host
        and relation.get("plain_source_host_label") == plain_host
        and int(relation.get("updates", -1)) == 2000
        and relation.get("training_protocol_fingerprint")
        == left["training_protocol_fingerprint"]
        == right["training_protocol_fingerprint"]
        and relation.get("manifest_sha256")
        == left["manifest_sha256"]
        == right["manifest_sha256"]
        and isinstance(relation.get("e0_core_sha256"), str)
        and len(relation["e0_core_sha256"]) == 64
        and isinstance(relation.get("step_core_sha256"), str)
        and len(relation["step_core_sha256"]) == 64
        and all(
            isinstance(relation.get(key), str) and len(relation[key]) == 64
            for key in (
                "method_runtime_receipt_sha256",
                "plain_runtime_receipt_sha256",
                "method_authorization_receipt_sha256",
            )
        )
        and relation.get("differences") == {}
        and relation.get("performance_values_read") is False
        and relation.get("paired_metric_control") is False
        and relation.get("confirmation20_opened") is False
    )
    return {
        "status": (
            "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION" if passed
            else "FAIL_EXACT_CROSS_HOST_RUNTIME_RELATION_MISMATCH"
        ),
        "method_source_host_label": method_host,
        "plain_source_host_label": plain_host,
        "runtime_twin_updates": relation.get("updates"),
        "e0_core_sha256": relation.get("e0_core_sha256"),
        "step_core_sha256": relation.get("step_core_sha256"),
        "performance_values_read": False,
    }


def runtime_pair_passed(result: dict[str, Any]) -> bool:
    return result.get("status") in PASS_STATUSES
