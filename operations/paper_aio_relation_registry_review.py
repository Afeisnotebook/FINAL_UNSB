"""Validate review-only runtime relations and propose a registry update.

This command is deliberately not an automatic authority.  It verifies frozen
relation candidates and writes an immutable proposed registry plus a compact
review receipt outside Git.  A Codex review must still apply the exact proposal
to the tracked registry with an explicit commit before any matched delta is
legal.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from operations.paper_aio_runtime_relation_successor import contains_performance_field
from operations.paper_aio_unified_evaluation_successor import _write_json, parse_lane_source
from research.paper_aio.protocol import file_sha256, object_sha256
from research.paper_aio.runtime_relation import relation_candidates


REGISTRY_SCHEMA = "final-unsb-paper-matched-runtime-relations-v1"
REVIEW_SCHEMA = "final-unsb-paper-runtime-relation-registry-review-v1"
STANDARD_STATUS = "PASS_EXACT_RUNTIME_RELATION"
CANDIDATE_STATUS = "PASS_EXACT_CROSS_HOST_CROSS_CODE_CANDIDATE_RELATION"
STCGR_ID = "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256_fields(value: dict[str, Any], fields: tuple[str, ...]) -> bool:
    return all(
        isinstance(value.get(field), str) and len(value[field]) == 64
        for field in fields
    )


def validate_relation_candidate(
    path: Path, *, expected_sha256: str, lane_id: str,
    method_source_host: str, plain_source_host: str,
) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or file_sha256(path) != expected_sha256:
        raise RuntimeError(f"relation candidate is absent or changed: {path}")
    value = read_json(path)
    common = (
        value.get("method_lane") == lane_id
        and value.get("method_source_host_label") == method_source_host
        and value.get("plain_source_host_label") == plain_source_host
        and int(value.get("updates", -1)) == 2000
        and value.get("differences") == {}
        and value.get("performance_values_read") is False
        and value.get("paired_metric_control") is False
        and value.get("confirmation20_opened") is False
        and not contains_performance_field(value)
        and _sha256_fields(
            value,
            ("manifest_sha256", "e0_core_sha256", "step_core_sha256"),
        )
    )
    if value.get("status") == STANDARD_STATUS:
        valid = (
            common
            and lane_id != STCGR_ID
            and _sha256_fields(
                value,
                (
                    "training_protocol_fingerprint",
                    "method_runtime_receipt_sha256",
                    "plain_runtime_receipt_sha256",
                    "method_authorization_receipt_sha256",
                ),
            )
            and isinstance(value.get("normalized_environment"), dict)
        )
    elif value.get("status") == CANDIDATE_STATUS:
        valid = (
            common
            and lane_id == STCGR_ID
            and value.get("candidate_id") == lane_id
            and value.get("proof_chain") == {
                "candidate_to_parent": "PASS_CROSS_CODE_CANDIDATE_RUNTIME",
                "parent_to_plain": "PASS_EXACT_RUNTIME_COHORT",
            }
            and _sha256_fields(
                value,
                (
                    "candidate_protocol_fingerprint",
                    "plain_training_protocol_fingerprint",
                    "candidate_runtime_gate_sha256",
                    "candidate_authorization_sha256",
                    "candidate_metadata_import_sha256",
                    "candidate_authority_sha256",
                    "candidate_parent_runtime_receipt_sha256",
                    "plain_runtime_receipt_sha256",
                ),
            )
        )
    else:
        valid = False
    if not valid:
        raise RuntimeError(f"invalid review-only runtime relation: {lane_id}")
    return value


def validate_registry(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if (
        value.get("schema") != REGISTRY_SCHEMA
        or value.get("status") != "ACTIVE_METRIC_BLIND_RELATIONS"
        or not isinstance(value.get("relations"), dict)
        or contains_performance_field(value)
    ):
        raise RuntimeError("runtime relation registry is invalid or metric-contaminated")
    return value


def propose_registry(
    registry: dict[str, Any], candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    result = json.loads(json.dumps(registry))
    relations = result["relations"]
    seen_new: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        lane = candidate["method_lane"]
        key = (
            lane,
            candidate["method_source_host_label"],
            candidate["plain_source_host_label"],
        )
        if key in seen_new:
            raise RuntimeError(f"duplicate proposed runtime relation: {key}")
        seen_new.add(key)
        existing = relation_candidates(result, lane)
        matching = [
            row for row in existing
            if (
                row.get("method_source_host_label") == key[1]
                and row.get("plain_source_host_label") == key[2]
            )
        ]
        if len(matching) > 1:
            raise RuntimeError(f"ambiguous existing runtime relation: {key}")
        if matching:
            if matching[0] != candidate:
                raise RuntimeError(f"existing runtime relation conflicts: {key}")
            continue
        if not existing:
            relations[lane] = candidate
        else:
            relations[lane] = [*existing, candidate]
    return result


def review(args: argparse.Namespace) -> dict[str, Any]:
    if len(args.candidate) != len(args.expected_candidate_sha256):
        raise RuntimeError("candidate paths and expected hashes must have equal length")
    expected_hosts = dict(parse_lane_source(value) for value in args.method_host)
    if len(expected_hosts) != len(args.method_host):
        raise RuntimeError("method host declarations contain duplicate lanes")
    required = set(args.required_lane)
    if required != set(expected_hosts) or len(args.candidate) != len(required):
        raise RuntimeError("required lanes, method hosts, and candidates must match exactly")
    candidates = []
    candidate_rows = []
    for path, expected_hash in zip(
        args.candidate, args.expected_candidate_sha256, strict=True,
    ):
        raw = read_json(path)
        lane = str(raw.get("method_lane", ""))
        if lane not in required:
            raise RuntimeError(f"unexpected runtime relation lane: {lane}")
        value = validate_relation_candidate(
            path, expected_sha256=expected_hash, lane_id=lane,
            method_source_host=expected_hosts[lane],
            plain_source_host=args.plain_source_host,
        )
        candidates.append(value)
        candidate_rows.append({
            "lane_id": lane,
            "method_source_host_label": expected_hosts[lane],
            "plain_source_host_label": args.plain_source_host,
            "path": str(path.resolve()),
            "sha256": expected_hash,
            "status": value["status"],
        })
    if {row["lane_id"] for row in candidate_rows} != required:
        raise RuntimeError("relation candidates do not cover every required lane")
    registry_path = args.registry.resolve()
    registry = validate_registry(registry_path)
    proposed = propose_registry(registry, candidates)
    output = args.output.resolve()
    proposed_path = output / "PROPOSED_RUNTIME_RELATION_REGISTRY.json"
    receipt_path = output / "RUNTIME_RELATION_REGISTRY_REVIEW.json"
    receipt = {
        "schema": REVIEW_SCHEMA,
        "status": "PASS_REVIEW_PROPOSAL_REQUIRES_EXPLICIT_GIT_ADMISSION",
        "base_registry": str(registry_path),
        "base_registry_sha256": file_sha256(registry_path),
        "candidate_relations": sorted(candidate_rows, key=lambda row: row["lane_id"]),
        "proposed_registry": str(proposed_path),
        "proposed_registry_object_sha256": object_sha256(proposed),
        "registry_edited": False,
        "comparison_authorized": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    for path, value in ((proposed_path, proposed), (receipt_path, receipt)):
        if path.is_file():
            if read_json(path) != value:
                raise RuntimeError(f"immutable registry review output changed: {path}")
        else:
            _write_json(path, value)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--registry", type=Path, required=True)
    value.add_argument("--candidate", type=Path, action="append", required=True)
    value.add_argument(
        "--expected-candidate-sha256", action="append", required=True,
    )
    value.add_argument("--required-lane", action="append", required=True)
    value.add_argument("--method-host", action="append", required=True)
    value.add_argument("--plain-source-host", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    print(json.dumps(review(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
