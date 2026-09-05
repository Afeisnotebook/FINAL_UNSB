"""Hash-bound paper theory bundle used by the post-result freeze review.

The bundle is pre-result metadata.  It proves which derivations and
formula-to-source audits a later paper claim was reviewed against; it never
reads evaluation values and cannot authorize confirmation access.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from .protocol import ROOT, file_sha256, object_sha256


SCHEMA = "final-unsb-paper-algorithm-theory-bundle-v1"
STATUS = "PRE_RESULT_THREE_OPERATOR_THEORY_BUNDLE_FROZEN"
DEFAULT_RELATIVE_PATH = Path("configs/PAPER_ALGORITHM_THEORY_BUNDLE.json")
METHOD_IDS = {
    "proposal": "ABL-G1-02B-PCRSMG-PROPOSAL-ONLY",
    "stcgr": "G4-01-STRATIFIED-TIME-CONDITIONAL-GF",
    "amtnc": "G2-01-ADAM-METRIC-TANGENTIAL-CONSENSUS",
}
REQUIRED_ARTIFACT_ROLES = {
    "proposal": {
        "derivation_card", "family_derivation", "formula_implementation_audit",
    },
    "stcgr": {
        "derivation_card", "formula_implementation_audit",
        "independent_operator_semantic_audit",
    },
    "amtnc": {"derivation_card", "formula_implementation_audit"},
}


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _artifact_path(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    value = (root / relative).resolve()
    try:
        value.relative_to(root)
    except ValueError as error:
        raise RuntimeError("paper theory artifact escaped the repository") from error
    if not value.is_file():
        raise RuntimeError(f"paper theory artifact is missing: {relative}")
    return value


def validate_theory_bundle(
    path: Path | None = None, *, root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / DEFAULT_RELATIVE_PATH if path is None else Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    value = _read(path)
    boundaries = value.get("claim_boundaries") or {}
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != STATUS
        or set((value.get("methods") or {})) != set(METHOD_IDS)
        or boundaries.get("pre_adam_conditional_mean_only") is not True
        or boundaries.get("expected_adam_displacement_unbiased_claimed") is not False
        or boundaries.get("full_markov_kernel_unbiased_claimed") is not False
        or boundaries.get("equal_flop_superiority_claimed") is not False
        or boundaries.get("terminal_singular_drift_repair_claimed") is not False
        or boundaries.get("full_data_benefit_claimed_before_e200") is not False
        or boundaries.get("unique_winner_predeclared") is not False
        or value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("paper algorithm theory bundle is incomplete or unsafe")

    canonical = value.get("canonical_map") or {}
    canonical_path = _artifact_path(root, str(canonical.get("path", "")))
    if canonical.get("sha256") != file_sha256(canonical_path):
        raise RuntimeError("paper theory canonical map changed")

    seen_paths = {str(canonical.get("path"))}
    for method, algorithm_id in METHOD_IDS.items():
        row = value["methods"].get(method) or {}
        artifacts = row.get("artifacts")
        if (
            row.get("algorithm_id") != algorithm_id
            or not isinstance(row.get("paper_role"), str)
            or not row["paper_role"].strip()
            or not isinstance(row.get("pre_adam_property"), str)
            or not row["pre_adam_property"].strip()
            or not isinstance(artifacts, list)
        ):
            raise RuntimeError(f"paper theory method entry is incomplete: {method}")
        roles = [str(item.get("role", "")) for item in artifacts if isinstance(item, dict)]
        if set(roles) != REQUIRED_ARTIFACT_ROLES[method] or len(roles) != len(set(roles)):
            raise RuntimeError(f"paper theory artifact roles are incomplete: {method}")
        for item in artifacts:
            relative = str(item.get("path", ""))
            if not relative or relative in seen_paths:
                raise RuntimeError("paper theory artifact paths are empty or duplicated")
            seen_paths.add(relative)
            artifact = _artifact_path(root, relative)
            if item.get("sha256") != file_sha256(artifact):
                raise RuntimeError(f"paper theory artifact changed: {relative}")
    return value


def theory_bundle_reference(
    path: Path | None = None, *, root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / DEFAULT_RELATIVE_PATH if path is None else Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    value = validate_theory_bundle(path, root=root)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("paper theory bundle must be inside the repository") from error
    artifact_rows = [value["canonical_map"]]
    for method in sorted(value["methods"]):
        artifact_rows.extend(value["methods"][method]["artifacts"])
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "object_sha256": object_sha256(value),
        "artifact_set_sha256": object_sha256(artifact_rows),
        "status": value["status"],
        "algorithm_ids": [METHOD_IDS[key] for key in sorted(METHOD_IDS)],
    }


def _artifact_relatives(value: dict[str, Any]) -> list[str]:
    result = [str(value["canonical_map"]["path"])]
    for method in sorted(value["methods"]):
        result.extend(
            str(item["path"]) for item in value["methods"][method]["artifacts"]
        )
    return result


def committed_theory_bundle_reference(
    path: Path | None = None, *, root: Path = ROOT,
) -> dict[str, Any]:
    """Require every theory byte to exist unchanged in the current Git tree."""
    root = Path(root).resolve()
    path = root / DEFAULT_RELATIVE_PATH if path is None else Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    value = validate_theory_bundle(path, root=root)
    try:
        bundle_relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("paper theory bundle must be inside the repository") from error
    relatives = [bundle_relative, *_artifact_relatives(value)]
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relatives],
        cwd=root, text=True,
    ).strip()
    if status:
        raise RuntimeError("paper theory bundle or artifact has uncommitted changes")
    for relative in relatives:
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"], cwd=root,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"paper theory artifact is not committed: {relative}"
            ) from error
        current_sha256 = file_sha256(root / relative)
        committed_bytes = (
            committed.encode("utf-8") if isinstance(committed, str) else committed
        )
        if hashlib.sha256(committed_bytes).hexdigest() != current_sha256:
            raise RuntimeError(
                f"paper theory artifact differs from Git HEAD: {relative}"
            )
    return theory_bundle_reference(path, root=root)


def validate_theory_bundle_reference(
    reference: object, *, root: Path = ROOT, require_committed: bool = False,
) -> dict[str, Any]:
    if not isinstance(reference, dict):
        raise RuntimeError("paper freeze lacks an algorithm theory bundle reference")
    relative = reference.get("path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("paper theory bundle reference has no path")
    factory = (
        committed_theory_bundle_reference if require_committed
        else theory_bundle_reference
    )
    expected = factory(Path(root) / relative, root=root)
    if reference != expected:
        raise RuntimeError("paper algorithm theory bundle reference changed")
    return expected
