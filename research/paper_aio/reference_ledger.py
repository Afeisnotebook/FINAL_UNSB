"""Hash-bound reference metadata for the post-result paper freeze.

This module validates repository bytes only.  It does not browse, read metrics,
approve empirical claims, or authorize confirmation access.  Volatile entries
remain behind the separate submission-day refresh requirement recorded in the
ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from .protocol import ROOT, file_sha256, object_sha256


SCHEMA = "final-unsb-paper-reference-ledger-v1"
STATUS = "PRE_RESULT_PRIMARY_METADATA_LOCK_CORE_STABLE_VOLATILE_REFRESH_REQUIRED"
DEFAULT_RELATIVE_PATH = Path("configs/PAPER_REFERENCE_LEDGER.json")


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _inside(root: Path, relative: str, *, role: str) -> Path:
    root = Path(root).resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"paper {role} escaped the repository") from error
    if not path.is_file():
        raise RuntimeError(f"paper {role} is missing: {relative}")
    return path


def validate_reference_ledger(
    path: Path | None = None, *, root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / DEFAULT_RELATIVE_PATH if path is None else Path(path)
    if not path.is_absolute():
        path = root / path
    value = _read(path.resolve())
    bib = value.get("bib") or {}
    entries = value.get("entries")
    groups = value.get("required_groups")
    volatile = value.get("submission_day_refresh_required")
    boundaries = value.get("scientific_boundaries") or {}
    authority = value.get("authority") or {}
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != STATUS
        or not isinstance(entries, list)
        or not entries
        or int(bib.get("entry_count", -1)) != len(entries)
        or not isinstance(groups, dict)
        or not groups
        or not isinstance(volatile, list)
        or not volatile
        or authority.get("metadata_lock_is_novelty_proof") is not False
        or authority.get(
            "volatile_entries_require_fresh_primary_source_review_before_submission"
        ) is not True
        or any(value is not False for value in boundaries.values())
        or set(boundaries) != {
            "performance_values_read",
            "training_or_queue_changed",
            "algorithm_selected",
            "confirmation20_opened",
            "empirical_claim_frozen",
        }
    ):
        raise RuntimeError("paper reference ledger is incomplete or unsafe")

    bib_path = _inside(root, str(bib.get("path", "")), role="bibliography")
    if bib.get("sha256") != file_sha256(bib_path):
        raise RuntimeError("paper bibliography changed")

    entry_keys = [str(item.get("citation_key", "")) for item in entries]
    if (
        any(not key for key in entry_keys)
        or len(entry_keys) != len(set(entry_keys))
        or any(
            item.get("metadata_status") != "verified_primary"
            or not str(item.get("primary_url", "")).startswith("https://")
            or not isinstance(item.get("year"), int)
            or not item.get("roles")
            for item in entries
        )
    ):
        raise RuntimeError("paper reference entries are incomplete or duplicated")

    grouped = [str(key) for keys in groups.values() for key in keys]
    if set(grouped) != set(entry_keys) or len(grouped) != len(set(grouped)):
        raise RuntimeError("paper reference groups do not cover the core exactly once")

    bib_keys = re.findall(
        r"^@\w+\{([^,]+),", bib_path.read_text(encoding="utf-8"), re.MULTILINE,
    )
    if len(bib_keys) != len(set(bib_keys)) or set(bib_keys) != set(entry_keys):
        raise RuntimeError("paper bibliography and reference ledger keys differ")

    volatile_keys = [str(item.get("working_key", "")) for item in volatile]
    if (
        any(not key for key in volatile_keys)
        or len(volatile_keys) != len(set(volatile_keys))
        or set(volatile_keys) & set(entry_keys)
        or any(
            item.get("status") != "volatile_not_in_core_bib"
            or not str(item.get("primary_url", "")).startswith("https://")
            for item in volatile
        )
    ):
        raise RuntimeError("volatile paper references are not safely isolated")
    return value


def reference_ledger_reference(
    path: Path | None = None, *, root: Path = ROOT,
) -> dict[str, Any]:
    root = Path(root).resolve()
    path = root / DEFAULT_RELATIVE_PATH if path is None else Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    value = validate_reference_ledger(path, root=root)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("paper reference ledger must be inside the repository") from error
    bib_path = _inside(root, value["bib"]["path"], role="bibliography")
    return {
        "path": relative,
        "sha256": file_sha256(path),
        "object_sha256": object_sha256(value),
        "status": value["status"],
        "bibliography_path": bib_path.relative_to(root).as_posix(),
        "bibliography_sha256": file_sha256(bib_path),
        "entry_count": len(value["entries"]),
        "submission_day_refresh_count": len(
            value["submission_day_refresh_required"]
        ),
    }


def committed_reference_ledger_reference(
    path: Path | None = None, *, root: Path = ROOT,
) -> dict[str, Any]:
    """Require the ledger and bibliography to equal their Git HEAD blobs."""
    root = Path(root).resolve()
    path = root / DEFAULT_RELATIVE_PATH if path is None else Path(path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    value = validate_reference_ledger(path, root=root)
    try:
        ledger_relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise RuntimeError("paper reference ledger must be inside the repository") from error
    relatives = [ledger_relative, str(value["bib"]["path"])]
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", *relatives],
        cwd=root, text=True,
    ).strip():
        raise RuntimeError("paper reference ledger or bibliography has uncommitted changes")
    for relative in relatives:
        try:
            committed = subprocess.check_output(
                ["git", "show", f"HEAD:{relative}"], cwd=root,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"paper reference artifact is not committed: {relative}"
            ) from error
        if hashlib.sha256(committed).hexdigest() != file_sha256(root / relative):
            raise RuntimeError(
                f"paper reference artifact differs from Git HEAD: {relative}"
            )
    return reference_ledger_reference(path, root=root)


def validate_reference_ledger_reference(
    reference: object, *, root: Path = ROOT, require_committed: bool = False,
) -> dict[str, Any]:
    if not isinstance(reference, dict):
        raise RuntimeError("paper freeze lacks a reference-ledger binding")
    relative = reference.get("path")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("paper reference-ledger binding has no path")
    factory = (
        committed_reference_ledger_reference
        if require_committed
        else reference_ledger_reference
    )
    expected = factory(Path(root) / relative, root=root)
    if reference != expected:
        raise RuntimeError("paper reference-ledger binding changed")
    return expected
