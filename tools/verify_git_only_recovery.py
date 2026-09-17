#!/usr/bin/env python3
"""Verify that a fresh Git clone contains the canonical FINAL_UNSB record."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    data = path.read_bytes()
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def load_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def verify_source_manifest() -> int:
    checked = 0
    manifest = ROOT / "SOURCE_MANIFEST.sha256"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        assert path.is_file(), f"missing source-manifest entry: {relative}"
        assert normalized_sha256(path) == expected, f"source hash mismatch: {relative}"
        checked += 1
    return checked


def main() -> int:
    recovery = load_json("configs/GIT_ONLY_RECOVERY_MANIFEST.json")
    handoff = load_json("FINAL_HANDOFF.json")
    archive = load_json("archive/paper_aio/final_v10/ARCHIVE_MANIFEST.json")

    assert recovery["status"] == "RESEARCH_CONTINUITY_COMPLETE_BINARY_RECOVERY_EXTERNAL"
    assert recovery["confirmation20_opened"] is False
    assert handoff["protocol"]["confirmation20_opened"] is False
    assert handoff["canonical_outcome"]["accepted_algorithms"] == ["proposal"]

    canonical = recovery["canonical_git_assets"]
    for key in (
        "human_entrypoint",
        "machine_entrypoint",
        "current_state_document",
        "claim_boundaries",
        "document_authority_registry",
        "disaster_recovery_guide",
        "fresh_context_prompt",
        "archive_manifest",
        "augmented_portfolio",
        "data_manifest",
        "source_integrity_manifest",
    ):
        assert (ROOT / canonical[key]).is_file(), f"missing canonical asset: {key}"

    assert sha256(ROOT / canonical["augmented_portfolio"]) == canonical["augmented_portfolio_sha256"]
    assert sha256(ROOT / canonical["data_manifest"]) == canonical["data_manifest_sha256"]

    registry = load_json(canonical["document_authority_registry"])
    assert registry["status"] == "CURRENT_AUTHORITY_EXPLICIT_HISTORICAL_EXECUTION_QUARANTINED"
    assert registry["current_phase"] == "DISCOVERY_ARCHIVED_CLAIM_REVIEW_PENDING_CONFIRMATION20_SEALED"
    assert registry["hard_interpretation_rules"]["old_pid_or_heartbeat_means_job_is_live"] is False

    project = load_json("PROJECT_CONTRACT.json")
    assert project["status"] == "DISCOVERY_COMPLETE_ARCHIVED_AWAITING_CLAIM_REVIEW"
    assert project["completion"]["live_training_authorized"] is False
    assert project["completion"]["confirmation20_opened"] is False

    portfolio = load_json("configs/FULL_DATA_METHOD_PORTFOLIO.json")
    assert portfolio["authority_scope"] == "HISTORICAL_APPEND_ONLY_OPERATIONAL_LEDGER_WITH_FINAL_OVERLAY"
    assert portfolio["nested_running_pid_and_waiter_fields_are_current"] is False
    assert portfolio["final_outcome_overlay"]["proposal"] == "pass_preregistered_full_data_gate"

    delivery = load_json("configs/PAPER_DELIVERY_COMPLETION_MATRIX.json")
    assert delivery["authority_scope"] == "HISTORICAL_APPEND_ONLY_DELIVERY_LEDGER_WITH_FINAL_OVERLAY"
    assert delivery["nested_waiter_pid_and_running_fields_are_current"] is False

    for entry in archive["entries"]:
        path = ROOT / entry["path"]
        assert path.is_file(), f"missing archive entry: {entry['path']}"
        assert path.stat().st_size == entry["bytes"], f"archive size mismatch: {entry['path']}"
        assert sha256(path) == entry["sha256"], f"archive hash mismatch: {entry['path']}"

    split_counts: dict[str, int] = {}
    with (ROOT / canonical["data_manifest"]).open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
    assert split_counts == {"train": 8553, "discovery": 480, "confirmation": 120}

    source_entries = verify_source_manifest()
    print(
        "PASS git-only research recovery: "
        f"{source_entries} source entries, {archive['entry_count']} archive entries, "
        "9153 data identities; confirmation20 sealed; binary assets external."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
