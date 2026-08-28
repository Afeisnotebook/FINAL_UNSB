#!/usr/bin/env python3
"""CPU-only coherence checks for the frozen project and optional data manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)
    print(f"PASS {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_text_sha256(path: Path) -> str:
    """Hash a committed text contract independently of checkout newlines."""
    data = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    project = load("PROJECT_CONTRACT.json")
    state = load("PROJECT_STATE.json")
    lanes = load("configs/FOUR_LANES.json")
    data = load("DATA_CONTRACT.json")
    budget = load("COMPUTE_BUDGET.json")
    evidence = load("evidence/EVIDENCE_SUMMARY.json")

    lane_ids = [lane["id"] for lane in lanes["lanes"]]
    check(lane_ids == state["active_lanes"], "state and lane order agree")
    check(len(lane_ids) == 4 and len(set(lane_ids)) == 4, "exactly four unique lanes")
    check(project["frozen"]["train_identities"] == data["totals"]["train"] == 8553,
          "train identity count is 8553")
    check(project["frozen"]["updates_per_lane"] == 8553 * 200,
          "updates per lane equal train identities times epochs")
    check(budget["parallel_lanes"] == 4, "compute budget has four parallel lanes")
    hj = next(lane for lane in lanes["lanes"] if lane["id"] == "P1_HJ_HANDOFF")
    check(hj["method"]["active_start_data_epoch"] == 1.6 and
          hj["method"]["active_end_data_epoch"] == 8.0,
          "HJ physical-epoch window is frozen")
    macro = next(lane for lane in lanes["lanes"] if lane["id"] == "P3_MACRO_MARGINAL")
    check(macro["method"]["A_domain_and_B_domain_independent"] is True,
          "macro lane is not DCUM")
    check(
        portable_text_sha256(ROOT / data["split_source"]) == data["frozen_split_sha256"],
        "frozen legacy split hash matches contract",
    )
    check(len(evidence["facts"]) == 5, "curated evidence contains five decision facts")

    final1 = load("evidence/raw/FINAL1_TA_E200_EVALUATION.json")
    check(abs(final1["macro_psnr_delta_db"] + 1.092243732049191) < 1e-12,
          "TA e200 negative evidence is exact")
    a3r2 = load("evidence/raw/A3R2_KCK_E10_AUDIT.json")
    check(a3r2["positive_time_triples"] == 0 and a3r2["positive_domains"] == 0,
          "KCK negative evidence is exact")
    hnek = load("evidence/raw/HNEK_G025_E200_FINAL.json")
    check(abs(hnek["decisions"][0]["macro_psnr_delta_db"] - 0.7883720592327812) < 1e-12,
          "HNEK e200 anchor is exact")

    manifest = args.manifest or ROOT / data["canonical_manifest"]
    if manifest.is_file():
        check(sha256(manifest) == data["canonical_manifest_sha256"],
              "canonical full manifest hash matches contract")
        with manifest.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        counts = Counter(row["split"] for row in rows)
        identities = {(row["domain"], row["stem"]) for row in rows}
        check(len(rows) == len(identities) == 9153, "manifest has 9153 unique identities")
        check(counts == Counter({"train": 8553, "discovery": 480, "confirmation": 120}),
              "manifest split totals match contract")
        check(all(row["input_sha256"] and row["target_sha256"] for row in rows),
              "manifest contains input and target content hashes")
        actual_domains = {
            domain: Counter(row["split"] for row in rows if row["domain"] == domain)
            for domain in data["domains"]
        }
        expected_domains = {
            domain: Counter(train=values["train"], discovery=80, confirmation=20)
            for domain, values in data["domains"].items()
        }
        check(actual_domains == expected_domains, "manifest per-domain splits match contract")

    print("ALL CONTRACT CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
