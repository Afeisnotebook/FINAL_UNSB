#!/usr/bin/env python3
"""CPU-only coherence checks for the active project and optional data manifest."""

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
    probes = load("configs/LOCAL_ROUTE1_PROBES.json")
    paper = load("configs/PAPER_AIO_UNPAIRED_V1.json")
    lanes = load("configs/FOUR_LANES.json")
    data = load("DATA_CONTRACT.json")
    budget = load("COMPUTE_BUDGET.json")
    evidence = load("evidence/EVIDENCE_SUMMARY.json")

    probe_ids = [probe["id"] for probe in probes["anchor_probes"]]
    check(
        project["status"]
        == "ACTIVE_FULL_DATA_PAPER_AND_ALGORITHM_RECONSTRUCTION",
        "full-data paper research and evidence-driven reconstruction are active",
    )
    check(probes["status"] == "ACTIVE_LOCAL_RESEARCH",
          "local long-horizon probes are active")
    check(probe_ids == state["active_probe_families"],
          "state and active probe order agree")
    check(len(probe_ids) == 4 and len(set(probe_ids)) == 4,
          "plain, HJ, HNEK and DT are four unique anchor probes")
    local = probes["local_view"]
    check(local["total_train"] == local["train_per_domain"] * local["domains"] == 150,
          "small25 local anchor contains 150 training identities")
    check(local["target_updates_per_lane"] == local["total_train"] * local["target_epochs"] == 30000,
          "local anchor e200 equals 30000 updates")
    check(project["frozen"]["updates_per_local_anchor_lane"] == 30000,
          "project and probe long-horizon clocks agree")
    offload = project.get("authorization_required") or {}
    required_decisions = {
        "decisions/DEC-20260830-ROUTE1-REMOTE-OFFLOAD.md",
        "decisions/DEC-20260830-ROUTE1-REMOTE5090.md",
        "decisions/DEC-20260830-ROUTE1-INDEPENDENT-PROBE-CONCURRENCY.md",
    }
    required_hosts = {
        "local GTX 1660", "192.168.0.30 RTX 4090A",
        "final-unsb-5090 RTX 5090A", "final-unsb-5090b RTX 5090B",
    }
    required_exclusions = {
        "confirmation20 access",
        "route2 handoff or exit search",
        "cross-host method-minus-plain comparisons",
        "paired metric training or scheduling control",
        "best checkpoint selection",
        "unprovided extra RTX 4090 hosts",
    }
    check(
        offload.get("status") == "GRANTED_FULL_DATA_PAPER_AND_ROUTE1_RECONSTRUCTION"
        and required_decisions.issubset(set(offload.get("decisions", [])))
        and "decisions/DEC-20260902-PAPER-AIO-ACTIVATION.md"
        in set(offload.get("decisions", []))
        and required_hosts == set(offload.get("hosts", []))
        and required_exclusions.issubset(set(offload.get("excludes", []))),
        "paper and route-1 multi-host work has explicit bounded authorization",
    )
    full = project["paper_full_frozen"]
    check(
        paper["status"] == "ACTIVE_FULL_DATA_PAPER_RESEARCH"
        and full["manifest_sha256"] == data["canonical_manifest_sha256"]
        and full["train_per_side"] == 8553
        and full["updates_per_lane"] == 1_710_600
        and full["batch_size"] == 1
        and full["confirmation_locked"] is True,
        "paper contract, protocol and full-data clock agree",
    )
    current = state.get("paper_aio_20260902") or {}
    check(
        state.get("phase") == "PAPER_AIO_FIRST_WAVE_AND_STCGR_SMALL25_RUNNING"
        and current.get("paired_metric_control") is False
        and current.get("cross_host_deltas_merged") is False
        and current.get("confirmation20_opened") is False,
        "active state preserves paper drift firewalls",
    )
    check(lanes["status"] == "SUSPENDED_NOT_CURRENT",
          "former four-lane server plan is suspended")
    check(budget["status"] == "SUSPENDED_SERVER_BUDGET_NOT_CURRENT",
          "server compute budget is suspended")

    # The suspended plan remains internally coherent provenance, but it is not
    # compared with active project state and cannot authorize execution.
    lane_ids = [lane["id"] for lane in lanes["lanes"]]
    check(len(lane_ids) == 4 and len(set(lane_ids)) == 4,
          "suspended server plan retains four unique historical lanes")
    hj = next(lane for lane in lanes["lanes"] if lane["id"] == "P1_HJ_HANDOFF")
    check(hj["method"]["active_start_data_epoch"] == 1.6 and
          hj["method"]["active_end_data_epoch"] == 8.0,
          "suspended HJ handoff window retains its historical identity")
    macro = next(lane for lane in lanes["lanes"] if lane["id"] == "P3_MACRO_MARGINAL")
    check(macro["method"]["A_domain_and_B_domain_independent"] is True,
          "suspended macro lane remains distinguishable from DCUM")
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
