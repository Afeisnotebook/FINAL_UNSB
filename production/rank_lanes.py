#!/usr/bin/env python3
"""Rank matched lane evaluations against P0_PLAIN without checkpoint picking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from production import common


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [common.load_json(path) for path in args.evaluations]
    by_lane = {item["lane_id"]: item for item in results}
    expected = {"P0_PLAIN", "P1_HJ_HANDOFF", "P2_HNEK", "P3_MACRO_MARGINAL"}
    if set(by_lane) != expected:
        raise RuntimeError(f"expected exactly four lanes, got {sorted(by_lane)}")
    identities = {
        (
            item["split"], item["checkpoint_metadata"]["epoch_completed"],
            item["protocol_sha256"], item["checkpoint_metadata"]["project_id"],
            item["checkpoint_metadata"]["seed"],
            item["checkpoint_metadata"]["manifest_sha256"],
            item["checkpoint_metadata"]["protocol_fingerprint"],
            item["checkpoint_metadata"]["steps_per_epoch"],
        ) for item in results
    }
    if len(identities) != 1:
        raise RuntimeError(f"evaluations are not matched: {identities}")
    plain = by_lane["P0_PLAIN"]["summary"]
    ranking = []
    for lane_id, item in by_lane.items():
        summary = item["summary"]
        per_domain_delta = {
            domain: summary["per_domain"][domain]["psnr"] - plain["per_domain"][domain]["psnr"]
            for domain in sorted(plain["per_domain"])
        }
        ranking.append({
            "lane_id": lane_id,
            "macro_psnr": summary["macro_psnr"],
            "macro_ssim": summary["macro_ssim"],
            "macro_psnr_delta_vs_plain": summary["macro_psnr"] - plain["macro_psnr"],
            "positive_domains": sum(value > 0 for value in per_domain_delta.values()),
            "worst_domain_delta": min(per_domain_delta.values()),
            "per_domain_psnr_delta_vs_plain": per_domain_delta,
        })
    ranking.sort(
        key=lambda row: (row["macro_psnr_delta_vs_plain"], row["positive_domains"], row["worst_domain_delta"]),
        reverse=True,
    )
    payload = {
        "schema_version": 1, "matched_identity": list(identities)[0],
        "selection_rule": "e200 primary; macro delta, positive domains, worst domain",
        "ranking": ranking,
        "provisional_candidate": next(row for row in ranking if row["lane_id"] != "P0_PLAIN"),
    }
    common.atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
