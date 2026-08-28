#!/usr/bin/env python3
"""Merge four server E0 reports into a bound long-run authorization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production import common


LANES = ["P0_PLAIN", "P1_HJ_HANDOFF", "P2_HNEK", "P3_MACRO_MARGINAL"]
RUNTIME_KEYS = ["torch", "torch_cuda", "cudnn", "gpu"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--e0", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reports = [common.load_json(path) for path in args.e0]
    by_lane = {report["lane_id"]: report for report in reports}
    if set(by_lane) != set(LANES):
        raise RuntimeError(f"expected one report per lane, got {sorted(by_lane)}")
    manifest_hashes = {report["manifest_sha256"] for report in reports}
    e0_hashes = {report["e0_network_state_sha256"] for report in reports}
    protocol_hashes = {report["protocol_fingerprint"] for report in reports}
    runtimes = {
        tuple(report["environment"].get(key) for key in RUNTIME_KEYS)
        for report in reports
    }
    if manifest_hashes != {common.file_sha256(common.ROOT / "manifests/FULL_DATA_MANIFEST.csv")}:
        raise RuntimeError(f"server manifest mismatch: {manifest_hashes}")
    if protocol_hashes != {common.protocol_fingerprint()}:
        raise RuntimeError(f"server protocol mismatch: {protocol_hashes}")
    if len(e0_hashes) != 1:
        raise RuntimeError(f"four-server e0 mismatch: {e0_hashes}")
    if len(runtimes) != 1:
        raise RuntimeError(f"four-server runtime mismatch: {runtimes}")
    runtime_tuple = next(iter(runtimes))
    payload = {
        "schema_version": 1, "authorized": True,
        "note": "Generated only after four-server preflight identity agreement.",
        "protocol_fingerprint": next(iter(protocol_hashes)),
        "data_manifest_sha256": next(iter(manifest_hashes)),
        "e0_network_state_sha256": next(iter(e0_hashes)),
        "runtime": dict(zip(RUNTIME_KEYS, runtime_tuple)),
        "lanes": LANES,
        "source_reports": [path.name for path in args.e0],
    }
    common.atomic_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
