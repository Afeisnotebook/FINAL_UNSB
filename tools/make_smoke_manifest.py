#!/usr/bin/env python3
"""Create a deterministic, non-scientific local subset from the canonical manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-domain", type=int, default=2)
    parser.add_argument("--discovery-per-domain", type=int, default=1)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        source = list(csv.DictReader(handle))
        fields = list(source[0])
    limits = {"train": args.train_per_domain, "discovery": args.discovery_per_domain}
    counts = defaultdict(int)
    selected = []
    for row in source:
        key = (row["domain"], row["split"])
        if row["split"] in limits and counts[key] < limits[row["split"]]:
            selected.append(row)
            counts[key] += 1
    domains = sorted({row["domain"] for row in source})
    expected = {
        (domain, split): limit for domain in domains for split, limit in limits.items()
    }
    if dict(counts) != expected:
        raise RuntimeError(f"smoke selection incomplete: {dict(counts)} != {expected}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(selected)
    summary = {
        "schema_version": 1, "non_scientific_smoke_only": True,
        "source_manifest_sha256": sha256(args.manifest),
        "manifest_sha256": sha256(args.output),
        "counts": {f"{domain}/{split}": counts[(domain, split)] for domain, split in expected},
    }
    Path(str(args.output) + ".json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
