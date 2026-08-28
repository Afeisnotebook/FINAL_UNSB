#!/usr/bin/env python3
"""Materialize read-only UNSB views using links and domain-prefixed names."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


def link_one(source: Path, target: Path, mode: str) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise RuntimeError(f"existing view points elsewhere: {target}")
        return "existing"
    attempts = [mode] if mode != "auto" else ["symlink", "hardlink"]
    errors = []
    for attempt in attempts:
        try:
            if attempt == "symlink":
                os.symlink(source, target)
            elif attempt == "hardlink":
                os.link(source, target)
            else:
                raise ValueError(f"unknown link mode: {attempt}")
            return attempt
        except OSError as error:
            errors.append(f"{attempt}: {error}")
    raise RuntimeError(f"cannot link {source} -> {target}: {'; '.join(errors)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--view-root", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "symlink", "hardlink"], default="auto")
    args = parser.parse_args()

    counts: dict[str, int] = {}
    methods: dict[str, int] = {}
    with args.manifest.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            phase = split
            suffix_a = Path(row["input_relpath"]).suffix.lower()
            suffix_b = Path(row["target_relpath"]).suffix.lower()
            name_a = f"{row['domain']}__{row['stem']}{suffix_a}"
            name_b = f"{row['domain']}__{row['stem']}{suffix_b}"
            source_a = args.data_root / row["input_relpath"]
            source_b = args.data_root / row["target_relpath"]
            if not source_a.is_file() or not source_b.is_file():
                raise RuntimeError(f"manifest path missing for {row['domain']}/{row['stem']}")
            method_a = link_one(source_a, args.view_root / f"{phase}A" / name_a, args.mode)
            method_b = link_one(source_b, args.view_root / f"{phase}B" / name_b, args.mode)
            counts[f"{phase}A"] = counts.get(f"{phase}A", 0) + 1
            counts[f"{phase}B"] = counts.get(f"{phase}B", 0) + 1
            methods[method_a] = methods.get(method_a, 0) + 1
            methods[method_b] = methods.get(method_b, 0) + 1

    summary = {"schema_version": 1, "counts": counts, "link_methods": methods}
    args.view_root.mkdir(parents=True, exist_ok=True)
    (args.view_root / "VIEW_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
