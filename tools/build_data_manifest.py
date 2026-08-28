#!/usr/bin/env python3
"""Build the canonical leak-free full-corpus manifest without reading pixels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_map(path: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS:
            if item.stem in result:
                raise RuntimeError(f"duplicate stem in {path}: {item.stem}")
            result[item.stem] = item
    return result


def frozen_splits(path: Path) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            split = row["split"]
            if split not in {"discovery", "confirmation"}:
                continue
            key = (row["domain"], row["stem"])
            if key in result:
                raise RuntimeError(f"duplicate frozen identity: {key}")
            result[key] = split
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--frozen-split",
        type=Path,
        default=ROOT / "manifests/frozen/legacy_split_manifest.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-content", action="store_true")
    args = parser.parse_args()

    contract = json.loads((ROOT / "DATA_CONTRACT.json").read_text(encoding="utf-8"))
    expected = contract["domains"]
    reserved = frozen_splits(args.frozen_split)
    rows: list[dict] = []
    counts: dict[str, dict[str, int]] = {}

    for domain, expected_counts in expected.items():
        input_dir = args.data_root / domain / "input"
        target_dir = args.data_root / domain / "target"
        if not input_dir.is_dir() or not target_dir.is_dir():
            raise RuntimeError(f"missing input/target directory for {domain}")
        inputs = image_map(input_dir)
        targets = image_map(target_dir)
        if set(inputs) != set(targets):
            missing_target = sorted(set(inputs) - set(targets))[:10]
            missing_input = sorted(set(targets) - set(inputs))[:10]
            raise RuntimeError(
                f"physical stem mismatch in {domain}; "
                f"missing_target={missing_target}, missing_input={missing_input}"
            )
        if len(inputs) != int(expected_counts["physical"]):
            raise RuntimeError(
                f"{domain} physical count {len(inputs)} != {expected_counts['physical']}"
            )
        counts[domain] = {"train": 0, "discovery": 0, "confirmation": 0}
        for stem in sorted(inputs):
            split = reserved.get((domain, stem), "train")
            counts[domain][split] += 1
            input_path = inputs[stem]
            target_path = targets[stem]
            rows.append(
                {
                    "domain": domain,
                    "split": split,
                    "stem": stem,
                    "input_relpath": input_path.relative_to(args.data_root).as_posix(),
                    "target_relpath": target_path.relative_to(args.data_root).as_posix(),
                    "input_bytes": input_path.stat().st_size,
                    "target_bytes": target_path.stat().st_size,
                    "input_sha256": sha256_file(input_path) if args.hash_content else "",
                    "target_sha256": sha256_file(target_path) if args.hash_content else "",
                }
            )
        wanted = {
            "train": int(expected_counts["train"]),
            "discovery": 80,
            "confirmation": 20,
        }
        if counts[domain] != wanted:
            raise RuntimeError(f"{domain} split counts {counts[domain]} != {wanted}")

    if len(reserved) != 600:
        raise RuntimeError(f"frozen reserved identities {len(reserved)} != 600")
    seen = {(row["domain"], row["stem"]) for row in rows}
    missing_reserved = sorted(set(reserved) - seen)
    if missing_reserved:
        raise RuntimeError(f"reserved identities absent from corpus: {missing_reserved[:10]}")

    split_order = {"train": 0, "discovery": 1, "confirmation": 2}
    rows.sort(key=lambda r: (split_order[r["split"]], r["domain"], r["stem"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": 1,
        "manifest": args.output.name,
        "manifest_sha256": sha256_file(args.output),
        "content_hashes_present": bool(args.hash_content),
        "counts": counts,
        "totals": {
            split: sum(domain_counts[split] for domain_counts in counts.values())
            for split in ("train", "discovery", "confirmation")
        },
    }
    sidecar = Path(str(args.output) + ".json")
    sidecar.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
