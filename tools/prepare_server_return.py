#!/usr/bin/env python3
"""Copy only compact, auditable lane artifacts into a Git return directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EPOCHS = (1, 10, 25, 50, 100, 150, 200)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--inbox", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lane_root = args.run_root / args.lane
    sources = [
        lane_root / "E0_IDENTITY.json",
        lane_root / "TRAIN_SUMMARY.json",
        args.inbox / f"{args.lane}_ENVIRONMENT.json",
    ]
    sources += [
        lane_root / "milestones" / f"full_state_e{epoch}.pt.json"
        for epoch in EPOCHS
    ]
    sources += [
        args.inbox / f"{args.lane}_DISCOVERY_E{epoch}.json"
        for epoch in EPOCHS
    ]
    missing = [str(path) for path in sources if not path.is_file()]
    if missing:
        raise RuntimeError(f"server return is incomplete: {missing}")
    args.output.mkdir(parents=True, exist_ok=True)
    index = []
    for source in sources:
        target = args.output / source.name
        shutil.copy2(source, target)
        index.append({"file": target.name, "sha256": sha256(target), "bytes": target.stat().st_size})
    payload = {"schema_version": 1, "lane_id": args.lane, "files": index}
    index_path = args.output / "RETURN_INDEX.json"
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
