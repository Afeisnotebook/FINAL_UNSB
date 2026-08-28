#!/usr/bin/env python3
"""Write a stable SHA-256 inventory for source, contracts and evidence."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache",
    "runs", "data_view", "local_validation",
}
EXCLUDED_FILES = {"SOURCE_MANIFEST.sha256"}


def digest(path: Path) -> str:
    data = path.read_bytes()
    # Keep the inventory stable across Windows/Linux clones, including text
    # control files without a normal suffix (e.g. .gitignore and *.example).
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "SOURCE_MANIFEST.sha256")
    args = parser.parse_args()
    rows = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        if path.name in EXCLUDED_FILES or relative.parts[:2] == ("reports", "inbox"):
            continue
        rows.append(f"{digest(path)}  {relative.as_posix()}")
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"wrote {len(rows)} entries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
