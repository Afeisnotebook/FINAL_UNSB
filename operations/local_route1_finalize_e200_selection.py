"""Materialize the immutable all-candidate seed-2026/e200 route-1 ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.final_selection import materialize_final_e200_selection


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    args = parser.parse_args()
    print(json.dumps(
        materialize_final_e200_selection(args.output, args.receipt),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

