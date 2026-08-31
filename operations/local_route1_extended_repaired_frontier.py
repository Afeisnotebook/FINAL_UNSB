"""CLI for the full-plus-proposal repaired-frontier adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.extended_repaired_frontier import (
    materialize_extended_repaired_frontier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        materialize_extended_repaired_frontier(
            args.output_root, output_path=args.output,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
