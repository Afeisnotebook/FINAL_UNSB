"""CLI for the complete two-candidate 5090 frontier adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.frontier_adjudication import adjudicate_frontier


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        adjudicate_frontier(args.receipt, args.output),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()

