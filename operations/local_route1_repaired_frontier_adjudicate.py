"""CLI for the semantic-repair-aware 5090 frontier adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.repaired_frontier_adjudication import (
    adjudicate_repaired_frontier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankable-receipt", type=Path, action="append", required=True)
    parser.add_argument(
        "--invalid-diagnostic-receipt", type=Path, action="append", required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        adjudicate_repaired_frontier(
            args.rankable_receipt,
            args.invalid_diagnostic_receipt,
            args.output,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
