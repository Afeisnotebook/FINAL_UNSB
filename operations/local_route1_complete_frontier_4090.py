"""CLI for the complete same-host 4090 route-1 frontier adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.complete_frontier import (
    materialize_complete_4090_frontier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        materialize_complete_4090_frontier(args.output),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
