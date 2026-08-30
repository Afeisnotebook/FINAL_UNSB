"""Freeze only the selected winner's source-bound mechanism ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.winner_ablations import (
    materialize_winner_ablation_definitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        materialize_winner_ablation_definitions(args.output),
        ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
