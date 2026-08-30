"""Adjudicate a proposal-only challenger after both frozen seed records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.ablation_challenger_selection import (
    adjudicate_ablation_challenger_selection,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--challenger-workspace", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(adjudicate_ablation_challenger_selection(
        args.output, args.challenger_workspace,
    ), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
