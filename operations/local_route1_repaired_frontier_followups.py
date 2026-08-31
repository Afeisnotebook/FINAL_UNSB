"""CLI for freezing evidence-qualified repaired-frontier follow-ups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.repaired_frontier_followups import (
    materialize_repaired_frontier_followups,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        materialize_repaired_frontier_followups(
            args.output,
            adjudication_path=args.adjudication,
            output_path=args.result,
        ),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
