"""Export the evidence-qualified 4090-to-5090 replay portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.cross_runtime_portfolio import (
    export_cross_runtime_portfolio,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontier", type=Path)
    parser.add_argument("--portable-output", type=Path)
    args = parser.parse_args(argv)
    result = export_cross_runtime_portfolio(
        args.output,
        frontier_path=args.frontier,
        output_path=args.portable_output,
    )
    print(json.dumps({
        "status": result["status"],
        "replay_candidate_ids": result["replay_candidate_ids"],
        "source_frontier_sha256": result["source_frontier_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
