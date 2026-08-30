"""Adjudicate the two frozen Generation-1 trajectories after matched e200."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.generation1_adjudication import adjudicate_generation1


DEFAULT_IDS = (
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
    "G1-02-SAMPLING-VARIANCE",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id", action="append", dest="candidate_ids")
    parser.add_argument(
        "--freeze-winner", action="store_true",
        help="write the immutable seed-validation freeze only when the winner passed all gates",
    )
    args = parser.parse_args()
    result = adjudicate_generation1(
        args.output,
        DEFAULT_IDS if args.candidate_ids is None else args.candidate_ids,
        freeze_winner=args.freeze_winner,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

