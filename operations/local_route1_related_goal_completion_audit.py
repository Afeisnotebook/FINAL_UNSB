"""CLI for the terminal related multi-algorithm Goal audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.related_goal_completion_audit import (
    materialize_related_goal_completion_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compatibility-delivery", type=Path, required=True)
    parser.add_argument("--related-delivery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = materialize_related_goal_completion_audit(
        args.compatibility_delivery, args.related_delivery, args.output,
    )
    print(json.dumps({
        "schema": result["schema"],
        "status": result["status"],
        "action_priority_candidate_id": result["action_priority_candidate_id"],
        "algorithm_set_status": result["algorithm_set_status"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

