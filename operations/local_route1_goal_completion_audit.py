"""Validate the retrieved route-1 terminal delivery against Goal requirements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.goal_completion_audit import (
    materialize_goal_completion_audit,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--delivery", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = materialize_goal_completion_audit(args.delivery, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
