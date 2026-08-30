"""Materialize final route-1 delivery only after every registered hard gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.final_delivery import materialize_final_delivery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        materialize_final_delivery(args.output), ensure_ascii=False, indent=2,
    ))


if __name__ == "__main__":
    main()
