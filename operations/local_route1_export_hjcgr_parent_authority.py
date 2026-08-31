"""Export source-bound HJCGR construction authority for a replay host."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.hj_multi_algorithm_frontier import (
    export_hjcgr_parent_authority,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--anchor-path", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = export_hjcgr_parent_authority(
        args.output_root,
        anchor_path=args.anchor_path,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
