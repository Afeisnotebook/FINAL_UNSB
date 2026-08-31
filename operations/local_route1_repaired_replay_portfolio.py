"""Export or register the source-bound repaired-algorithm replay portfolio."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.repaired_replay_portfolio import (
    export_portable_authority,
    register_portable_replay,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--export", action="store_true")
    mode.add_argument("--register", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--authority-output", type=Path)
    parser.add_argument("--candidate-id")
    parser.add_argument("--source-repo", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.export:
        value = export_portable_authority(
            args.output,
            adjudication_path=args.adjudication,
            output_path=args.authority_output,
        )
    else:
        if (
            args.authority is None or not args.candidate_id
            or args.source_repo is None or args.python is None
        ):
            raise SystemExit(
                "--register requires --authority, --candidate-id, --source-repo and --python"
            )
        value = register_portable_replay(
            args.output,
            authority_path=args.authority,
            candidate_id=args.candidate_id,
            source_repo=args.source_repo,
            python=args.python,
        )
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
