"""Run target-blind negative-candidate defect audits and revision routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.candidate_defect_audit import (
    adjudicate_revision_need,
    audit_candidate_defect,
)


DEFAULT_IDS = [
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
    "G1-02-SAMPLING-VARIANCE",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--train-view", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    args = parser.parse_args()
    if args.aggregate:
        result = adjudicate_revision_need(args.output, DEFAULT_IDS)
    else:
        if not args.candidate_id or not args.train_view or not args.manifest:
            raise SystemExit("candidate audit requires --candidate-id --train-view --manifest")
        result = audit_candidate_defect(
            output_root=args.output, candidate_id=args.candidate_id,
            train_view=args.train_view, manifest_path=args.manifest,
            gpu=args.gpu, samples=args.samples,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
