"""Run the target-blind antithetic Gaussian G/F gradient audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.antithetic_gradient_audit import (
    run_antithetic_gradient_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 100, 200])
    parser.add_argument("--pairs", type=int, default=8)
    args = parser.parse_args()
    result = run_antithetic_gradient_audit(
        output_root=args.output,
        train_view=args.train_view,
        manifest_path=args.manifest,
        gpu=args.gpu,
        epochs=tuple(args.epochs),
        pairs=args.pairs,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

