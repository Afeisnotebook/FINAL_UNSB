from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.local_route1.stratified_time_audit import (
    DEFAULT_EPOCHS,
    DEFAULT_REPLICATES,
    run_stratified_time_fixed_state_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--train-view", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lanes", nargs="+", default=["proposal", "hjcgr"])
    parser.add_argument("--epochs", nargs="+", type=int, default=list(DEFAULT_EPOCHS))
    parser.add_argument("--replicates", type=int, default=DEFAULT_REPLICATES)
    args = parser.parse_args()
    result = run_stratified_time_fixed_state_audit(
        source_root=args.source_root,
        train_view=args.train_view,
        manifest_path=args.manifest,
        output=args.output,
        gpu=args.gpu,
        lanes=tuple(args.lanes),
        epochs=tuple(args.epochs),
        replicates=args.replicates,
    )
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "lane_summaries": result["lane_summaries"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
