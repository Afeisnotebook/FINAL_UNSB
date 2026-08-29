"""Command line entry point for the isolated local route-1 workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .anchors import run_anchor, summarize_anchors
from .candidate_runner import run_candidate, summarize_candidate
from .causal_audit import DEFAULT_HORIZONS, run_audit_job
from .gates import run_cpu_gates, run_gpu_gates
from .lineage import write_lineage
from .protocol import ROOT
from .runtime import write_json
from .seed_validation import (
    run_seed_validation_lane,
    seed_validation_status,
    summarize_seed_validation,
)
from .stages import derive_from_completed_atlas, prepare_audit_queue, validate_candidate_ready


DEFAULT_OUTPUT = ROOT.parent / "runs" / "FINAL_UNSB_LOCAL_ROUTE1_E200"
DEFAULT_TRAIN_VIEW = ROOT.parent / "FOUR_METHOD_MOTIVATION_20260813" / "frozen" / "data_views_v2" / "allinone_100"
DEFAULT_DATA_ROOT = Path(r"E:\UNSB_abl\full_dataset")
DEFAULT_MANIFEST = ROOT / "manifests" / "frozen" / "legacy_split_manifest.csv"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--stage", required=True,
        choices=[
            "lineage", "gate", "anchors", "evaluate", "audit", "derive",
            "candidate", "seed_validate",
        ],
    )
    value.add_argument("--lane", choices=["plain", "hj", "hnek", "dt"])
    value.add_argument("--candidate-id")
    value.add_argument(
        "--candidate-action", choices=["status", "train", "evaluate"], default="status",
        help="candidate stage action; status never launches training",
    )
    value.add_argument("--validation-seed", type=int, choices=[2027, 2028])
    value.add_argument("--validation-lane", choices=["plain", "candidate"])
    value.add_argument(
        "--validation-action", choices=["status", "train", "evaluate"], default="status",
    )
    value.add_argument("--resume", action="store_true")
    value.add_argument("--cpu-only", action="store_true", help="gate stage only")
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--train-view", type=Path, default=DEFAULT_TRAIN_VIEW)
    value.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--engineering-stop-after-epoch", type=int, help="debug only; never a scientific early-stop rule")
    value.add_argument("--audit-probe", choices=["hj", "hnek", "dt"])
    value.add_argument("--audit-epoch", type=int)
    value.add_argument(
        "--audit-horizons", default=",".join(str(value) for value in DEFAULT_HORIZONS),
        help="comma-separated actual-update horizons; scientific default is 1,8,32,200",
    )
    value.add_argument(
        "--audit-label-horizons", default="200",
        help="comma-separated horizons labeled with discovery70 only after both branches",
    )
    value.add_argument(
        "--training-worktree", type=Path,
        help="immutable training worktree used to prove audit/training core equivalence",
    )
    return value


def _integer_csv(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(item.strip()) for item in str(value).split(",") if item.strip()}))
    if not parsed or any(item <= 0 for item in parsed):
        raise ValueError("horizons must be a non-empty comma-separated list of positive integers")
    return parsed


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "lineage":
        path = write_lineage(args.output, args.manifest.resolve())
        print(json.dumps({"status": "COMPLETE", "lineage": str(path)}, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "gate":
        cpu = run_cpu_gates(
            manifest_path=args.manifest.resolve(), train_view=args.train_view.resolve(),
            data_root=args.data_root.resolve(),
        )
        write_json(args.output / "gates" / "CPU_GATE.json", cpu)
        if cpu["status"] != "PASS":
            print(json.dumps(cpu, ensure_ascii=False, indent=2))
            return 2
        if args.cpu_only:
            print(json.dumps(cpu, ensure_ascii=False, indent=2))
            return 0
        gpu = run_gpu_gates(
            output_root=args.output, manifest_path=args.manifest.resolve(),
            train_view=args.train_view.resolve(), data_root=args.data_root.resolve(),
            gpu=args.gpu,
        )
        print(json.dumps({"cpu": cpu, "gpu": gpu}, ensure_ascii=False, indent=2))
        return 0 if gpu["status"] == "PASS" else 3
    if args.stage == "anchors":
        if not args.lane:
            raise SystemExit("--stage anchors requires --lane plain|hj|hnek|dt")
        result = run_anchor(
            probe_id=args.lane, output_root=args.output,
            train_view=args.train_view.resolve(), data_root=args.data_root.resolve(),
            manifest_path=args.manifest.resolve(), gpu=args.gpu, resume=args.resume,
            engineering_stop_after_epoch=args.engineering_stop_after_epoch,
        )
        if args.lane in ("hnek", "dt") and result["final_data_epoch"] == 200:
            result["evaluation"] = summarize_anchors(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "evaluate":
        result = summarize_anchors(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.stage == "audit":
        if args.audit_probe or args.audit_epoch is not None:
            if not args.audit_probe or args.audit_epoch is None:
                raise SystemExit("executing an audit requires both --audit-probe and --audit-epoch")
            result = run_audit_job(
                output_root=args.output,
                probe=args.audit_probe,
                epoch=args.audit_epoch,
                train_view=args.train_view.resolve(),
                data_root=args.data_root.resolve(),
                manifest_path=args.manifest.resolve(),
                gpu=args.gpu,
                horizons=_integer_csv(args.audit_horizons),
                label_horizons=_integer_csv(args.audit_label_horizons),
                training_root=(
                    None if args.training_worktree is None
                    else args.training_worktree.resolve()
                ),
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        result = prepare_audit_queue(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "READY" else 4
    if args.stage == "derive":
        result = derive_from_completed_atlas(args.output)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] != "BLOCKED_CAUSAL_ATLAS_INCOMPLETE" else 5
    if args.stage == "candidate":
        if not args.candidate_id:
            raise SystemExit("--stage candidate requires --candidate-id")
        if args.candidate_action == "status":
            result = validate_candidate_ready(args.output, args.candidate_id)
            return_code = 0 if str(result.get("status", "")).startswith("READY_") else 6
        elif args.candidate_action == "train":
            result = run_candidate(
                output_root=args.output,
                candidate_id=args.candidate_id,
                train_view=args.train_view.resolve(),
                data_root=args.data_root.resolve(),
                manifest_path=args.manifest.resolve(),
                gpu=args.gpu,
                resume=args.resume,
                engineering_stop_after_epoch=args.engineering_stop_after_epoch,
            )
            return_code = 0
        else:
            result = summarize_candidate(args.output, args.candidate_id)
            return_code = 0
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return return_code
    if args.stage == "seed_validate":
        if not args.candidate_id or args.validation_seed is None:
            raise SystemExit(
                "--stage seed_validate requires --candidate-id and --validation-seed"
            )
        if args.validation_action == "status":
            result = seed_validation_status(
                args.output, args.candidate_id, args.validation_seed,
            )
        elif args.validation_action == "evaluate":
            result = summarize_seed_validation(
                args.output, args.candidate_id, args.validation_seed,
            )
        else:
            if not args.validation_lane:
                raise SystemExit(
                    "training seed validation requires --validation-lane plain|candidate"
                )
            result = run_seed_validation_lane(
                output_root=args.output,
                candidate_id=args.candidate_id,
                seed=args.validation_seed,
                lane=args.validation_lane,
                train_view=args.train_view.resolve(),
                data_root=args.data_root.resolve(),
                manifest_path=args.manifest.resolve(),
                gpu=args.gpu,
                resume=args.resume,
                engineering_stop_after_epoch=args.engineering_stop_after_epoch,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
