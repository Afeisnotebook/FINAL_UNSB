"""CLI for the full-data paper benchmark and target-blind terminal audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from research.local_route1.runtime import load_model_state, write_json

from .adjudicate import adjudicate
from .evaluate import evaluate_live_model
from .gates import (
    authorize_lane,
    create_runtime_twin_receipt,
    external_gate_status,
    run_evaluation_repeat_gate,
    run_preflight,
    run_resume_gate,
    run_zero_intervention_gate,
)
from .protocol import ROOT, lane_spec, load_protocol
from .runtime import (
    _annotated_rows,
    load_full_state,
    manifest_report,
    prepare_lane,
    train_lane,
)
from .terminal_audit import append_audit, audit_model


DEFAULT_OUTPUT = ROOT.parent / "runs" / "FINAL_UNSB_PAPER_AIO_V1"
DEFAULT_MANIFEST = ROOT / "manifests" / "FULL_DATA_MANIFEST.csv"
DEFAULT_DATA_ROOT = Path(r"E:\UNSB_abl\full_dataset")
DEFAULT_VIEW = ROOT.parent / "paper_views" / "FINAL_UNSB_FULL_AIO_V1"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument(
        "--stage", required=True,
        choices=[
            "preflight", "materialize", "runtime-twin", "resume-gate", "external-gate",
            "zero-intervention-gate", "authorize", "train", "evaluate",
            "evaluation-repeat-gate", "terminal-audit", "adjudicate",
        ],
    )
    value.add_argument("--lane", choices=["plain", "proposal", "hjcgr", "amtnc", "cyclegan", "cut", "ddsb"])
    value.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    value.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    value.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    value.add_argument("--train-view", type=Path, default=DEFAULT_VIEW)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--engineering-stop-after-updates", type=int)
    value.add_argument("--host-label", default="local")
    value.add_argument("--peer-receipt", type=Path)
    value.add_argument("--epoch", type=int)
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--audit-replicates", type=int, default=32)
    value.add_argument("--audit-samples-per-domain", type=int, default=1)
    value.add_argument("--audit-gradient-replicates", type=int, default=4)
    value.add_argument("--skip-content-hashes", action="store_true")
    value.add_argument("--node-role", choices=["training", "audit_only"], default="training")
    value.add_argument(
        "--matched-plain-mode",
        choices=["same_runtime_output_root", "exact_cross_4090_cohort"],
    )
    value.add_argument("--runtime-receipt", type=Path)
    return value


def _materialize(args) -> dict:
    report = manifest_report(args.manifest.resolve(), data_root=args.data_root.resolve())
    command = [
        sys.executable, str(ROOT / "tools" / "materialize_views.py"),
        "--manifest", str(args.manifest.resolve()),
        "--data-root", str(args.data_root.resolve()),
        "--view-root", str(args.train_view.resolve()),
        "--mode", "auto",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    gate = run_preflight(
        output_root=args.output.resolve(), manifest_path=args.manifest.resolve(),
        data_root=None if args.skip_content_hashes else args.data_root.resolve(),
        train_view=args.train_view.resolve(),
        node_role=args.node_role,
    )
    return {"manifest": report, "preflight": gate}


def _load_checkpoint_model(args):
    if not args.lane or not args.checkpoint:
        raise SystemExit("stage requires --lane and --checkpoint")
    spec = lane_spec(args.lane)
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model, primary, secondary, rows = prepare_lane(
        output_root=args.output.resolve(), train_view=args.train_view.resolve(),
        manifest_path=args.manifest.resolve(), spec=spec, gpu=args.gpu,
    )
    load_full_state(
        args.checkpoint.resolve(), model=model, spec=spec, primary=primary,
        secondary=secondary, expected_metadata=payload["metadata"],
    )
    return model, spec, rows, payload, primary, secondary


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        result = run_preflight(
            output_root=args.output, manifest_path=args.manifest.resolve(),
            data_root=None if args.skip_content_hashes else args.data_root.resolve(),
            train_view=args.train_view.resolve() if args.train_view.exists() else None,
            node_role=args.node_role,
        )
    elif args.stage == "materialize":
        result = _materialize(args)
    elif args.stage == "runtime-twin":
        result = create_runtime_twin_receipt(
            output_root=args.output, train_view=args.train_view.resolve(),
            data_root=args.data_root.resolve(), manifest_path=args.manifest.resolve(),
            gpu=args.gpu, host_label=args.host_label,
            peer_receipt=None if args.peer_receipt is None else args.peer_receipt.resolve(),
        )
    elif args.stage == "resume-gate":
        if not args.lane:
            raise SystemExit("resume-gate requires --lane")
        result = run_resume_gate(
            output_root=args.output, train_view=args.train_view.resolve(),
            data_root=args.data_root.resolve(), manifest_path=args.manifest.resolve(),
            gpu=args.gpu, lane_id=args.lane,
        )
    elif args.stage == "zero-intervention-gate":
        result = run_zero_intervention_gate(
            output_root=args.output, train_view=args.train_view.resolve(),
            manifest_path=args.manifest.resolve(), gpu=args.gpu,
        )
    elif args.stage == "authorize":
        if not args.lane:
            raise SystemExit("authorize requires --lane")
        result = authorize_lane(
            output_root=args.output, lane_id=args.lane,
            matched_plain_mode=args.matched_plain_mode,
            runtime_receipt=(
                None if args.runtime_receipt is None else args.runtime_receipt.resolve()
            ),
        )
    elif args.stage == "external-gate":
        if args.lane not in ("cut", "ddsb"):
            raise SystemExit("external-gate requires --lane cut|ddsb")
        result = external_gate_status(args.output, args.lane)
    elif args.stage == "train":
        if not args.lane:
            raise SystemExit("train requires --lane")
        result = train_lane(
            lane_id=args.lane, output_root=args.output,
            train_view=args.train_view.resolve(), data_root=args.data_root.resolve(),
            manifest_path=args.manifest.resolve(), gpu=args.gpu, resume=args.resume,
            engineering_stop_after_updates=args.engineering_stop_after_updates,
        )
    elif args.stage == "evaluate":
        if args.epoch is None:
            raise SystemExit("evaluate requires --epoch")
        model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
        result = evaluate_live_model(
            model=model, spec=spec, rows=rows, data_root=args.data_root.resolve(),
            protocol_hash=payload["metadata"]["protocol_fingerprint"],
            epoch=args.epoch, lane_root=args.output / "lanes" / spec.id,
        )
    elif args.stage == "evaluation-repeat-gate":
        model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
        result = run_evaluation_repeat_gate(
            output_root=args.output, model=model, spec=spec, rows=rows,
            data_root=args.data_root.resolve(),
            protocol_hash=payload["metadata"]["protocol_fingerprint"],
        )
    elif args.stage == "terminal-audit":
        model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
        result = audit_model(
            model=model, spec=spec, rows=rows, data_root=args.data_root.resolve(),
            protocol_hash=payload["metadata"]["protocol_fingerprint"],
            checkpoint_label=args.checkpoint.name,
            replicates=args.audit_replicates,
            samples_per_domain=args.audit_samples_per_domain,
            primary=primary, secondary=secondary,
            gradient_replicates=args.audit_gradient_replicates,
        )
        append_audit(args.output / "TERMINAL_AUDIT.jsonl", result)
    else:
        result = adjudicate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
