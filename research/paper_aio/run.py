"""CLI for the full-data paper benchmark and target-blind terminal audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

from .adjudicate import adjudicate
from .candidate_lock import materialize_candidate_lock
from .candidate_runtime import (
    authorize_candidate,
    load_candidate_spec,
    run_candidate_runtime_gate,
    train_candidate,
)
from .complexity import profile_model
from .distribution import profile_distribution
from .evaluate import evaluate_live_model
from .freeze import create_review_draft, materialize_freeze_receipt
from .gates import (
    authorize_lane,
    create_runtime_twin_receipt,
    external_gate_status,
    run_evaluation_repeat_gate,
    run_preflight,
    run_resume_gate,
    run_zero_intervention_gate,
)
from .protocol import ROOT, evaluation_bundle_fingerprint, lane_spec
from .runtime import (
    _annotated_rows,
    load_full_state,
    manifest_report,
    prepare_lane,
    train_lane,
)
from .runtime_relation import materialize_exact_runtime_relation
from .terminal_audit import append_audit, audit_model
from .unified import (
    candidate_spec_from_portable_authority,
    evaluate_input_reference,
    evaluate_imported_checkpoint,
    export_checkpoint_receipt,
    lock_unified_evaluation_cohort,
)


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
            "candidate-lock",
            "candidate-runtime-gate",
            "checkpoint-export", "input-evaluate", "unified-evaluate", "unified-lock",
            "complexity", "distribution", "freeze-draft", "freeze-materialize",
            "runtime-relation",
        ],
    )
    value.add_argument("--lane", choices=["input", "plain", "proposal", "hjcgr", "amtnc", "cyclegan", "cut", "ddsb", "candidate"])
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
    value.add_argument("--capacity-override-receipt", type=Path)
    value.add_argument(
        "--matched-plain-mode",
        choices=["same_runtime_output_root", "exact_cross_4090_cohort"],
    )
    value.add_argument("--runtime-receipt", type=Path)
    value.add_argument("--candidate-id")
    value.add_argument("--candidate-terminal-receipt", type=Path)
    value.add_argument("--candidate-trajectory", type=Path)
    value.add_argument("--candidate-derivation-card", type=Path)
    value.add_argument("--candidate-implementation", type=Path)
    value.add_argument("--candidate-runtime-gate", type=Path)
    value.add_argument("--candidate-authority", type=Path)
    value.add_argument("--parent-output", type=Path)
    value.add_argument("--parent-runtime-receipt", type=Path)
    value.add_argument("--parent-e0", type=Path)
    value.add_argument("--parent-scientific-git-commit")
    value.add_argument("--parent-protocol-fingerprint")
    value.add_argument(
        "--parent-readiness-mode",
        choices=["complete_e200", "authorized_running"],
        default="complete_e200",
        help=(
            "authorized_running permits candidate training to overlap a healthy "
            "same-host plain; complete e200 remains mandatory before adjudication"
        ),
    )
    value.add_argument("--source-sidecar", type=Path)
    value.add_argument("--source-receipt", type=Path)
    value.add_argument("--receipt-output", type=Path)
    value.add_argument("--source-host-label")
    value.add_argument("--copied-checkpoint", type=Path)
    value.add_argument("--freeze-receipt", type=Path)
    value.add_argument("--portfolio", type=Path)
    value.add_argument("--review-decision", type=Path)
    value.add_argument("--paper-claim", action="append", default=[])
    value.add_argument("--method-runtime-receipt", type=Path)
    value.add_argument("--plain-runtime-receipt", type=Path)
    value.add_argument("--method-authorization-receipt", type=Path)
    value.add_argument("--method-source-host-label")
    value.add_argument("--plain-source-host-label")
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
        capacity_override=args.capacity_override_receipt,
        host_label=args.host_label,
    )
    return {"manifest": report, "preflight": gate}


def _load_checkpoint_model(args):
    if not args.lane or not args.checkpoint:
        raise SystemExit("stage requires --lane and --checkpoint")
    payload = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if args.lane == "candidate":
        if not args.candidate_id:
            raise SystemExit("candidate checkpoint stage requires --candidate-id")
        if args.candidate_authority is not None:
            metadata = payload.get("metadata") or {}
            spec, _ = candidate_spec_from_portable_authority(
                authority_path=args.candidate_authority.resolve(),
                candidate_id=args.candidate_id,
                exported_lane=payload.get("lane"),
                training_git_commit=str(metadata.get("git_commit", "")),
                training_protocol_fingerprint=str(
                    metadata.get("protocol_fingerprint", "")
                ),
            )
        else:
            spec, _ = load_candidate_spec(args.output.resolve(), args.candidate_id)
    else:
        spec = lane_spec(args.lane)
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
            capacity_override=args.capacity_override_receipt,
            host_label=args.host_label,
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
        if args.lane == "candidate":
            raise SystemExit("candidate resume is part of --stage candidate-runtime-gate")
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
        if args.lane == "candidate":
            if not args.candidate_id:
                raise SystemExit("candidate authorization requires --candidate-id")
            result = authorize_candidate(args.output, args.candidate_id)
        else:
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
        if args.lane == "candidate":
            if not args.candidate_id:
                raise SystemExit("candidate training requires --candidate-id")
            result = train_candidate(
                candidate_id=args.candidate_id, output_root=args.output,
                train_view=args.train_view.resolve(), data_root=args.data_root.resolve(),
                manifest_path=args.manifest.resolve(), gpu=args.gpu, resume=args.resume,
                engineering_stop_after_updates=args.engineering_stop_after_updates,
            )
        else:
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
            protocol_hash=evaluation_bundle_fingerprint(),
            epoch=args.epoch, lane_root=args.output / "lanes" / spec.id,
        )
    elif args.stage == "evaluation-repeat-gate":
        model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
        result = run_evaluation_repeat_gate(
            output_root=args.output, model=model, spec=spec, rows=rows,
            data_root=args.data_root.resolve(),
            protocol_hash=evaluation_bundle_fingerprint(),
        )
    elif args.stage == "terminal-audit":
        model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
        result = audit_model(
            model=model, spec=spec, rows=rows, data_root=args.data_root.resolve(),
            protocol_hash=evaluation_bundle_fingerprint(),
            checkpoint_label=args.checkpoint.name,
            replicates=args.audit_replicates,
            samples_per_domain=args.audit_samples_per_domain,
            primary=primary, secondary=secondary,
            gradient_replicates=args.audit_gradient_replicates,
        )
        append_audit(args.output / "TERMINAL_AUDIT.jsonl", result)
    elif args.stage == "complexity":
        if args.receipt_output is None:
            raise SystemExit("complexity requires --receipt-output")
        model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
        result = profile_model(
            model=model, spec=spec, rows=rows, primary=primary,
            secondary=secondary, data_root=args.data_root.resolve(),
            checkpoint=args.checkpoint.resolve(),
            checkpoint_metadata=payload["metadata"],
            destination=args.receipt_output.resolve(),
            candidate_authority=(
                args.candidate_authority.resolve()
                if args.candidate_authority is not None else None
            ),
        )
    elif args.stage == "distribution":
        if args.receipt_output is None or args.freeze_receipt is None or args.lane is None:
            raise SystemExit(
                "distribution requires --lane, --receipt-output and --freeze-receipt"
            )
        if args.lane == "input":
            manifest_report(args.manifest.resolve(), data_root=args.data_root.resolve())
            result = profile_distribution(
                model=None, spec=None, rows=_annotated_rows(args.manifest.resolve()),
                data_root=args.data_root.resolve(),
                destination=args.receipt_output.resolve(),
                freeze_receipt=args.freeze_receipt.resolve(), checkpoint=None,
                checkpoint_step=None, checkpoint_metadata=None, gpu=args.gpu,
            )
        else:
            model, spec, rows, payload, primary, secondary = _load_checkpoint_model(args)
            result = profile_distribution(
                model=model, spec=spec, rows=rows,
                data_root=args.data_root.resolve(),
                destination=args.receipt_output.resolve(),
                freeze_receipt=args.freeze_receipt.resolve(),
                checkpoint=args.checkpoint.resolve(),
                checkpoint_step=int(payload.get("step", -1)),
                checkpoint_metadata=payload.get("metadata"),
                gpu=args.gpu,
            )
    elif args.stage == "freeze-draft":
        if args.portfolio is None or args.receipt_output is None:
            raise SystemExit("freeze-draft requires --portfolio and --receipt-output")
        result = create_review_draft(
            portfolio=args.portfolio.resolve(), claims=args.paper_claim,
            destination=args.receipt_output.resolve(),
        )
    elif args.stage == "freeze-materialize":
        if (
            args.portfolio is None
            or args.review_decision is None
            or args.receipt_output is None
        ):
            raise SystemExit(
                "freeze-materialize requires --portfolio, --review-decision and "
                "--receipt-output"
            )
        result = materialize_freeze_receipt(
            portfolio=args.portfolio.resolve(),
            review_decision=args.review_decision.resolve(),
            destination=args.receipt_output.resolve(),
        )
    elif args.stage == "runtime-relation":
        required = {
            "--lane": args.lane,
            "--method-runtime-receipt": args.method_runtime_receipt,
            "--plain-runtime-receipt": args.plain_runtime_receipt,
            "--method-authorization-receipt": args.method_authorization_receipt,
            "--method-source-host-label": args.method_source_host_label,
            "--plain-source-host-label": args.plain_source_host_label,
            "--receipt-output": args.receipt_output,
        }
        missing = [name for name, item in required.items() if item is None]
        if missing:
            raise SystemExit("runtime-relation requires " + ", ".join(missing))
        result = materialize_exact_runtime_relation(
            lane_id=args.lane,
            method_source_host_label=args.method_source_host_label,
            plain_source_host_label=args.plain_source_host_label,
            method_runtime_receipt=args.method_runtime_receipt,
            plain_runtime_receipt=args.plain_runtime_receipt,
            method_authorization_receipt=args.method_authorization_receipt,
            destination=args.receipt_output,
        )
    elif args.stage == "candidate-lock":
        required = {
            "--candidate-id": args.candidate_id,
            "--candidate-terminal-receipt": args.candidate_terminal_receipt,
            "--candidate-trajectory": args.candidate_trajectory,
            "--candidate-derivation-card": args.candidate_derivation_card,
            "--candidate-implementation": args.candidate_implementation,
            "--candidate-runtime-gate": args.candidate_runtime_gate,
            "--parent-output": args.parent_output,
            "--parent-scientific-git-commit": args.parent_scientific_git_commit,
            "--parent-protocol-fingerprint": args.parent_protocol_fingerprint,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise SystemExit("candidate-lock requires " + ", ".join(missing))
        result = materialize_candidate_lock(
            output_root=args.output,
            candidate_id=args.candidate_id,
            terminal_receipt=args.candidate_terminal_receipt,
            trajectory=args.candidate_trajectory,
            derivation_card=args.candidate_derivation_card,
            implementation=args.candidate_implementation,
            runtime_gate=args.candidate_runtime_gate,
            parent_output=args.parent_output,
            parent_scientific_git_commit=args.parent_scientific_git_commit,
            parent_protocol_fingerprint=args.parent_protocol_fingerprint,
            parent_readiness_mode=args.parent_readiness_mode,
        )
    elif args.stage == "candidate-runtime-gate":
        required = {
            "--candidate-id": args.candidate_id,
            "--candidate-terminal-receipt": args.candidate_terminal_receipt,
            "--candidate-trajectory": args.candidate_trajectory,
            "--candidate-derivation-card": args.candidate_derivation_card,
            "--candidate-implementation": args.candidate_implementation,
            "--parent-output": args.parent_output,
            "--parent-runtime-receipt": args.parent_runtime_receipt,
            "--parent-e0": args.parent_e0,
            "--parent-scientific-git-commit": args.parent_scientific_git_commit,
            "--parent-protocol-fingerprint": args.parent_protocol_fingerprint,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise SystemExit("candidate-runtime-gate requires " + ", ".join(missing))
        result = run_candidate_runtime_gate(
            output_root=args.output,
            candidate_id=args.candidate_id,
            terminal_receipt=args.candidate_terminal_receipt,
            trajectory=args.candidate_trajectory,
            derivation_card=args.candidate_derivation_card,
            implementation=args.candidate_implementation,
            parent_output=args.parent_output,
            parent_runtime_receipt=args.parent_runtime_receipt,
            parent_e0=args.parent_e0,
            parent_scientific_git_commit=args.parent_scientific_git_commit,
            parent_protocol_fingerprint=args.parent_protocol_fingerprint,
            train_view=args.train_view.resolve(),
            data_root=args.data_root.resolve(),
            manifest_path=args.manifest.resolve(),
            gpu=args.gpu,
            capacity_override=args.capacity_override_receipt,
            host_label=args.host_label,
            parent_readiness_mode=args.parent_readiness_mode,
        )
    elif args.stage == "checkpoint-export":
        required = {
            "--lane": args.lane,
            "--epoch": args.epoch,
            "--checkpoint": args.checkpoint,
            "--source-sidecar": args.source_sidecar,
            "--source-host-label": args.source_host_label,
            "--receipt-output": args.receipt_output,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise SystemExit("checkpoint-export requires " + ", ".join(missing))
        lane_id = args.candidate_id if args.lane == "candidate" else args.lane
        if not lane_id:
            raise SystemExit("candidate checkpoint-export requires --candidate-id")
        result = export_checkpoint_receipt(
            checkpoint=args.checkpoint,
            sidecar=args.source_sidecar,
            lane_id=lane_id,
            epoch=args.epoch,
            host_label=args.source_host_label,
            destination=args.receipt_output,
        )
    elif args.stage == "input-evaluate":
        result = evaluate_input_reference(
            output_root=args.output, data_root=args.data_root,
            manifest_path=args.manifest, gpu=args.gpu,
        )
    elif args.stage == "unified-evaluate":
        required = {
            "--lane": args.lane,
            "--source-receipt": args.source_receipt,
            "--copied-checkpoint": args.copied_checkpoint,
        }
        missing = [key for key, value in required.items() if value is None]
        if missing:
            raise SystemExit("unified-evaluate requires " + ", ".join(missing))
        if args.lane == "candidate" and not args.candidate_id:
            raise SystemExit("candidate unified-evaluate requires --candidate-id")
        if args.lane == "candidate" and args.candidate_authority is None:
            raise SystemExit("candidate unified-evaluate requires --candidate-authority")
        result = evaluate_imported_checkpoint(
            output_root=args.output,
            export_receipt=args.source_receipt,
            copied_checkpoint=args.copied_checkpoint,
            train_view=args.train_view.resolve(),
            data_root=args.data_root.resolve(),
            manifest_path=args.manifest.resolve(),
            gpu=args.gpu,
            candidate_id=args.candidate_id if args.lane == "candidate" else None,
            candidate_authority=(
                args.candidate_authority.resolve()
                if args.candidate_authority is not None else None
            ),
        )
    elif args.stage == "unified-lock":
        result = lock_unified_evaluation_cohort(args.output)
    else:
        result = adjudicate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
