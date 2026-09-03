"""Explicit one-session authorization boundary for the sealed confirmation20.

This module does not evaluate or enumerate confirmation images while creating
drafts or authorizations.  It only permits a later evaluator to claim one
recoverable logical session after the paper set, claims and discovery80
distribution evidence have all been frozen and reviewed.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

import torch

from production.metrics import (
    METRIC_SEMANTICS,
    build_rollout_bundle,
    bundle_hash,
    psnr_unit,
    ssim_unit,
    to_unit,
)
from research.local_route1.runtime import capture_rng, full_state_hash, restore_rng
from research.local_route1.runtime import write_json

from .distribution import (
    DISTRIBUTION_COHORT_SCHEMA,
    _model_state_sha256,
    committed_freeze_identity,
)
from .evaluate import (
    _lpips,
    _prediction,
    aggregate_metric_rows,
    evaluation_input_hash,
    read_image,
    replicate_stochasticity,
    validate_evaluation_result,
)
from .gates import environment_record
from .protocol import (
    EVALUATION_SCHEMA,
    ROOT,
    LaneSpec,
    evaluation_bundle_fingerprint,
    file_sha256,
    load_protocol,
    object_sha256,
)


DRAFT_SCHEMA = "final-unsb-paper-confirmation20-review-draft-v1"
DRAFT_STATUS = "PENDING_EXPLICIT_CONFIRMATION20_REVIEW"
REVIEW_SCHEMA = "final-unsb-paper-confirmation20-review-decision-v1"
REVIEW_STATUS = "APPROVE_ONE_RECOVERABLE_CONFIRMATION20_SESSION"
AUTHORIZATION_SCHEMA = "final-unsb-paper-confirmation20-authorization-v1"
AUTHORIZATION_STATUS = "AUTHORIZED_ONE_RECOVERABLE_CONFIRMATION20_SESSION"
SESSION_SCHEMA = "final-unsb-paper-confirmation20-session-v1"
SESSION_STATUS = "OPEN_CONFIRMATION20_SESSION_IN_PROGRESS"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _repo_relative(path: Path, *, label: str) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise RuntimeError(f"{label} must be inside the repository") from error


def _committed_json(path: Path, *, label: str) -> tuple[dict[str, Any], str, str]:
    path = Path(path).resolve()
    relative = _repo_relative(path, label=label)
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True,
    ).strip():
        raise RuntimeError(f"{label} has uncommitted changes")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative],
        cwd=ROOT, text=True,
    ).strip()
    if len(commit) != 40:
        raise RuntimeError(f"{label} has no committed Git identity")
    committed = json.loads(subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT, text=True,
    ))
    current = _read(path)
    if object_sha256(committed) != object_sha256(current):
        raise RuntimeError(f"working {label} differs from its committed Git blob")
    return current, commit, relative


def validate_distribution_cohort(
    *, path: Path, freeze: dict[str, Any], freeze_path: Path,
    freeze_commit: str,
) -> dict[str, Any]:
    path = Path(path).resolve()
    value = _read(path)
    receipt_rows = value.get("receipts")
    if (
        value.get("schema") != DISTRIBUTION_COHORT_SCHEMA
        or value.get("status") != "PASS_COMPLETE_FROZEN_DISTRIBUTION_COHORT"
        or Path(value.get("freeze_receipt", "")).resolve()
        != Path(freeze_path).resolve()
        or value.get("freeze_receipt_sha256") != file_sha256(freeze_path)
        or value.get("freeze_git_commit") != freeze_commit
        or value.get("lanes") != sorted(freeze["distribution_lanes"])
        or not isinstance(receipt_rows, list)
        or value.get("all_lanes_one_runtime") is not True
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("distribution cohort is not eligible for confirmation review")
    seen = set()
    for row in receipt_rows:
        lane = str(row.get("lane_id", ""))
        receipt = Path(row.get("receipt", "")).resolve()
        if (
            not lane or lane in seen or not receipt.is_file()
            or file_sha256(receipt) != row.get("receipt_sha256")
            or object_sha256(_read(receipt)) != row.get("receipt_object_sha256")
        ):
            raise RuntimeError("distribution cohort receipt identity changed")
        payload = _read(receipt)
        checkpoint_value = payload.get("checkpoint")
        if lane != "input":
            checkpoint = Path(str(checkpoint_value)).resolve()
            if (
                not checkpoint.is_file()
                or file_sha256(checkpoint) != payload.get("checkpoint_sha256")
                or payload.get("checkpoint_unchanged") is not True
            ):
                raise RuntimeError(
                    "frozen e200 checkpoint is unavailable before confirmation"
                )
        elif checkpoint_value is not None:
            raise RuntimeError("Input distribution receipt unexpectedly has a checkpoint")
        seen.add(lane)
    if seen != set(freeze["distribution_lanes"]):
        raise RuntimeError("distribution cohort receipt set is incomplete")
    return value


def _review_basis(
    *, freeze_receipt: Path, distribution_cohort: Path,
) -> tuple[dict[str, Any], str, dict[str, Any], dict[str, Any]]:
    freeze_path = Path(freeze_receipt).resolve()
    freeze, freeze_commit = committed_freeze_identity(
        freeze_path, lane_id="input",
    )
    cohort = validate_distribution_cohort(
        path=distribution_cohort, freeze=freeze, freeze_path=freeze_path,
        freeze_commit=freeze_commit,
    )
    basis = {
        "freeze_receipt": str(freeze_path),
        "freeze_receipt_sha256": file_sha256(freeze_path),
        "freeze_git_commit": freeze_commit,
        "distribution_cohort": str(Path(distribution_cohort).resolve()),
        "distribution_cohort_sha256": file_sha256(distribution_cohort),
        "distribution_lanes": sorted(freeze["distribution_lanes"]),
        "paper_claims_sha256": freeze["paper_claims_sha256"],
    }
    basis["confirmation_session_id"] = object_sha256(basis)
    return freeze, freeze_commit, cohort, basis


def create_confirmation_review_draft(
    *, freeze_receipt: Path, distribution_cohort: Path, destination: Path,
) -> dict[str, Any]:
    _, _, _, basis = _review_basis(
        freeze_receipt=freeze_receipt, distribution_cohort=distribution_cohort,
    )
    result = {
        "schema": DRAFT_SCHEMA,
        "status": DRAFT_STATUS,
        **basis,
        "review_requirements": [
            "confirm all frozen lanes completed discovery80 distribution evaluation",
            "confirm algorithm, baseline, e200 result and claim set did not change",
            "confirm confirmation20 will be opened once for the entire frozen lane set",
            "confirm confirmation results cannot revise, select or stop an algorithm",
            "commit a separate human/Codex review decision; this draft cannot authorize",
        ],
        "human_approval_recorded": False,
        "codex_scientific_review_recorded": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    write_json(Path(destination).resolve(), result)
    return result


def materialize_confirmation_authorization(
    *, freeze_receipt: Path, distribution_cohort: Path,
    review_decision: Path, destination: Path,
) -> dict[str, Any]:
    _, _, _, basis = _review_basis(
        freeze_receipt=freeze_receipt, distribution_cohort=distribution_cohort,
    )
    review, review_commit, review_relative = _committed_json(
        review_decision, label="confirmation review decision",
    )
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != REVIEW_STATUS
        or any(review.get(key) != value for key, value in basis.items())
        or review.get("human_approval_recorded") is not True
        or review.get("codex_scientific_review_recorded") is not True
        or review.get("algorithm_or_baseline_changed_after_freeze") is not False
        or review.get("paper_claim_changed_after_freeze") is not False
        or review.get("best_checkpoint_selection") is not False
        or review.get("paired_metric_control") is not False
        or review.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("committed confirmation review decision is invalid")
    destination = Path(destination).resolve()
    _repo_relative(destination, label="confirmation authorization")
    result = {
        "schema": AUTHORIZATION_SCHEMA,
        "status": AUTHORIZATION_STATUS,
        **basis,
        "review_decision": review_relative,
        "review_decision_sha256": file_sha256(review_decision),
        "review_decision_git_commit": review_commit,
        "one_logical_session_only": True,
        "recover_same_session_after_failure": True,
        "all_frozen_lanes_required": True,
        "results_may_not_change_frozen_claims_or_methods": True,
        "confirmation_authorized": True,
        "confirmation20_opened": False,
    }
    if destination.is_file():
        if object_sha256(_read(destination)) != object_sha256(result):
            raise RuntimeError("confirmation authorization already exists and differs")
    else:
        write_json(destination, result)
    return result


def committed_confirmation_authorization(
    path: Path,
) -> tuple[dict[str, Any], str]:
    value, commit, _ = _committed_json(path, label="confirmation authorization")
    _, _, _, basis = _review_basis(
        freeze_receipt=Path(value.get("freeze_receipt", "")),
        distribution_cohort=Path(value.get("distribution_cohort", "")),
    )
    review_path = (ROOT / str(value.get("review_decision", ""))).resolve()
    review, review_commit, _ = _committed_json(
        review_path, label="confirmation review decision",
    )
    if (
        value.get("schema") != AUTHORIZATION_SCHEMA
        or value.get("status") != AUTHORIZATION_STATUS
        or any(value.get(key) != expected for key, expected in basis.items())
        or value.get("review_decision_sha256") != file_sha256(review_path)
        or value.get("review_decision_git_commit") != review_commit
        or review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != REVIEW_STATUS
        or any(review.get(key) != expected for key, expected in basis.items())
        or review.get("human_approval_recorded") is not True
        or review.get("codex_scientific_review_recorded") is not True
        or review.get("algorithm_or_baseline_changed_after_freeze") is not False
        or review.get("paper_claim_changed_after_freeze") is not False
        or review.get("best_checkpoint_selection") is not False
        or review.get("paired_metric_control") is not False
        or review.get("confirmation20_opened") is not False
        or value.get("one_logical_session_only") is not True
        or value.get("recover_same_session_after_failure") is not True
        or value.get("all_frozen_lanes_required") is not True
        or value.get("results_may_not_change_frozen_claims_or_methods") is not True
        or value.get("confirmation_authorized") is not True
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("committed confirmation authorization is invalid or stale")
    return value, commit


def claim_confirmation_session(
    *, authorization: Path, output_root: Path,
) -> dict[str, Any]:
    """Atomically claim or recover the one authorized logical session.

    Claiming marks the sealed split as opened but does not enumerate rows or
    load images. A process restart may recover only this exact session.
    """
    authorization = Path(authorization).resolve()
    value, commit = committed_confirmation_authorization(authorization)
    path = Path(output_root).resolve() / "gates" / "CONFIRMATION20_SESSION.json"
    complete = path.parent / "CONFIRMATION20_COMPLETE.json"
    if complete.exists():
        raise RuntimeError("confirmation20 logical session is already complete")
    result = {
        "schema": SESSION_SCHEMA,
        "status": SESSION_STATUS,
        "confirmation_session_id": value["confirmation_session_id"],
        "authorization": str(authorization),
        "authorization_sha256": file_sha256(authorization),
        "authorization_git_commit": commit,
        "lanes": value["distribution_lanes"],
        "one_logical_session_only": True,
        "same_session_recovery_allowed": True,
        "confirmation_authorized": True,
        "confirmation20_opened": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError:
        existing = _read(path)
        if object_sha256(existing) != object_sha256(result):
            raise RuntimeError("confirmation20 was already claimed by another session")
        return {**existing, "recovered_existing_session": True}
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return {**result, "recovered_existing_session": False}


def select_confirmation(
    rows: list[dict[str, Any]], *, session: dict[str, Any], count_per_domain: int = 20,
) -> list[dict[str, Any]]:
    """The sole row selector for an already-open authorized session."""
    if (
        session.get("schema") != SESSION_SCHEMA
        or session.get("status") != SESSION_STATUS
        or session.get("confirmation_authorized") is not True
        or session.get("confirmation20_opened") is not True
        or int(count_per_domain) != 20
    ):
        raise RuntimeError("confirmation20 access requires the authorized open session")
    selected = []
    domains = sorted({str(row["domain"]) for row in rows})
    if len(domains) != 6:
        raise RuntimeError("confirmation20 requires six manifest domains")
    for domain in domains:
        candidates = [
            row for row in rows
            if str(row["domain"]) == domain and row["split"] == "confirmation"
        ]
        candidates.sort(key=lambda row: int(row["order"]))
        take = candidates[:20]
        if len(take) != 20:
            raise RuntimeError(f"{domain}: confirmation20 split is incomplete")
        selected.extend(take)
    if len(selected) != 120 or any(row["split"] != "confirmation" for row in selected):
        raise RuntimeError("confirmation20 selection is incomplete")
    return selected


def validate_open_session(
    *, session_receipt: Path, authorization: Path,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    authorization = Path(authorization).resolve()
    auth, auth_commit = committed_confirmation_authorization(authorization)
    session = _read(session_receipt)
    if Path(session_receipt).resolve().parent.joinpath(
        "CONFIRMATION20_COMPLETE.json"
    ).exists():
        raise RuntimeError("confirmation20 logical session is already complete")
    if (
        session.get("schema") != SESSION_SCHEMA
        or session.get("status") != SESSION_STATUS
        or session.get("confirmation_session_id") != auth["confirmation_session_id"]
        or Path(session.get("authorization", "")).resolve() != authorization
        or session.get("authorization_sha256") != file_sha256(authorization)
        or session.get("authorization_git_commit") != auth_commit
        or session.get("lanes") != auth["distribution_lanes"]
        or session.get("one_logical_session_only") is not True
        or session.get("same_session_recovery_allowed") is not True
        or session.get("confirmation_authorized") is not True
        or session.get("confirmation20_opened") is not True
    ):
        raise RuntimeError("confirmation20 session receipt is invalid or stale")
    return session, auth, auth_commit


@torch.no_grad()
def evaluate_confirmation_lane(
    *, model, spec: LaneSpec | None, rows: list[dict[str, Any]], data_root: Path,
    authorization: Path, session_receipt: Path, destination: Path,
    checkpoint: Path | None, checkpoint_step: int | None,
    checkpoint_metadata: dict[str, Any] | None, gpu: int,
) -> dict[str, Any]:
    """Evaluate one frozen e200 lane inside the already-open logical session."""
    session, auth, auth_commit = validate_open_session(
        session_receipt=session_receipt, authorization=authorization,
    )
    lane_id = "input" if model is None else str(spec.id if spec else "")
    if lane_id not in auth["distribution_lanes"]:
        raise RuntimeError("confirmation lane is absent from the frozen algorithm set")
    if model is not None and spec is None:
        raise RuntimeError("confirmation model lane requires a frozen lane spec")
    if model is not None and (
        int(checkpoint_step or -1) != 1_710_600
        or checkpoint_metadata is None
        or checkpoint_metadata.get("paired_controller_access") is not False
        or checkpoint_metadata.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("confirmation evaluation requires a safe fixed e200 checkpoint")
    selected = select_confirmation(rows, session=session, count_per_domain=20)
    protocol = load_protocol()
    protocol_hash = evaluation_bundle_fingerprint(protocol)
    checkpoint_path = None if checkpoint is None else Path(checkpoint).resolve()
    checkpoint_before = None if checkpoint_path is None else file_sha256(checkpoint_path)
    device = (
        torch.device(
            f"cuda:{int(gpu)}"
            if int(gpu) >= 0 and torch.cuda.is_available() else "cpu"
        )
        if model is None else torch.device(model.device)
    )
    family = "input" if model is None else spec.family
    replicates = 1 if model is None or family != "unsb" else int(
        protocol["evaluation"]["terminal_replicates"]
    )
    primary_nfe = 0 if model is None else (
        int(protocol["evaluation"]["primary_unsb_nfe"])
        if family == "unsb" else 1
    )
    saved_rng = capture_rng()
    rng_identity_before = full_state_hash(saved_rng)
    modes = {} if model is None else {
        name: getattr(model, "net" + name).training for name in model.model_names
    }
    model_identity_before = None if model is None else {
        name: _model_state_sha256(getattr(model, "net" + name))
        for name in model.model_names
    }
    image_rows = []
    try:
        if model is not None:
            model.eval()
        perceptual = _lpips(device)
        for row in selected:
            image_size = int(protocol["evaluation"]["image_size"])
            source = read_image(
                Path(data_root) / row["input_relpath"], size=image_size,
            ).to(device)
            target = read_image(
                Path(data_root) / row["target_relpath"], size=image_size,
            ).to(device)
            for replicate in range(replicates):
                if model is None:
                    endpoint = source
                    crn_sha256 = None
                else:
                    bundle = build_rollout_bundle(
                        protocol_hash=protocol_hash, domain=row["domain"],
                        stem=row["stem"], replicate=replicate,
                        latent_dim=4 * int(getattr(model.opt, "ngf", 64)),
                        height=image_size, width=image_size,
                        num_timesteps=int(protocol["unsb"]["num_timesteps"]),
                    )
                    endpoint = _prediction(
                        model, spec, source, bundle, nfe=primary_nfe,
                    ).clamp(-1.0, 1.0)
                    crn_sha256 = bundle_hash(bundle)
                image_rows.append({
                    "domain": row["domain"], "stem": row["stem"],
                    "order": int(row["order"]), "replicate": replicate,
                    "nfe": primary_nfe,
                    "psnr": psnr_unit(to_unit(endpoint), to_unit(target)),
                    "ssim": ssim_unit(to_unit(endpoint), to_unit(target)),
                    "lpips": float(perceptual(endpoint, target).item()),
                    "crn_bundle_sha256": crn_sha256,
                })
    finally:
        for name, was_training in modes.items():
            getattr(model, "net" + name).train(was_training)
        restore_rng(saved_rng)
    rng_identity_after = full_state_hash(capture_rng())
    model_identity_after = None if model is None else {
        name: _model_state_sha256(getattr(model, "net" + name))
        for name in model.model_names
    }
    if (
        rng_identity_before != rng_identity_after
        or model_identity_before != model_identity_after
    ):
        raise RuntimeError("confirmation evaluation changed model or RNG state")
    aggregate = aggregate_metric_rows(image_rows)
    replicate_cells = [
        {
            "replicate": replicate,
            **aggregate_metric_rows([
                row for row in image_rows if int(row["replicate"]) == replicate
            ]),
        }
        for replicate in range(replicates)
    ]
    stochasticity = replicate_stochasticity(replicate_cells)
    result = {
        "schema": EVALUATION_SCHEMA,
        "lane_id": lane_id,
        "family": family,
        "split": "confirmation",
        "count_per_domain": 20,
        "replicates": replicates,
        "nfe_values": [primary_nfe],
        "primary_nfe": primary_nfe,
        "primary_replicate": int(protocol["evaluation"]["primary_replicate"]),
        "top_level_replicate_aggregation": "mean_over_fixed_replicates",
        "metric_semantics": METRIC_SEMANTICS,
        "metric_semantics_sha256": object_sha256(METRIC_SEMANTICS),
        "protocol_fingerprint": protocol_hash,
        "evaluation_input_sha256": object_sha256({
            "base": evaluation_input_hash(selected, protocol_hash),
            "split": "confirmation",
            "session": session["confirmation_session_id"],
        }),
        **{key: aggregate[key] for key in ("macro_psnr", "macro_ssim", "macro_lpips")},
        "domains": aggregate["domains"],
        "replicate_cells": replicate_cells,
        "stochasticity": stochasticity,
        "nfe_cells": {str(primary_nfe): {
            **aggregate, "replicate_cells": replicate_cells,
            "stochasticity": stochasticity,
        }},
        "images": image_rows,
        "lpips_requested": True,
        "lpips_available": True,
        "epoch": 200,
        "updates": 0 if model is None else 1_710_600,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "checkpoint_sha256": checkpoint_before,
        "confirmation_session_id": session["confirmation_session_id"],
        "authorization": str(Path(authorization).resolve()),
        "authorization_sha256": file_sha256(Path(authorization).resolve()),
        "authorization_git_commit": auth_commit,
        "environment": environment_record(),
        "training_checkpoint_read_only": True,
        "model_state_sha256_before": model_identity_before,
        "model_state_sha256_after": model_identity_after,
        "rng_state_sha256_before": rng_identity_before,
        "rng_state_sha256_after": rng_identity_after,
        "metric_values_used_for_training_or_scheduling": False,
        "results_may_not_change_frozen_claims_or_methods": True,
        "confirmation20_opened": True,
    }
    validate_evaluation_result(
        result, lane_id=lane_id, family=family, count_per_domain=20,
        replicates=replicates, nfe_values=[primary_nfe], include_lpips=True,
        input_reference=model is None, expected_split="confirmation",
        confirmation20_opened=True,
    )
    checkpoint_after = None if checkpoint_path is None else file_sha256(checkpoint_path)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("confirmation evaluation changed the source checkpoint")
    result["checkpoint_unchanged"] = True
    destination = Path(destination).resolve()
    if destination.is_file():
        if object_sha256(_read(destination)) != object_sha256(result):
            raise RuntimeError("confirmation lane result already exists and differs")
    else:
        write_json(destination, result)
    return result


def lock_confirmation_cohort(
    *, authorization: Path, session_receipt: Path,
    results: list[Path], destination: Path,
) -> dict[str, Any]:
    """Finalize the one session only after every frozen lane is complete."""
    destination = Path(destination).resolve()
    expected_destination = (
        Path(session_receipt).resolve().parent / "CONFIRMATION20_COMPLETE.json"
    )
    if destination != expected_destination:
        raise RuntimeError("confirmation completion receipt must close the session gate")
    if destination.exists():
        raise RuntimeError("confirmation20 logical session is already complete")
    session, auth, auth_commit = validate_open_session(
        session_receipt=session_receipt, authorization=authorization,
    )
    paths = [Path(path).resolve() for path in results]
    if not paths or len(paths) != len(set(paths)):
        raise RuntimeError("confirmation cohort requires unique lane results")
    rows = {}
    common_environment = None
    common_input = None
    common_samples = None
    from .evaluate import evaluation_sample_identity

    protocol = load_protocol()
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"confirmation result is missing: {path}")
        value = _read(path)
        lane = str(value.get("lane_id", ""))
        family = str(value.get("family", ""))
        if not lane or lane in rows:
            raise RuntimeError("confirmation result lane set is duplicated or empty")
        replicates = 1 if family != "unsb" else int(
            protocol["evaluation"]["terminal_replicates"]
        )
        primary_nfe = 0 if lane == "input" else (
            int(protocol["evaluation"]["primary_unsb_nfe"])
            if family == "unsb" else 1
        )
        validate_evaluation_result(
            value, lane_id=lane, family=family, count_per_domain=20,
            replicates=replicates, nfe_values=[primary_nfe], include_lpips=True,
            input_reference=lane == "input", expected_split="confirmation",
            confirmation20_opened=True,
        )
        if (
            int(value.get("epoch", -1)) != 200
            or value.get("confirmation_session_id")
            != session["confirmation_session_id"]
            or Path(value.get("authorization", "")).resolve()
            != Path(authorization).resolve()
            or value.get("authorization_sha256") != file_sha256(authorization)
            or value.get("authorization_git_commit") != auth_commit
            or value.get("training_checkpoint_read_only") is not True
            or value.get("checkpoint_unchanged") is not True
            or value.get("model_state_sha256_before")
            != value.get("model_state_sha256_after")
            or value.get("rng_state_sha256_before")
            != value.get("rng_state_sha256_after")
            or value.get("metric_values_used_for_training_or_scheduling") is not False
            or value.get("results_may_not_change_frozen_claims_or_methods") is not True
            or value.get("confirmation20_opened") is not True
        ):
            raise RuntimeError(f"confirmation result violates frozen policy: {path}")
        environment = value.get("environment")
        input_identity = value.get("evaluation_input_sha256")
        sample_identity = evaluation_sample_identity(value)
        if common_environment is None:
            common_environment = environment
            common_input = input_identity
            common_samples = sample_identity
        elif (
            environment != common_environment
            or input_identity != common_input
            or sample_identity != common_samples
        ):
            raise RuntimeError("confirmation results do not share one runtime and split")
        rows[lane] = {
            "lane_id": lane,
            "result": str(path),
            "result_sha256": file_sha256(path),
            "result_object_sha256": object_sha256(value),
        }
    expected = set(auth["distribution_lanes"])
    if set(rows) != expected:
        raise RuntimeError("confirmation cohort does not exactly cover the frozen lanes")
    result = {
        "schema": "final-unsb-paper-confirmation20-cohort-v1",
        "status": "COMPLETE_ONE_TIME_CONFIRMATION20_COHORT",
        "confirmation_session_id": session["confirmation_session_id"],
        "authorization": str(Path(authorization).resolve()),
        "authorization_sha256": file_sha256(authorization),
        "authorization_git_commit": auth_commit,
        "lanes": sorted(rows),
        "results": [rows[lane] for lane in sorted(rows)],
        "evaluation_input_sha256": common_input,
        "environment": common_environment,
        "all_frozen_lanes_complete": True,
        "all_lanes_one_evaluator_runtime": True,
        "algorithm_or_claim_revision_authorized": False,
        "metric_values_used_for_training_or_scheduling": False,
        "confirmation20_opened_once": True,
        "confirmation20_opened": True,
    }
    write_json(destination, result)
    return result
