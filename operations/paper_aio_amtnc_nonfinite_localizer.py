"""Replay a protected AM-TNC state and localize non-finite replica geometry.

This is a diagnostic-only runner.  It writes only below ``--diagnostic-output``
and never saves or edits the source lane.  It does not read paired targets or
metrics and it deliberately re-raises no scientific result: its only question
is which player/tensor first makes the frozen AM-TNC geometry non-finite.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import torch

from models.route1 import amtnc as amtnc_module
from research.local_route1.runtime import (
    assert_finite,
    file_sha256,
    full_state_hash,
    model_state,
    write_json,
)
from research.paper_aio.protocol import (
    ROOT,
    lane_spec,
    load_protocol,
    step_to_epoch,
)
from research.paper_aio.runtime import (
    lane_metadata,
    load_full_state,
    optimizer_step,
    prepare_lane,
)


PLAYERS = ("D", "E", "GF")
IDENTITY_KEYS = (
    "project_id", "lane_id", "lane_config_sha256", "seed",
    "manifest_sha256", "protocol_fingerprint", "e0_scientific_state_sha256",
    "git_commit", "steps_per_data_epoch", "target_updates", "batch_size",
    "sampling_measure", "paired_controller_access", "confirmation20_opened",
)


def replay_expected_metadata(
    payload: dict[str, Any],
    current_metadata: dict[str, Any],
    *,
    required_parent_git_commit: str | None = None,
    required_parent_protocol_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Bind a diagnostic replay across an explicitly named source transition.

    The ordinary path remains strict against the current checkout.  A patched
    diagnostic may admit a parent checkpoint only when both changing identity
    fields are supplied and every scientific/data identity outside those two
    fields still equals the current checkout.
    """
    if (required_parent_git_commit is None) != (
        required_parent_protocol_fingerprint is None
    ):
        raise RuntimeError(
            "parent git commit and protocol fingerprint must be supplied together"
        )
    parent = payload.get("metadata", {})
    if required_parent_git_commit is None:
        return {key: current_metadata[key] for key in IDENTITY_KEYS}
    if parent.get("git_commit") != required_parent_git_commit:
        raise RuntimeError("diagnostic parent git commit mismatch")
    if (
        parent.get("protocol_fingerprint")
        != required_parent_protocol_fingerprint
    ):
        raise RuntimeError("diagnostic parent protocol fingerprint mismatch")
    for key in IDENTITY_KEYS:
        if key in {"git_commit", "protocol_fingerprint"}:
            continue
        if parent.get(key) != current_metadata.get(key):
            raise RuntimeError(f"diagnostic cross-version identity mismatch for {key}")
    return {key: parent[key] for key in IDENTITY_KEYS}


def _finite_summary(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    result: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "finite_count": finite_count,
        "nonfinite_count": int(detached.numel()) - finite_count,
        "all_finite": finite_count == int(detached.numel()),
    }
    if result["all_finite"] and detached.numel():
        result["absmax"] = float(detached.abs().max().item())
    else:
        result["absmax"] = None
    return result


def summarize_replica_geometry(
    first: tuple[torch.Tensor | None, ...],
    second: tuple[torch.Tensor | None, ...],
    scales: tuple[torch.Tensor, ...],
    *,
    parameter_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Describe the first tensor-level source of non-finite geometry."""
    if len(first) != len(second) or len(first) != len(scales):
        raise ValueError("replica diagnostic structures differ")
    names = list(parameter_names or (f"parameter_{index}" for index in range(len(first))))
    if len(names) != len(first):
        raise ValueError("replica diagnostic parameter names differ")

    records = []
    category_counts: dict[str, int] = {}
    first_issue = None
    for index, (left_optional, right_optional, scale) in enumerate(
        zip(first, second, scales)
    ):
        left = torch.zeros_like(scale) if left_optional is None else left_optional
        right = torch.zeros_like(scale) if right_optional is None else right_optional
        mean = (left + right) * 0.5
        difference = (left - right) * 0.5
        adam_mean = scale * mean
        adam_difference = scale * difference
        mean_square = adam_mean * adam_mean
        difference_square = adam_difference * adam_difference
        cross_product = adam_mean * adam_difference

        summaries = {
            "first_gradient": _finite_summary(left),
            "second_gradient": _finite_summary(right),
            "adam_scale": _finite_summary(scale),
            "replica_mean": _finite_summary(mean),
            "replica_difference": _finite_summary(difference),
            "scaled_mean": _finite_summary(adam_mean),
            "scaled_difference": _finite_summary(adam_difference),
            "scaled_mean_square": _finite_summary(mean_square),
            "scaled_difference_square": _finite_summary(difference_square),
            "scaled_cross_product": _finite_summary(cross_product),
        }
        source_finite = (
            summaries["first_gradient"]["all_finite"]
            and summaries["second_gradient"]["all_finite"]
        )
        if not source_finite:
            category = "SOURCE_GRADIENT_NONFINITE"
        elif not summaries["adam_scale"]["all_finite"]:
            category = "ADAM_SCALE_NONFINITE"
        elif not (
            summaries["replica_mean"]["all_finite"]
            and summaries["replica_difference"]["all_finite"]
        ):
            category = "REPLICA_MEAN_OR_DIFFERENCE_OVERFLOW"
        elif not (
            summaries["scaled_mean"]["all_finite"]
            and summaries["scaled_difference"]["all_finite"]
        ):
            category = "ADAM_METRIC_MULTIPLICATION_OVERFLOW"
        elif not (
            summaries["scaled_mean_square"]["all_finite"]
            and summaries["scaled_difference_square"]["all_finite"]
            and summaries["scaled_cross_product"]["all_finite"]
        ):
            category = "FLOAT32_GEOMETRY_PRODUCT_OVERFLOW"
        else:
            category = "FINITE_TENSOR_CONTRIBUTION"
        category_counts[category] = category_counts.get(category, 0) + 1
        if category != "FINITE_TENSOR_CONTRIBUTION":
            record = {
                "parameter_index": index,
                "parameter_name": names[index],
                "category": category,
                "summaries": summaries,
            }
            records.append(record)
            if first_issue is None:
                first_issue = record

        del mean, difference, adam_mean, adam_difference
        del mean_square, difference_square, cross_product

    return {
        "parameter_count": len(first),
        "category_counts": category_counts,
        "first_issue": first_issue,
        "all_issues": records,
    }


def _player_parameter_names(model) -> dict[str, list[str]]:
    def names(label: str, network) -> list[str]:
        return [f"{label}.{name}" for name, parameter in network.named_parameters() if parameter.requires_grad]

    return {
        "D": names("D", model.netD),
        "E": names("E", model.netE),
        "GF": names("G", model.netG) + names("F", model.netF),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_checkpoint = args.checkpoint.resolve()
    diagnostic_output = args.diagnostic_output.resolve()
    source_lane = source_checkpoint.parent
    if (
        diagnostic_output == source_lane
        or diagnostic_output.is_relative_to(source_lane)
        or source_checkpoint.is_relative_to(diagnostic_output)
    ):
        raise RuntimeError("diagnostic output must be separate from the source lane")
    diagnostic_output.mkdir(parents=True, exist_ok=True)
    receipt_path = diagnostic_output / "AMTNC_NONFINITE_LOCALIZATION.json"

    source_sha_before = file_sha256(source_checkpoint)
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    if int(payload["step"]) >= int(payload["target_steps"]):
        raise RuntimeError("diagnostic expects an incomplete protected checkpoint")
    e0 = torch.load(args.e0.resolve(), map_location="cpu", weights_only=False)
    protocol = load_protocol()
    spec = lane_spec("amtnc", protocol)
    model, primary, secondary, _ = prepare_lane(
        output_root=diagnostic_output,
        train_view=args.train_view.resolve(),
        manifest_path=args.manifest.resolve(),
        spec=spec,
        gpu=args.gpu,
        e0=e0,
    )
    current_metadata = lane_metadata(
        spec=spec,
        manifest_path=args.manifest.resolve(),
        e0=e0,
        train_view=args.train_view.resolve(),
        data_root=args.data_root.resolve(),
    )
    expected_metadata = replay_expected_metadata(
        payload,
        current_metadata,
        required_parent_git_commit=args.required_parent_git_commit,
        required_parent_protocol_fingerprint=(
            args.required_parent_protocol_fingerprint
        ),
    )
    load_full_state(
        source_checkpoint,
        model=model,
        spec=spec,
        primary=primary,
        secondary=secondary,
        expected_metadata=expected_metadata,
    )
    source_scientific_sha = full_state_hash(payload)
    parameter_names = _player_parameter_names(model)

    original_combine = amtnc_module._combine_optional_gradients
    call_index = 0
    current_zero_step = int(payload["step"])
    failure: dict[str, Any] | None = None

    def diagnostic_combine(first, second, scales):
        nonlocal call_index, failure
        player = PLAYERS[call_index % len(PLAYERS)]
        update_offset = call_index // len(PLAYERS)
        try:
            return original_combine(first, second, scales)
        except RuntimeError as error:
            failure = {
                "exception_type": type(error).__name__,
                "exception": str(error),
                "player": player,
                "zero_based_update": current_zero_step,
                "completed_updates_before_failure": current_zero_step,
                "replay_update_offset": update_offset,
                "geometry": summarize_replica_geometry(
                    first,
                    second,
                    scales,
                    parameter_names=parameter_names[player],
                ),
            }
            raise
        finally:
            call_index += 1

    amtnc_module._combine_optional_gradients = diagnostic_combine
    started = time.time()
    attempted = 0
    caught: RuntimeError | None = None
    precision_fallback_count = 0
    first_precision_fallback: dict[str, Any] | None = None
    try:
        for offset in range(args.max_updates):
            current_zero_step = int(payload["step"]) + offset
            model.set_train_epoch(step_to_epoch(current_zero_step, protocol))
            model.set_search_step(current_zero_step, int(payload["target_steps"]))
            optimizer_step(model, spec, primary, secondary)
            attempted += 1
            for player, diagnostics in model._amtnc_last_geometry.items():
                if diagnostics.get("metric_product_precision") != (
                    "float64_overflow_fallback"
                ):
                    continue
                precision_fallback_count += 1
                if first_precision_fallback is None:
                    first_precision_fallback = {
                        "player": player,
                        "zero_based_update": current_zero_step,
                        "replay_update_offset": offset,
                        "diagnostics": dict(diagnostics),
                    }
    except RuntimeError as error:
        caught = error
        if failure is None:
            failure = {
                "exception_type": type(error).__name__,
                "exception": str(error),
                "player": "OUTSIDE_AMTNC_GEOMETRY_COMBINER",
                "zero_based_update": current_zero_step,
                "completed_updates_before_failure": current_zero_step,
                "replay_update_offset": attempted,
                "geometry": None,
            }
    finally:
        amtnc_module._combine_optional_gradients = original_combine

    source_sha_after = file_sha256(source_checkpoint)
    overflow_safe_mismatch = None
    if args.expect_overflow_safe:
        if failure is not None:
            overflow_safe_mismatch = "UNEXPECTED_RUNTIME_FAILURE"
        elif attempted != int(args.max_updates):
            overflow_safe_mismatch = "REPLAY_DID_NOT_REACH_REQUESTED_LIMIT"
        elif first_precision_fallback is None:
            overflow_safe_mismatch = "EXPECTED_PRECISION_FALLBACK_NOT_OBSERVED"
        elif (
            args.required_first_fallback_offset is not None
            and first_precision_fallback["replay_update_offset"]
            != int(args.required_first_fallback_offset)
        ):
            overflow_safe_mismatch = "FIRST_PRECISION_FALLBACK_OFFSET_MISMATCH"
        elif (
            args.required_first_fallback_player is not None
            and first_precision_fallback["player"]
            != args.required_first_fallback_player
        ):
            overflow_safe_mismatch = "FIRST_PRECISION_FALLBACK_PLAYER_MISMATCH"

    if failure is not None:
        status = "FAILURE_LOCALIZED"
    elif args.expect_overflow_safe and overflow_safe_mismatch is None:
        status = "OVERFLOW_SAFE_REPLAY_COMPLETE"
    elif args.expect_overflow_safe:
        status = "OVERFLOW_SAFE_REPLAY_MISMATCH"
    else:
        status = "NOT_REPRODUCED_WITHIN_LIMIT"

    post_replay_state_finite = None
    if status == "OVERFLOW_SAFE_REPLAY_COMPLETE":
        replay_state = model_state(model)
        assert_finite(replay_state)
        post_replay_state_finite = True
        del replay_state

    receipt = {
        "schema": "final-unsb-paper-amtnc-nonfinite-localization-v1",
        "status": status,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256_before": source_sha_before,
        "source_checkpoint_sha256_after": source_sha_after,
        "source_checkpoint_unchanged": source_sha_before == source_sha_after,
        "source_scientific_state_sha256": source_scientific_sha,
        "source_step": int(payload["step"]),
        "max_replay_updates": int(args.max_updates),
        "completed_replay_updates": attempted,
        "elapsed_seconds": time.time() - started,
        "failure": failure,
        "precision_fallback_count": precision_fallback_count,
        "first_precision_fallback": first_precision_fallback,
        "overflow_safe_mismatch": overflow_safe_mismatch,
        "post_replay_state_finite": post_replay_state_finite,
        "caught_expected_geometry_failure": bool(
            caught is not None and "replica geometry is nonfinite" in str(caught)
        ),
        "diagnostic_code_sha256": file_sha256(Path(__file__)),
        "amtnc_operator_sha256": file_sha256(ROOT / "src/models/route1/amtnc.py"),
        "runtime_sha256": file_sha256(ROOT / "research/paper_aio/runtime.py"),
        "writes_confined_to_diagnostic_output": True,
        "performance_values_read": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
        "scientific_lane_modified": False,
    }
    write_json(receipt_path, receipt)
    return receipt


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", type=Path, required=True)
    value.add_argument("--e0", type=Path, required=True)
    value.add_argument("--diagnostic-output", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--max-updates", type=int, default=8553)
    value.add_argument("--expect-overflow-safe", action="store_true")
    value.add_argument("--required-first-fallback-offset", type=int)
    value.add_argument("--required-first-fallback-player", choices=PLAYERS)
    value.add_argument("--required-parent-git-commit")
    value.add_argument("--required-parent-protocol-fingerprint")
    return value


def main() -> int:
    args = parser().parse_args()
    if args.max_updates < 1:
        raise SystemExit("--max-updates must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] in {
        "FAILURE_LOCALIZED", "OVERFLOW_SAFE_REPLAY_COMPLETE",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
