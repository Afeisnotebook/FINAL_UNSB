"""Attach fixed DCLGAN evidence to the completed paper portfolio.

This successor is deliberately not a dependency of the core paper delivery.
It waits for both the core portfolio and the independently evaluated DCLGAN
trajectory, profiles only the fixed DCLGAN e200 checkpoint, and writes a new
hash-bound augmented portfolio.  No result is inspected while either upstream
is incomplete, and no training or scheduling decision is made here.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from operations import paper_aio_dclgan_evaluation_successor as dcl_eval
from operations.paper_aio_final_delivery_successor import (
    COMPLETE_STATUS as BASE_COMPLETE_STATUS,
    PORTFOLIO_SCHEMA as BASE_PORTFOLIO_SCHEMA,
    STATE_SCHEMA as BASE_STATE_SCHEMA,
    _complexity_summary,
    validate_complexity_receipt,
)
from research.paper_aio.protocol import file_sha256, protocol_fingerprint


CONTRACT_SCHEMA = "final-unsb-paper-dclgan-portfolio-addendum-contract-v1"
STATE_SCHEMA = "final-unsb-paper-dclgan-portfolio-addendum-state-v1"
ADDENDUM_SCHEMA = "final-unsb-paper-external-baseline-addendum-v1"
AUGMENTED_PORTFOLIO_SCHEMA = "final-unsb-paper-full-data-algorithm-portfolio-v2"
COMPLETE_STATUS = "COMPLETE_DCLGAN_PAPER_PORTFOLIO_ADDENDUM"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    os.replace(temporary, path)


def _immutable_json(path: Path, value: dict[str, Any]) -> None:
    path = Path(path)
    if path.is_file():
        if _read(path) != value:
            raise RuntimeError(f"immutable DCLGAN addendum differs: {path}")
        return
    _atomic_json(path, value)


def _git(repo: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=repo, text=True, stderr=subprocess.DEVNULL,
    ).strip()


def _bound_artifact(
    state: dict[str, Any], *, field: str, expected_path: Path,
) -> bool:
    expected_path = Path(expected_path).resolve()
    return (
        Path(str(state.get(field, ""))).resolve() == expected_path
        and expected_path.is_file()
        and isinstance(state.get(f"{field}_sha256"), str)
        and file_sha256(expected_path) == state[f"{field}_sha256"]
    )


def base_portfolio_decision(output: Path) -> str:
    """Inspect only completion metadata until the core portfolio is complete."""
    output = Path(output).resolve()
    state_path = output / "operations" / "FINAL_DELIVERY_STATE.json"
    if not state_path.is_file():
        return "WAIT"
    state = _read(state_path)
    status = str(state.get("status", ""))
    if status.startswith(("BLOCKED", "FAIL", "FATAL")):
        return "BLOCKED"
    portfolio = output / "PAPER_ALGORITHM_PORTFOLIO.json"
    valid = (
        status == BASE_COMPLETE_STATUS
        and state.get("schema") == BASE_STATE_SCHEMA
        and state.get("performance_values_in_control_state") is False
        and state.get("metric_values_used_for_training_or_scheduling") is False
        and state.get("paired_metric_control") is False
        and state.get("best_checkpoint_selection") is False
        and state.get("paper_claims_frozen") is False
        and state.get("confirmation_authorized") is False
        and state.get("confirmation20_opened") is False
        and _bound_artifact(state, field="portfolio", expected_path=portfolio)
    )
    return "READY" if valid else ("BLOCKED" if status == BASE_COMPLETE_STATUS else "WAIT")


def dclgan_result_decision(output: Path) -> str:
    """Inspect only completion metadata until DCLGAN fixed evaluation completes."""
    output = Path(output).resolve()
    state_path = output / "operations" / "DCLGAN_EVALUATION_STATE.json"
    if not state_path.is_file():
        return "WAIT"
    state = _read(state_path)
    status = str(state.get("status", ""))
    if status.startswith(("BLOCKED", "FAIL", "FATAL")):
        return "BLOCKED"
    result = output / "DCLGAN_PAPER_RESULT.json"
    valid = (
        status == dcl_eval.COMPLETE_STATUS
        and state.get("schema") == dcl_eval.STATE_SCHEMA
        and state.get("performance_values_in_control_state") is False
        and state.get("paired_metric_control") is False
        and state.get("best_checkpoint_selection") is False
        and state.get("confirmation20_opened") is False
        and _bound_artifact(state, field="result", expected_path=result)
    )
    return "READY" if valid else (
        "BLOCKED" if status == dcl_eval.COMPLETE_STATUS else "WAIT"
    )


def validate_base_portfolio(path: Path) -> dict[str, Any]:
    value = _read(path)
    if (
        value.get("schema") != BASE_PORTFOLIO_SCHEMA
        or value.get("status")
        != "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_AWAITING_CONFIRMATION_DECISION"
        or value.get("paper_claims_frozen") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("cross_non_equivalent_runtime_delta") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
        or "dclgan" in (value.get("external_baselines") or {})
    ):
        raise RuntimeError("invalid base paper portfolio for DCLGAN addendum")
    return value


def validate_dclgan_result(path: Path) -> dict[str, Any]:
    value = _read(path)
    epochs = [
        int(row.get("epoch", -1)) for row in value.get("trajectory", [])
        if isinstance(row, dict)
    ]
    terminal = value.get("terminal") or {}
    if (
        value.get("schema") != dcl_eval.RESULT_SCHEMA
        or value.get("status") != "COMPLETE_FIXED_E200_EXTERNAL_BASELINE"
        or value.get("lane_id") != "dclgan"
        or value.get("primary_epoch") != 200
        or value.get("fixed_epochs") != list(dcl_eval.EPOCHS)
        or epochs != list(dcl_eval.EPOCHS)
        or terminal.get("epoch") != 200
        or value.get("comparison_scope")
        != "standalone_fixed_protocol_no_matched_delta_claim"
        or value.get("performance_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("paired_metric_control") is not False
        or value.get("cross_non_equivalent_runtime_delta") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("invalid fixed DCLGAN paper result")
    receipts = value.get("evaluation_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(dcl_eval.EPOCHS):
        raise RuntimeError("incomplete DCLGAN evaluation receipt set")
    for row in receipts:
        for field in ("receipt", "metric"):
            artifact = Path(str(row.get(field, ""))).resolve()
            if (
                not artifact.is_file()
                or file_sha256(artifact) != row.get(f"{field}_sha256")
            ):
                raise RuntimeError(f"DCLGAN {field} artifact changed")
    return value


def _terminal_checkpoint_sha256(result: dict[str, Any]) -> str:
    matches = [
        row for row in result["evaluation_receipts"]
        if int(row.get("epoch", -1)) == 200
    ]
    if len(matches) != 1:
        raise RuntimeError("DCLGAN result lacks exactly one e200 checkpoint")
    digest = matches[0].get("checkpoint_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise RuntimeError("DCLGAN e200 checkpoint identity is invalid")
    return digest


def build_augmented_portfolio(
    *, base: dict[str, Any], dclgan: dict[str, Any], complexity: dict[str, Any],
    source_hashes: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if complexity.get("lane_id") != "dclgan":
        raise RuntimeError("DCLGAN complexity receipt has the wrong lane")
    augmented = copy.deepcopy(base)
    augmented["schema"] = AUGMENTED_PORTFOLIO_SCHEMA
    augmented["status"] = (
        "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_WITH_DCLGAN_"
        "AWAITING_CONFIRMATION_DECISION"
    )
    augmented["external_baselines"] = dict(augmented["external_baselines"])
    augmented["external_baselines"]["dclgan"] = copy.deepcopy(dclgan)
    augmented["complexity"] = dict(augmented["complexity"])
    augmented["complexity"]["dclgan"] = _complexity_summary(complexity)
    augmented["source_artifact_sha256"] = {
        **dict(augmented.get("source_artifact_sha256") or {}),
        **source_hashes,
    }
    augmented["dclgan_is_nonblocking_post_core_addendum"] = True
    augmented["paper_claims_frozen"] = False
    augmented["confirmation_authorized"] = False
    augmented["confirmation20_opened"] = False
    addendum = {
        "schema": ADDENDUM_SCHEMA,
        "status": "COMPLETE_FIXED_DCLGAN_EXTERNAL_BASELINE_ADDENDUM",
        "lane_id": "dclgan",
        "primary_epoch": 200,
        "fixed_epochs": list(dcl_eval.EPOCHS),
        "result": copy.deepcopy(dclgan),
        "complexity": _complexity_summary(complexity),
        "source_artifact_sha256": dict(source_hashes),
        "core_paper_delivery_was_not_blocked": True,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "cross_non_equivalent_runtime_delta": False,
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    return augmented, addendum


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    head = _git(repo, "rev-parse", "HEAD")
    if head != args.required_control_git_commit:
        raise RuntimeError("DCLGAN addendum control checkout moved")
    if _git(repo, "status", "--porcelain"):
        raise RuntimeError("DCLGAN addendum control checkout is dirty")
    evaluation_contract = (
        args.dclgan_evaluation_output.resolve() / "operations"
        / "DCLGAN_EVALUATION_CONTRACT.json"
    )
    if not evaluation_contract.is_file():
        raise RuntimeError("DCLGAN evaluation contract is missing")
    if not 30 <= int(args.poll_seconds) <= 600 or float(args.timeout_hours) < 24:
        raise ValueError("unsafe DCLGAN addendum waiting policy")
    sources = [
        Path(__file__).resolve(),
        repo / "research" / "paper_aio" / "complexity.py",
        repo / "research" / "paper_aio" / "runtime.py",
        repo / "operations" / "paper_aio_dclgan_evaluation_successor.py",
    ]
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "control_repo": str(repo),
        "control_git_commit": head,
        "control_source_sha256": {
            str(path.relative_to(repo)).replace("\\", "/"): file_sha256(path)
            for path in sources
        },
        "base_delivery_output": str(args.base_delivery_output.resolve()),
        "dclgan_evaluation_output": str(args.dclgan_evaluation_output.resolve()),
        "dclgan_evaluation_contract": str(evaluation_contract),
        "dclgan_evaluation_contract_sha256": file_sha256(evaluation_contract),
        "output": str(args.output.resolve()),
        "gpu_lock": str(args.gpu_lock.resolve()),
        "gpu": int(args.gpu),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "core_delivery_dependency": False,
        "performance_values_available_to_scheduler": False,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify_contract(value: dict[str, Any]) -> dict[str, Any]:
    repo = Path(value["control_repo"])
    if (
        _git(repo, "rev-parse", "HEAD") != value["control_git_commit"]
        or _git(repo, "status", "--porcelain")
    ):
        raise RuntimeError("DCLGAN addendum control checkout changed")
    for relative, digest in value["control_source_sha256"].items():
        if file_sha256(repo / relative) != digest:
            raise RuntimeError(f"DCLGAN addendum control source changed: {relative}")
    evaluation_contract_path = Path(value["dclgan_evaluation_contract"])
    if (
        file_sha256(evaluation_contract_path)
        != value["dclgan_evaluation_contract_sha256"]
    ):
        raise RuntimeError("DCLGAN evaluation contract changed")
    evaluation_contract = _read(evaluation_contract_path)
    dcl_eval.verify_contract(evaluation_contract)
    return evaluation_contract


def _profile_complexity(
    *, evaluation_contract: dict[str, Any], dclgan_result: dict[str, Any],
    destination: Path,
) -> dict[str, Any]:
    rows = dcl_eval.validate_import_lane(
        Path(evaluation_contract["import_root"]),
        source_host_label=evaluation_contract["source_host_label"],
        required_training_commit=evaluation_contract["adapter_git_commit"],
        required_adapter_fingerprint=evaluation_contract["adapter_fingerprint"],
    )
    source = next((row for row in rows if int(row["epoch"]) == 200), None)
    if source is None:
        raise RuntimeError("imported DCLGAN lane lacks e200")
    checkpoint = Path(source["checkpoint"]).resolve()
    expected_checkpoint = _terminal_checkpoint_sha256(dclgan_result)
    if file_sha256(checkpoint) != expected_checkpoint:
        raise RuntimeError("DCLGAN result and imported e200 checkpoint differ")
    adapter = dcl_eval.load_frozen_adapter(Path(evaluation_contract["adapter_repo"]))
    upstream = adapter.verify_upstream(Path(evaluation_contract["upstream_root"]))
    if upstream["commit"] != evaluation_contract["upstream_commit"]:
        raise RuntimeError("DCLGAN upstream identity changed before complexity")
    manifest = Path(evaluation_contract["manifest"])
    model, stream, payload = adapter._load_evaluation_runtime(
        upstream_root=Path(evaluation_contract["upstream_root"]),
        manifest_path=manifest,
        train_view=Path(evaluation_contract["train_view"]),
        output_root=Path(evaluation_contract["output"]) / "complexity_runtime",
        checkpoint=checkpoint,
        gpu=int(evaluation_contract["gpu"]),
    )
    from research.paper_aio.complexity import profile_model

    value = profile_model(
        model=model,
        spec=adapter.dclgan_lane_spec(),
        rows=adapter.annotated_manifest_rows(manifest),
        primary=stream,
        secondary=stream,
        data_root=Path(evaluation_contract["data_root"]),
        checkpoint=checkpoint,
        checkpoint_metadata=payload["metadata"],
        destination=destination,
    )
    del model
    adapter.torch.cuda.empty_cache()
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    proposed = _contract(args)
    output = Path(proposed["output"])
    contract_path = output / "operations" / "DCLGAN_PORTFOLIO_ADDENDUM_CONTRACT.json"
    state_path = output / "operations" / "DCLGAN_PORTFOLIO_ADDENDUM_STATE.json"
    lock_path = output / "operations" / "DCLGAN_PORTFOLIO_ADDENDUM.lock"
    if contract_path.is_file():
        if _read(contract_path) != proposed:
            raise RuntimeError("DCLGAN portfolio addendum contract changed")
    else:
        _atomic_json(contract_path, proposed)
    base = {
        "schema": STATE_SCHEMA,
        "pid": os.getpid(),
        "contract": str(contract_path.resolve()),
        "contract_sha256": file_sha256(contract_path),
        "performance_values_available_to_scheduler": False,
        "metric_values_used_for_training_or_scheduling": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    }
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with lock_path.open("a+", encoding="utf-8") as handle:
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        heartbeat = dcl_eval.Heartbeat(state_path, base, proposed["poll_seconds"])
        heartbeat.start()
        try:
            while True:
                evaluation_contract = _verify_contract(proposed)
                decisions = {
                    "base_portfolio": base_portfolio_decision(
                        Path(proposed["base_delivery_output"])
                    ),
                    "dclgan_result": dclgan_result_decision(
                        Path(proposed["dclgan_evaluation_output"])
                    ),
                }
                if "BLOCKED" in decisions.values():
                    raise RuntimeError(f"DCLGAN addendum dependency blocked: {decisions}")
                if all(value == "READY" for value in decisions.values()):
                    break
                if time.time() - started > proposed["timeout_hours"] * 3600:
                    raise TimeoutError("DCLGAN portfolio addendum successor timed out")
                heartbeat.update(
                    status="WAITING_FOR_CORE_PORTFOLIO_AND_DCLGAN_FIXED_RESULT",
                    dependencies=decisions,
                    performance_values_read=False,
                )
                time.sleep(proposed["poll_seconds"])

            base_path = (
                Path(proposed["base_delivery_output"])
                / "PAPER_ALGORITHM_PORTFOLIO.json"
            )
            dclgan_path = (
                Path(proposed["dclgan_evaluation_output"])
                / "DCLGAN_PAPER_RESULT.json"
            )
            base_portfolio = validate_base_portfolio(base_path)
            dclgan_result = validate_dclgan_result(dclgan_path)
            complexity_path = output / "complexity" / "dclgan.json"
            gpu_lock = Path(proposed["gpu_lock"])
            gpu_lock.parent.mkdir(parents=True, exist_ok=True)
            with gpu_lock.open("a+", encoding="utf-8") as gpu_handle:
                while True:
                    try:
                        fcntl.flock(gpu_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        heartbeat.update(
                            status="WAITING_FOR_SHARED_EVALUATION_GPU",
                            performance_values_read=True,
                        )
                        time.sleep(proposed["poll_seconds"])
                    else:
                        break
                if not complexity_path.is_file():
                    heartbeat.update(
                        status="PROFILING_FIXED_DCLGAN_E200_COMPLEXITY",
                        performance_values_read=True,
                    )
                    _profile_complexity(
                        evaluation_contract=evaluation_contract,
                        dclgan_result=dclgan_result,
                        destination=complexity_path,
                    )
            complexity = validate_complexity_receipt(
                complexity_path,
                lane_id="dclgan",
                checkpoint_sha256=_terminal_checkpoint_sha256(dclgan_result),
                expected_protocol_fingerprint=protocol_fingerprint(
                    Path(evaluation_contract["manifest"])
                ),
            )
            source_hashes = {
                "base_portfolio": file_sha256(base_path),
                "dclgan_result": file_sha256(dclgan_path),
                "dclgan_evaluation_contract": file_sha256(
                    Path(proposed["dclgan_evaluation_contract"])
                ),
                "dclgan_complexity": file_sha256(complexity_path),
            }
            augmented, addendum = build_augmented_portfolio(
                base=base_portfolio,
                dclgan=dclgan_result,
                complexity=complexity,
                source_hashes=source_hashes,
            )
            augmented_path = output / "PAPER_ALGORITHM_PORTFOLIO_WITH_DCLGAN.json"
            addendum_path = output / "PAPER_EXTERNAL_BASELINE_ADDENDUM.json"
            _immutable_json(augmented_path, augmented)
            _immutable_json(addendum_path, addendum)
            final = {
                **base,
                "status": COMPLETE_STATUS,
                "augmented_portfolio": str(augmented_path.resolve()),
                "augmented_portfolio_sha256": file_sha256(augmented_path),
                "addendum": str(addendum_path.resolve()),
                "addendum_sha256": file_sha256(addendum_path),
                "complexity": str(complexity_path.resolve()),
                "complexity_sha256": file_sha256(complexity_path),
                "performance_values_read": True,
                "performance_values_in_control_state": False,
                "core_paper_delivery_was_not_blocked": True,
                "paper_claims_frozen": False,
                "confirmation_authorized": False,
            }
            heartbeat.update(**final)
            return final
        except Exception as error:
            heartbeat.update(
                status="FAIL_CLOSED_REQUIRES_CODEX_AUDIT",
                error_type=type(error).__name__,
                error_message=str(error),
                metric_values_used_for_training_or_scheduling=False,
                paired_metric_control=False,
                confirmation20_opened=False,
            )
            raise
        finally:
            heartbeat.close()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--base-delivery-output", type=Path, required=True)
    value.add_argument("--dclgan-evaluation-output", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--gpu-lock", type=Path, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    print(json.dumps(run(parser().parse_args()), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
