"""Run source-bound e200 ablations if a frontier replay becomes the winner."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from operations import local_route1_candidate_executor as support
from operations.local_route1_candidate_terminal_receipt import materialize_receipt
from operations.local_route1_cross_version_adjudicate import (
    _validate_receipt,
    adjudicate as rank_receipts,
)
from operations.local_route1_winner_ablation_adjudicate import (
    adjudicate as adjudicate_ablations,
)
from operations.local_route1_winner_ablation_successor import WinnerAblationSuccessor
from research.local_route1.frontier_final_delivery import (
    FINAL_SELECTION,
    _same_host_selection,
)
from research.local_route1.runtime import write_json
from research.local_route1.winner_ablations import (
    WINNER_FAMILIES,
    materialize_winner_ablation_definitions,
)


SCHEMA = "final-unsb-route1-frontier-winner-ablation-successor-contract-v1"
RESULT = "FRONTIER_WINNER_ABLATION_RESULT.json"
POST_SELECTION = "ROUTE1_FRONTIER_POST_ABLATION_SELECTION.json"
SOURCE_RELATIVES = (
    "operations/local_route1_frontier_winner_ablation_successor.py",
    "operations/local_route1_winner_ablation_successor.py",
    "operations/local_route1_winner_ablation_adjudicate.py",
    "operations/local_route1_candidate_executor.py",
    "operations/local_route1_candidate_terminal_receipt.py",
    "operations/local_route1_cross_version_adjudicate.py",
    "research/local_route1/frontier_final_delivery.py",
    "research/local_route1/winner_ablations.py",
    "research/local_route1/generation1_gates.py",
    "src/models/route1/pcnr.py",
    "src/models/route1/pcnr_ablation.py",
    "src/models/route1_pcnr_model.py",
    "src/models/route1_pcnr_ablation_model.py",
    "src/models/route1/ammcrb.py",
    "src/models/route1/ammcrb_ablation.py",
    "src/models/route1_ammcrb_model.py",
    "src/models/route1_ammcrb_ablation_model.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier winner ablation worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "repo": str(repo),
        "git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SOURCE_RELATIVES
        },
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "baseline_environment_record": str(args.baseline_environment_record.resolve()),
        "baseline_environment_record_sha256": support.file_sha256(
            args.baseline_environment_record.resolve()
        ),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parallel_e200_executors": 1,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "requires_complete_cross_host_result": True,
        "requires_selected_algorithm_specific_ablations": True,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("frontier winner ablation contract schema mismatch")
    repo = Path(contract["repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "git_commit"
    ):
        raise RuntimeError("frontier winner ablation worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("frontier winner ablation worktree is dirty")
    for relative, expected in contract.get("source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"frontier winner ablation source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("frontier winner ablation manifest changed")
    if support.file_sha256(Path(contract["baseline_environment_record"])) != contract.get(
        "baseline_environment_record_sha256"
    ):
        raise RuntimeError("frontier winner ablation environment record changed")
    fixed = {
        "batch_size": 1,
        "target_data_epochs": 200,
        "maximum_parallel_e200_executors": 1,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "requires_complete_cross_host_result": True,
        "requires_selected_algorithm_specific_ablations": True,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    for key, expected in fixed.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"frontier winner ablation contract changed: {key}")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("frontier winner ablation polling is too frequent")
    if int(contract.get("timeout_seconds", 0)) < 43200:
        raise RuntimeError("frontier winner ablation timeout is too short")


class FrontierWinnerAblationSuccessor(WinnerAblationSuccessor):
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "FRONTIER_WINNER_ABLATION_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "FRONTIER_WINNER_ABLATION_SUCCESSOR_EVENTS.jsonl"
        self.started = time.time()

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-frontier-winner-ablation-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(), "batch_size": 1,
            "target_data_epochs": 200,
            "maximum_parallel_e200_executors": 1,
            "selection_seeds": [2026],
            "deferred_seed_validation": [2027, 2028],
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-frontier-winner-ablation-successor-event-v1",
            "time": support.now(), "event": event,
            "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_cross_result(self) -> dict[str, Any]:
        result_path = self.operations / "FRONTIER_CROSS_HOST_RESULT.json"
        fatal_path = self.operations / "FRONTIER_CROSS_HOST_SUCCESSOR_FATAL.json"
        while not result_path.is_file():
            if fatal_path.is_file():
                raise RuntimeError(f"frontier cross-host successor failed: {fatal_path}")
            if time.time() - self.started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("timed out waiting for frontier cross-host result")
            state_path = self.operations / "FRONTIER_CROSS_HOST_SUCCESSOR_STATE.json"
            state = _read_json(state_path) if state_path.is_file() else {}
            self.state(
                "WAITING_FOR_TERMINAL_FRONTIER_CROSS_HOST_RESULT",
                cross_host_status=state.get("status"),
                cross_host_data_epoch=state.get("data_epoch"),
            )
            time.sleep(int(self.contract["poll_seconds"]))
        result = _read_json(result_path)
        if result.get("confirmation20_opened") is not False or result.get(
            "paired_controller_access"
        ) is not False:
            raise RuntimeError("frontier cross-host result violates target-blind scope")
        return result

    def _base_delivery(self) -> tuple[dict[str, Any], dict[str, Any]]:
        candidate_path = self.run_root / "final" / "CANDIDATE.json"
        results_path = self.run_root / "final" / "RESULTS.json"
        if not candidate_path.is_file() or not results_path.is_file():
            raise RuntimeError("pre-frontier final delivery is not complete")
        candidate = _read_json(candidate_path)
        results = _read_json(results_path)
        if results.get("selected_candidate_id") != candidate.get("candidate_id"):
            raise RuntimeError("pre-frontier candidate/results identity mismatch")
        return candidate, results

    def _write_reuse_result(
        self, *, base_id: str, selection: dict[str, Any], selection_path: Path,
        receipt_path: Path, base_results: dict[str, Any],
    ) -> dict[str, Any]:
        ablation_path = self.operations / "WINNER_ABLATION_ADJUDICATION.json"
        evidence = base_results.get("winner_ablation_results")
        if not ablation_path.is_file() or not isinstance(evidence, dict):
            raise RuntimeError("retained pre-frontier winner lacks its own e200 ablations")
        result = {
            "schema": "final-unsb-route1-frontier-winner-ablation-result-v1",
            "status": "REUSED_PRE_FRONTIER_SELECTED_WINNER_ABLATIONS",
            "selected_candidate_id": base_id,
            "selected_receipt_path": str(receipt_path),
            "selected_receipt_sha256": support.file_sha256(receipt_path),
            "post_ablation_selection_path": str(selection_path),
            "post_ablation_selection_sha256": support.file_sha256(selection_path),
            "post_ablation_selection": selection,
            "winner_ablation_adjudication_path": str(ablation_path),
            "winner_ablation_adjudication_sha256": support.file_sha256(ablation_path),
            "winner_ablation_evidence": evidence,
            "new_frontier_ablation_e200_executors": 0,
            "paired_metrics_used_only_after_complete_e200_trajectories": True,
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        write_json(self.operations / RESULT, result)
        return result

    def run(self) -> int:
        self.event("FRONTIER_WINNER_ABLATION_SUCCESSOR_START", contract=str(self.contract_path))
        cross_result = self.wait_cross_result()
        base_candidate, base_results = self._base_delivery()
        base_id = str(base_candidate["candidate_id"])
        selection, receipt_path, receipt = _same_host_selection(
            self.run_root, base_id, cross_result,
        )
        selection_path = self.operations / FINAL_SELECTION
        selected_id = str(receipt["candidate_id"])
        if selected_id == base_id:
            result = self._write_reuse_result(
                base_id=base_id, selection=selection, selection_path=selection_path,
                receipt_path=receipt_path, base_results=base_results,
            )
            self.state(result["status"], selected_candidate_id=base_id)
            return 0

        if selected_id not in WINNER_FAMILIES:
            raise RuntimeError("new frontier winner has no source-bound ablation family")
        frozen = materialize_winner_ablation_definitions(
            self.run_root,
            selection_path=selection_path,
            freeze_filename="FRONTIER_WINNER_ABLATION_FREEZE.json",
        )
        candidate_ids = [
            frozen["ablation_candidate_ids"][role]
            for role in ("proposal_only", "observable_only")
        ]
        self.event("FRONTIER_WINNER_ABLATIONS_SOURCE_FROZEN", candidate_ids=candidate_ids)
        self.run_gates(candidate_ids)
        self.event("FRONTIER_WINNER_ABLATION_GATES_PASS", candidate_ids=candidate_ids)
        self.run_e200(candidate_ids)

        receipts = {
            candidate_id: self.operations / "terminal_receipts" / f"{candidate_id}.json"
            for candidate_id in candidate_ids
        }
        for candidate_id, path in receipts.items():
            materialize_receipt(self.run_root, candidate_id, path)
        proposal_id = frozen["ablation_candidate_ids"]["proposal_only"]
        observable_id = frozen["ablation_candidate_ids"]["observable_only"]
        ablation_path = self.operations / "FRONTIER_WINNER_ABLATION_ADJUDICATION.json"
        ablation = adjudicate_ablations(
            output_root=self.run_root,
            cross_adjudication_path=selection_path,
            proposal_receipt_path=receipts[proposal_id],
            observable_receipt_path=receipts[observable_id],
            full_receipt_path=receipt_path,
            output_path=ablation_path,
        )

        base_receipt_path = self.operations / "terminal_receipts" / f"{base_id}.json"
        post_path = self.operations / POST_SELECTION
        post = rank_receipts(
            [base_receipt_path, receipt_path, receipts[proposal_id]], post_path,
        )
        post.update({
            "selection_scope": "same_host_4090_frontier_full_and_source_bound_ablation",
            "source_frontier_selection_sha256": support.file_sha256(selection_path),
            "source_winner_ablation_adjudication_sha256": support.file_sha256(ablation_path),
            "observable_only_excluded_from_candidate_ranking": True,
            "cross_host_deltas_merged": False,
            "additional_seed_replication_deferred": [2027, 2028],
            "cross_seed_stability_claimed": False,
        })
        write_json(post_path, post)
        final_id = str(post["selected_candidate_id"])
        final_receipt_path = self.operations / "terminal_receipts" / f"{final_id}.json"
        final_receipt = _validate_receipt(final_receipt_path)
        common = {
            "schema": "final-unsb-route1-frontier-winner-ablation-result-v1",
            "frontier_full_candidate_id": selected_id,
            "selected_candidate_id": final_id,
            "selected_algorithm_fingerprint": final_receipt["algorithm_fingerprint"],
            "selected_receipt_path": str(final_receipt_path),
            "selected_receipt_sha256": support.file_sha256(final_receipt_path),
            "post_ablation_selection_path": str(post_path),
            "post_ablation_selection_sha256": support.file_sha256(post_path),
            "post_ablation_selection": post,
            "ablation_candidate_ids": frozen["ablation_candidate_ids"],
            "new_frontier_ablation_e200_executors": 2,
            "paired_metrics_used_only_after_complete_e200_trajectories": True,
            "paired_metrics_used_for_training_or_control": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        }
        if final_id == base_id:
            base_ablation_path = (
                self.operations / "WINNER_ABLATION_ADJUDICATION.json"
            )
            base_evidence = base_results.get("winner_ablation_results")
            if not base_ablation_path.is_file() or not isinstance(base_evidence, dict):
                raise RuntimeError(
                    "reselected pre-frontier winner lacks its own e200 ablations"
                )
            result = {
                **common,
                "status": (
                    "PRE_FRONTIER_SELECTED_WINNER_RETAINED_AFTER_"
                    "FRONTIER_ABLATIONS"
                ),
                "winner_ablation_adjudication_path": str(base_ablation_path),
                "winner_ablation_adjudication_sha256": support.file_sha256(
                    base_ablation_path
                ),
                "winner_ablation_evidence": base_evidence,
                "frontier_challenger_ablation_adjudication_path": str(
                    ablation_path
                ),
                "frontier_challenger_ablation_adjudication_sha256": (
                    support.file_sha256(ablation_path)
                ),
                "frontier_challenger_ablation_evidence": ablation["roles"],
            }
        else:
            result = {
                **common,
                "status": "FRONTIER_SELECTED_ALGORITHM_ABLATIONS_COMPLETE",
                "winner_ablation_adjudication_path": str(ablation_path),
                "winner_ablation_adjudication_sha256": support.file_sha256(
                    ablation_path
                ),
                "winner_ablation_evidence": ablation["roles"],
            }
        write_json(self.operations / RESULT, result)
        self.state(
            result["status"], selected_candidate_id=final_id,
            frontier_full_candidate_id=selected_id,
        )
        self.event("FRONTIER_WINNER_ABLATION_SUCCESSOR_COMPLETE", winner=final_id)
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--baseline-environment-record", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=1209600)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "repo", "run_root", "train_view", "data_root", "manifest", "python",
            "baseline_environment_record",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    contract = _read_json(args.contract)
    run_root = Path(contract["run_root"])
    try:
        with support.executor_lock(
            run_root / "operations" / "FRONTIER_WINNER_ABLATION_SUCCESSOR.lock"
        ):
            return FrontierWinnerAblationSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "FRONTIER_WINNER_ABLATION_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-frontier-winner-ablation-successor-fatal-v1",
                "updated": support.now(), "status": "FAILED",
                "error": repr(error), "traceback": traceback.format_exc(),
                "supervisor_pid": os.getpid(),
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
