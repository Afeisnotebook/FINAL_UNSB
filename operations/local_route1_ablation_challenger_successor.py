"""Durably seed-validate an unexpected winner-ablation challenger.

The ordinary winner-ablation successor deliberately refuses final delivery
when the proposal-only e200 trajectory outranks the frozen full operator.  This
successor closes the expensive execution gap without changing the selection:
it freezes that already-complete proposal identity and runs the registered
host-matched seed protocol.  A later posthoc adjudicator must compare the two
frozen multi-seed records before any final candidate can change.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from operations import local_route1_candidate_executor as support
except ModuleNotFoundError:  # direct execution from operations/
    import local_route1_candidate_executor as support  # type: ignore[no-redef]

from operations.local_route1_winner_ablation_adjudicate import (
    SCHEMA as ABLATION_SCHEMA,
)


SCHEMA = "final-unsb-route1-ablation-challenger-successor-contract-v1"
CHALLENGER_STATUS = "ABLATION_CHALLENGER_REQUIRES_FROZEN_SEED_VALIDATION"
COMPLETE_SEED_STATUSES = ("ROUTE1_SUSTAINED_LOCAL", "MULTI_SEED_NOT_SUSTAINED")
SUCCESSOR_SOURCE_RELATIVES = (
    "operations/local_route1_ablation_challenger_successor.py",
)
ABLATION_REPO_SOURCE_RELATIVES = (
    "operations/local_route1_seed_executor.py",
    "research/local_route1/candidate_runner.py",
    "research/local_route1/seed_validation.py",
)
WORKSPACE_SCHEMA = "final-unsb-route1-ablation-challenger-seed-workspace-v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def challenger_from_adjudication(payload: dict[str, Any]) -> str | None:
    """Return the frozen proposal id only for a source-bound challenge."""
    if payload.get("schema") != ABLATION_SCHEMA:
        raise RuntimeError("winner-ablation adjudication schema mismatch")
    for key in (
        "paired_metrics_used_for_training_or_control",
        "paired_controller_access",
        "confirmation20_opened",
    ):
        if payload.get(key) is not False:
            raise RuntimeError(f"winner-ablation adjudication violates {key}=false")
    status = payload.get("status")
    if status == "COMPLETE_NO_SELECTION_CHANGE":
        if payload.get("proposal_only_out_ranks_full") is not False:
            raise RuntimeError("no-change ablation record contradicts its ranking flag")
        return None
    if status != CHALLENGER_STATUS:
        raise RuntimeError("winner-ablation adjudication is incomplete")
    if (
        payload.get("proposal_only_out_ranks_full") is not True
        or payload.get("selection_change_blocked_pending_seed_validation") is not True
        or payload.get("selection_changed") is not False
    ):
        raise RuntimeError("ablation challenger was not fail-closed before seed validation")
    roles = payload.get("roles")
    if not isinstance(roles, dict) or not isinstance(roles.get("proposal_only"), dict):
        raise RuntimeError("ablation challenger record lacks proposal-only identity")
    return support.safe_candidate_id(str(roles["proposal_only"]["candidate_id"]))


def materialize_challenger_seed_workspace(
    source_root: Path, workspace_root: Path, candidate_id: str,
) -> dict[str, Any]:
    """Copy the immutable registration authority into a seed-isolated root.

    Seed validation historically stores one candidate under
    ``seed_validation/seed<N>``.  The full winner already owns that namespace,
    so an ablation challenger must not reuse or overwrite it.  This workspace
    keeps the same card, gate, causal evidence and e0 hashes while giving the
    challenger a separate matched-plain/candidate seed namespace.
    """
    source_root = Path(source_root).resolve()
    workspace_root = Path(workspace_root).resolve()
    candidate_id = support.safe_candidate_id(candidate_id)
    relatives = (
        Path("audit/LONG_CAUSAL_MATRIX.json"),
        Path("audit/LONG_REVERSAL_ATLAS.jsonl"),
        Path("derive/HYPOTHESIS_LEDGER.json"),
        Path("derive/cards") / f"{candidate_id}.json",
        Path("derive/implementations") / f"{candidate_id}.json",
        Path("derive/gates") / f"{candidate_id}.json",
        Path("shared_e0/e0.pt"),
        Path("shared_e0/e0.pt.json"),
        Path("candidates") / candidate_id / "CANDIDATE_TRAJECTORY.json",
    )
    sources = {}
    for relative in relatives:
        source = source_root / relative
        if not source.is_file():
            raise RuntimeError(f"challenger seed authority is missing: {relative}")
        sources[relative.as_posix()] = support.file_sha256(source)
    record_path = workspace_root / "CHALLENGER_SEED_WORKSPACE.json"
    if record_path.is_file():
        record = _read_json(record_path)
        if (
            record.get("schema") != WORKSPACE_SCHEMA
            or record.get("source_root") != str(source_root)
            or record.get("candidate_id") != candidate_id
            or record.get("source_sha256") != sources
        ):
            raise RuntimeError("existing challenger seed workspace identity changed")
        for relative, expected in sources.items():
            if support.file_sha256(workspace_root / relative) != expected:
                raise RuntimeError(
                    f"challenger seed workspace authority changed: {relative}"
                )
        return record
    if workspace_root.exists() and any(workspace_root.iterdir()):
        raise RuntimeError("unregistered challenger seed workspace is not empty")
    for relative in relatives:
        destination = workspace_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(str(destination) + ".tmp")
        shutil.copy2(source_root / relative, temporary)
        if support.file_sha256(temporary) != sources[relative.as_posix()]:
            raise RuntimeError(f"challenger seed workspace copy failed: {relative}")
        temporary.replace(destination)
    record = {
        "schema": WORKSPACE_SCHEMA,
        "created": support.now(),
        "source_root": str(source_root),
        "workspace_root": str(workspace_root),
        "candidate_id": candidate_id,
        "source_sha256": sources,
        "full_winner_seed_namespace_reused": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    support.atomic_json(record_path, record)
    return record


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    successor_repo = args.successor_repo.resolve()
    ablation_repo = args.ablation_repo.resolve()
    for label, repo in (
        ("successor", successor_repo), ("ablation", ablation_repo),
    ):
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"{label} worktree must be clean")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "successor_repo": str(successor_repo),
        "successor_git_commit": support.run_text(
            ["git", "rev-parse", "HEAD"], cwd=successor_repo,
        ),
        "successor_source_sha256": {
            relative: support.file_sha256(successor_repo / relative)
            for relative in SUCCESSOR_SOURCE_RELATIVES
        },
        "ablation_repo": str(ablation_repo),
        "ablation_repo_git_commit": support.run_text(
            ["git", "rev-parse", "HEAD"], cwd=ablation_repo,
        ),
        "ablation_repo_source_sha256": {
            relative: support.file_sha256(ablation_repo / relative)
            for relative in ABLATION_REPO_SOURCE_RELATIVES
        },
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "seed_order": [2027, 2028],
        "selection_change_before_seed_adjudication": False,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("ablation challenger successor contract schema mismatch")
    for label, path_key, commit_key, source_key in (
        (
            "successor", "successor_repo", "successor_git_commit",
            "successor_source_sha256",
        ),
        (
            "ablation", "ablation_repo", "ablation_repo_git_commit",
            "ablation_repo_source_sha256",
        ),
    ):
        repo = Path(contract[path_key])
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
            commit_key
        ):
            raise RuntimeError(f"{label} worktree moved")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"{label} worktree is dirty")
        for relative, expected in contract.get(source_key, {}).items():
            if support.file_sha256(repo / relative) != expected:
                raise RuntimeError(f"{label} source changed: {relative}")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("ablation challenger manifest changed")
    if contract.get("seed_order") != [2027, 2028]:
        raise RuntimeError("ablation challenger seed order changed")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("ablation challenger poll interval too short")
    if int(contract.get("timeout_seconds", 0)) < 3600:
        raise RuntimeError("ablation challenger timeout too short")
    for key in (
        "selection_change_before_seed_adjudication",
        "paired_metric_scheduling", "paired_controller_access",
        "confirmation20_opened",
    ):
        if contract.get(key) is not False:
            raise RuntimeError(f"ablation challenger successor requires {key}=false")


class AblationChallengerSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["ablation_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "ABLATION_CHALLENGER_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "ABLATION_CHALLENGER_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-ablation-challenger-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(),
            "selection_changed": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-ablation-challenger-successor-event-v1",
            "time": support.now(), "event": event,
            "supervisor_pid": os.getpid(),
            "selection_changed": False,
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_for_adjudication(self) -> tuple[str | None, dict[str, Any]]:
        path = self.operations / "WINNER_ABLATION_ADJUDICATION.json"
        started = time.time()
        while True:
            payload = _read_json(path) if path.is_file() else None
            self.state(
                "WAITING_FOR_WINNER_ABLATION_ADJUDICATION",
                adjudication_status=None if payload is None else payload.get("status"),
                elapsed_seconds=time.time() - started,
            )
            if payload is not None:
                return challenger_from_adjudication(payload), payload
            if time.time() - started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("ablation challenger successor timed out")
            time.sleep(int(self.contract["poll_seconds"]))

    def freeze(self, workspace: Path, candidate_id: str) -> None:
        code = (
            "import json; from research.local_route1.candidate_runner import "
            "freeze_for_seed_validation as f; "
            f"print(json.dumps(f({str(workspace)!r}, {candidate_id!r})))"
        )
        result = subprocess.run(
            [self.contract["python"], "-c", code], cwd=self.repo,
            env=_env(self.repo), capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"ablation challenger freeze failed:\n{result.stdout}\n{result.stderr}"
            )

    def run_seed(self, workspace: Path, candidate_id: str, seed: int) -> None:
        workspace_operations = workspace / "operations"
        workspace_operations.mkdir(parents=True, exist_ok=True)
        path = workspace_operations / f"SEED_EXECUTOR_CONTRACT_{candidate_id}_s{seed}.json"
        if not path.is_file():
            command = [
                self.contract["python"], "operations/local_route1_seed_executor.py",
                "--init-contract", "--contract", str(path),
                "--main-repo", str(self.repo), "--seed-repo", str(self.repo),
                "--candidate-id", candidate_id, "--validation-seed", str(seed),
                "--run-root", str(workspace),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", self.contract["python"],
            ]
            result = subprocess.run(
                command, cwd=self.repo, env=_env(self.repo),
                capture_output=True, text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(
                    f"challenger seed{seed} contract failed:\n"
                    f"{result.stdout}\n{result.stderr}"
                )
        stdout = self.operations / f"ABLATION_CHALLENGER_SEED_s{seed}.stdout.log"
        stderr = self.operations / f"ABLATION_CHALLENGER_SEED_s{seed}.stderr.log"
        with stdout.open("a", encoding="utf-8") as out, stderr.open(
            "a", encoding="utf-8"
        ) as err:
            process = subprocess.Popen(
                [self.contract["python"], "operations/local_route1_seed_executor.py",
                 "--contract", str(path)],
                cwd=self.repo, env=_env(self.repo), stdout=out, stderr=err,
            )
            while process.poll() is None:
                child_path = workspace_operations / (
                    f"SEED_EXECUTION_STATE_{candidate_id}_s{seed}.json"
                )
                child = _read_json(child_path) if child_path.is_file() else {}
                self.state(
                    "ABLATION_CHALLENGER_FROZEN_SEED_RUNNING",
                    challenger_candidate_id=candidate_id, seed=seed,
                    child_pid=process.pid, child_status=child.get("status"),
                    plain_data_epoch=child.get("plain_data_epoch"),
                    candidate_data_epoch=child.get("candidate_data_epoch"),
                )
                time.sleep(30)
            returncode = int(process.wait())
        if returncode:
            raise RuntimeError(
                f"ablation challenger seed{seed} failed with exit code {returncode}"
            )

    def seed_summary(self, workspace: Path, candidate_id: str) -> dict[str, Any]:
        code = (
            "import json; from research.local_route1.seed_validation import "
            "summarize_multi_seed_validation as f; "
            f"print(json.dumps(f({str(workspace)!r}, {candidate_id!r})))"
        )
        result = subprocess.run(
            [self.contract["python"], "-c", code], cwd=self.repo,
            env=_env(self.repo), capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"challenger seed summary failed:\n{result.stdout}\n{result.stderr}"
            )
        return json.loads(result.stdout)

    def run(self) -> int:
        self.event("ABLATION_CHALLENGER_SUCCESSOR_START", contract=str(self.contract_path))
        challenger, adjudication = self.wait_for_adjudication()
        if challenger is None:
            self.state(
                "INAPPLICABLE_NO_ABLATION_CHALLENGER",
                adjudication_status=adjudication["status"], seeds_started=False,
            )
            return 0
        workspace = (
            self.run_root / "ablation_challenger_seed_validation" / challenger
        )
        workspace_record = materialize_challenger_seed_workspace(
            self.run_root, workspace, challenger,
        )
        self.freeze(workspace, challenger)
        self.event(
            "ABLATION_CHALLENGER_IDENTITY_FROZEN", candidate_id=challenger,
            workspace=str(workspace),
        )
        self.run_seed(workspace, challenger, 2027)
        aggregate = self.seed_summary(workspace, challenger)
        if aggregate["status"] == "WAITING_FOR_AUTHORIZED_SEED2028":
            self.run_seed(workspace, challenger, 2028)
            aggregate = self.seed_summary(workspace, challenger)
        if aggregate.get("status") not in COMPLETE_SEED_STATUSES:
            raise RuntimeError("ablation challenger seed adjudication did not complete")
        self.state(
            "ABLATION_CHALLENGER_MULTI_SEED_COMPLETE_SELECTION_STILL_FROZEN",
            challenger_candidate_id=challenger,
            multi_seed_status=aggregate["status"],
            classification=aggregate.get("classification"),
            included_seeds=aggregate.get("included_seeds"),
            challenger_seed_workspace=str(workspace),
            challenger_seed_workspace_record_sha256=support.file_sha256(
                workspace / "CHALLENGER_SEED_WORKSPACE.json"
            ),
            source_authority_sha256=workspace_record["source_sha256"],
            final_selection_adjudication_required=True,
        )
        self.event(
            "ABLATION_CHALLENGER_SUCCESSOR_COMPLETE", candidate_id=challenger,
        )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--successor-repo", type=Path)
    value.add_argument("--ablation-repo", type=Path)
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=604800)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = (
            "successor_repo", "ablation_repo", "run_root", "train_view",
            "data_root", "manifest", "python",
        )
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            raise SystemExit(f"--init-contract missing arguments: {missing}")
        contract = default_contract(args)
        validate_contract(contract)
        support.atomic_json(args.contract.resolve(), contract)
        print(json.dumps({"status": "CONTRACT_INITIALIZED", **contract}, indent=2))
        return 0
    contract = _read_json(args.contract.resolve())
    run_root = Path(contract["run_root"])
    try:
        with support.executor_lock(
            run_root / "operations" / "ABLATION_CHALLENGER_SUCCESSOR.lock"
        ):
            return AblationChallengerSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(
            run_root / "operations" / "ABLATION_CHALLENGER_SUCCESSOR_FATAL.json",
            {
                "schema": "final-unsb-route1-ablation-challenger-successor-fatal-v1",
                "time": support.now(), "status": "FAILED", "error": repr(error),
                "traceback": traceback.format_exc(), "supervisor_pid": os.getpid(),
                "selection_changed": False,
                "paired_metric_scheduling": False,
                "paired_controller_access": False,
                "confirmation20_opened": False,
            },
        )
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
