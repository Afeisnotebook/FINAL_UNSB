"""Durable post-e200 successor for candidates frozen under different code cores.

Each candidate is verified inside its own immutable worktree.  Only signed
terminal receipts cross the version boundary.  A positive cross-version winner
is then frozen and seed-validated inside that same worktree; no candidate is
ever loaded under a sibling's training core.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any

try:
    from operations import local_route1_candidate_executor as support
except ModuleNotFoundError:  # direct execution from operations/
    import local_route1_candidate_executor as support  # type: ignore[no-redef]

from operations.local_route1_cross_version_adjudicate import adjudicate


SCHEMA = "final-unsb-route1-cross-version-successor-contract-v1"
EXPECTED_IDS = (
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
    "G1-02B-PLAYER-CONDITIONAL-RSMG",
)
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
REPO_SOURCE_RELATIVES = (
    "operations/local_route1_candidate_terminal_receipt.py",
    "operations/local_route1_generation1_adjudicate.py",
    "operations/local_route1_seed_executor.py",
)
SUCCESSOR_SOURCE_RELATIVES = (
    "operations/local_route1_cross_version_successor.py",
    "operations/local_route1_cross_version_adjudicate.py",
    "operations/local_route1_candidate_terminal_receipt.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _candidate_env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def _parse_candidate_repo(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate-repo must use CANDIDATE_ID=/absolute/worktree")
    candidate_id, raw_path = value.split("=", 1)
    return support.safe_candidate_id(candidate_id), Path(raw_path).resolve()


def _candidate_status(
    *, python: Path, repo: Path, candidate_id: str, run_root: Path,
    train_view: Path, data_root: Path, manifest: Path,
) -> dict[str, Any]:
    command = [
        str(python), "-m", "research.local_route1.run",
        "--stage", "candidate", "--candidate-action", "status",
        "--candidate-id", candidate_id,
        "--output", str(run_root),
        "--train-view", str(train_view),
        "--data-root", str(data_root),
        "--manifest", str(manifest),
        "--gpu", "0",
    ]
    result = subprocess.run(
        command, cwd=repo, env=_candidate_env(repo), capture_output=True,
        text=True, encoding="utf-8", errors="replace", check=False, timeout=600,
    )
    if result.returncode:
        raise RuntimeError(
            f"candidate status failed for {candidate_id}:\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(result.stdout)


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    successor_repo = args.successor_repo.resolve()
    if support.run_text(["git", "status", "--porcelain"], cwd=successor_repo):
        raise RuntimeError("successor worktree must be clean")
    repositories = dict(_parse_candidate_repo(value) for value in args.candidate_repo)
    if tuple(repositories) != EXPECTED_IDS:
        raise RuntimeError("cross-version successor requires the frozen BVCP/PC-RSMG ids")
    candidate_records = []
    for candidate_id, repo in repositories.items():
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"candidate receipt worktree is dirty: {candidate_id}")
        missing = [relative for relative in REPO_SOURCE_RELATIVES if not (repo / relative).is_file()]
        if missing:
            raise RuntimeError(f"candidate receipt worktree lacks sources: {missing}")
        status = _candidate_status(
            python=args.python.resolve(), repo=repo, candidate_id=candidate_id,
            run_root=args.run_root.resolve(), train_view=args.train_view.resolve(),
            data_root=args.data_root.resolve(), manifest=args.manifest.resolve(),
        )
        candidate_records.append({
            "candidate_id": candidate_id,
            "repo": str(repo),
            "verification_git_commit": support.run_text(
                ["git", "rev-parse", "HEAD"], cwd=repo,
            ),
            "algorithm_fingerprint": status["algorithm_fingerprint"],
            "candidate_fingerprint": status["candidate_fingerprint"],
            "candidate_training_core_fingerprint": status[
                "candidate_training_core_fingerprint"
            ],
            "source_sha256": {
                relative: support.file_sha256(repo / relative)
                for relative in REPO_SOURCE_RELATIVES
            },
        })
    receipt_hashes = {
        record["source_sha256"][REPO_SOURCE_RELATIVES[0]]
        for record in candidate_records
    }
    expected_receipt = support.file_sha256(
        successor_repo / "operations" / "local_route1_candidate_terminal_receipt.py"
    )
    if receipt_hashes != {expected_receipt}:
        raise RuntimeError("candidate worktrees do not share the frozen receipt verifier")
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
        "candidates": candidate_records,
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "data_root": str(args.data_root.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": EXPECTED_MANIFEST,
        "python": str(args.python.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "seed_order": [2027, 2028],
        "seed2028_requires_seed2027_sign_inconsistency": True,
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("cross-version successor contract schema mismatch")
    if [row.get("candidate_id") for row in contract.get("candidates", [])] != list(EXPECTED_IDS):
        raise RuntimeError("cross-version candidate set changed")
    if contract.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("cross-version manifest identity changed")
    if support.file_sha256(Path(contract["manifest"])) != EXPECTED_MANIFEST:
        raise RuntimeError("cross-version manifest file changed")
    successor_repo = Path(contract["successor_repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=successor_repo) != contract.get(
        "successor_git_commit"
    ):
        raise RuntimeError("cross-version successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=successor_repo):
        raise RuntimeError("cross-version successor worktree is dirty")
    for relative, expected in contract.get("successor_source_sha256", {}).items():
        if support.file_sha256(successor_repo / relative) != expected:
            raise RuntimeError(f"cross-version successor source changed: {relative}")
    expected_receipt_hash = support.file_sha256(
        successor_repo / "operations" / "local_route1_candidate_terminal_receipt.py"
    )
    for record in contract["candidates"]:
        repo = Path(record["repo"])
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != record.get(
            "verification_git_commit"
        ):
            raise RuntimeError(f"candidate verifier worktree moved: {record['candidate_id']}")
        if support.run_text(["git", "status", "--porcelain"], cwd=repo):
            raise RuntimeError(f"candidate verifier worktree dirty: {record['candidate_id']}")
        for relative, expected in record.get("source_sha256", {}).items():
            if support.file_sha256(repo / relative) != expected:
                raise RuntimeError(f"candidate verifier source changed: {record['candidate_id']}")
        if record["source_sha256"][REPO_SOURCE_RELATIVES[0]] != expected_receipt_hash:
            raise RuntimeError("candidate verifier receipt source differs")
    if int(contract.get("poll_seconds", 0)) < 15:
        raise RuntimeError("cross-version successor poll interval too short")
    if int(contract.get("timeout_seconds", 0)) < 3600:
        raise RuntimeError("cross-version successor timeout too short")
    if contract.get("seed_order") != [2027, 2028]:
        raise RuntimeError("seed order changed")
    for key in ("paired_metric_scheduling", "paired_controller_access", "confirmation20_opened"):
        if contract.get(key) is not False:
            raise RuntimeError(f"cross-version successor requires {key}=false")


class CrossVersionSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = contract_path.resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.repo = Path(self.contract["successor_repo"])
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "CROSS_VERSION_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "CROSS_VERSION_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-cross-version-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(),
            "candidate_ids": list(EXPECTED_IDS),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-cross-version-successor-event-v1",
            "time": support.now(), "event": event,
            "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def _record(self, candidate_id: str) -> dict[str, Any]:
        return next(row for row in self.contract["candidates"] if row["candidate_id"] == candidate_id)

    def wait_for_candidates(self) -> None:
        started = time.time()
        while True:
            complete = [
                candidate_id for candidate_id in EXPECTED_IDS
                if (self.run_root / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json").is_file()
            ]
            fatal = [
                candidate_id for candidate_id in EXPECTED_IDS
                if (self.operations / f"CANDIDATE_EXECUTOR_FATAL_{candidate_id}.json").is_file()
            ]
            self.state(
                "WAITING_FOR_SOURCE_BOUND_E200_TRAJECTORIES",
                complete_candidate_ids=complete,
                pending_candidate_ids=[value for value in EXPECTED_IDS if value not in complete],
                elapsed_seconds=time.time() - started,
            )
            if fatal:
                raise RuntimeError(f"candidate executor fatal record observed: {fatal}")
            if len(complete) == len(EXPECTED_IDS):
                return
            if time.time() - started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("cross-version successor timed out")
            time.sleep(int(self.contract["poll_seconds"]))

    def create_receipt(self, candidate_id: str) -> Path:
        record = self._record(candidate_id)
        repo = Path(record["repo"])
        receipt = self.operations / "terminal_receipts" / f"{candidate_id}.json"
        command = [
            self.contract["python"],
            "operations/local_route1_candidate_terminal_receipt.py",
            "--output", str(self.run_root),
            "--candidate-id", candidate_id,
            "--receipt", str(receipt),
        ]
        result = subprocess.run(
            command, cwd=repo, env=_candidate_env(repo), capture_output=True,
            text=True, encoding="utf-8", errors="replace", check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"terminal receipt failed for {candidate_id}:\n{result.stdout}\n{result.stderr}"
            )
        self.event("SOURCE_BOUND_TERMINAL_RECEIPT_ACCEPTED", candidate_id=candidate_id)
        return receipt

    def freeze_single_winner(self, candidate_id: str) -> None:
        repo = Path(self._record(candidate_id)["repo"])
        result = subprocess.run([
            self.contract["python"],
            "operations/local_route1_generation1_adjudicate.py",
            "--output", str(self.run_root),
            "--candidate-id", candidate_id,
            "--freeze-winner",
        ], cwd=repo, env=_candidate_env(repo), capture_output=True, text=True, check=False)
        if result.returncode:
            raise RuntimeError(
                f"winner source-identity freeze failed:\n{result.stdout}\n{result.stderr}"
            )

    def run_seed(self, candidate_id: str, seed: int) -> None:
        repo = Path(self._record(candidate_id)["repo"])
        contract_path = self.operations / f"SEED_EXECUTOR_CONTRACT_{candidate_id}_s{seed}.json"
        if not contract_path.is_file():
            command = [
                self.contract["python"], "operations/local_route1_seed_executor.py",
                "--init-contract", "--contract", str(contract_path),
                "--main-repo", str(repo), "--seed-repo", str(repo),
                "--candidate-id", candidate_id, "--validation-seed", str(seed),
                "--run-root", str(self.run_root),
                "--train-view", self.contract["train_view"],
                "--data-root", self.contract["data_root"],
                "--manifest", self.contract["manifest"],
                "--python", self.contract["python"],
            ]
            result = subprocess.run(
                command, cwd=repo, env=_candidate_env(repo), capture_output=True,
                text=True, check=False,
            )
            if result.returncode:
                raise RuntimeError(f"seed{seed} contract failed:\n{result.stdout}\n{result.stderr}")
        stdout = self.operations / f"SEED_EXECUTOR_{candidate_id}_s{seed}.stdout.log"
        stderr = self.operations / f"SEED_EXECUTOR_{candidate_id}_s{seed}.stderr.log"
        with stdout.open("a", encoding="utf-8") as out, stderr.open("a", encoding="utf-8") as err:
            process = subprocess.Popen(
                [self.contract["python"], "operations/local_route1_seed_executor.py",
                 "--contract", str(contract_path)],
                cwd=repo, env=_candidate_env(repo), stdout=out, stderr=err,
            )
            while process.poll() is None:
                child_state_path = self.operations / f"SEED_EXECUTION_STATE_{candidate_id}_s{seed}.json"
                child = _read_json(child_state_path) if child_state_path.is_file() else {}
                self.state(
                    "FROZEN_SOURCE_IDENTITY_SEED_RUNNING",
                    candidate_id=candidate_id, seed=seed, child_pid=process.pid,
                    child_status=child.get("status"),
                    plain_data_epoch=child.get("plain_data_epoch"),
                    candidate_data_epoch=child.get("candidate_data_epoch"),
                )
                time.sleep(30)
            returncode = int(process.wait())
        if returncode:
            raise RuntimeError(f"seed{seed} executor failed with exit code {returncode}")

    def seed_summary(self, candidate_id: str) -> dict[str, Any]:
        repo = Path(self._record(candidate_id)["repo"])
        code = (
            "import json; from research.local_route1.seed_validation import "
            "summarize_multi_seed_validation as f; "
            f"print(json.dumps(f({str(self.run_root)!r}, {candidate_id!r})))"
        )
        result = subprocess.run(
            [self.contract["python"], "-c", code], cwd=repo,
            env=_candidate_env(repo), capture_output=True, text=True, check=False,
        )
        if result.returncode:
            raise RuntimeError(f"seed summary failed:\n{result.stdout}\n{result.stderr}")
        return json.loads(result.stdout)

    def run(self) -> int:
        self.event("CROSS_VERSION_SUCCESSOR_START", contract=str(self.contract_path))
        self.wait_for_candidates()
        receipts = [self.create_receipt(candidate_id) for candidate_id in EXPECTED_IDS]
        cross_path = self.operations / "CROSS_VERSION_E200_ADJUDICATION.json"
        result = adjudicate(receipts, cross_path)
        winner = str(result["selected_candidate_id"])
        self.event("CROSS_VERSION_E200_ADJUDICATED", status=result["status"], winner=winner)
        if result["status"] != "SEED2026_WINNER_REQUIRES_SOURCE_IDENTITY_SEED_FREEZE":
            self.state(
                "CROSS_VERSION_NEGATIVE_CAUSAL_ADJUDICATION_REQUIRED",
                selected_fallback=winner, automatic_revision_started=False,
                fixed_window_or_handoff_started=False,
            )
            return 0
        self.freeze_single_winner(winner)
        self.run_seed(winner, 2027)
        aggregate = self.seed_summary(winner)
        if aggregate["status"] == "WAITING_FOR_AUTHORIZED_SEED2028":
            self.run_seed(winner, 2028)
            aggregate = self.seed_summary(winner)
        self.state(
            "CROSS_VERSION_MULTI_SEED_ADJUDICATION_COMPLETE",
            candidate_id=winner, final_status=aggregate["status"],
            classification=aggregate.get("classification"),
            included_seeds=aggregate.get("included_seeds"),
            final_delivery_required=True,
        )
        self.event("CROSS_VERSION_SUCCESSOR_COMPLETE", winner=winner)
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--successor-repo", type=Path)
    value.add_argument("--candidate-repo", action="append", default=[])
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--data-root", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=172800)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = ("successor_repo", "run_root", "train_view", "data_root", "manifest", "python")
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
        with support.executor_lock(run_root / "operations" / "CROSS_VERSION_SUCCESSOR.lock"):
            return CrossVersionSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(run_root / "operations" / "CROSS_VERSION_SUCCESSOR_FATAL.json", {
            "schema": "final-unsb-route1-cross-version-successor-fatal-v1",
            "time": support.now(), "status": "FAILED", "error": repr(error),
            "traceback": traceback.format_exc(), "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        })
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
