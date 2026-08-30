"""Durably audit a fully negative cross-version e200 outcome.

Each candidate's target-blind defect is measured in its own compatible
worktree.  Compact audit records are then aggregated without importing sibling
candidate code.  The process never invents a revision, threshold, window or
handoff; a mathematically eligible revision is returned to the research agent
for evidence-driven derivation.
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

from research.local_route1.candidate_defect_audit import (
    GENERATION1_NEGATIVE_STATUS,
    adjudicate_cross_version_revision_need,
)


SCHEMA = "final-unsb-route1-cross-version-negative-successor-contract-v1"
EXPECTED_IDS = (
    "G1-01-ROLLOUT-DISTRIBUTION-SPEED",
    "G1-02B-PLAYER-CONDITIONAL-RSMG",
)
CANDIDATE_SOURCE_RELATIVES = (
    "operations/local_route1_candidate_defect_audit.py",
    "research/local_route1/candidate_defect_audit.py",
)
SUCCESSOR_SOURCE_RELATIVES = (
    "operations/local_route1_cross_version_negative_successor.py",
    "research/local_route1/candidate_defect_audit.py",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _env(repo: Path) -> dict[str, str]:
    value = dict(os.environ)
    prefix = os.pathsep.join((str(repo), str(repo / "src")))
    current = value.get("PYTHONPATH")
    value["PYTHONPATH"] = prefix if not current else prefix + os.pathsep + current
    return value


def _parse_candidate_repo(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--candidate-repo must use CANDIDATE_ID=/absolute/worktree")
    candidate_id, path = value.split("=", 1)
    return support.safe_candidate_id(candidate_id), Path(path).resolve()


def default_contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.successor_repo.resolve()
    candidate_repos = dict(_parse_candidate_repo(value) for value in args.candidate_repo)
    if tuple(candidate_repos) != EXPECTED_IDS:
        raise RuntimeError("negative successor candidate set/order changed")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("negative successor worktree must be clean")
    records = []
    for candidate_id, candidate_repo in candidate_repos.items():
        if support.run_text(["git", "status", "--porcelain"], cwd=candidate_repo):
            raise RuntimeError(f"candidate audit worktree is dirty: {candidate_id}")
        records.append({
            "candidate_id": candidate_id,
            "repo": str(candidate_repo),
            "git_commit": support.run_text(
                ["git", "rev-parse", "HEAD"], cwd=candidate_repo,
            ),
            "source_sha256": {
                relative: support.file_sha256(candidate_repo / relative)
                for relative in CANDIDATE_SOURCE_RELATIVES
            },
        })
    audit_hashes = {
        record["source_sha256"]["research/local_route1/candidate_defect_audit.py"]
        for record in records
    }
    if audit_hashes != {
        support.file_sha256(repo / "research/local_route1/candidate_defect_audit.py")
    }:
        raise RuntimeError("candidate worktrees do not share the frozen defect auditor")
    return {
        "schema": SCHEMA,
        "created": support.now(),
        "successor_repo": str(repo),
        "successor_git_commit": support.run_text(["git", "rev-parse", "HEAD"], cwd=repo),
        "successor_source_sha256": {
            relative: support.file_sha256(repo / relative)
            for relative in SUCCESSOR_SOURCE_RELATIVES
        },
        "candidates": records,
        "run_root": str(args.run_root.resolve()),
        "train_view": str(args.train_view.resolve()),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": support.file_sha256(args.manifest.resolve()),
        "python": str(args.python.resolve()),
        "samples": int(args.samples),
        "poll_seconds": int(args.poll_seconds),
        "timeout_seconds": int(args.timeout_seconds),
        "paired_metric_scheduling": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }


def validate_contract(contract: dict[str, Any]) -> None:
    if contract.get("schema") != SCHEMA:
        raise RuntimeError("negative successor contract schema mismatch")
    if [row.get("candidate_id") for row in contract.get("candidates", [])] != list(
        EXPECTED_IDS
    ):
        raise RuntimeError("negative successor candidate identities changed")
    repo = Path(contract["successor_repo"])
    if support.run_text(["git", "rev-parse", "HEAD"], cwd=repo) != contract.get(
        "successor_git_commit"
    ):
        raise RuntimeError("negative successor worktree moved")
    if support.run_text(["git", "status", "--porcelain"], cwd=repo):
        raise RuntimeError("negative successor worktree is dirty")
    for relative, expected in contract.get("successor_source_sha256", {}).items():
        if support.file_sha256(repo / relative) != expected:
            raise RuntimeError(f"negative successor source changed: {relative}")
    expected_audit = support.file_sha256(
        repo / "research/local_route1/candidate_defect_audit.py"
    )
    for record in contract["candidates"]:
        candidate_repo = Path(record["repo"])
        if support.run_text(["git", "rev-parse", "HEAD"], cwd=candidate_repo) != record.get(
            "git_commit"
        ):
            raise RuntimeError(f"candidate audit worktree moved: {record['candidate_id']}")
        if support.run_text(["git", "status", "--porcelain"], cwd=candidate_repo):
            raise RuntimeError(f"candidate audit worktree dirty: {record['candidate_id']}")
        for relative, expected in record.get("source_sha256", {}).items():
            if support.file_sha256(candidate_repo / relative) != expected:
                raise RuntimeError(f"candidate audit source changed: {record['candidate_id']}")
        if record["source_sha256"][
            "research/local_route1/candidate_defect_audit.py"
        ] != expected_audit:
            raise RuntimeError("candidate audit source differs across code versions")
    if support.file_sha256(Path(contract["manifest"])) != contract.get("manifest_sha256"):
        raise RuntimeError("negative successor manifest changed")
    if int(contract.get("samples", 0)) < 4 or int(contract["samples"]) % 2:
        raise RuntimeError("negative successor requires an even sample count >=4")
    if int(contract.get("poll_seconds", 0)) < 15 or int(contract.get("timeout_seconds", 0)) < 3600:
        raise RuntimeError("negative successor polling/timeout contract is unsafe")
    for key in ("paired_metric_scheduling", "paired_controller_access", "confirmation20_opened"):
        if contract.get(key) is not False:
            raise RuntimeError(f"negative successor requires {key}=false")


class CrossVersionNegativeSuccessor:
    def __init__(self, contract_path: Path):
        self.contract_path = Path(contract_path).resolve()
        self.contract = _read_json(self.contract_path)
        validate_contract(self.contract)
        self.run_root = Path(self.contract["run_root"])
        self.operations = self.run_root / "operations"
        self.state_path = self.operations / "CROSS_VERSION_NEGATIVE_SUCCESSOR_STATE.json"
        self.events_path = self.operations / "CROSS_VERSION_NEGATIVE_SUCCESSOR_EVENTS.jsonl"

    def state(self, status: str, **fields: Any) -> None:
        support.atomic_json(self.state_path, {
            "schema": "final-unsb-route1-cross-version-negative-successor-state-v1",
            "updated": support.now(), "status": status,
            "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def event(self, event: str, **fields: Any) -> None:
        support.append_jsonl(self.events_path, {
            "schema": "final-unsb-route1-cross-version-negative-successor-event-v1",
            "time": support.now(), "event": event, "supervisor_pid": os.getpid(),
            "paired_metric_scheduling": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **fields,
        })

    def wait_for_cross(self) -> dict[str, Any]:
        path = self.operations / "CROSS_VERSION_E200_ADJUDICATION.json"
        started = time.time()
        while True:
            cross = _read_json(path) if path.is_file() else None
            self.state(
                "WAITING_FOR_CROSS_VERSION_E200_ADJUDICATION",
                cross_status=None if cross is None else cross.get("status"),
                elapsed_seconds=time.time() - started,
            )
            if cross is not None:
                return cross
            if time.time() - started > int(self.contract["timeout_seconds"]):
                raise TimeoutError("negative successor timed out waiting for e200 adjudication")
            time.sleep(int(self.contract["poll_seconds"]))

    def run_audits(self) -> None:
        processes = []
        for record in self.contract["candidates"]:
            candidate_id = record["candidate_id"]
            repo = Path(record["repo"])
            stdout = self.operations / f"NEGATIVE_DEFECT_AUDIT_{candidate_id}.stdout.log"
            stderr = self.operations / f"NEGATIVE_DEFECT_AUDIT_{candidate_id}.stderr.log"
            out = stdout.open("a", encoding="utf-8")
            err = stderr.open("a", encoding="utf-8")
            command = [
                self.contract["python"], "operations/local_route1_candidate_defect_audit.py",
                "--output", str(self.run_root), "--candidate-id", candidate_id,
                "--train-view", self.contract["train_view"],
                "--manifest", self.contract["manifest"], "--gpu", "0",
                "--samples", str(self.contract["samples"]),
            ]
            process = subprocess.Popen(
                command, cwd=repo, env=_env(repo), stdout=out, stderr=err,
            )
            processes.append((candidate_id, process, out, err))
        while any(process.poll() is None for _, process, _, _ in processes):
            self.state(
                "TARGET_BLIND_E200_DEFECT_AUDITS_RUNNING",
                children={candidate_id: process.pid for candidate_id, process, _, _ in processes},
            )
            time.sleep(30)
        failures = []
        for candidate_id, process, out, err in processes:
            out.close(); err.close()
            if int(process.wait()) != 0:
                failures.append(candidate_id)
        if failures:
            raise RuntimeError(f"target-blind defect audits failed: {failures}")

    def run(self) -> int:
        self.event("CROSS_VERSION_NEGATIVE_SUCCESSOR_START", contract=str(self.contract_path))
        cross = self.wait_for_cross()
        if cross.get("status") != GENERATION1_NEGATIVE_STATUS:
            self.state(
                "INAPPLICABLE_POSITIVE_CROSS_VERSION_WINNER",
                cross_status=cross.get("status"), audits_started=False,
            )
            return 0
        self.run_audits()
        result = adjudicate_cross_version_revision_need(
            self.run_root, list(EXPECTED_IDS),
        )
        self.event("CROSS_VERSION_NEGATIVE_DEFECTS_ADJUDICATED", status=result["status"])
        if result["status"] == "REVISION_DERIVATION_REQUIRED":
            self.state(
                "MATHEMATICAL_REVISION_DERIVATION_REQUIRED",
                selected_parent_candidate_id=result["selected_candidate_id"],
                revision_applicable_candidate_ids=result[
                    "revision_applicable_candidate_ids"
                ],
                automatic_revision_started=False,
                fixed_window_or_handoff_started=False,
            )
        else:
            self.state(
                "NO_REVISION_APPLICABLE_FINAL_FALLBACK_REQUIRES_FINAL_ABLATION_ADJUDICATION",
                selected_fallback=result["selected_candidate_id"],
                automatic_revision_started=False,
                fixed_window_or_handoff_started=False,
            )
        return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--contract", type=Path, required=True)
    value.add_argument("--init-contract", action="store_true")
    value.add_argument("--successor-repo", type=Path)
    value.add_argument("--candidate-repo", action="append", default=[])
    value.add_argument("--run-root", type=Path)
    value.add_argument("--train-view", type=Path)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--python", type=Path)
    value.add_argument("--samples", type=int, default=32)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-seconds", type=int, default=172800)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.init_contract:
        required = ("successor_repo", "run_root", "train_view", "manifest", "python")
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
            run_root / "operations" / "CROSS_VERSION_NEGATIVE_SUCCESSOR.lock"
        ):
            return CrossVersionNegativeSuccessor(args.contract).run()
    except Exception as error:
        support.atomic_json(run_root / "operations" / "CROSS_VERSION_NEGATIVE_SUCCESSOR_FATAL.json", {
            "schema": "final-unsb-route1-cross-version-negative-successor-fatal-v1",
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
