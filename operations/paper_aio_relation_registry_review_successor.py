"""Durably turn two completed relation candidates into a Git review proposal.

The successor is metric-blind and never edits the tracked registry.  It waits
for the Proposal and ST-CGR review-only successor states, validates their exact
terminal identities, and invokes the deterministic registry review interface.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any


_BOOTSTRAP_ROOT = Path(__file__).resolve().parents[1]
if str(_BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(_BOOTSTRAP_ROOT))

from operations.paper_aio_relation_registry_review import (  # noqa: E402
    CANDIDATE_STATE_SCHEMA,
    REVIEW_SCHEMA,
    STANDARD_STATE_SCHEMA,
    STCGR_ID,
    read_json,
    review,
)
from operations.paper_aio_runtime_relation_successor import (  # noqa: E402
    contains_performance_field,
)
from research.paper_aio.protocol import file_sha256  # noqa: E402


CONTRACT_SCHEMA = "final-unsb-paper-relation-registry-review-successor-contract-v1"
STATE_SCHEMA = "final-unsb-paper-relation-registry-review-successor-state-v1"
PROPOSAL_COMPLETE = "COMPLETE_REVIEW_ONLY_RUNTIME_RELATION_CANDIDATE"
STCGR_COMPLETE = "COMPLETE_REVIEW_ONLY_CANDIDATE_CONTROL_RELATION"
SOURCE_RELATIVES = (
    "operations/paper_aio_relation_registry_review_successor.py",
    "operations/paper_aio_relation_registry_review.py",
    "operations/paper_aio_runtime_relation_successor.py",
    "operations/paper_aio_unified_evaluation_successor.py",
    "research/paper_aio/runtime_relation.py",
    "research/paper_aio/protocol.py",
)


try:
    import fcntl
except ImportError:  # pragma: no cover - deployment is POSIX, unit tests are portable.
    fcntl = None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def relation_state_release(
    path: Path,
    *,
    expected_schema: str,
    expected_status: str,
    lane_field: str,
    expected_lane: str,
    method_host: str,
    plain_host: str,
) -> str:
    if not Path(path).is_file():
        return "WAIT"
    value = read_json(path)
    if (
        value.get("performance_values_read") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
        or value.get("registry_edited") is not False
        or value.get("comparison_authorized") is not False
        or contains_performance_field(value)
    ):
        return "BLOCKED"
    status = str(value.get("status", ""))
    if status.startswith(("BLOCKED", "FAIL", "FATAL")):
        return "BLOCKED"
    if status != expected_status:
        return "WAIT"
    valid = (
        value.get("schema") == expected_schema
        and value.get(lane_field) == expected_lane
        and value.get("method_source_host_label") == method_host
        and value.get("plain_source_host_label") == plain_host
        and value.get("exact_runtime_equivalence") is True
        and Path(str(value.get("relation_candidate", ""))).is_absolute()
        and isinstance(value.get("relation_candidate_sha256"), str)
        and len(value["relation_candidate_sha256"]) == 64
    )
    return "READY" if valid else "BLOCKED"


def _contract(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != args.required_control_git_commit or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("registry-review successor checkout is not frozen")
    registry = args.registry.resolve()
    if not registry.is_file():
        raise RuntimeError("registry-review successor base registry is absent")
    return {
        "schema": CONTRACT_SCHEMA,
        "status": "FROZEN_WAITING",
        "repo": str(repo),
        "control_git_commit": commit,
        "control_source_sha256": {
            relative: file_sha256(repo / relative) for relative in SOURCE_RELATIVES
        },
        "registry": str(registry),
        "registry_sha256": file_sha256(registry),
        "proposal_state": str(args.proposal_state.resolve()),
        "stcgr_state": str(args.stcgr_state.resolve()),
        "proposal_method_host": args.proposal_method_host,
        "stcgr_method_host": args.stcgr_method_host,
        "plain_source_host": args.plain_source_host,
        "output": str(args.output.resolve()),
        "poll_seconds": int(args.poll_seconds),
        "timeout_hours": float(args.timeout_hours),
        "tracked_registry_edited": False,
        "comparison_authorized": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _verify_control(contract: dict[str, Any]) -> None:
    repo = Path(contract["repo"])
    if _git(repo, "rev-parse", "HEAD") != contract["control_git_commit"] or _git(
        repo, "status", "--porcelain"
    ):
        raise RuntimeError("registry-review successor checkout moved")
    for relative, expected in contract["control_source_sha256"].items():
        if file_sha256(repo / relative) != expected:
            raise RuntimeError(f"registry-review source changed: {relative}")
    registry = Path(contract["registry"])
    if not registry.is_file() or file_sha256(registry) != contract["registry_sha256"]:
        raise RuntimeError("registry-review base registry changed")


def _release(contract: dict[str, Any]) -> dict[str, str]:
    return {
        "proposal": relation_state_release(
            Path(contract["proposal_state"]),
            expected_schema=STANDARD_STATE_SCHEMA,
            expected_status=PROPOSAL_COMPLETE,
            lane_field="lane_id",
            expected_lane="proposal",
            method_host=contract["proposal_method_host"],
            plain_host=contract["plain_source_host"],
        ),
        "stcgr": relation_state_release(
            Path(contract["stcgr_state"]),
            expected_schema=CANDIDATE_STATE_SCHEMA,
            expected_status=STCGR_COMPLETE,
            lane_field="candidate_id",
            expected_lane=STCGR_ID,
            method_host=contract["stcgr_method_host"],
            plain_host=contract["plain_source_host"],
        ),
    }


def _review_args(contract: dict[str, Any]) -> SimpleNamespace:
    proposal_state_path = Path(contract["proposal_state"])
    stcgr_state_path = Path(contract["stcgr_state"])
    proposal_state = read_json(proposal_state_path)
    stcgr_state = read_json(stcgr_state_path)
    return SimpleNamespace(
        registry=Path(contract["registry"]),
        candidate=[
            Path(proposal_state["relation_candidate"]),
            Path(stcgr_state["relation_candidate"]),
        ],
        expected_candidate_sha256=[
            proposal_state["relation_candidate_sha256"],
            stcgr_state["relation_candidate_sha256"],
        ],
        candidate_state=[proposal_state_path, stcgr_state_path],
        expected_candidate_state_sha256=[
            file_sha256(proposal_state_path),
            file_sha256(stcgr_state_path),
        ],
        required_lane=["proposal", STCGR_ID],
        method_host=[
            f"proposal={contract['proposal_method_host']}",
            f"{STCGR_ID}={contract['stcgr_method_host']}",
        ],
        plain_source_host=contract["plain_source_host"],
        output=Path(contract["output"]) / "review",
    )


def _state(contract: dict[str, Any], *, status: str, releases: dict[str, str], **extra):
    return {
        "schema": STATE_SCHEMA,
        "status": status,
        "pid": os.getpid(),
        "relation_releases": releases,
        "tracked_registry_edited": False,
        "comparison_authorized": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        **extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 30 <= args.poll_seconds <= 600:
        raise ValueError("poll interval must be in [30,600] seconds")
    if args.timeout_hours < 24:
        raise ValueError("timeout must be at least 24 hours")
    contract = _contract(args)
    output = Path(contract["output"])
    operations = output / "operations"
    contract_path = operations / "REGISTRY_REVIEW_SUCCESSOR_CONTRACT.json"
    state_path = operations / "REGISTRY_REVIEW_SUCCESSOR_STATE.json"
    if contract_path.is_file():
        if read_json(contract_path) != contract:
            raise RuntimeError("registry-review successor contract changed")
    else:
        _write_json(contract_path, contract)
    lock_path = operations / "REGISTRY_REVIEW_SUCCESSOR.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with lock_path.open("w", encoding="utf-8") as lock:
        if fcntl is None:
            raise RuntimeError("registry-review successor requires POSIX file locking")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            _verify_control(contract)
            releases = _release(contract)
            if "BLOCKED" in releases.values():
                raise RuntimeError("relation candidate release failed closed")
            if set(releases.values()) == {"READY"}:
                receipt = review(_review_args(contract))
                if receipt.get("schema") != REVIEW_SCHEMA:
                    raise RuntimeError("registry review returned the wrong schema")
                receipt_path = (
                    Path(contract["output"])
                    / "review"
                    / "RUNTIME_RELATION_REGISTRY_REVIEW.json"
                )
                proposed_path = Path(receipt["proposed_registry"])
                result = _state(
                    contract,
                    status="COMPLETE_RUNTIME_RELATION_REGISTRY_REVIEW_PROPOSAL",
                    releases=releases,
                    review_receipt=str(receipt_path),
                    review_receipt_sha256=file_sha256(receipt_path),
                    proposed_registry=str(proposed_path),
                    proposed_registry_sha256=file_sha256(proposed_path),
                    explicit_codex_git_admission_still_required=True,
                )
                _write_json(state_path, result)
                return result
            if time.time() - started > contract["timeout_hours"] * 3600:
                raise TimeoutError("registry-review successor timed out")
            _write_json(
                state_path,
                _state(
                    contract,
                    status="WAITING_FOR_BOTH_REVIEW_ONLY_RELATION_CANDIDATES",
                    releases=releases,
                    elapsed_seconds=time.time() - started,
                ),
            )
            time.sleep(contract["poll_seconds"])


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo", type=Path, required=True)
    value.add_argument("--required-control-git-commit", required=True)
    value.add_argument("--registry", type=Path, required=True)
    value.add_argument("--proposal-state", type=Path, required=True)
    value.add_argument("--stcgr-state", type=Path, required=True)
    value.add_argument("--proposal-method-host", default="5090C")
    value.add_argument("--stcgr-method-host", default="5090A")
    value.add_argument("--plain-source-host", default="5090B_MATCHED_PLAIN")
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--poll-seconds", type=int, default=60)
    value.add_argument("--timeout-hours", type=float, default=720)
    return value


def main() -> int:
    args = parser().parse_args()
    try:
        print(json.dumps(run(args), ensure_ascii=False, indent=2))
        return 0
    except Exception as error:
        output = args.output.resolve()
        contract_path = (
            output / "operations" / "REGISTRY_REVIEW_SUCCESSOR_CONTRACT.json"
        )
        if contract_path.is_file():
            contract = read_json(contract_path)
            _write_json(
                output / "operations" / "REGISTRY_REVIEW_SUCCESSOR_STATE.json",
                _state(
                    contract,
                    status="BLOCKED_FAIL_CLOSED",
                    releases=_release(contract),
                    error_type=type(error).__name__,
                ),
            )
        print(f"registry-review successor failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
