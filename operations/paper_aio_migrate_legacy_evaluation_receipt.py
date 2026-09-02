"""Migrate the legacy paper repeated-evaluation receipt without re-evaluating.

The first paper runner overloaded ``protocol_fingerprint`` with the frozen
evaluation-bundle fingerprint.  Its authorization gate, however, compared the
same field with the training protocol fingerprint.  A deterministic repeated
evaluation could therefore pass and still be rejected as stale.

This control-plane migration is deliberately narrow.  It verifies the frozen
scientific checkout, the training-bound preflight and exact-resume receipts,
and the bit-identical repeated evaluation before separating the two identities.
It does not load a checkpoint, inspect a metric value, or authorize training.
The unchanged scientific runner must still issue the final lane authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any


SCHEMA = "final-unsb-paper-legacy-evaluation-receipt-migration-v1"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def migrate(
    *, output: Path, lane: str, scientific_repo: Path,
    required_scientific_commit: str, training_fingerprint: str,
    evaluation_fingerprint: str,
) -> dict[str, Any]:
    output = output.resolve()
    scientific_repo = scientific_repo.resolve()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=scientific_repo, text=True,
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=scientific_repo, text=True,
    ).strip()
    if head != required_scientific_commit or dirty:
        raise RuntimeError("frozen scientific checkout identity changed")

    gates = output / "gates"
    preflight_path = gates / "PREFLIGHT.json"
    resume_path = gates / f"RESUME_GATE_{lane}.json"
    repeat_path = gates / f"EVALUATION_REPEAT_{lane}.json"
    preflight = _read(preflight_path)
    resume = _read(resume_path)
    repeat = _read(repeat_path)

    if (
        preflight.get("status") != "PASS"
        or preflight.get("node_role") != "training"
        or preflight.get("protocol_fingerprint") != training_fingerprint
        or preflight.get("manifest", {}).get("content_hashes_verified") is not True
        or preflight.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("training preflight is not an eligible frozen receipt")
    if (
        resume.get("status") != "PASS"
        or resume.get("lane_id") != lane
        or resume.get("protocol_fingerprint") != training_fingerprint
        or resume.get("continuous_core_sha256") != resume.get("resumed_core_sha256")
        or int(resume.get("total_updates", -1)) != 1000
        or int(resume.get("split_updates", -1)) != 500
        or resume.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("lane exact-resume receipt is not eligible")
    if (
        repeat.get("schema") != "final-unsb-paper-evaluation-repeat-gate-v1"
        or repeat.get("status") != "PASS"
        or repeat.get("lane_id") != lane
        or repeat.get("split") != "discovery"
        or repeat.get("first_result_sha256") != repeat.get("second_result_sha256")
        or repeat.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("repeated-evaluation receipt is not deterministic and eligible")

    old_hash = _sha256(repeat_path)
    observed_training = repeat.get("protocol_fingerprint")
    observed_evaluation = repeat.get("evaluation_bundle_fingerprint")
    if observed_evaluation is None:
        if observed_training != evaluation_fingerprint:
            raise RuntimeError("legacy receipt does not contain the frozen evaluation fingerprint")
        repeat["evaluation_bundle_fingerprint"] = observed_training
        repeat["protocol_fingerprint"] = training_fingerprint
    elif (
        observed_training != training_fingerprint
        or observed_evaluation != evaluation_fingerprint
    ):
        raise RuntimeError("already-separated receipt identities do not match requirements")

    repeat["migration"] = {
        "schema": SCHEMA,
        "legacy_receipt_sha256": old_hash,
        "scientific_git_commit": head,
        "training_protocol_fingerprint": training_fingerprint,
        "evaluation_bundle_fingerprint": evaluation_fingerprint,
        "checkpoint_loaded": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
        "migrated_unix_time": time.time(),
    }
    _write(repeat_path, repeat)
    return repeat


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lane", choices=("plain", "proposal", "amtnc", "hjcgr"), required=True)
    parser.add_argument("--scientific-repo", type=Path, required=True)
    parser.add_argument("--required-scientific-commit", required=True)
    parser.add_argument("--training-fingerprint", required=True)
    parser.add_argument("--evaluation-fingerprint", required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    result = migrate(
        output=args.output,
        lane=args.lane,
        scientific_repo=args.scientific_repo,
        required_scientific_commit=args.required_scientific_commit,
        training_fingerprint=args.training_fingerprint,
        evaluation_fingerprint=args.evaluation_fingerprint,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
