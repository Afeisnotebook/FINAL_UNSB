"""Verify a frozen route-1 milestone and emit compact acceptance evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TRAINING_COMMIT = "0da2a37086cca5bc4ad4488bb07c53096a7152ed"
EXPECTED_PROTOCOL = "b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2"
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
EXPECTED_CRN = "3eded31c265a38f34c48b7e7216a140244e2314513b4fff396f8c64dee76dbb4"
DOMAINS = (
    "FoggyCityscapes", "LowLightTrafficData", "RSCityscapes",
    "RainCityscapes", "RainDS-syn", "SnowTrafficData",
)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def command(argv: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=300, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def scientific_state_sha256(checkpoint: Path, training_repo: Path) -> str:
    checkpoint_literal = json.dumps(str(checkpoint.resolve()))
    code = (
        "import torch; "
        "from research.local_route1.runtime import full_state_hash; "
        f"p=torch.load({checkpoint_literal},map_location='cpu',weights_only=False); "
        "print(full_state_hash(p))"
    )
    return command([sys.executable, "-c", code], cwd=training_repo)


def validate_metric(
    metric: dict[str, Any], *, lane: str, epoch: int, require_lpips: bool,
) -> dict[str, Any]:
    expected = {
        "schema": "local-route1-discovery70-crn-single-rollout-v1",
        "split": "discovery",
        "count_per_domain": 70,
        "replicates": 1,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "evaluation_input_sha256": EXPECTED_CRN,
        "confirmation20_opened": False,
        "probe_id": lane,
        "epoch": epoch,
        "updates": epoch * 150,
        "data_epoch": epoch,
    }
    for key, value in expected.items():
        if metric.get(key) != value:
            raise RuntimeError(f"metric identity mismatch for {key}")
    if require_lpips and (
        metric.get("lpips_requested") is not True
        or metric.get("lpips_available") is not True
        or metric.get("macro_lpips") is None
    ):
        raise RuntimeError("required LPIPS metric is unavailable")
    images = metric.get("images")
    if not isinstance(images, list) or len(images) != 420:
        raise RuntimeError("metric does not contain exactly 420 images")
    counts = {domain: 0 for domain in DOMAINS}
    for row in images:
        domain = row.get("domain")
        if domain not in counts:
            raise RuntimeError(f"unexpected discovery domain {domain}")
        counts[domain] += 1
        if not row.get("crn_bundle_sha256"):
            raise RuntimeError("image-level CRN identity is missing")
    if any(value != 70 for value in counts.values()):
        raise RuntimeError(f"discovery domain counts differ from 70: {counts}")
    domains = metric.get("domains", {})
    if set(domains) != set(DOMAINS):
        raise RuntimeError("metric domain summary set mismatch")
    for domain in DOMAINS:
        if int(domains[domain].get("n", -1)) != 70:
            raise RuntimeError(f"metric summary count mismatch for {domain}")
    return {
        domain: {
            "psnr": domains[domain]["psnr"],
            "ssim": domains[domain]["ssim"],
            "lpips": domains[domain].get("lpips"),
        }
        for domain in DOMAINS
    }


def verify(
    *, run_root: Path, training_repo: Path, lane: str, epoch: int,
    require_lpips: bool,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    training_repo = training_repo.resolve()
    head = command(["git", "rev-parse", "HEAD"], cwd=training_repo)
    if head != EXPECTED_TRAINING_COMMIT:
        raise RuntimeError("frozen training worktree commit mismatch")
    if command(["git", "status", "--porcelain"], cwd=training_repo):
        raise RuntimeError("frozen training worktree is dirty")
    checkpoint = run_root / "anchors" / lane / "milestones" / f"e{epoch:03d}.pt"
    sidecar_path = Path(str(checkpoint) + ".json")
    metric_path = run_root / "anchors" / lane / "metrics" / f"e{epoch:03d}.json"
    for path in (checkpoint, sidecar_path, metric_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    sidecar = read_json(sidecar_path)
    if sidecar.get("schema") != "final-unsb-local-route1-full-state-v1":
        raise RuntimeError("checkpoint sidecar schema mismatch")
    if sidecar.get("probe_id") != lane:
        raise RuntimeError("checkpoint lane mismatch")
    if int(sidecar.get("step", -1)) != epoch * 150:
        raise RuntimeError("checkpoint update mismatch")
    if int(sidecar.get("physical_epoch_completed", -1)) != epoch:
        raise RuntimeError("checkpoint data-epoch mismatch")
    metadata = sidecar.get("metadata", {})
    expected_metadata = {
        "probe_id": lane,
        "seed": 2026,
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"checkpoint metadata mismatch for {key}")
    checkpoint_hash = file_sha256(checkpoint)
    if checkpoint_hash != sidecar.get("full_state_sha256"):
        raise RuntimeError("checkpoint file hash differs from sidecar")
    scientific_hash = scientific_state_sha256(checkpoint, training_repo)
    if scientific_hash != sidecar.get("scientific_state_sha256"):
        raise RuntimeError("checkpoint scientific-state hash differs from sidecar")
    metric = read_json(metric_path)
    per_domain = validate_metric(
        metric, lane=lane, epoch=epoch, require_lpips=require_lpips,
    )
    return {
        "schema": "final-unsb-route1-milestone-verification-v1",
        "recorded": now(),
        "status": "ACCEPTED_MILESTONE",
        "identity": {
            "probe_id": lane,
            "seed": 2026,
            "data_epoch": epoch,
            "updates": epoch * 150,
            "git_commit": EXPECTED_TRAINING_COMMIT,
            "protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest_sha256": EXPECTED_MANIFEST,
            "evaluation_input_sha256": EXPECTED_CRN,
        },
        "checkpoint": {
            "path": str(checkpoint),
            "file_sha256": checkpoint_hash,
            "scientific_state_sha256": scientific_hash,
            "sidecar_sha256": file_sha256(sidecar_path),
        },
        "discovery70_metric": {
            "path": str(metric_path),
            "file_sha256": file_sha256(metric_path),
            "images": len(metric["images"]),
            "macro_psnr": metric["macro_psnr"],
            "macro_ssim": metric["macro_ssim"],
            "macro_lpips": metric.get("macro_lpips"),
            "per_domain": per_domain,
        },
        "integrity": {
            "checkpoint_file_hash_matches_sidecar": True,
            "scientific_state_hash_matches_sidecar": True,
            "metric_protocol_matches": True,
            "evaluation_bundle_matches_frozen_crn": True,
            "paired_metric_used_for_training_control": False,
            "confirmation20_opened": False,
        },
        "claim_boundary": "This accepts one completed milestone only; it does not rank algorithms or control training.",
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-root", type=Path, required=True)
    value.add_argument("--training-repo", type=Path, required=True)
    value.add_argument("--lane", required=True)
    value.add_argument("--epoch", type=int, required=True)
    value.add_argument("--require-lpips", action="store_true")
    value.add_argument("--output", type=Path)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = verify(
        run_root=args.run_root, training_repo=args.training_repo,
        lane=args.lane, epoch=args.epoch, require_lpips=bool(args.require_lpips),
    )
    if args.output:
        atomic_json(args.output.resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
