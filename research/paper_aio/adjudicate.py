"""Build incomplete-safe paper tables without best-checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research.local_route1.runtime import write_json

from .protocol import lane_spec, load_protocol


def _read(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric(output_root: Path, lane: str, epoch: int) -> dict | None:
    path = Path(output_root) / "lanes" / lane / "metrics" / f"e{epoch:03d}.json"
    return _read(path) if path.is_file() else None


def _domain_delta(method: dict, plain: dict) -> dict:
    return {
        domain: {
            "psnr": method["domains"][domain]["psnr"] - plain["domains"][domain]["psnr"],
            "ssim": method["domains"][domain]["ssim"] - plain["domains"][domain]["ssim"],
            "lpips": (
                None if method["domains"][domain]["lpips"] is None or plain["domains"][domain]["lpips"] is None
                else method["domains"][domain]["lpips"] - plain["domains"][domain]["lpips"]
            ),
        }
        for domain in sorted(plain["domains"])
    }


def adjudicate(output_root: Path) -> dict:
    protocol = load_protocol()
    output_root = Path(output_root)
    plain_metrics = {
        epoch: _metric(output_root, "plain", epoch)
        for epoch in protocol["training"]["milestone_epochs"]
    }
    lanes = []
    for row in protocol["lanes"]:
        lane = row["id"]
        terminal = _metric(output_root, lane, 200)
        entry = {
            "lane_id": lane,
            "role": row["role"],
            "backend": row["backend"],
            "status": "COMPLETE_E200" if terminal is not None else "INCOMPLETE",
            "terminal": None if terminal is None else {
                key: terminal[key] for key in ("macro_psnr", "macro_ssim", "macro_lpips")
            },
        }
        if lane_spec(lane, protocol).family == "unsb" and lane != "plain":
            trajectory = []
            for epoch in protocol["training"]["late_epochs"]:
                method = _metric(output_root, lane, epoch)
                plain = plain_metrics.get(epoch)
                if method is None or plain is None:
                    continue
                deltas = _domain_delta(method, plain)
                psnr = [value["psnr"] for value in deltas.values()]
                trajectory.append({
                    "epoch": epoch,
                    "macro_psnr_delta": method["macro_psnr"] - plain["macro_psnr"],
                    "macro_ssim_delta": method["macro_ssim"] - plain["macro_ssim"],
                    "macro_lpips_delta": (
                        None if method["macro_lpips"] is None or plain["macro_lpips"] is None
                        else method["macro_lpips"] - plain["macro_lpips"]
                    ),
                    "positive_domains": sum(value > 0 for value in psnr),
                    "worst_domain_delta": min(psnr),
                    "domain_delta": deltas,
                })
            entry["late_trajectory"] = trajectory
            if len(trajectory) == 3:
                lpips = [row["macro_lpips_delta"] for row in trajectory]
                late_mean = float(np.mean([row["macro_psnr_delta"] for row in trajectory]))
                terminal_delta = float(trajectory[-1]["macro_psnr_delta"])
                accepted = (
                    late_mean > 0 and terminal_delta > 0
                    and sum(row["positive_domains"] >= 4 for row in trajectory) >= 2
                    and float(np.mean([row["worst_domain_delta"] for row in trajectory])) > -1.0
                    and float(np.mean([row["macro_ssim_delta"] for row in trajectory])) >= 0
                    and all(value is not None and value <= 0 for value in lpips)
                )
                entry["scientific_gate"] = {
                    "status": "PASS" if accepted else "FAIL",
                    "late_three_macro_psnr_delta": late_mean,
                    "e200_macro_psnr_delta": terminal_delta,
                    "confirmation20_opened": False,
                }
        lanes.append(entry)
    complete_required = all(
        next(row for row in lanes if row["lane_id"] == lane)["status"] == "COMPLETE_E200"
        for lane in ("plain", "proposal", "cyclegan")
    )
    result = {
        "schema": "final-unsb-paper-results-v1",
        "status": "FIRST_WAVE_COMPLETE" if complete_required else "FIRST_WAVE_INCOMPLETE",
        "primary_epoch": 200,
        "best_checkpoint_selection": False,
        "lanes": lanes,
        "cross_5090_delta_merged": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }
    write_json(output_root / "PAPER_RESULTS.json", result)
    algorithm_set = {
        "schema": "final-unsb-paper-algorithm-set-v1",
        "status": "FROZEN" if result["status"] == "FIRST_WAVE_COMPLETE" else "INCOMPLETE",
        "accepted_algorithms": [
            row["lane_id"] for row in lanes
            if row.get("scientific_gate", {}).get("status") == "PASS"
        ],
        "multiple_algorithms_allowed": True,
        "confirmation20_opened": False,
    }
    write_json(output_root / "ALGORITHM_SET.json", algorithm_set)
    return {"results": result, "algorithm_set": algorithm_set}
