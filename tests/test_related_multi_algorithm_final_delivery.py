from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.local_route1 import related_multi_algorithm_final_delivery as delivery
from research.local_route1.protocol import file_sha256


def _write(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _row(candidate_id: str, late: float, e200: float) -> dict:
    return {
        "candidate_id": candidate_id,
        "source_role": "test",
        "classification": "strict_sustained",
        "trajectory_status": "LONG_HORIZON_POSITIVE_CANDIDATE",
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "training_git_commit": "a" * 40,
        "ranking_fields": {
            "late_three_mean_macro_psnr_delta": late,
            "e200_macro_psnr_delta": e200,
            "late_points_with_four_of_six_positive_domains": 3,
            "late_average_worst_domain_delta": -0.2,
            "candidate_best_to_terminal_three_point_rolling_drawdown": 0.1,
            "late_mean_macro_ssim_delta": 0.01,
            "late_mean_macro_lpips_delta": -0.01,
        },
        "receipt_path": "unused",
        "receipt_sha256": "unused",
        "trajectory_path": "unused",
        "trajectory_sha256": "unused",
        "median_epoch_wall_seconds": 10.0,
    }


def _related_row(candidate_id: str, late: float, e200: float) -> dict:
    trajectory = {
        "candidate_id": candidate_id,
        "status": "LONG_HORIZON_POSITIVE_CANDIDATE",
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }
    return {
        "candidate_id": candidate_id,
        "classification": "strict_sustained_local_signal",
        "strict_checks": {"late_three_positive": True},
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"candidate-{candidate_id}",
        "training_git_commit": "b" * 40,
        "base_e0_scientific_state_sha256": "e0",
        "base_protocol_fingerprint": "protocol",
        "manifest_sha256": "manifest",
        "late_three_mean_macro_psnr_delta": late,
        "e200_macro_psnr_delta": e200,
        "late_points_with_four_of_six_positive_domains": 3,
        "late_average_worst_domain_delta": -0.2,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
        "rolling_drawdown_db": 0.1,
        "median_epoch_wall_seconds": 10.0,
        "terminal_receipt_path": "unused",
        "terminal_receipt_sha256": "unused",
        "trajectory_path": "unused",
        "trajectory_sha256": "unused",
        "trajectory_snapshot": trajectory,
    }


def test_final_delivery_keeps_multiple_viable_algorithms(monkeypatch, tmp_path: Path):
    complete_pointer_path = _write(tmp_path / "complete-pointer.json", {"ok": True})
    base_path = _write(tmp_path / "base.json", {"ok": True})
    related_paths = {
        key: _write(tmp_path / f"{key}.json", {"ok": True})
        for key in ("remote4090", "remote5090", "combined")
    }
    base = {
        "same_host_authority": {
            "base_e0_scientific_state_sha256": "e0",
            "base_protocol_fingerprint": "protocol",
            "manifest_sha256": "manifest",
        },
        "ranking": [
            _row("BASE", 0.4, 0.3),
            _row(delivery.PROPOSAL, 0.45, 0.35),
        ],
    }
    host4090 = {"ranking": [
        _related_row(delivery.HPCGR, 0.8, 0.5),
        _related_row(delivery.HJCGR, 0.6, 0.4),
    ]}
    host5090 = {"ranking": []}
    combined = {
        "status": "MULTIPLE_VIABLE_ALGORITHMS",
        "algorithms": [],
        "cross_host_deltas_merged": False,
        "cross_runtime_is_not_cross_seed": True,
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }

    monkeypatch.setattr(
        delivery, "_complete_delivery",
        lambda _root: ({"status": "complete"}, complete_pointer_path),
    )
    monkeypatch.setattr(delivery, "_base_frontier", lambda _root: (base, base_path))
    monkeypatch.setattr(
        delivery, "_related_inputs",
        lambda _root: (host4090, host5090, combined, related_paths),
    )
    monkeypatch.setattr(
        delivery, "_candidate_domain_trajectory",
        lambda _root, candidate_id: {"candidate_id": candidate_id},
    )

    sources = {}
    _write(tmp_path / "evidence" / "ANCHOR_TRAJECTORIES.json", {
        "schema": "local-route1-anchor-summary-v1",
        "summaries": [
            {
                "probe_id": "hnek",
                "complete_e200": True,
                "late_three_mean_macro_psnr_delta": 0.5,
                "trajectory": [{"epoch": 200, "macro_psnr_delta": 0.25}],
            },
            {
                "probe_id": "hj",
                "complete_e200": True,
                "late_three_mean_macro_psnr_delta": 0.2,
                "trajectory": [{"epoch": 200, "macro_psnr_delta": 0.1}],
            },
        ],
    })

    for candidate_id in (
        "BASE", delivery.PROPOSAL, delivery.HPCGR, delivery.HJCGR,
    ):
        receipt_path = _write(
            tmp_path / "operations" / "terminal_receipts" / f"{candidate_id}.json",
            {"candidate_id": candidate_id},
        )
        trajectory_path = _write(
            tmp_path / "candidates" / candidate_id / "CANDIDATE_TRAJECTORY.json",
            {"candidate_id": candidate_id},
        )
        card_path = _write(
            tmp_path / "derive" / "cards" / f"{candidate_id}.json",
            {
                "candidate_id": candidate_id,
                "name": candidate_id,
                "formula": f"formula-{candidate_id}",
            },
        )
        implementation_path = _write(
            tmp_path / "derive" / "implementations" / f"{candidate_id}.json",
            {"candidate_id": candidate_id, "model": candidate_id, "method": {}},
        )
        sources[candidate_id] = (
            {
                "candidate_id": candidate_id,
                "algorithm_fingerprint": f"algorithm-{candidate_id}",
                "candidate_fingerprint": f"candidate-{candidate_id}",
                "training_git_commit": "c" * 40,
            },
            receipt_path,
            {"candidate_id": candidate_id},
            trajectory_path,
            {"candidate_id": candidate_id, "name": candidate_id},
            card_path,
            {"candidate_id": candidate_id, "model": candidate_id, "method": {}},
        )
        assert file_sha256(implementation_path)

    monkeypatch.setattr(
        delivery, "_selected_source",
        lambda _root, row: sources[row["candidate_id"]],
    )

    pointer = delivery.materialize_related_multi_algorithm_final_delivery(tmp_path)
    assert pointer["status"] == "RELATED_MULTI_ALGORITHM_FINAL_DELIVERY_COMPLETE"
    assert pointer["action_priority_candidate_id"] == delivery.HPCGR
    assert pointer["strict_viable_candidate_count"] == 4
    algorithm_set = json.loads(
        (tmp_path / delivery.FINAL_SUBDIR / "ALGORITHM_SET.json").read_text()
    )
    assert algorithm_set["status"] == "MULTIPLE_VIABLE_ALGORITHMS"
    assert algorithm_set["algorithm_discovery_collapsed_to_single_candidate"] is False
    assert set(algorithm_set["strict_viable_candidate_ids"]) == {
        "BASE", delivery.PROPOSAL, delivery.HPCGR, delivery.HJCGR,
    }
    assert algorithm_set["action_priority_is_not_scientific_exclusivity"] is True
    decomposition = algorithm_set["mechanism_gain_source_decomposition"]
    by_id = {row["candidate_id"]: row for row in decomposition["members"]}
    assert by_id[delivery.PROPOSAL][
        "matched_compositional_increment_over_parent"
    ]["late_three_macro_psnr_delta"] == 0.45
    assert by_id[delivery.HPCGR][
        "matched_compositional_increment_over_parent"
    ]["late_three_macro_psnr_delta"] == pytest.approx(0.3)
    assert by_id[delivery.HJCGR][
        "matched_compositional_increment_over_parent"
    ]["e200_macro_psnr_delta"] == pytest.approx(0.3)
    assert decomposition["shared_estimator_positive_increment_count"] == 3
    assert decomposition[
        "matched_increment_is_not_additive_causal_attribution"
    ] is True
    assert delivery.materialize_related_multi_algorithm_final_delivery(tmp_path) == pointer
