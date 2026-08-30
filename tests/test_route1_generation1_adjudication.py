from __future__ import annotations

import math
import json
from pathlib import Path
from types import SimpleNamespace

from research.local_route1 import generation1_adjudication as adjudication
from research.local_route1.generation1_adjudication import (
    NEGATIVE_STATUS,
    POSITIVE_STATUS,
    adjudicate_generation1,
    trajectory_rank_key,
)


def _trajectory(candidate_id: str, *, late: float, final: float, cost_guard: float = 0.1):
    return {
        "candidate_id": candidate_id,
        "late_three_mean_macro_psnr_delta": late,
        "e200_macro_psnr_delta": final,
        "late_points_with_four_of_six_positive_domains": 2,
        "late_average_worst_domain_delta": -0.5,
        "candidate_best_to_terminal_three_point_rolling_drawdown": cost_guard,
        "late_mean_macro_ssim_delta": 0.01,
        "late_mean_macro_lpips_delta": -0.01,
    }


def test_generation1_ranking_prioritizes_late_three_before_terminal_or_cost():
    sustained = _trajectory("sustained", late=0.2, final=0.1)
    terminal = _trajectory("terminal", late=0.1, final=0.8)
    assert trajectory_rank_key(sustained, measured_epoch_seconds=99.0) < trajectory_rank_key(
        terminal, measured_epoch_seconds=1.0
    )


def test_generation1_ranking_uses_terminal_then_registered_guards():
    stronger_terminal = _trajectory("a", late=0.2, final=0.2)
    weaker_terminal = _trajectory("b", late=0.2, final=0.1)
    assert trajectory_rank_key(stronger_terminal) < trajectory_rank_key(weaker_terminal)


def test_generation1_ranking_treats_missing_scientific_values_as_worst():
    complete = _trajectory("complete", late=0.0, final=0.0)
    missing = {"candidate_id": "missing"}
    assert trajectory_rank_key(complete) < trajectory_rank_key(
        missing, measured_epoch_seconds=math.inf
    )


def _registration(root: Path, candidate_id: str):
    card = root / "cards" / f"{candidate_id}.json"
    implementation = root / "implementations" / f"{candidate_id}.json"
    return SimpleNamespace(
        algorithm_fingerprint=f"algorithm-{candidate_id}",
        candidate_fingerprint=f"execution-{candidate_id}",
        card_path=card,
        implementation_path=implementation,
    )


def _complete_payload(candidate_id: str, *, late: float, status: str):
    return {
        "schema": adjudication.TRAJECTORY_SCHEMA,
        "candidate_id": candidate_id,
        "algorithm_fingerprint": f"algorithm-{candidate_id}",
        "candidate_fingerprint": f"execution-{candidate_id}",
        "status": status,
        "trajectory": [{"epoch": 150}, {"epoch": 175}, {"epoch": 200}],
        **_trajectory(candidate_id, late=late, final=late),
        "paired_metrics_used_for_training_or_gate": False,
        "confirmation20_opened": False,
    }


def test_generation1_adjudication_waits_for_every_e200_trajectory(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adjudication, "load_candidate_registration",
        lambda root, candidate_id, require_gate: _registration(root, candidate_id),
    )
    result = adjudicate_generation1(tmp_path, ("first", "second"))
    assert result["status"] == "WAITING_FOR_ALL_MATCHED_E200_TRAJECTORIES"
    assert result["ranking_performed"] is False
    assert {row["candidate_id"] for row in result["pending"]} == {"first", "second"}


def test_generation1_adjudication_freezes_only_ranked_positive_winner(tmp_path, monkeypatch):
    monkeypatch.setattr(
        adjudication, "load_candidate_registration",
        lambda root, candidate_id, require_gate: _registration(root, candidate_id),
    )
    frozen = []
    monkeypatch.setattr(
        adjudication, "freeze_for_seed_validation",
        lambda root, candidate_id: frozen.append(candidate_id) or {"candidate_id": candidate_id},
    )
    for candidate_id, late, status in (
        ("positive", 0.2, POSITIVE_STATUS),
        ("negative", 0.4, NEGATIVE_STATUS),
    ):
        root = tmp_path / "candidates" / candidate_id
        root.mkdir(parents=True)
        (root / "CANDIDATE_TRAJECTORY.json").write_text(
            json.dumps(_complete_payload(candidate_id, late=late, status=status)),
            encoding="utf-8",
        )
    result = adjudicate_generation1(
        tmp_path, ("positive", "negative"), freeze_winner=True,
    )
    assert result["status"] == "SEED2026_WINNER_READY_FOR_FROZEN_SEED2027"
    assert result["selected_candidate_id"] == "positive"
    assert result["ranking"][0]["candidate_id"] == "negative"
    assert frozen == ["positive"]
