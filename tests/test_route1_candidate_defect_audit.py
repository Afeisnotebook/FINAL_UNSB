import json

import pytest
import torch

from research.local_route1.candidate_defect_audit import (
    CROSS_VERSION_NEGATIVE_STATUS,
    CROSS_VERSION_FINAL_OUTCOME_SCHEMA,
    GENERATION1_NEGATIVE_STATUS,
    _GradientTraceAccumulator,
    _ReplicaGeometryAccumulator,
    _adam_update_space_scales,
    _mean_gradients,
    adjudicate_cross_version_revision_need,
    adjudicate_revision_need,
)
from research.local_route1.candidates import DEFECT_ADJUDICATION_SCHEMA
from research.local_route1.protocol import file_sha256


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_gradient_trace_accumulator_and_replica_mean_are_exact():
    rows = [
        (torch.tensor([1.0, 3.0]),),
        (torch.tensor([3.0, 1.0]),),
        (torch.tensor([2.0, 4.0]),),
        (torch.tensor([4.0, 2.0]),),
    ]
    native = _GradientTraceAccumulator()
    replicated = _GradientTraceAccumulator()
    for row in rows:
        native.add(row)
    replicated.add(_mean_gradients(rows[0], rows[1]))
    replicated.add(_mean_gradients(rows[2], rows[3]))
    assert native.trace_variance() > 0.0
    assert replicated.trace_variance() == pytest.approx(1.0)


def test_replica_geometry_exactly_separates_parallel_and_orthogonal_noise():
    orthogonal = _ReplicaGeometryAccumulator()
    orthogonal.add(
        (torch.tensor([3.0, 1.0]),),
        (torch.tensor([1.0, 3.0]),),
    )
    orthogonal_summary = orthogonal.summary()
    assert orthogonal_summary["parallel_fraction_of_difference"] == pytest.approx(0.0)
    assert orthogonal_summary["orthogonal_fraction_of_difference"] == pytest.approx(1.0)

    parallel = _ReplicaGeometryAccumulator()
    parallel.add(
        (torch.tensor([3.0, 3.0]),),
        (torch.tensor([1.0, 1.0]),),
        scales=(torch.tensor([1.0, 2.0]),),
    )
    parallel_summary = parallel.summary()
    assert parallel_summary["parallel_fraction_of_difference"] == pytest.approx(1.0)
    assert parallel_summary["orthogonal_fraction_of_difference"] == pytest.approx(0.0)


def test_adam_update_space_scale_uses_the_frozen_pre_step_second_moment():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    optimizer = torch.optim.Adam([parameter], eps=1e-8)
    optimizer.state[parameter]["exp_avg_sq"] = torch.tensor([4.0, 9.0])
    scale, = _adam_update_space_scales((parameter,), (optimizer,))
    assert scale.tolist() == pytest.approx([
        1.0 / (2.0 + 1e-8), 1.0 / (3.0 + 1e-8),
    ])


def test_revision_need_stops_or_routes_only_from_target_blind_e200_evidence(tmp_path):
    ids = ["G1-FIRST", "G1-SECOND"]
    generation1_path = tmp_path / "operations" / "GENERATION1_E200_ADJUDICATION.json"
    _write(generation1_path, {
        "status": GENERATION1_NEGATIVE_STATUS,
        "selected_candidate_id": ids[0],
        "ranking": [
            {"rank": 1, "candidate_id": ids[0]},
            {"rank": 2, "candidate_id": ids[1]},
        ],
    })
    for candidate_id in ids:
        _write(tmp_path / "candidates" / candidate_id / "TARGET_BLIND_DEFECT_ADJUDICATION.json", {
            "schema": DEFECT_ADJUDICATION_SCHEMA,
            "candidate_id": candidate_id,
            "data_epoch_adjudicated": 200,
            "target_blind_defect_reduced": False,
            "long_horizon_benefit_reversed": True,
            "revision_applicable": False,
            "paired_target_used_to_compute_defect": False,
            "paired_metric_used_for_training_or_control": False,
            "confirmation20_opened": False,
        })
    result = adjudicate_revision_need(tmp_path, ids)
    assert result["status"] == "NO_REVISION_APPLICABLE_FINAL_FALLBACK"
    assert result["selected_candidate_id"] == ids[0]
    assert result["source_generation1_adjudication_sha256"] == file_sha256(generation1_path)
    first_path = tmp_path / "candidates" / ids[0] / "TARGET_BLIND_DEFECT_ADJUDICATION.json"
    first = json.loads(first_path.read_text())
    first["target_blind_defect_reduced"] = True
    first["revision_applicable"] = True
    _write(first_path, first)
    result = adjudicate_revision_need(tmp_path, ids)
    assert result["status"] == "REVISION_DERIVATION_REQUIRED"
    assert result["selected_candidate_id"] == ids[0]
    assert result["automatic_revision_started"] is False


def test_cross_version_revision_routing_accepts_only_source_bound_defects(tmp_path):
    ids = ["G1-FIRST", "G1-SECOND"]
    cross_path = tmp_path / "operations" / "CROSS_VERSION_E200_ADJUDICATION.json"
    ranking = [
        {
            "rank": rank,
            "candidate_id": candidate_id,
            "algorithm_fingerprint": f"algorithm-{candidate_id}",
            "trajectory_sha256": f"trajectory-{candidate_id}",
        }
        for rank, candidate_id in enumerate(ids, start=1)
    ]
    _write(cross_path, {
        "status": CROSS_VERSION_NEGATIVE_STATUS,
        "selected_candidate_id": ids[0],
        "ranking": ranking,
    })
    for row in ranking:
        candidate_id = row["candidate_id"]
        _write(
            tmp_path / "candidates" / candidate_id
            / "TARGET_BLIND_DEFECT_ADJUDICATION.json",
            {
                "schema": DEFECT_ADJUDICATION_SCHEMA,
                "candidate_id": candidate_id,
                "algorithm_fingerprint": row["algorithm_fingerprint"],
                "data_epoch_adjudicated": 200,
                "source_candidate_trajectory_sha256": row["trajectory_sha256"],
                "source_checkpoint_unchanged_after_audit": True,
                "target_blind_defect_reduced": candidate_id == ids[1],
                "long_horizon_benefit_reversed": True,
                "revision_applicable": candidate_id == ids[1],
                "paired_target_used_to_compute_defect": False,
                "paired_metric_used_for_training_or_control": False,
                "confirmation20_opened": False,
            },
        )
    result = adjudicate_cross_version_revision_need(tmp_path, ids)
    assert result["schema"] == CROSS_VERSION_FINAL_OUTCOME_SCHEMA
    assert result["status"] == "REVISION_DERIVATION_REQUIRED"
    assert result["selected_candidate_id"] == ids[1]
    assert result["source_cross_version_adjudication_sha256"] == file_sha256(cross_path)
    assert result["automatic_revision_started"] is False

    defect_path = (
        tmp_path / "candidates" / ids[1]
        / "TARGET_BLIND_DEFECT_ADJUDICATION.json"
    )
    defect = json.loads(defect_path.read_text())
    defect["algorithm_fingerprint"] = "changed"
    _write(defect_path, defect)
    with pytest.raises(RuntimeError, match="algorithm_fingerprint"):
        adjudicate_cross_version_revision_need(tmp_path, ids)
