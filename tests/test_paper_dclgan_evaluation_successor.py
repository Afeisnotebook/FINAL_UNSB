import json
from pathlib import Path

import pytest

from operations import paper_aio_dclgan_evaluation_successor as evaluator


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _import_lane(tmp_path: Path) -> tuple[Path, str, str]:
    root = tmp_path / "imports"
    host = "5090B"
    training_commit = "a" * 40
    fingerprint = "b" * 64
    rows = []
    lane = root / "sources" / host / "dclgan"
    for epoch in evaluator.EPOCHS:
        checkpoint = lane / f"e{epoch:03d}.pt"
        sidecar = lane / f"e{epoch:03d}.pt.json"
        receipt = lane / f"e{epoch:03d}.export.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"checkpoint-{epoch}".encode())
        sidecar.write_bytes(f"sidecar-{epoch}".encode())
        receipt_value = {
            "schema": "final-unsb-paper-dclgan-checkpoint-export-v1",
            "status": "ACCEPTED_SOURCE_BOUND_DCLGAN_CHECKPOINT",
            "lane_id": "dclgan",
            "epoch": epoch,
            "updates": epoch * evaluator.STEPS_PER_EPOCH,
            "source_host_label": host,
            "source_checkpoint": f"/remote/e{epoch:03d}.pt",
            "source_sidecar": f"/remote/e{epoch:03d}.pt.json",
            "checkpoint_sha256": evaluator.file_sha256(checkpoint),
            "sidecar_sha256": evaluator.file_sha256(sidecar),
            "scientific_state_sha256": "c" * 64,
            "training_git_commit": training_commit,
            "training_protocol_fingerprint": fingerprint,
            "manifest_sha256": "d" * 64,
            "upstream_commit": "e" * 40,
            "paired_metric_control": False,
            "confirmation20_opened": False,
        }
        _write_json(receipt, receipt_value)
        rows.append({
            "epoch": epoch,
            "export_receipt": str(receipt.resolve()),
            "export_receipt_sha256": evaluator.file_sha256(receipt),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": evaluator.file_sha256(checkpoint),
            "sidecar": str(sidecar.resolve()),
            "sidecar_sha256": evaluator.file_sha256(sidecar),
            "scientific_state_sha256": "c" * 64,
        })
    _write_json(lane / "IMPORT_LANE.json", {
        "schema": evaluator.IMPORT_LANE_SCHEMA,
        "status": "COMPLETE_VERIFIED_IMPORTED_LANE",
        "source_host_label": host,
        "lane_id": "dclgan",
        "epochs": list(evaluator.EPOCHS),
        "imports": rows,
        "checkpoint_copy_performed": True,
        "source_checkpoint_mutation": False,
        "performance_values_read": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    return root, training_commit, fingerprint


def test_dclgan_fixed_evaluation_schedule() -> None:
    assert evaluator.expected_schedule(100) == {
        "count_per_domain": 70,
        "replicates": 1,
        "nfe_values": [1],
        "include_lpips": True,
    }
    assert evaluator.expected_schedule(200)["replicates"] == 5
    assert evaluator.expected_schedule(200)["count_per_domain"] == 80
    with pytest.raises(ValueError):
        evaluator.expected_schedule(199)


def test_validate_imported_dclgan_epoch_set(tmp_path: Path) -> None:
    root, commit, fingerprint = _import_lane(tmp_path)
    rows = evaluator.validate_import_lane(
        root, source_host_label="5090B",
        required_training_commit=commit,
        required_adapter_fingerprint=fingerprint,
    )
    assert [row["epoch"] for row in rows] == list(evaluator.EPOCHS)
    Path(rows[0]["checkpoint"]).write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="files changed"):
        evaluator.validate_import_lane(
            root, source_host_label="5090B",
            required_training_commit=commit,
            required_adapter_fingerprint=fingerprint,
        )


def test_common_reference_requires_environment_sample_and_crn_identity(
    tmp_path: Path,
) -> None:
    images = [{
        "domain": "haze", "stem": "a", "order": 0,
        "replicate": 0, "nfe": 1, "crn_bundle_sha256": "bundle",
    }]
    metric = {
        "evaluation_input_sha256": "input",
        "protocol_fingerprint": "protocol",
        "unified_environment": {"torch": "same"},
        "unified_evaluator_protocol_fingerprint": "evaluator",
        "images": images,
    }
    reference = dict(metric)
    reference["images"] = [{**images[0], "nfe": 5}]
    path = tmp_path / "reference" / "lanes" / "plain" / "metrics" / "e100.json"
    _write_json(path, reference)
    evaluator.validate_common_reference(
        metric, reference_output=tmp_path / "reference", epoch=100,
    )
    metric["unified_environment"] = {"torch": "different"}
    with pytest.raises(RuntimeError, match="identity differs"):
        evaluator.validate_common_reference(
            metric, reference_output=tmp_path / "reference", epoch=100,
        )


def test_crn_identity_rejects_inconsistent_nfe_bundle() -> None:
    metric = {"images": [
        {"domain": "haze", "stem": "a", "order": 0, "replicate": 0,
         "nfe": 1, "crn_bundle_sha256": "one"},
        {"domain": "haze", "stem": "a", "order": 0, "replicate": 0,
         "nfe": 5, "crn_bundle_sha256": "two"},
    ]}
    with pytest.raises(RuntimeError, match="inconsistent CRN"):
        evaluator.crn_identity(metric)


def test_dclgan_result_is_absolute_fixed_e200_not_matched_delta(tmp_path: Path) -> None:
    evaluations = []
    for epoch in evaluator.EPOCHS:
        metric = tmp_path / "metrics" / f"e{epoch:03d}.json"
        _write_json(metric, {
            "macro_psnr": 20.0 + epoch / 1000,
            "macro_ssim": 0.8,
            "macro_lpips": 0.2,
            "stochasticity": {"replicate_count": 5 if epoch == 200 else 1},
            "domains": {"haze": {"psnr": 20.0, "ssim": 0.8, "lpips": 0.2}},
        })
        receipt = tmp_path / "receipts" / f"e{epoch:03d}.json"
        _write_json(receipt, {"epoch": epoch})
        evaluations.append({
            "epoch": epoch,
            "receipt": str(receipt),
            "receipt_sha256": evaluator.file_sha256(receipt),
            "metric": str(metric),
            "metric_sha256": evaluator.file_sha256(metric),
            "checkpoint_sha256": "a" * 64,
        })
    result = evaluator.build_result(tmp_path / "output", evaluations)
    assert result["status"] == "COMPLETE_FIXED_E200_EXTERNAL_BASELINE"
    assert result["comparison_scope"] == "standalone_fixed_protocol_no_matched_delta_claim"
    assert result["terminal"]["epoch"] == 200
    assert result["best_checkpoint_selection"] is False
    assert result["confirmation20_opened"] is False
