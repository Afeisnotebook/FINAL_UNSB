from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import torch

from production.metrics import METRIC_SEMANTICS
from research.local_route1.runtime import full_state_hash, seed_everything
from research.paper_aio.evaluate import (
    aggregate_metric_rows,
    replicate_stochasticity,
    validate_evaluation_result,
)
from research.paper_aio.protocol import (
    FULL_STATE_SCHEMA,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    REQUIRED_PAPER_TABLE,
    file_sha256,
    lane_spec,
    object_sha256,
    protocol_fingerprint,
)
from research.paper_aio.gates import environment_record
from research.paper_aio.unified import (
    INPUT_RECEIPT_SCHEMA,
    REQUIRED_FIRST_WAVE,
    UNIFIED_EPOCHS,
    UNIFIED_RECEIPT_SCHEMA,
    _expected_evaluation,
    candidate_spec_from_portable_authority,
    export_checkpoint_receipt,
    lock_unified_evaluation_cohort,
)


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path


DOMAINS = (
    "FoggyCityscapes", "LowLightTrafficData", "RainCityscapes",
    "RainDS-syn", "RSCityscapes", "SnowTrafficData",
)


def _complete_metric(*, lane_id: str, family: str, epoch: int) -> dict:
    expected = _expected_evaluation(epoch, family)
    images = []
    for domain in DOMAINS:
        for order in range(expected["count_per_domain"]):
            for replicate in range(expected["replicates"]):
                digest = hashlib.sha256(
                    f"{epoch}|{domain}|{order}|{replicate}".encode()
                ).hexdigest()
                for nfe in expected["nfe_values"]:
                    images.append({
                        "domain": domain,
                        "stem": f"{domain}-{order:03d}",
                        "order": order,
                        "replicate": replicate,
                        "nfe": nfe,
                        "psnr": 20.0 + nfe / 100.0,
                        "ssim": 0.7,
                        "lpips": 0.2,
                        "crn_bundle_sha256": digest,
                    })
    cells = {}
    for nfe in expected["nfe_values"]:
        rows = [row for row in images if row["nfe"] == nfe]
        aggregate = aggregate_metric_rows(rows)
        replicate_cells = [
            {
                "replicate": replicate,
                **aggregate_metric_rows([
                    row for row in rows if row["replicate"] == replicate
                ]),
            }
            for replicate in range(expected["replicates"])
        ]
        cells[str(nfe)] = {
            **aggregate,
            "replicate_cells": replicate_cells,
            "stochasticity": replicate_stochasticity(replicate_cells),
        }
    primary_nfe = 5 if family == "unsb" else 1
    primary = cells[str(primary_nfe)]
    return {
        "schema": "final-unsb-paper-aio-evaluation-v1",
        "lane_id": lane_id,
        "split": "discovery",
        **expected,
        "primary_nfe": primary_nfe,
        "primary_replicate": 0,
        "top_level_replicate_aggregation": "mean_over_fixed_replicates",
        "metric_semantics": METRIC_SEMANTICS,
        "metric_semantics_sha256": object_sha256(METRIC_SEMANTICS),
        "protocol_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "evaluation_input_sha256": f"input-{epoch}",
        **{
            key: primary[key]
            for key in ("macro_psnr", "macro_ssim", "macro_lpips")
        },
        "domains": primary["domains"],
        "replicate_cells": primary["replicate_cells"],
        "stochasticity": primary["stochasticity"],
        "nfe_cells": cells,
        "images": images,
        "lpips_requested": True,
        "lpips_available": True,
        "confirmation20_opened": False,
    }


def _complete_input_metric() -> dict:
    images = []
    for domain in DOMAINS:
        for order in range(80):
            images.append({
                "domain": domain,
                "stem": f"{domain}-{order:03d}",
                "order": order,
                "replicate": 0,
                "nfe": 0,
                "psnr": 12.0,
                "ssim": 0.4,
                "lpips": 0.5,
                "crn_bundle_sha256": None,
            })
    aggregate = aggregate_metric_rows(images)
    replicate_cell = {"replicate": 0, **aggregate}
    stochasticity = replicate_stochasticity([replicate_cell])
    return {
        "schema": "final-unsb-paper-aio-evaluation-v1",
        "lane_id": "input",
        "split": "discovery",
        "count_per_domain": 80,
        "replicates": 1,
        "nfe_values": [0],
        "primary_nfe": 0,
        "primary_replicate": 0,
        "top_level_replicate_aggregation": "mean_over_fixed_replicates",
        "metric_semantics": METRIC_SEMANTICS,
        "metric_semantics_sha256": object_sha256(METRIC_SEMANTICS),
        "protocol_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "evaluation_input_sha256": "input-200",
        **{
            key: aggregate[key]
            for key in ("macro_psnr", "macro_ssim", "macro_lpips")
        },
        "domains": aggregate["domains"],
        "replicate_cells": [replicate_cell],
        "stochasticity": stochasticity,
        "nfe_cells": {"0": {
            **aggregate,
            "replicate_cells": [replicate_cell],
            "stochasticity": stochasticity,
        }},
        "images": images,
        "lpips_requested": True,
        "lpips_available": True,
        "evaluation_only_reference": True,
        "confirmation20_opened": False,
    }


def test_complete_metric_payload_is_recomputed_and_validated() -> None:
    metric = _complete_metric(lane_id="plain", family="unsb", epoch=200)
    validated = validate_evaluation_result(
        metric, lane_id="plain", family="unsb", count_per_domain=80,
        replicates=5, nfe_values=[1, 2, 3, 4, 5], include_lpips=True,
    )
    assert validated["image_cells"] == 6 * 80 * 5 * 5
    assert len(validated["sample_identity"]) == 6 * 80
    assert len(validated["crn_identity"]) == 6 * 80 * 5


def test_metric_payload_rejects_aggregate_or_crn_tampering() -> None:
    metric = _complete_metric(lane_id="plain", family="unsb", epoch=200)
    metric["macro_psnr"] += 0.01
    with pytest.raises(RuntimeError, match="primary.macro_psnr differs"):
        validate_evaluation_result(
            metric, lane_id="plain", family="unsb", count_per_domain=80,
            replicates=5, nfe_values=[1, 2, 3, 4, 5], include_lpips=True,
        )

    metric = _complete_metric(lane_id="plain", family="unsb", epoch=200)
    metric["images"][1]["crn_bundle_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="changes CRN bundle across NFE"):
        validate_evaluation_result(
            metric, lane_id="plain", family="unsb", count_per_domain=80,
            replicates=5, nfe_values=[1, 2, 3, 4, 5], include_lpips=True,
        )


def test_portable_candidate_authority_binds_metric_free_source_identity(
    tmp_path: Path,
) -> None:
    authority_path = (
        Path(__file__).resolve().parents[1]
        / "configs" / "PAPER_STCGR_EVALUATION_AUTHORITY.json"
    )
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    spec, digest = candidate_spec_from_portable_authority(
        authority_path=authority_path,
        candidate_id=authority["candidate_id"],
        exported_lane=authority["lane"],
        training_git_commit=authority["training_identity"]["git_commit"],
        training_protocol_fingerprint=(
            authority["training_identity"]["protocol_fingerprint"]
        ),
    )
    assert spec.to_dict() == authority["lane"]
    assert digest == file_sha256(authority_path)

    contaminated = dict(authority)
    contaminated["macro_psnr"] = 1.0
    contaminated_path = _write(tmp_path / "contaminated.json", contaminated)
    with pytest.raises(RuntimeError, match="authority is invalid"):
        candidate_spec_from_portable_authority(
            authority_path=contaminated_path,
            candidate_id=authority["candidate_id"],
            exported_lane=authority["lane"],
            training_git_commit=authority["training_identity"]["git_commit"],
            training_protocol_fingerprint=(
                authority["training_identity"]["protocol_fingerprint"]
            ),
        )


def test_checkpoint_export_is_source_and_scientific_state_bound(tmp_path: Path) -> None:
    checkpoint = tmp_path / "e200.pt"
    payload = {
        "schema": FULL_STATE_SCHEMA,
        "lane": lane_spec("plain").to_dict(),
        "step": 1_710_600,
        "target_steps": 1_710_600,
        "model": {"networks": {"G": {"weight": torch.tensor([1.0])}}},
        "rng": {"python": (3, (), None)},
        "samplers": {"primary": {}, "secondary": {}},
        "metadata": {
            "git_commit": "1" * 40,
            "protocol_fingerprint": "2" * 64,
            "manifest_sha256": "3" * 64,
            "paired_controller_access": False,
            "confirmation20_opened": False,
        },
    }
    torch.save(payload, checkpoint)
    sidecar = _write(tmp_path / "e200.pt.json", {
        "schema": FULL_STATE_SCHEMA,
        "lane_id": "plain",
        "step": 1_710_600,
        "physical_epoch_completed": 200,
        "full_state_sha256": file_sha256(checkpoint),
        "scientific_state_sha256": full_state_hash(payload),
    })
    destination = tmp_path / "export.json"
    result = export_checkpoint_receipt(
        checkpoint=checkpoint, sidecar=sidecar, lane_id="plain", epoch=200,
        host_label="4090A", destination=destination,
    )
    assert result["status"] == "ACCEPTED_SOURCE_BOUND_CHECKPOINT_EXPORT"
    assert result["checkpoint_sha256"] == file_sha256(checkpoint)
    assert result["confirmation20_opened"] is False
    payload["step"] = 1
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="file hash mismatch"):
        export_checkpoint_receipt(
            checkpoint=checkpoint, sidecar=sidecar, lane_id="plain", epoch=200,
            host_label="4090A", destination=destination,
        )


def test_unified_cohort_requires_all_first_wave_lanes_and_fixed_protocol(
    tmp_path: Path,
) -> None:
    seed_everything(2026)
    environment = environment_record()
    evaluator = protocol_fingerprint()
    input_metric = _write(
        tmp_path / "lanes" / "input" / "metrics" / "e200.json",
        _complete_input_metric(),
    )
    _write(
        tmp_path / "gates" / "UNIFIED_EVALUATION_input_e200.json",
        {
            "schema": INPUT_RECEIPT_SCHEMA,
            "status": "PASS_UNIFIED_INPUT_EVALUATION",
            "lane_id": "input",
            "epoch": 200,
            "metric": str(input_metric.resolve()),
            "metric_sha256": file_sha256(input_metric),
            "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
            "unified_evaluator_protocol_fingerprint": evaluator,
            "unified_environment": environment,
            "evaluation_only_reference": True,
            "training_checkpoint_read_only": True,
            "paired_metric_control": False,
            "cross_host_training_delta_merged": False,
            "confirmation20_opened": False,
        },
    )
    for lane_id in REQUIRED_FIRST_WAVE:
        family = lane_spec(lane_id).family
        source_host = (
            "5090A" if lane_id == "plain" else
            "5090C" if lane_id == "proposal" else
            "5090B"
        )
        training_fingerprint = (
            "e5704e445a51dd9c5c12369c94df01cf9532364a71c806b9914ef3963994b07b"
            if lane_id in ("plain", "proposal") else "external-training"
        )
        training_manifest = (
            "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
        )
        for epoch in UNIFIED_EPOCHS:
            source_hash = hashlib.sha256(
                f"{lane_id}|{epoch}|checkpoint".encode()
            ).hexdigest()
            metric_value = _complete_metric(
                lane_id=lane_id, family=family, epoch=epoch,
            )
            metric_value.update({
                "epoch": epoch,
                "source_host_label": source_host,
                "training_protocol_fingerprint": training_fingerprint,
                "manifest_sha256": training_manifest,
                "source_checkpoint_sha256": source_hash,
                "source_checkpoint_sha256_after_evaluation": source_hash,
                "training_checkpoint_read_only": True,
                "training_checkpoint_read_only_verified_by_rehash": True,
                "cross_host_training_delta_merged": False,
            })
            metric = _write(
                tmp_path / "lanes" / lane_id / "metrics" / f"e{epoch:03d}.json",
                metric_value,
            )
            _write(
                tmp_path / "gates" / f"UNIFIED_EVALUATION_{lane_id}_e{epoch:03d}.json",
                {
                    "schema": UNIFIED_RECEIPT_SCHEMA,
                    "status": "PASS_UNIFIED_READ_ONLY_EVALUATION",
                    "lane_id": lane_id,
                    "epoch": epoch,
                    "source_host_label": source_host,
                    "training_protocol_fingerprint": training_fingerprint,
                    "manifest_sha256": training_manifest,
                    "source_checkpoint_sha256": source_hash,
                    "source_checkpoint_sha256_after_evaluation": source_hash,
                    "metric": str(metric.resolve()),
                    "metric_sha256": file_sha256(metric),
                    "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
                    "unified_evaluator_protocol_fingerprint": evaluator,
                    "unified_environment": environment,
                    "training_checkpoint_read_only": True,
                    "training_checkpoint_read_only_verified_by_rehash": True,
                    "paired_metric_control": False,
                    "cross_host_training_delta_merged": False,
                    "confirmation20_opened": False,
                },
            )
    result = lock_unified_evaluation_cohort(tmp_path)
    assert result["status"] == "PASS_FIRST_WAVE_UNIFIED_EVALUATION_COHORT"
    assert result["required_lanes"] == list(REQUIRED_PAPER_TABLE)
    assert result["training_hosts_remain_separate"] is True
    assert result["proposal_plain_runtime_relation"]["status"] == (
        "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"
    )
    bad = tmp_path / "gates" / "UNIFIED_EVALUATION_cut_e200.json"
    value = json.loads(bad.read_text(encoding="utf-8"))
    value["cross_host_training_delta_merged"] = True
    _write(bad, value)
    with pytest.raises(RuntimeError, match="invalid unified"):
        lock_unified_evaluation_cohort(tmp_path)

    value["cross_host_training_delta_merged"] = False
    cut_metric = Path(value["metric"])
    cut_value = json.loads(cut_metric.read_text(encoding="utf-8"))
    cut_value["images"][0]["crn_bundle_sha256"] = "f" * 64
    _write(cut_metric, cut_value)
    value["metric_sha256"] = file_sha256(cut_metric)
    _write(bad, value)
    with pytest.raises(RuntimeError, match="do not share exact samples/CRN"):
        lock_unified_evaluation_cohort(tmp_path)
