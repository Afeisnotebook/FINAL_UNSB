from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research.local_route1.runtime import full_state_hash, seed_everything
from research.paper_aio.protocol import (
    FULL_STATE_SCHEMA,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    REQUIRED_PAPER_TABLE,
    file_sha256,
    lane_spec,
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
        {
            "count_per_domain": 80,
            "replicates": 1,
            "nfe_values": [0],
            "protocol_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
            "evaluation_only_reference": True,
            "confirmation20_opened": False,
        },
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
            expected = _expected_evaluation(epoch, family)
            metric = _write(
                tmp_path / "lanes" / lane_id / "metrics" / f"e{epoch:03d}.json",
                {
                    **expected,
                    "protocol_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
                    "source_host_label": source_host,
                    "training_protocol_fingerprint": training_fingerprint,
                    "manifest_sha256": training_manifest,
                    "training_checkpoint_read_only": True,
                    "cross_host_training_delta_merged": False,
                    "confirmation20_opened": False,
                },
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
                    "metric": str(metric.resolve()),
                    "metric_sha256": file_sha256(metric),
                    "evaluation_bundle_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
                    "unified_evaluator_protocol_fingerprint": evaluator,
                    "unified_environment": environment,
                    "training_checkpoint_read_only": True,
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
