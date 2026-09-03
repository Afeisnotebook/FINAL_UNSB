import json
from pathlib import Path

import pytest

from operations.paper_aio_local_terminal_audit_successor import (
    AUDIT_GRADIENT_REPLICATES,
    AUDIT_REPLICATES,
    AUDIT_SCHEMA,
    RECEIPT_SCHEMA as AUDIT_RECEIPT_SCHEMA,
)
from research.paper_aio.protocol import (
    EVALUATION_SCHEMA,
    EXPECTED_MANIFEST_SHA256,
    FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
    file_sha256,
)
from research.paper_aio.terminal_adjudicate import (
    BINDING_SCHEMA,
    PROBES,
    adjudicate_terminal_pathology,
)
from research.paper_aio.unified import UNIFIED_RECEIPT_SCHEMA


DOMAINS = [f"domain-{index}" for index in range(6)]
EPOCHS = (100, 150, 200)


def _write(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def _audit_value(lane_id: str, epoch: int, mechanisms: dict):
    records = []
    for domain in DOMAINS:
        mechanism = mechanisms.get(domain)
        spectral_value = 0.8 if epoch >= 150 and mechanism == "spectral" else 1.0
        amplification = 1.2 if epoch >= 150 and mechanism == "amplification" else 1.0
        steps = []
        for time_index in range(5):
            spectrum = {
                "top_eigenvalue": spectral_value,
                "trace": spectral_value,
                "effective_rank": spectral_value,
                "effective_rank_definition": "participation_ratio_trace_squared_over_frobenius_squared",
                "normalization": "unbiased_sample_covariance_nonzero_spectrum_n_minus_1",
                "sample_count": AUDIT_REPLICATES,
                "flattened_dimension": 49152,
                "eigenvalues": [spectral_value] * AUDIT_REPLICATES,
            }
            steps.append(
                {
                    "time_index": time_index,
                    "increment_spectrum": spectrum,
                    "endpoint_spectrum": spectrum,
                    "endpoint_direction_cosine_to_mean": 0.5,
                    "endpoint_direction_definition": "endpoint_minus_bridge_state",
                    "local_jacobian_top_singular_proxy": amplification,
                    "rollout_jacobian_top_singular_proxy": amplification,
                    "jvp_initial_direction": (
                        "lane_blind_crn_bridge_noise_same_sample_time"
                    ),
                    "perturbation_gain_to_final_output": amplification,
                }
            )
        records.append(
            {
                "domain": domain,
                "stem": "sample",
                "steps": steps,
                "nfe4_to_nfe5_output_rms_mean": 0.1,
                "nfe4_to_nfe5_output_rms_std": 0.01,
            }
        )
    return {
        "schema": AUDIT_SCHEMA,
        "status": "TARGET_BLIND_AUDIT_COMPLETE",
        "lane_id": lane_id,
        "replicates": AUDIT_REPLICATES,
        "samples_per_domain": 1,
        "records": records,
        "gradient_stratum_audit": {
            "status": "TARGET_BLIND_NATIVE_OBJECTIVE_GRADIENT_AUDIT_COMPLETE",
            "cross_time_common_sampler_state": True,
            "cross_time_common_rng_state": True,
            "forward_mode": "training_for_every_replicate",
            "parent_requires_grad_flags_restored": True,
            "strata": [
                {
                    "time_index": time_index,
                    "replicates": AUDIT_GRADIENT_REPLICATES,
                    "gradient_mean_norm": 1.0,
                    "gradient_variance_trace": 1.0,
                    "gradient_variance_normalization": "unbiased_sample_covariance_trace_n_minus_1",
                    "gradient_second_moment": 2.0,
                    "adam_preconditioned_norm_mean": 1.0,
                    "adam_preconditioned_norm_std": 0.1,
                    "adam_preconditioned_norm_std_normalization": "sample_std_n_minus_1",
                    "loss_component_gradient_cosines_first_batch": {
                        "gan_sb": 0.1,
                        "gan_nce": 0.2,
                        "sb_nce": 0.3,
                    },
                }
                for time_index in range(5)
            ],
        },
        "rollout_jacobian_definition": (
            "full numerical frozen NFE5 map from X_t to final endpoint"
        ),
        "parent_state_sha256_before": "state",
        "parent_state_sha256_after": "state",
        "parent_rng_sha256_before": "rng",
        "parent_rng_sha256_after": "rng",
        "paired_labels_attached": False,
        "terminal_pathology_confirmed": False,
        "confirmation20_opened": False,
    }


def _materialize_audits(root: Path, mechanism_by_probe: dict):
    for probe_id, probe in PROBES.items():
        for epoch in EPOCHS:
            cell = root / "probes" / probe_id / f"e{epoch:03d}"
            audit = cell / "TERMINAL_AUDIT.jsonl"
            _write(
                audit,
                _audit_value(
                    probe["lane_id"], epoch, mechanism_by_probe.get(probe_id, {})
                ),
            )
            _write(
                cell / "AUDIT_RECEIPT.json",
                {
                    "schema": AUDIT_RECEIPT_SCHEMA,
                    "status": "PASS_FIXED_TARGET_BLIND_TERMINAL_AUDIT",
                    "probe_id": probe_id,
                    "host_label": probe["source_host_label"],
                    "lane_id": probe["lane_id"],
                    "epoch": epoch,
                    "checkpoint_sha256": "checkpoint",
                    "export_receipt_sha256": "export",
                    "audit_sha256": file_sha256(audit),
                    "parent_state_and_rng_unchanged": True,
                    "performance_values_read": False,
                    "paired_metric_control": False,
                    "confirmation20_opened": False,
                },
            )


def _metric(lane_id: str, epoch: int, decline_domains: set[str]):
    count = 80 if epoch == 200 else 70
    replicates = 5 if epoch == 200 else 1
    images = []
    for domain in DOMAINS:
        for replicate in range(replicates):
            for order in range(count):
                psnr = 20.0
                if epoch == 100:
                    psnr = 19.0
                elif epoch == 200 and domain in decline_domains and order < 70:
                    psnr = 19.9
                elif epoch == 200 and order >= 70:
                    psnr = 100.0
                images.append(
                    {
                        "domain": domain,
                        "stem": f"{order:03d}",
                        "order": order,
                        "replicate": replicate,
                        "nfe": 5,
                        "psnr": psnr,
                        "ssim": 0.5,
                        "lpips": None,
                        "crn_bundle_sha256": f"crn-{domain}-{order}-r{replicate}",
                    }
                )
    return {
        "schema": EVALUATION_SCHEMA,
        "lane_id": lane_id,
        "split": "discovery",
        "epoch": epoch,
        "count_per_domain": count,
        "replicates": replicates,
        "primary_nfe": 5,
        "nfe_values": [5],
        "protocol_fingerprint": FROZEN_EVALUATION_BUNDLE_FINGERPRINT,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_host_label": next(
            row["source_host_label"]
            for row in PROBES.values()
            if row["lane_id"] == lane_id
        ),
        "source_checkpoint_sha256": f"checkpoint-e{epoch}",
        "training_protocol_fingerprint": "training-protocol",
        "unified_evaluator_protocol_fingerprint": "evaluator",
        "unified_environment": {"runtime": "one-test-evaluator"},
        "training_checkpoint_read_only": True,
        "cross_host_training_delta_merged": False,
        "confirmation20_opened": False,
        "images": images,
    }


def _materialize_metrics(root: Path, decline_by_probe: dict) -> Path:
    bindings = {"schema": BINDING_SCHEMA, "probes": {}}
    environment = {"runtime": "one-test-evaluator"}
    for probe_id, probe in PROBES.items():
        bindings["probes"][probe_id] = {}
        for epoch in EPOCHS:
            cell = root / probe_id / f"e{epoch:03d}"
            metric_path = cell / "metric.json"
            receipt_path = cell / "receipt.json"
            _write(
                metric_path,
                _metric(probe["lane_id"], epoch, decline_by_probe.get(probe_id, set())),
            )
            _write(
                receipt_path,
                {
                    "schema": UNIFIED_RECEIPT_SCHEMA,
                    "status": "PASS_UNIFIED_READ_ONLY_EVALUATION",
                    "lane_id": probe["lane_id"],
                    "source_host_label": probe["source_host_label"],
                    "epoch": epoch,
                    "metric_sha256": file_sha256(metric_path),
                    "metric": str(metric_path.resolve()),
                    "evaluation_schema": EVALUATION_SCHEMA,
                    "evaluation_bundle_fingerprint": (
                        FROZEN_EVALUATION_BUNDLE_FINGERPRINT
                    ),
                    "manifest_sha256": EXPECTED_MANIFEST_SHA256,
                    "source_checkpoint_sha256": f"checkpoint-e{epoch}",
                    "training_protocol_fingerprint": "training-protocol",
                    "unified_environment": environment,
                    "unified_evaluator_protocol_fingerprint": "evaluator",
                    "training_checkpoint_read_only": True,
                    "paired_metric_control": False,
                    "cross_host_training_delta_merged": False,
                    "confirmation20_opened": False,
                },
            )
            bindings["probes"][probe_id][f"e{epoch:03d}"] = {
                "receipt": str(receipt_path),
                "metric": str(metric_path),
            }
    path = root / "METRIC_BINDINGS.json"
    _write(path, bindings)
    return path


def test_shared_spectral_precursor_confirms_and_uses_common70(tmp_path):
    supported = set(DOMAINS[:3])
    mechanisms = {
        "4090A_plain": {domain: "spectral" for domain in supported},
        "5090C_proposal": {domain: "spectral" for domain in supported},
    }
    declines = {
        "4090A_plain": supported,
        "5090C_proposal": supported,
    }
    audit_root = tmp_path / "audits"
    _materialize_audits(audit_root, mechanisms)
    binding = _materialize_metrics(tmp_path / "metrics", declines)
    result = adjudicate_terminal_pathology(
        audit_root=audit_root,
        metric_bindings=binding,
        destination=tmp_path / "decision.json",
    )
    assert result["terminal_pathology_confirmed"] is True
    assert result["confirmed_mechanisms"] == ["spectral_collapse"]
    assert result["mechanism_support"]["spectral_collapse"]["support_cell_count"] == 6
    supported_cells = [
        row
        for row in result["cells"]
        if row["spectral_collapse_precursor"] and row["future_decline_label"]
    ]
    assert all(
        row["future_psnr_delta_db"] == pytest.approx(-0.1) for row in supported_cells
    )


def test_union_of_two_different_weak_mechanisms_does_not_confirm(tmp_path):
    mechanisms = {
        "4090A_plain": {domain: "spectral" for domain in DOMAINS[:3]},
        "5090C_proposal": {domain: "amplification" for domain in DOMAINS[3:]},
    }
    declines = {
        "4090A_plain": set(DOMAINS[:3]),
        "5090C_proposal": set(DOMAINS[3:]),
    }
    audit_root = tmp_path / "audits"
    _materialize_audits(audit_root, mechanisms)
    binding = _materialize_metrics(tmp_path / "metrics", declines)
    result = adjudicate_terminal_pathology(
        audit_root=audit_root,
        metric_bindings=binding,
        destination=tmp_path / "decision.json",
    )
    assert result["terminal_pathology_confirmed"] is False
    assert result["confirmed_mechanisms"] == []
    assert result["next_action"] == "do not add a terminal repair module"


def test_paired_binding_is_not_parsed_before_all_target_blind_audits(tmp_path):
    malformed_binding = tmp_path / "malformed.json"
    malformed_binding.write_text("not-json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="target-blind terminal audit is incomplete"):
        adjudicate_terminal_pathology(
            audit_root=tmp_path / "missing-audits",
            metric_bindings=malformed_binding,
            destination=tmp_path / "decision.json",
        )


def test_terminal_audit_requires_the_same_fixed_sample_across_epochs(tmp_path):
    audit_root = tmp_path / "audits"
    _materialize_audits(audit_root, {})
    changed = audit_root / "probes" / "4090A_plain" / "e150" / "TERMINAL_AUDIT.jsonl"
    value = json.loads(changed.read_text(encoding="utf-8"))
    value["records"][0]["stem"] = "different-sample"
    _write(changed, value)
    receipt = changed.parent / "AUDIT_RECEIPT.json"
    receipt_value = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_value["audit_sha256"] = file_sha256(changed)
    _write(receipt, receipt_value)
    binding = _materialize_metrics(tmp_path / "metrics", {})
    with pytest.raises(RuntimeError, match="sample identity changed"):
        adjudicate_terminal_pathology(
            audit_root=audit_root,
            metric_bindings=binding,
            destination=tmp_path / "decision.json",
        )


def test_metric_receipt_must_bind_the_exact_metric_path(tmp_path):
    audit_root = tmp_path / "audits"
    _materialize_audits(audit_root, {})
    binding = _materialize_metrics(tmp_path / "metrics", {})
    bindings = json.loads(binding.read_text(encoding="utf-8"))
    receipt_path = Path(bindings["probes"]["4090A_plain"]["e100"]["receipt"])
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["metric"] = str(tmp_path / "different.json")
    _write(receipt_path, receipt)
    with pytest.raises(RuntimeError, match="invalid unified metric receipt"):
        adjudicate_terminal_pathology(
            audit_root=audit_root,
            metric_bindings=binding,
            destination=tmp_path / "decision.json",
        )
