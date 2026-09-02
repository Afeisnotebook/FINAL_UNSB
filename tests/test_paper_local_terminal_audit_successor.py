import json

import pytest

from operations.paper_aio_local_terminal_audit_successor import (
    AUDIT_GRADIENT_REPLICATES,
    AUDIT_REPLICATES,
    AUDIT_SCHEMA,
    PROBES,
    _audit_result,
)


def _valid_result():
    spectrum = {
        "top_eigenvalue": 1.0,
        "trace": 1.0,
        "effective_rank": 1.0,
        "eigenvalues": [1.0] * AUDIT_REPLICATES,
    }
    step = {
        "increment_spectrum": spectrum,
        "endpoint_spectrum": spectrum,
        "endpoint_direction_cosine_to_mean": 0.5,
        "endpoint_direction_definition": "endpoint_minus_bridge_state",
        "local_jacobian_top_singular_proxy": 1.0,
        "rollout_jacobian_top_singular_proxy": 2.0,
        "jvp_initial_direction": "lane_blind_crn_bridge_noise_same_sample_time",
        "perturbation_gain_to_final_output": 1.5,
    }
    return {
        "schema": AUDIT_SCHEMA,
        "status": "TARGET_BLIND_AUDIT_COMPLETE",
        "replicates": AUDIT_REPLICATES,
        "samples_per_domain": 1,
        "records": [
            {
                "domain": f"domain-{domain}",
                "stem": "sample",
                "steps": [
                    {"time_index": time_index, **step} for time_index in range(5)
                ],
                "nfe4_to_nfe5_output_rms_mean": 0.1,
                "nfe4_to_nfe5_output_rms_std": 0.01,
            }
            for domain in range(6)
        ],
        "gradient_stratum_audit": {
            "status": "TARGET_BLIND_NATIVE_OBJECTIVE_GRADIENT_AUDIT_COMPLETE",
            "strata": [
                {"time_index": time_index, "replicates": AUDIT_GRADIENT_REPLICATES}
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


def test_full_data_terminal_probe_set_is_fixed_and_multi_algorithm():
    assert {(row["host_label"], row["import_lane"]) for row in PROBES} == {
        ("4090A", "plain"),
        ("4090A", "amtnc"),
        ("5090C", "proposal"),
        ("5090A", "G4-01-STRATIFIED-TIME-CONDITIONAL-GF"),
    }


def test_terminal_result_gate_requires_unchanged_state_rng_and_no_labels(tmp_path):
    path = tmp_path / "TERMINAL_AUDIT.jsonl"
    value = _valid_result()
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    assert _audit_result(path) == value
    value["paired_labels_attached"] = True
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="boundary"):
        _audit_result(path)


def test_terminal_result_gate_rejects_local_only_jacobian(tmp_path):
    path = tmp_path / "TERMINAL_AUDIT.jsonl"
    value = _valid_result()
    del value["records"][0]["steps"][0]["rollout_jacobian_top_singular_proxy"]
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="rollout_jacobian"):
        _audit_result(path)
