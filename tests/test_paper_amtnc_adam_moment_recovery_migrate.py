from __future__ import annotations

import copy

import torch

from operations.paper_aio_amtnc_adam_moment_recovery_migrate import (
    dynamics_only_hash,
    migrate_payloads,
)
from operations.paper_aio_amtnc_adam_moment_incident import (
    EXCEPTION_LINE,
    repeated_failure_count,
)


def _payloads():
    metadata = {
        "git_commit": "a" * 40,
        "protocol_fingerprint": "b" * 64,
        "e0_scientific_state_sha256": "c" * 64,
    }
    e0 = {
        "schema": "e0",
        "metadata": copy.deepcopy(metadata),
        "model": {"value": torch.tensor([1.0])},
        "rng": {"seed": 2026},
    }
    checkpoint = {
        "schema": "full",
        "metadata": copy.deepcopy(metadata),
        "model": {
            "optimizers": [{
                "state": {2: {"exp_avg_sq": torch.tensor([3.0])}},
                "param_groups": [{"params": [2]}],
            }],
        },
        "rng": {"seed": 2026},
        "samplers": {"primary": 4, "secondary": 5},
    }
    return e0, checkpoint


def test_migration_changes_only_source_lineage_metadata():
    e0, checkpoint = _payloads()
    parent_e0_dynamics = dynamics_only_hash(e0)
    parent_checkpoint_dynamics = dynamics_only_hash(checkpoint)
    migrated_e0, migrated_checkpoint = migrate_payloads(
        e0,
        checkpoint,
        new_git_commit="d" * 40,
        new_protocol_fingerprint="e" * 64,
    )

    assert dynamics_only_hash(migrated_e0) == parent_e0_dynamics
    assert dynamics_only_hash(migrated_checkpoint) == parent_checkpoint_dynamics
    assert migrated_e0["metadata"]["git_commit"] == "d" * 40
    assert migrated_checkpoint["metadata"]["protocol_fingerprint"] == "e" * 64
    assert migrated_checkpoint["metadata"]["e0_scientific_state_sha256"] != (
        checkpoint["metadata"]["e0_scientific_state_sha256"]
    )
    assert e0["metadata"]["git_commit"] == "a" * 40
    assert checkpoint["metadata"]["protocol_fingerprint"] == "b" * 64


def test_incident_counter_requires_identical_complete_exception_lines():
    text = "\n".join((
        EXCEPTION_LINE,
        "RuntimeError: a different failure",
        EXCEPTION_LINE,
        "prefix " + EXCEPTION_LINE,
    ))
    assert repeated_failure_count(text) == 2
