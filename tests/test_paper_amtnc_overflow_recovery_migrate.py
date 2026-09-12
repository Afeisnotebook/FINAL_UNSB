from __future__ import annotations

import copy

import torch

from operations.paper_aio_amtnc_overflow_recovery_migrate import (
    dynamics_only_hash,
    migrate_payloads,
    validate_localization,
    validate_replay,
)
from research.local_route1.runtime import full_state_hash


def _e0():
    return {
        "schema": "final-unsb-paper-aio-common-e0-v1",
        "metadata": {
            "git_commit": "old",
            "protocol_fingerprint": "old-protocol",
        },
        "model": {"value": torch.tensor([1.0])},
        "rng": {"value": torch.tensor([2], dtype=torch.uint8)},
        "samplers": {"cursor": 3},
    }


def _checkpoint(e0_hash):
    return {
        "schema": "final-unsb-paper-aio-full-state-v1",
        "lane": {"id": "amtnc"},
        "step": 1_522_434,
        "physical_epoch_completed": 178,
        "target_steps": 1_710_600,
        "model": {
            "networks": {"G": {"weight": torch.tensor([3.0])}},
            "optimizers": [{"state": {}}],
            "schedulers": [{}],
            "method": {"active": True},
        },
        "rng": {"value": torch.tensor([4], dtype=torch.uint8)},
        "samplers": {"primary": {"cursor": 5}, "secondary": {"cursor": 6}},
        "metadata": {
            "git_commit": "old",
            "protocol_fingerprint": "old-protocol",
            "e0_scientific_state_sha256": e0_hash,
        },
    }


def test_migration_changes_only_source_lineage_metadata():
    e0 = _e0()
    checkpoint = _checkpoint(full_state_hash(e0))
    old_e0 = copy.deepcopy(e0)
    old_checkpoint = copy.deepcopy(checkpoint)
    migrated_e0, migrated_checkpoint = migrate_payloads(
        e0, checkpoint, new_git_commit="new",
        new_protocol_fingerprint="new-protocol",
    )
    assert e0["metadata"] == old_e0["metadata"]
    assert checkpoint["metadata"] == old_checkpoint["metadata"]
    assert dynamics_only_hash(migrated_e0) == dynamics_only_hash(e0)
    assert dynamics_only_hash(migrated_checkpoint) == dynamics_only_hash(checkpoint)
    assert migrated_e0["metadata"]["git_commit"] == "new"
    assert migrated_checkpoint["metadata"]["protocol_fingerprint"] == "new-protocol"
    assert migrated_checkpoint["metadata"]["e0_scientific_state_sha256"] == (
        full_state_hash(migrated_e0)
    )


def test_replay_and_localizer_bind_the_same_overflow_event():
    checkpoint_sha = "checkpoint"
    scientific_sha = "state"
    replay = {
        "status": "OVERFLOW_SAFE_REPLAY_COMPLETE",
        "completed_replay_updates": 6250,
        "post_replay_state_finite": True,
        "source_checkpoint_unchanged": True,
        "scientific_lane_modified": False,
        "performance_values_read": False,
        "confirmation20_opened": False,
        "precision_fallback_count": 1,
        "first_precision_fallback": {
            "replay_update_offset": 6245,
            "player": "GF",
        },
        "source_checkpoint_sha256_before": checkpoint_sha,
        "source_checkpoint_sha256_after": checkpoint_sha,
        "source_scientific_state_sha256": scientific_sha,
    }
    localizer = {
        "status": "FAILURE_LOCALIZED",
        "source_checkpoint_unchanged": True,
        "scientific_lane_modified": False,
        "performance_values_read": False,
        "confirmation20_opened": False,
        "source_checkpoint_sha256_before": checkpoint_sha,
        "source_checkpoint_sha256_after": checkpoint_sha,
        "source_scientific_state_sha256": scientific_sha,
        "failure": {
            "player": "GF",
            "replay_update_offset": 6245,
            "geometry": {
                "category_counts": {"FLOAT32_GEOMETRY_PRODUCT_OVERFLOW": 3}
            },
        },
    }
    validate_replay(
        replay, checkpoint_sha256=checkpoint_sha,
        scientific_state_sha256=scientific_sha,
    )
    validate_localization(
        localizer, checkpoint_sha256=checkpoint_sha,
        scientific_state_sha256=scientific_sha,
    )
