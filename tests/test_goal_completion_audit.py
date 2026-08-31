from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from operations.local_route1_complete_final_result_relay import EXTRA_FILES
from operations import local_route1_goal_completion_audit_successor as successor
from research.local_route1.complete_frontier_final_delivery import (
    ALTERNATES_SCHEMA,
    CANDIDATE_SCHEMA,
    POINTER,
    POINTER_SCHEMA,
    PUBLISHED_FILES,
    RESEARCH_FRONTIER_SCHEMA,
    RESULTS_SCHEMA,
)
from research.local_route1.goal_completion_audit import (
    RELAY_MANIFEST_SCHEMA,
    audit_complete_delivery,
    materialize_goal_completion_audit,
)
from research.local_route1.protocol import file_sha256


DOMAINS = [f"domain-{index}" for index in range(6)]


def _write(path: Path, value: dict | str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def _trajectory() -> list[dict]:
    rows = []
    for epoch in (150, 175, 200):
        domains = {}
        for domain in DOMAINS:
            domains[domain] = {
                role: {"psnr": 1.0, "ssim": 0.5, "lpips": 0.2}
                for role in ("candidate", "plain", "delta")
            }
        rows.append({
            "data_epoch": epoch,
            "updates": epoch * 150,
            "candidate_macro": {"psnr": 1.0, "ssim": 0.5, "lpips": 0.2},
            "plain_macro": {"psnr": 0.9, "ssim": 0.4, "lpips": 0.3},
            "macro_delta": {"psnr": 0.1, "ssim": 0.1, "lpips": -0.1},
            "positive_domains": 6,
            "worst_domain_delta": 0.1,
            "domains": domains,
        })
    return rows


def _evidence(candidate_id: str) -> dict:
    return {
        "candidate_id": candidate_id,
        "receipt": {"candidate_id": candidate_id},
        "trajectory": {"candidate_id": candidate_id},
        "derivation_card": {"candidate_id": candidate_id},
        "implementation": {"candidate_id": candidate_id},
        "absolute_relative_domain_trajectory": _trajectory(),
    }


def _rehash(root: Path) -> None:
    pointer = json.loads((root / POINTER).read_text(encoding="utf-8"))
    pointer["final_file_sha256"] = {
        name: file_sha256(root / name) for name in PUBLISHED_FILES
    }
    _write(root / POINTER, pointer)
    manifest = json.loads((root / "RELAY_MANIFEST.json").read_text(encoding="utf-8"))
    manifest["file_sha256"] = {
        name: file_sha256(root / name)
        for name in (POINTER, *PUBLISHED_FILES, *EXTRA_FILES)
    }
    _write(root / "RELAY_MANIFEST.json", manifest)


def _delivery(root: Path) -> Path:
    selected = "SELECTED"
    ids_4090 = [selected, "ALT-A", "ALT-B"]
    ids_5090 = ["SOURCE-A"]
    boundaries = {
        "scientific_conclusion": "selected is the current action priority",
        "engineering_failures": "invalid implementations are diagnostics",
        "proxy_distortion": "local proxy does not overwrite the 4090 authority",
        "untested_hypotheses": ["seed stability", "full data", "confirmation20"],
    }
    candidate = {
        "schema": CANDIDATE_SCHEMA,
        "status": "FINAL_CURRENT_BEST_ROUTE1_CANDIDATE",
        "candidate_id": selected,
        "classification": "strict_sustained",
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "selected_fixed_checkpoint": {
            "data_epoch": 200, "updates": 30000,
            "best_checkpoint_selection": False,
        },
        "mathematics": {
            "unsb_object": "native update",
            "formula": "u",
            "identity_or_unbiased_condition": "E[u]=g",
            "target_inaccessibility_proof": "training state only",
        },
        "source_files": ["model.py"],
        "complexity": {
            "compute_cost": "2x", "memory_cost": "O(P)",
            "recovery_state_cost": "one state",
        },
        "risk": {
            "expected_applicable_state": {"condition": "variance high"},
            "falsifying_experiment": "e200 negative",
            "single_seed_only": True,
            "cross_seed_stability_claimed": False,
        },
        "reproduction": {
            "seed2026_e200": "python executor.py",
            "deferred_seed_validation": [2027, 2028],
        },
        "absolute_relative_domain_trajectory": _trajectory(),
        "mechanism_evidence": {
            "kind": "same_host_three_role_ablation",
            "roles": {
                "proposal_only": {}, "observable_only": {},
                "projected_or_full": {},
            },
        },
        "target_data_epochs": 200,
        "target_updates": 30000,
        "training_batch_size": 1,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    alternates = {
        "schema": ALTERNATES_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected,
        "alternates": [
            {"candidate_id": "ALT-A"}, {"candidate_id": "ALT-B"},
        ],
        "cross_host_deltas_merged": False,
        "confirmation20_opened": False,
    }
    historical = {
        "status": "COMPLETE_LONG_HORIZON_PROBE_CAUSAL_AND_DERIVATION_EVIDENCE",
        "dt_hj_hnek_anchor_trajectories": {
            "summaries": [
                {"probe_id": probe, "complete_e200": True}
                for probe in ("dt", "hj", "hnek")
            ],
        },
        "proxy_calibration": {"status": "CALIBRATED", "passing_probes": ["hj"]},
        "long_causal_matrix_summary": {
            "status": "COMPLETE_CAUSAL_AUDIT",
            "reversal_rows": 474,
            "sampling_variance_rows": 140,
            "paired_labels_joined_only_after_branches": True,
            "paired_metrics_accessed_by_controller": False,
        },
        "hypothesis_ledger_summary": [
            {"candidate_id": candidate_id} for candidate_id in ids_4090
        ],
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    research = {
        "schema": RESEARCH_FRONTIER_SCHEMA,
        "status": "COMPLETE_MULTI_CANDIDATE_ROUTE1_RESEARCH_FRONTIER",
        "action_priority_candidate_id": selected,
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "remote4090_same_host_frontier": [
            {"candidate_id": candidate_id} for candidate_id in ids_4090
        ],
        "remote5090_source_host_frontier": [
            {"candidate_id": candidate_id} for candidate_id in ids_5090
        ],
        "remote4090_complete_candidate_evidence": [
            _evidence(candidate_id) for candidate_id in ids_4090
        ],
        "remote5090_complete_candidate_evidence": [
            _evidence(candidate_id) for candidate_id in ids_5090
        ],
        "historical_probe_causal_and_derivation_evidence": historical,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_formula_or_training_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    results = {
        "schema": RESULTS_SCHEMA,
        "status": "COMPLETE",
        "selected_candidate_id": selected,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "conclusion_boundaries": boundaries,
        "cross_host_deltas_merged": False,
        "paired_metrics_used_for_training_or_control": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    report = (
        "# report\n\n## 证据边界\n科学结论\n工程失败\nproxy失真\n"
        "未验证\nRESEARCH_FRONTIER.json\n"
    )
    values: dict[str, dict | str] = {
        "CANDIDATE.json": candidate,
        "ALTERNATES.json": alternates,
        "RESULTS.json": results,
        "RESEARCH_FRONTIER.json": research,
        "FINAL_ROUTE1_REPORT.md": report,
    }
    for name, value in values.items():
        _write(root / name, value)
    for name in EXTRA_FILES:
        _write(root / name, {"name": name})
    pointer = {
        "schema": POINTER_SCHEMA,
        "status": "COMPLETE_FRONTIER_FINAL_DELIVERY_COMPLETE",
        "selected_candidate_id": selected,
        "final_file_sha256": {
            name: file_sha256(root / name) for name in PUBLISHED_FILES
        },
        "complete_4090_frontier_sha256": file_sha256(root / EXTRA_FILES[0]),
        "portable_5090_frontier_sha256": file_sha256(root / EXTRA_FILES[1]),
        "canonical_candidate_is_action_priority_only": True,
        "algorithm_discovery_collapsed_to_single_candidate": False,
        "research_frontier_unique_candidate_count": 4,
        "research_frontier_host_scoped_row_count": 4,
        "cross_host_deltas_merged": False,
        "selection_seeds": [2026],
        "deferred_seed_validation": [2027, 2028],
        "cross_seed_stability_claimed": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    _write(root / POINTER, pointer)
    _write(root / "RELAY_MANIFEST.json", {
        "schema": RELAY_MANIFEST_SCHEMA,
        "status": "COMPLETE_EXACT_FINAL_DELIVERY_RETRIEVED",
        "selected_candidate_id": selected,
        "file_sha256": {
            name: file_sha256(root / name)
            for name in (POINTER, *PUBLISHED_FILES, *EXTRA_FILES)
        },
        "credentials_persisted": False,
        "checkpoint_transfer": False,
        "cross_host_deltas_merged": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    })
    return root


def test_goal_completion_audit_proves_terminal_artifacts_but_reserves_git_gate(
    tmp_path: Path,
):
    delivery = _delivery(tmp_path)
    result = audit_complete_delivery(delivery)
    assert result["terminal_artifact_requirements_proven"] is True
    assert result["final_repository_commit_and_push_required"] is True
    assert result["completion_claim_allowed"] is False
    assert result["candidate"]["trajectory"]["terminal_updates"] == 30000
    assert result["historical_evidence"]["reversal_rows"] == 474
    output = tmp_path / "audit" / "GOAL_COMPLETION_AUDIT.json"
    assert materialize_goal_completion_audit(delivery, output) == result
    assert json.loads(output.read_text(encoding="utf-8")) == result


def test_goal_completion_audit_rejects_missing_true_e200_trajectory(tmp_path: Path):
    delivery = _delivery(tmp_path)
    candidate = json.loads((delivery / "CANDIDATE.json").read_text(encoding="utf-8"))
    candidate["absolute_relative_domain_trajectory"] = candidate[
        "absolute_relative_domain_trajectory"
    ][:-1]
    _write(delivery / "CANDIDATE.json", candidate)
    _rehash(delivery)
    with pytest.raises(RuntimeError, match="does not terminate at e200"):
        audit_complete_delivery(delivery)


def test_goal_completion_audit_rejects_single_candidate_collapse(tmp_path: Path):
    delivery = _delivery(tmp_path)
    research = json.loads((delivery / "RESEARCH_FRONTIER.json").read_text(encoding="utf-8"))
    research["algorithm_discovery_collapsed_to_single_candidate"] = True
    _write(delivery / "RESEARCH_FRONTIER.json", research)
    _rehash(delivery)
    with pytest.raises(RuntimeError, match="research-frontier identity changed"):
        audit_complete_delivery(delivery)


def test_goal_completion_audit_successor_contract_is_source_bound_and_posthoc(
    monkeypatch, tmp_path: Path,
):
    def fake_run_text(command, *, cwd):
        if command[:2] == ["git", "status"]:
            return ""
        if command[:2] == ["git", "rev-parse"]:
            return "a" * 40
        raise AssertionError(command)

    monkeypatch.setattr(successor.support, "run_text", fake_run_text)
    monkeypatch.setattr(successor.support, "file_sha256", lambda _path: "b" * 64)
    contract = successor.default_contract(Namespace(
        repo=tmp_path / "repo",
        delivery=tmp_path / "delivery",
        output=tmp_path / "audit.json",
        state=tmp_path / "state.json",
        poll_seconds=60,
        timeout_seconds=1209600,
    ))
    successor.validate_contract(contract)
    assert contract["selection_seeds"] == [2026]
    assert contract["deferred_seed_validation"] == [2027, 2028]
    assert contract["cross_host_deltas_merged"] is False
    assert contract["paired_controller_access"] is False
    assert contract["confirmation20_opened"] is False
    assert "password" not in json.dumps(contract).lower()
