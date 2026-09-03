from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from research.paper_aio import confirmation
from research.paper_aio.run import parser


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _basis(tmp_path: Path, monkeypatch):
    freeze_path = _write(tmp_path / "FREEZE.json", {"freeze": True})
    freeze = {
        "distribution_lanes": ["input", "plain"],
        "paper_claims_sha256": "p" * 64,
    }
    monkeypatch.setattr(
        confirmation, "committed_freeze_identity",
        lambda _path, lane_id: (freeze, "f" * 40),
    )
    receipts = []
    for lane in freeze["distribution_lanes"]:
        checkpoint = None
        payload = {"lane_id": lane, "checkpoint": None}
        if lane != "input":
            checkpoint = tmp_path / f"{lane}.pt"
            checkpoint.write_bytes(b"fixed e200")
            payload.update({
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_sha256": confirmation.file_sha256(checkpoint),
                "checkpoint_unchanged": True,
            })
        path = _write(tmp_path / f"{lane}.json", payload)
        receipts.append({
            "lane_id": lane, "receipt": str(path.resolve()),
            "receipt_sha256": confirmation.file_sha256(path),
            "receipt_object_sha256": confirmation.object_sha256(payload),
        })
    cohort = {
        "schema": confirmation.DISTRIBUTION_COHORT_SCHEMA,
        "status": "PASS_COMPLETE_FROZEN_DISTRIBUTION_COHORT",
        "freeze_receipt": str(freeze_path.resolve()),
        "freeze_receipt_sha256": confirmation.file_sha256(freeze_path),
        "freeze_git_commit": "f" * 40,
        "lanes": ["input", "plain"],
        "receipts": receipts,
        "all_lanes_one_runtime": True,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    cohort_path = _write(tmp_path / "DISTRIBUTION_COHORT.json", cohort)
    return freeze_path, cohort_path


def test_confirmation_draft_requires_complete_distribution_and_cannot_authorize(
    tmp_path: Path, monkeypatch,
) -> None:
    freeze_path, cohort_path = _basis(tmp_path, monkeypatch)
    result = confirmation.create_confirmation_review_draft(
        freeze_receipt=freeze_path, distribution_cohort=cohort_path,
        destination=tmp_path / "CONFIRMATION_DRAFT.json",
    )
    assert result["status"] == confirmation.DRAFT_STATUS
    assert result["distribution_lanes"] == ["input", "plain"]
    assert len(result["confirmation_session_id"]) == 64
    assert result["confirmation_authorized"] is False
    assert result["confirmation20_opened"] is False

    receipt_path = Path(json.loads(cohort_path.read_text())["receipts"][0]["receipt"])
    receipt_path.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt identity changed"):
        confirmation.create_confirmation_review_draft(
            freeze_receipt=freeze_path, distribution_cohort=cohort_path,
            destination=tmp_path / "OTHER.json",
        )


def test_confirmation_authorization_requires_committed_review(
    tmp_path: Path, monkeypatch,
) -> None:
    basis = {
        "freeze_receipt": str((tmp_path / "FREEZE.json").resolve()),
        "freeze_receipt_sha256": "f" * 64,
        "freeze_git_commit": "g" * 40,
        "distribution_cohort": str((tmp_path / "COHORT.json").resolve()),
        "distribution_cohort_sha256": "d" * 64,
        "distribution_lanes": ["input", "plain"],
        "paper_claims_sha256": "p" * 64,
        "confirmation_session_id": "s" * 64,
    }
    monkeypatch.setattr(
        confirmation, "_review_basis", lambda **_kwargs: ({}, "", {}, basis),
    )
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr(confirmation, "ROOT", root)
    review = _write(root / "review.json", {
        "schema": confirmation.REVIEW_SCHEMA,
        "status": confirmation.REVIEW_STATUS,
        **basis,
        "human_approval_recorded": True,
        "codex_scientific_review_recorded": True,
        "algorithm_or_baseline_changed_after_freeze": False,
        "paper_claim_changed_after_freeze": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    monkeypatch.setattr(
        confirmation, "_committed_json",
        lambda _path, label: (json.loads(review.read_text()), "r" * 40, "review.json"),
    )
    result = confirmation.materialize_confirmation_authorization(
        freeze_receipt=tmp_path / "FREEZE.json",
        distribution_cohort=tmp_path / "COHORT.json", review_decision=review,
        destination=root / "AUTHORIZATION.json",
    )
    assert result["confirmation_authorized"] is True
    assert result["confirmation20_opened"] is False
    assert result["one_logical_session_only"] is True

    changed = json.loads(review.read_text())
    changed["best_checkpoint_selection"] = True
    review.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="review decision is invalid"):
        confirmation.materialize_confirmation_authorization(
            freeze_receipt=tmp_path / "FREEZE.json",
            distribution_cohort=tmp_path / "COHORT.json", review_decision=review,
            destination=root / "OTHER.json",
        )


def test_confirmation_authorization_real_git_identity_chain(
    tmp_path: Path, monkeypatch,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    basis = {
        "freeze_receipt": str((tmp_path / "FREEZE.json").resolve()),
        "freeze_receipt_sha256": "f" * 64,
        "freeze_git_commit": "g" * 40,
        "distribution_cohort": str((tmp_path / "COHORT.json").resolve()),
        "distribution_cohort_sha256": "d" * 64,
        "distribution_lanes": ["input", "plain"],
        "paper_claims_sha256": "p" * 64,
        "confirmation_session_id": "s" * 64,
    }
    monkeypatch.setattr(confirmation, "ROOT", root)
    monkeypatch.setattr(
        confirmation, "_review_basis", lambda **_kwargs: ({}, "", {}, basis),
    )
    review = _write(root / "review.json", {
        "schema": confirmation.REVIEW_SCHEMA,
        "status": confirmation.REVIEW_STATUS,
        **basis,
        "human_approval_recorded": True,
        "codex_scientific_review_recorded": True,
        "algorithm_or_baseline_changed_after_freeze": False,
        "paper_claim_changed_after_freeze": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    })
    subprocess.run(["git", "add", "review.json"], cwd=root, check=True)
    subprocess.run([
        "git", "-c", "user.name=Confirmation Test", "-c",
        "user.email=confirmation@example.invalid", "commit", "-q", "-m", "review",
    ], cwd=root, check=True)
    authorization = root / "authorization.json"
    confirmation.materialize_confirmation_authorization(
        freeze_receipt=tmp_path / "FREEZE.json",
        distribution_cohort=tmp_path / "COHORT.json", review_decision=review,
        destination=authorization,
    )
    subprocess.run(["git", "add", "authorization.json"], cwd=root, check=True)
    subprocess.run([
        "git", "-c", "user.name=Confirmation Test", "-c",
        "user.email=confirmation@example.invalid", "commit", "-q", "-m", "authorize",
    ], cwd=root, check=True)
    value, commit = confirmation.committed_confirmation_authorization(authorization)
    assert len(commit) == 40
    assert value["confirmation_session_id"] == "s" * 64


def test_confirmation_session_is_atomic_and_only_same_session_can_recover(
    tmp_path: Path, monkeypatch,
) -> None:
    authorization = _write(tmp_path / "AUTH.json", {"fixed": True})
    value = {
        "confirmation_session_id": "s" * 64,
        "distribution_lanes": ["input", "plain"],
    }
    monkeypatch.setattr(
        confirmation, "committed_confirmation_authorization",
        lambda _path: (value, "a" * 40),
    )
    first = confirmation.claim_confirmation_session(
        authorization=authorization, output_root=tmp_path / "output",
    )
    second = confirmation.claim_confirmation_session(
        authorization=authorization, output_root=tmp_path / "output",
    )
    assert first["recovered_existing_session"] is False
    assert second["recovered_existing_session"] is True
    assert first["confirmation20_opened"] is True

    monkeypatch.setattr(
        confirmation, "committed_confirmation_authorization",
        lambda _path: ({**value, "confirmation_session_id": "x" * 64}, "a" * 40),
    )
    with pytest.raises(RuntimeError, match="another session"):
        confirmation.claim_confirmation_session(
            authorization=authorization, output_root=tmp_path / "output",
        )


def test_confirmation_rows_are_unaddressable_without_open_session() -> None:
    rows = []
    for domain_index in range(6):
        domain = f"domain{domain_index}"
        for order in range(20):
            rows.append({
                "domain": domain, "split": "confirmation", "order": str(order),
                "stem": f"c{order}",
            })
        rows.append({
            "domain": domain, "split": "discovery", "order": "0", "stem": "d0",
        })
    with pytest.raises(RuntimeError, match="authorized open session"):
        confirmation.select_confirmation(rows, session={}, count_per_domain=20)
    session = {
        "schema": confirmation.SESSION_SCHEMA,
        "status": confirmation.SESSION_STATUS,
        "confirmation_authorized": True,
        "confirmation20_opened": True,
    }
    selected = confirmation.select_confirmation(
        rows, session=session, count_per_domain=20,
    )
    assert len(selected) == 120
    assert {row["split"] for row in selected} == {"confirmation"}


def test_confirmation_cli_exposes_review_and_authorization_but_not_open() -> None:
    choices = set(parser()._option_string_actions["--stage"].choices)
    assert {
        "confirmation-draft", "confirmation-authorize", "confirmation-claim",
        "confirmation-evaluate", "confirmation-lock",
    } <= choices
    assert "confirmation-open" not in choices
    args = parser().parse_args([
        "--stage", "confirmation-evaluate", "--lane", "input",
        "--confirmation-authorization", "AUTH.json",
        "--confirmation-session", "SESSION.json",
        "--receipt-output", "input_confirmation.json",
    ])
    assert args.confirmation_authorization == Path("AUTH.json")
    assert args.confirmation_session == Path("SESSION.json")


def test_confirmation_input_evaluation_uses_only_open_session_and_fixed_split(
    tmp_path: Path, monkeypatch,
) -> None:
    authorization = _write(tmp_path / "AUTH.json", {"authorized": True})
    session_receipt = _write(tmp_path / "SESSION.json", {"session": True})
    session = {
        "schema": confirmation.SESSION_SCHEMA,
        "status": confirmation.SESSION_STATUS,
        "confirmation_session_id": "s" * 64,
        "confirmation_authorized": True,
        "confirmation20_opened": True,
    }
    auth = {"distribution_lanes": ["input"]}
    monkeypatch.setattr(
        confirmation, "validate_open_session",
        lambda **_kwargs: (session, auth, "a" * 40),
    )
    monkeypatch.setattr(
        confirmation, "read_image",
        lambda _path, size: torch.zeros(1, 3, 4, 4),
    )

    class Perceptual:
        def __call__(self, _left, _right):
            return torch.tensor([0.0])

    monkeypatch.setattr(confirmation, "_lpips", lambda _device: Perceptual())
    monkeypatch.setattr(confirmation, "environment_record", lambda: {"runtime": "one"})
    rows = [
        {
            "domain": f"domain{domain}", "split": "confirmation",
            "order": str(order), "stem": f"stem{order}",
            "input_relpath": "input.png", "target_relpath": "target.png",
        }
        for domain in range(6) for order in range(20)
    ]
    result = confirmation.evaluate_confirmation_lane(
        model=None, spec=None, rows=rows, data_root=tmp_path,
        authorization=authorization, session_receipt=session_receipt,
        destination=tmp_path / "input_confirmation.json",
        checkpoint=None, checkpoint_step=None, checkpoint_metadata=None, gpu=-1,
    )
    assert result["split"] == "confirmation"
    assert result["count_per_domain"] == 20
    assert result["confirmation20_opened"] is True
    assert result["checkpoint_unchanged"] is True
    assert len(result["images"]) == 120
    complete = confirmation.lock_confirmation_cohort(
        authorization=authorization, session_receipt=session_receipt,
        results=[tmp_path / "input_confirmation.json"],
        destination=tmp_path / "CONFIRMATION20_COMPLETE.json",
    )
    assert complete["status"] == "COMPLETE_ONE_TIME_CONFIRMATION20_COHORT"
    assert complete["lanes"] == ["input"]
    assert complete["confirmation20_opened_once"] is True
