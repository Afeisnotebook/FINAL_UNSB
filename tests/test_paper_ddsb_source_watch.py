from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations.paper_aio_ddsb_source_watch import (
    DEFAULT_AUTHORITY_URLS,
    DEFAULT_GITHUB_QUERIES,
    authority_repository_candidates,
    evaluate_sources,
    freeze_contract,
    github_repository_candidates,
    process_alive,
    proposed_contract,
)


def _args(output: Path) -> SimpleNamespace:
    return SimpleNamespace(
        output=output,
        authority_url=None,
        github_query=None,
        poll_seconds=21600,
        timeout_hours=720.0,
        request_timeout_seconds=30.0,
        once=False,
    )


def test_authority_link_is_review_required_not_automatic_acceptance(tmp_path: Path) -> None:
    document = b"""
    <h1>Degradation-Aware Dynamic Schrodinger Bridge for Unpaired Image Restoration</h1>
    <a href="https://github.com/example/DDSB">Code</a>
    """
    github_empty = json.dumps({"total_count": 0, "items": []}).encode()

    def fetcher(url: str, **_kwargs) -> bytes:
        return github_empty if "api.github.com" in url else document

    result = evaluate_sources(proposed_contract(_args(tmp_path)), fetcher=fetcher, now=1.0)
    assert result["status"] == "AUTHORITATIVE_SOURCE_CANDIDATE_REVIEW_REQUIRED"
    assert result["official_repository_candidates"][0]["repository_url"] == (
        "https://github.com/example/DDSB"
    )
    assert result["training_authorized"] is False
    assert result["training_started"] is False


def test_unrelated_lab_repository_outside_title_window_is_ignored() -> None:
    document = (
        '<a href="https://github.com/example/unrelated">Code</a>'
        + "x" * 6000
        + "Degradation-Aware Dynamic Schrodinger Bridge for Unpaired Image Restoration"
    )
    assert authority_repository_candidates(
        document, authority_url="https://example.test/publications"
    ) == []


def test_github_candidate_is_unverified_until_authorship_review(tmp_path: Path) -> None:
    authority = b"<h1>Degradation-Aware Dynamic Schrodinger Bridge for Unpaired Image Restoration</h1>"
    github = json.dumps(
        {
            "total_count": 1,
            "items": [
                {
                    "name": "DDSB",
                    "full_name": "someone/DDSB",
                    "description": "DDSB for unpaired image restoration",
                    "html_url": "https://github.com/someone/DDSB",
                    "owner": {"login": "someone"},
                }
            ],
        }
    ).encode()

    def fetcher(url: str, **_kwargs) -> bytes:
        return github if "api.github.com" in url else authority

    result = evaluate_sources(proposed_contract(_args(tmp_path)), fetcher=fetcher, now=2.0)
    assert result["status"] == "UNVERIFIED_SOURCE_CANDIDATE_REVIEW_REQUIRED"
    assert result["unverified_repository_candidates"][0]["full_name"] == "someone/DDSB"
    assert result["paper_status"] == "REPRODUCTION_INCOMPLETE"
    assert result["manual_formula_source_implementation_review_required"] is True


def test_unrelated_github_hit_is_not_a_candidate() -> None:
    payload = {
        "items": [
            {
                "name": "DD-SB",
                "full_name": "someone/DD-SB",
                "description": "an unrelated diffusion project",
                "html_url": "https://github.com/someone/DD-SB",
            }
        ]
    }
    assert github_repository_candidates(payload) == []


def test_total_network_failure_is_transient_not_scientific_failure(tmp_path: Path) -> None:
    def fail(_url: str, **_kwargs) -> bytes:
        raise TimeoutError("offline")

    result = evaluate_sources(proposed_contract(_args(tmp_path)), fetcher=fail, now=3.0)
    assert result["status"] == "TRANSIENT_NETWORK_ERROR"
    assert result["paper_status"] == "REPRODUCTION_INCOMPLETE"
    assert result["training_authorized"] is False
    assert len(result["network_errors"]) == len(DEFAULT_AUTHORITY_URLS) + len(
        DEFAULT_GITHUB_QUERIES
    )


def test_contract_is_frozen_and_fail_closed(tmp_path: Path) -> None:
    proposed = proposed_contract(_args(tmp_path))
    assert proposed["automatic_source_acceptance"] is False
    assert proposed["automatic_training_authorization"] is False
    path = freeze_contract(tmp_path, proposed)
    assert path.is_file()
    assert freeze_contract(tmp_path, proposed) == path
    changed = dict(proposed)
    changed["poll_seconds"] = 3600
    with pytest.raises(RuntimeError, match="contract changed"):
        freeze_contract(tmp_path, changed)


def test_process_liveness_supports_durable_supervision() -> None:
    assert process_alive(os.getpid()) is True
