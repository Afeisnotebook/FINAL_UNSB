from __future__ import annotations

import copy

import pytest

from operations import paper_aio_manuscript_tables as tables


DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _late(method: str, plain_host: str) -> list[dict]:
    return [
        {
            "epoch": epoch,
            "macro_psnr_delta": 0.1,
            "macro_ssim_delta": 0.01,
            "macro_lpips_delta": -0.01,
            "candidate_macro_psnr": 20.1,
            "plain_macro_psnr": 20.0,
            "positive_domains": 6,
            "worst_domain_delta": 0.01,
            "crn_exact": True,
            "runtime_relation": {"status": "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION"},
            "domain_delta": {
                domain: {"psnr": 0.1, "ssim": 0.01, "lpips": -0.01}
                for domain in DOMAINS
            },
        }
        for epoch in (150, 175, 200)
    ]


def _entry(lane: str, *, algorithm: bool = False) -> dict:
    value = {
        "lane_id": lane,
        "comparison_scope": "fixture",
        "terminal": {"macro_psnr": 20.0, "macro_ssim": 0.8, "macro_lpips": 0.2},
    }
    if algorithm:
        value["scientific_gate"] = {"status": "PASS"}
        value["late_trajectory"] = _late(lane, "plain")
    return value


def _portfolio() -> dict:
    methods = {
        key: {
            "algorithm_id": f"algorithm-{key}",
            "matched_plain": "host/plain",
            "result": _entry(key, algorithm=True),
        }
        for key in ("proposal", "amtnc", "stcgr")
    }
    complexity = {
        lane: {
            "parameters": {"unique_parameters": 10},
            "training_step": {"median_ms": 2.0},
            "inference": {"nfe": {"5": 1.0}},
            "flops": {"reported": False},
            "environment": {"gpu": "fixture"},
        }
        for lane in ("plain", "proposal", "amtnc", "stcgr", "cut", "cyclegan")
    }
    labels = {
        "input": "Input",
        "cyclegan": "CycleGAN (official-loss, controlled shared backbone)",
        "cut": "CUT (official-loss, controlled exposure reproduction)",
        "plain": "Plain UNSB",
        "proposal": "Proposal-only",
        "amtnc": "AM-TNC",
        "stcgr": "ST-CGR",
        "dclgan": "DCLGAN (official-source, controlled exposure reproduction)",
    }
    return {
        "schema": "final-unsb-paper-full-data-algorithm-portfolio-v1",
        "status": "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_AWAITING_CONFIRMATION_DECISION",
        "primary_epoch": 200,
        "methods": methods,
        "external_baselines": {
            "input": _entry("input"), "cut": _entry("cut"),
            "cyclegan": _entry("cyclegan"),
        },
        "plain_control": _entry("plain"),
        "complexity": complexity,
        "source_artifact_sha256": {"first_wave_results": "a" * 64},
        "baseline_reporting_tiers": {
            "main_table_metadata": {
                key: {
                    "paper_label": label,
                    "reproduction_or_comparison_scope": f"frozen scope for {key}",
                }
                for key, label in labels.items()
            },
        },
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def test_builds_fixed_main_sustained_domain_and_complexity_tables() -> None:
    portfolio = tables.validate_portfolio(_portfolio())
    result = tables.build_tables(
        portfolio=portfolio, claims=["claim one"], portfolio_sha256="b" * 64,
    )
    assert set(result) == set(tables.TABLE_FILES)
    assert result["MAIN_E200.csv"].splitlines()[1].startswith("input,")
    assert "plain,Plain UNSB,plain_control" in result["MAIN_E200.csv"]
    assert "CycleGAN (official-loss, controlled shared backbone)" in result["MAIN_E200.csv"]
    assert "## Reproduction and comparison scope" in result["PAPER_RESULT_SUMMARY.md"]
    assert len(result["ALGORITHM_SUSTAINED.csv"].splitlines()) == 1 + 3 * 3
    assert len(result["ALGORITHM_DOMAIN_DELTAS.csv"].splitlines()) == 1 + 3 * 3 * 6
    assert "algorithm-proposal" in result["ALGORITHM_SUSTAINED.csv"]
    assert "Source portfolio SHA256" in result["PAPER_RESULT_SUMMARY.md"]
    assert "claim one" in result["PAPER_CLAIMS.csv"]


def test_accepts_augmented_dclgan_without_calling_it_matched() -> None:
    portfolio = _portfolio()
    portfolio["schema"] = "final-unsb-paper-full-data-algorithm-portfolio-v2"
    portfolio["status"] = "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_WITH_DCLGAN_AWAITING_CONFIRMATION_DECISION"
    portfolio["external_baselines"]["dclgan"] = {
        "lane_id": "dclgan",
        "comparison_scope": "standalone_fixed_protocol_no_matched_delta_claim",
        "terminal": {"epoch": 200, "macro_psnr": 19.0, "macro_ssim": 0.7, "macro_lpips": 0.3},
    }
    portfolio["complexity"]["dclgan"] = copy.deepcopy(portfolio["complexity"]["cut"])
    main = tables.build_tables(
        portfolio=tables.validate_portfolio(portfolio), claims=["fixed"],
        portfolio_sha256="c" * 64,
    )["MAIN_E200.csv"]
    dclgan = next(line for line in main.splitlines() if line.startswith("dclgan,"))
    assert "DCLGAN (official-source, controlled exposure reproduction)" in dclgan
    assert "standalone_fixed_protocol_no_matched_delta_claim" in dclgan
    assert ",," in dclgan


def test_rejects_best_checkpoint_or_incomplete_domain_trajectory() -> None:
    portfolio = _portfolio()
    portfolio["best_checkpoint_selection"] = True
    with pytest.raises(RuntimeError, match="incomplete or unsafe"):
        tables.validate_portfolio(portfolio)
    portfolio = _portfolio()
    portfolio["methods"]["proposal"]["result"]["late_trajectory"][0]["domain_delta"].pop("d6")
    with pytest.raises(RuntimeError, match="six-domain"):
        tables.build_tables(
            portfolio=tables.validate_portfolio(portfolio), claims=["fixed"],
            portfolio_sha256="d" * 64,
        )


def test_rejects_unlabelled_or_undisclosed_main_table_row() -> None:
    portfolio = _portfolio()
    portfolio["baseline_reporting_tiers"]["main_table_metadata"].pop("cyclegan")
    with pytest.raises(RuntimeError, match="reporting metadata is incomplete: cyclegan"):
        tables.build_tables(
            portfolio=tables.validate_portfolio(portfolio), claims=["fixed"],
            portfolio_sha256="e" * 64,
        )

    portfolio = _portfolio()
    portfolio["baseline_reporting_tiers"]["main_table_metadata"]["cut"][
        "reproduction_or_comparison_scope"
    ] = ""
    with pytest.raises(RuntimeError, match="reporting metadata is incomplete: cut"):
        tables.build_tables(
            portfolio=tables.validate_portfolio(portfolio), claims=["fixed"],
            portfolio_sha256="f" * 64,
        )


def test_immutable_artifact_refuses_drift(tmp_path) -> None:
    path = tmp_path / "table.csv"
    tables._immutable_text(path, "a\n")
    tables._immutable_text(path, "a\n")
    with pytest.raises(RuntimeError, match="differs"):
        tables._immutable_text(path, "b\n")
