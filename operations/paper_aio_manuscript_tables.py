"""Export a frozen paper portfolio into deterministic manuscript tables.

This is post-freeze reporting code.  It cannot run from a draft portfolio: the
exact algorithm/baseline/claim freeze receipt must already be committed to Git.
The exporter never trains, schedules, ranks or selects a checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from research.paper_aio.distribution import committed_freeze_identity
from research.paper_aio.protocol import ROOT, file_sha256, portable_source_sha256


SCHEMA = "final-unsb-paper-manuscript-table-receipt-v2"
STATUS = "COMPLETE_FROZEN_E200_MANUSCRIPT_TABLES"
BRANCH_SCHEMA = "final-unsb-paper-manuscript-result-branch-v1"
BRANCH_CONTRACT_SCHEMA = "final-unsb-paper-manuscript-branching-contract-v1"
BRANCH_CONTRACT_STATUS = "PRE_RESULT_STRUCTURE_FROZEN_NO_EMPIRICAL_CLAIM"
BRANCH_FILE = "MANUSCRIPT_RESULT_BRANCH.json"
CANONICAL_BRANCH_CONTRACT = ROOT / "configs" / "PAPER_MANUSCRIPT_BRANCHING_CONTRACT.json"
PORTFOLIO_SCHEMAS = {
    "final-unsb-paper-full-data-algorithm-portfolio-v1",
    "final-unsb-paper-full-data-algorithm-portfolio-v2",
}
PORTFOLIO_STATUSES = {
    "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_AWAITING_CONFIRMATION_DECISION",
    "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_WITH_DCLGAN_AWAITING_CONFIRMATION_DECISION",
}
TABLE_FILES = (
    "MAIN_E200.csv",
    "ALGORITHM_SUSTAINED.csv",
    "ALGORITHM_DOMAIN_DELTAS.csv",
    "COMPLEXITY.csv",
    "PAPER_CLAIMS.csv",
    "PAPER_RESULT_SUMMARY.md",
)
OUTPUT_FILES = (*TABLE_FILES, BRANCH_FILE)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _committed_file_identity(path: Path, *, label: str) -> tuple[str, str]:
    path = Path(path).resolve()
    relative = path.relative_to(ROOT.resolve()).as_posix()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True,
    ).strip()
    if status:
        raise RuntimeError(f"{label} has uncommitted changes")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT,
        text=True,
    ).strip()
    if len(commit) != 40:
        raise RuntimeError(f"{label} has no committed Git identity")
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
    )
    committed = committed.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    if committed_sha256 != portable_source_sha256(path):
        raise RuntimeError(f"working {label} differs from its Git blob")
    return commit, committed_sha256


def _committed_script_identity() -> tuple[str, str]:
    return _committed_file_identity(Path(__file__), label="manuscript exporter")


def _number(value: Any, *, label: str, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"invalid numeric paper field: {label}")
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")):
        raise RuntimeError(f"non-finite paper field: {label}")
    return result


def validate_portfolio(
    value: dict[str, Any], *, expected_sha256: str | None = None,
    source_path: Path | None = None,
) -> dict[str, Any]:
    if source_path is not None and expected_sha256 is not None:
        if file_sha256(source_path) != expected_sha256:
            raise RuntimeError("frozen source portfolio hash changed")
    if (
        value.get("schema") not in PORTFOLIO_SCHEMAS
        or value.get("status") not in PORTFOLIO_STATUSES
        or int(value.get("primary_epoch", -1)) != 200
        or value.get("paper_claims_frozen") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("best_checkpoint_selection") is not False
        or value.get("cross_non_equivalent_runtime_delta") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation20_opened") is not False
        or not isinstance(value.get("methods"), dict)
        or not isinstance(value.get("external_baselines"), dict)
        or not isinstance(value.get("plain_control"), dict)
        or not isinstance(value.get("complexity"), dict)
    ):
        raise RuntimeError("frozen paper portfolio is incomplete or unsafe")
    for name, digest in (value.get("source_artifact_sha256") or {}).items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError("paper portfolio has an invalid source-artifact hash")
    return value


def validate_branch_contract(value: dict[str, Any]) -> dict[str, Any]:
    method_keys = value.get("method_keys")
    matrix = value.get("branch_matrix")
    bindings = value.get("source_bindings")
    if (
        value.get("schema") != BRANCH_CONTRACT_SCHEMA
        or value.get("status") != BRANCH_CONTRACT_STATUS
        or method_keys != ["proposal", "stcgr", "amtnc"]
        or not isinstance(matrix, list)
        or not isinstance(bindings, dict)
        or value.get("hard_boundaries", {}).get("performance_values_read_while_creating_contract") is not False
        or value.get("hard_boundaries", {}).get("full_data_benefit_predeclared") is not False
        or value.get("hard_boundaries", {}).get("unique_winner_predeclared") is not False
        or value.get("hard_boundaries", {}).get("confirmation20_opened") is not False
    ):
        raise RuntimeError("manuscript branching contract is incomplete or unsafe")
    observed: set[tuple[bool, bool, bool]] = set()
    for row in matrix:
        if not isinstance(row, dict) or not isinstance(row.get("route"), str) or not row["route"]:
            raise RuntimeError("manuscript branching contract has an invalid route")
        key = tuple(row.get(name) for name in method_keys)
        if any(not isinstance(item, bool) for item in key):
            raise RuntimeError("manuscript branching contract has a non-boolean disposition")
        observed.add(key)  # type: ignore[arg-type]
    expected = {
        (proposal, stcgr, amtnc)
        for proposal in (False, True)
        for stcgr in (False, True)
        for amtnc in (False, True)
    }
    if len(matrix) != 8 or observed != expected:
        raise RuntimeError("manuscript branching contract does not cover all outcomes exactly once")
    for name, binding in bindings.items():
        if not isinstance(binding, dict):
            raise RuntimeError(f"invalid manuscript source binding: {name}")
        path = ROOT / str(binding.get("path", ""))
        digest = binding.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or file_sha256(path) != digest:
            raise RuntimeError(f"manuscript source binding changed: {name}")
    return value


def resolve_manuscript_branch(
    portfolio: dict[str, Any], contract: dict[str, Any],
) -> dict[str, Any]:
    dispositions: dict[str, str] = {}
    outcome: list[bool] = []
    for key in contract["method_keys"]:
        method = portfolio["methods"].get(key)
        if not isinstance(method, dict) or not isinstance(method.get("result"), dict):
            raise RuntimeError(f"paper portfolio lacks manuscript method: {key}")
        status = (method["result"].get("scientific_gate") or {}).get("status")
        if status not in {"PASS", "FAIL"}:
            raise RuntimeError(f"paper method lacks a terminal disposition: {key}")
        dispositions[key] = status
        outcome.append(status == "PASS")
    matches = [
        row for row in contract["branch_matrix"]
        if [row[key] for key in contract["method_keys"]] == outcome
    ]
    if len(matches) != 1:
        raise RuntimeError("paper dispositions do not resolve one manuscript branch")
    return {
        "schema": BRANCH_SCHEMA,
        "status": "COMPLETE_POST_FREEZE_RESULT_BRANCH_RESOLVED",
        "method_dispositions": dispositions,
        "route": matches[0]["route"],
        "branch_was_pre_registered": True,
        "unique_winner_inferred": False,
        "negative_result_route_preserved": True,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }


def _terminal(entry: dict[str, Any], *, lane: str) -> dict[str, Any]:
    terminal = entry.get("terminal")
    if not isinstance(terminal, dict):
        raise RuntimeError(f"paper lane lacks an e200 terminal result: {lane}")
    if "epoch" in terminal and int(terminal["epoch"]) != 200:
        raise RuntimeError(f"paper lane terminal is not e200: {lane}")
    return {
        "macro_psnr": _number(terminal.get("macro_psnr"), label=f"{lane}.psnr"),
        "macro_ssim": _number(terminal.get("macro_ssim"), label=f"{lane}.ssim"),
        "macro_lpips": _number(
            terminal.get("macro_lpips"), label=f"{lane}.lpips", optional=True,
        ),
    }


def _main_rows(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reporting = portfolio.get("baseline_reporting_tiers") or {}
    reporting_metadata = reporting.get("main_table_metadata") or {}

    def add(
        *, key: str, category: str, entry: dict[str, Any],
        algorithm_id: str | None = None, matched_plain: str | None = None,
    ) -> None:
        lane = str(entry.get("lane_id", key))
        terminal = _terminal(entry, lane=lane)
        metadata = reporting_metadata.get(key)
        if (
            not isinstance(metadata, dict)
            or not isinstance(metadata.get("paper_label"), str)
            or not metadata["paper_label"].strip()
            or not isinstance(metadata.get("reproduction_or_comparison_scope"), str)
            or not metadata["reproduction_or_comparison_scope"].strip()
        ):
            raise RuntimeError(f"paper reporting metadata is incomplete: {key}")
        rows.append({
            "row_id": key,
            "paper_label": metadata["paper_label"],
            "category": category,
            "lane_id": lane,
            "algorithm_id": algorithm_id or lane,
            "comparison_scope": entry.get("comparison_scope"),
            "matched_plain": matched_plain,
            "scientific_gate": (entry.get("scientific_gate") or {}).get("status"),
            "reproduction_or_comparison_scope": metadata[
                "reproduction_or_comparison_scope"
            ],
            "e200_macro_psnr": terminal["macro_psnr"],
            "e200_macro_ssim": terminal["macro_ssim"],
            "e200_macro_lpips": terminal["macro_lpips"],
        })

    for key, entry in portfolio["external_baselines"].items():
        if not isinstance(entry, dict):
            raise RuntimeError(f"invalid external baseline entry: {key}")
        add(key=key, category="external_baseline", entry=entry)
    add(key="plain", category="plain_control", entry=portfolio["plain_control"])
    for key, method in portfolio["methods"].items():
        if not isinstance(method, dict) or not isinstance(method.get("result"), dict):
            raise RuntimeError(f"invalid algorithm entry: {key}")
        add(
            key=key, category="algorithm", entry=method["result"],
            algorithm_id=str(method.get("algorithm_id", key)),
            matched_plain=str(method.get("matched_plain", "")) or None,
        )
    order = {
        "input": 0, "cyclegan": 1, "cut": 2, "dclgan": 3, "plain": 4,
        "proposal": 5, "amtnc": 6, "stcgr": 7,
    }
    return sorted(rows, key=lambda row: (order.get(row["row_id"], 100), row["row_id"]))


def _algorithm_rows(
    portfolio: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trajectory_rows: list[dict[str, Any]] = []
    domain_rows: list[dict[str, Any]] = []
    for key in sorted(portfolio["methods"]):
        method = portfolio["methods"][key]
        result = method["result"]
        trajectory = result.get("late_trajectory")
        if not isinstance(trajectory, list) or [row.get("epoch") for row in trajectory] != [150, 175, 200]:
            raise RuntimeError(f"algorithm lacks the frozen late-three trajectory: {key}")
        for row in trajectory:
            epoch = int(row["epoch"])
            relation = row.get("runtime_relation") or {}
            trajectory_rows.append({
                "method_id": key,
                "algorithm_id": method.get("algorithm_id", key),
                "matched_plain": method.get("matched_plain"),
                "epoch": epoch,
                "macro_psnr_delta": _number(row.get("macro_psnr_delta"), label=f"{key}.e{epoch}.psnr_delta"),
                "macro_ssim_delta": _number(row.get("macro_ssim_delta"), label=f"{key}.e{epoch}.ssim_delta"),
                "macro_lpips_delta": _number(row.get("macro_lpips_delta"), label=f"{key}.e{epoch}.lpips_delta"),
                "candidate_macro_psnr": _number(row.get("candidate_macro_psnr"), label=f"{key}.e{epoch}.candidate_psnr"),
                "plain_macro_psnr": _number(row.get("plain_macro_psnr"), label=f"{key}.e{epoch}.plain_psnr"),
                "positive_domains": int(row.get("positive_domains", -1)),
                "worst_domain_delta": _number(row.get("worst_domain_delta"), label=f"{key}.e{epoch}.worst_domain"),
                "crn_exact": row.get("crn_exact") is True,
                "runtime_relation_status": relation.get("status"),
            })
            domains = row.get("domain_delta")
            if not isinstance(domains, dict) or len(domains) != 6:
                raise RuntimeError(f"algorithm lacks six-domain deltas: {key}.e{epoch}")
            for domain in sorted(domains):
                delta = domains[domain]
                if not isinstance(delta, dict):
                    raise RuntimeError(f"invalid domain delta: {key}.e{epoch}.{domain}")
                domain_rows.append({
                    "method_id": key,
                    "algorithm_id": method.get("algorithm_id", key),
                    "matched_plain": method.get("matched_plain"),
                    "epoch": epoch,
                    "domain": domain,
                    "psnr_delta": _number(delta.get("psnr"), label=f"{key}.e{epoch}.{domain}.psnr"),
                    "ssim_delta": _number(delta.get("ssim"), label=f"{key}.e{epoch}.{domain}.ssim"),
                    "lpips_delta": _number(delta.get("lpips"), label=f"{key}.e{epoch}.{domain}.lpips"),
                })
    return trajectory_rows, domain_rows


def _complexity_rows(portfolio: dict[str, Any]) -> list[dict[str, Any]]:
    plain = portfolio["complexity"].get("plain")
    if not isinstance(plain, dict):
        raise RuntimeError("complexity table lacks the controlled plain profile")
    plain_training = plain.get("training_step") or {}
    plain_median = _number(
        plain_training.get("median_ms"), label="plain.training_step.median_ms",
    )
    if plain_median is None or plain_median <= 0:
        raise RuntimeError("plain training-step median must be positive")
    rows = []
    for lane in sorted(portfolio["complexity"]):
        value = portfolio["complexity"][lane]
        if not isinstance(value, dict):
            raise RuntimeError(f"invalid complexity entry: {lane}")
        parameters = value.get("parameters") or {}
        training = value.get("training_step") or {}
        training_median = _number(
            training.get("median_ms"), label=f"{lane}.training_step.median_ms",
        )
        training_p90 = _number(
            training.get("p90_ms"), label=f"{lane}.training_step.p90_ms",
        )
        training_peak = training.get("peak_allocated_bytes")
        if (
            training_median is None or training_median <= 0
            or training_p90 is None or training_p90 <= 0
            or isinstance(training_peak, bool) or not isinstance(training_peak, int)
            or training_peak < 0
        ):
            raise RuntimeError(f"invalid controlled training-cost profile: {lane}")
        inference = value.get("inference") or {}
        nfe = inference.get("nfe") or {}
        reference_nfe = "5" if "5" in nfe else "1"
        reference = nfe.get(reference_nfe)
        if not isinstance(reference, dict):
            raise RuntimeError(f"complexity profile lacks reference inference NFE: {lane}")
        inference_median = _number(
            reference.get("median_ms"), label=f"{lane}.inference.nfe{reference_nfe}.median_ms",
        )
        inference_peak = reference.get("peak_allocated_bytes")
        if (
            inference_median is None or inference_median <= 0
            or isinstance(inference_peak, bool) or not isinstance(inference_peak, int)
            or inference_peak < 0
        ):
            raise RuntimeError(f"invalid controlled inference-cost profile: {lane}")
        rows.append({
            "lane_id": lane,
            "unique_parameters": parameters.get("unique_parameters"),
            "training_step_median_ms": training_median,
            "training_step_p90_ms": training_p90,
            "training_step_vs_plain": training_median / plain_median,
            "training_peak_allocated_bytes": training_peak,
            "inference_reference_nfe": int(reference_nfe),
            "inference_median_ms": inference_median,
            "inference_peak_allocated_bytes": inference_peak,
            "inference_json": json.dumps(inference, sort_keys=True, separators=(",", ":")),
            "flops_json": json.dumps(value.get("flops") or {}, sort_keys=True, separators=(",", ":")),
            "environment_json": json.dumps(value.get("environment") or {}, sort_keys=True, separators=(",", ":")),
        })
    return rows


def _csv_text(rows: list[dict[str, Any]], fields: Iterable[str]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _summary_markdown(
    main_rows: list[dict[str, Any]], trajectory_rows: list[dict[str, Any]],
    claims: list[str], portfolio_sha256: str,
) -> str:
    lines = [
        "# Frozen full-data paper result summary", "",
        f"Source portfolio SHA256: `{portfolio_sha256}`", "",
        "Primary checkpoint: e200. No best-checkpoint selection.", "",
        "## Main e200 macro table", "",
        "| Row | Paper label | Category | PSNR | SSIM | LPIPS | Scientific gate |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in main_rows:
        lpips = "" if row["e200_macro_lpips"] is None else f"{row['e200_macro_lpips']:.6f}"
        lines.append(
            f"| {row['row_id']} | {row['paper_label']} | {row['category']} | "
            f"{row['e200_macro_psnr']:.6f} | "
            f"{row['e200_macro_ssim']:.6f} | {lpips} | {row['scientific_gate'] or ''} |"
        )
    lines.extend(["", "## Reproduction and comparison scope", ""])
    lines.extend(
        f"- **{row['paper_label']}** (`{row['row_id']}`): "
        f"{row['reproduction_or_comparison_scope']}"
        for row in main_rows
    )
    lines.extend(["", "## Frozen sustained algorithm deltas", "",
                  "| Method | Epoch | PSNR delta | SSIM delta | LPIPS delta | Positive domains | Worst domain |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for row in trajectory_rows:
        lines.append(
            f"| {row['method_id']} | {row['epoch']} | {row['macro_psnr_delta']:.6f} | "
            f"{row['macro_ssim_delta']:.6f} | {row['macro_lpips_delta']:.6f} | "
            f"{row['positive_domains']} | {row['worst_domain_delta']:.6f} |"
        )
    lines.extend(["", "## Frozen paper claims", ""])
    lines.extend(f"- {claim}" for claim in claims)
    lines.extend(["", "These tables are descriptive post-freeze artifacts and cannot control training, scheduling or confirmation access.", ""])
    return "\n".join(lines)


def build_tables(
    *, portfolio: dict[str, Any], claims: list[str], portfolio_sha256: str,
) -> dict[str, str]:
    main = _main_rows(portfolio)
    trajectory, domains = _algorithm_rows(portfolio)
    complexity = _complexity_rows(portfolio)
    claim_rows = [{"claim_index": index + 1, "claim": claim} for index, claim in enumerate(claims)]
    return {
        "MAIN_E200.csv": _csv_text(main, (
            "row_id", "paper_label", "category", "lane_id", "algorithm_id",
            "comparison_scope", "matched_plain", "scientific_gate",
            "reproduction_or_comparison_scope", "e200_macro_psnr",
            "e200_macro_ssim", "e200_macro_lpips",
        )),
        "ALGORITHM_SUSTAINED.csv": _csv_text(trajectory, (
            "method_id", "algorithm_id", "matched_plain", "epoch",
            "macro_psnr_delta", "macro_ssim_delta", "macro_lpips_delta",
            "candidate_macro_psnr", "plain_macro_psnr", "positive_domains",
            "worst_domain_delta", "crn_exact", "runtime_relation_status",
        )),
        "ALGORITHM_DOMAIN_DELTAS.csv": _csv_text(domains, (
            "method_id", "algorithm_id", "matched_plain", "epoch", "domain",
            "psnr_delta", "ssim_delta", "lpips_delta",
        )),
        "COMPLEXITY.csv": _csv_text(complexity, (
            "lane_id", "unique_parameters", "training_step_median_ms",
            "training_step_p90_ms", "training_step_vs_plain",
            "training_peak_allocated_bytes", "inference_reference_nfe",
            "inference_median_ms", "inference_peak_allocated_bytes",
            "inference_json", "flops_json", "environment_json",
        )),
        "PAPER_CLAIMS.csv": _csv_text(claim_rows, ("claim_index", "claim")),
        "PAPER_RESULT_SUMMARY.md": _summary_markdown(main, trajectory, claims, portfolio_sha256),
    }


def _immutable_text(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"frozen manuscript artifact differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="")
    os.replace(temporary, path)


def run(
    *, freeze_receipt: Path, output: Path,
    branch_contract: Path = CANONICAL_BRANCH_CONTRACT,
) -> dict[str, Any]:
    script_commit, script_sha256 = _committed_script_identity()
    branch_contract = Path(branch_contract).resolve()
    if branch_contract != CANONICAL_BRANCH_CONTRACT.resolve():
        raise RuntimeError("manuscript exporter requires the canonical branch contract")
    contract_commit, contract_source_sha256 = _committed_file_identity(
        branch_contract, label="manuscript branching contract",
    )
    branch_contract_value = validate_branch_contract(_read(branch_contract))
    freeze_receipt = Path(freeze_receipt).resolve()
    freeze, freeze_commit = committed_freeze_identity(freeze_receipt, lane_id="input")
    portfolio_path = Path(freeze["source_portfolio_path"]).resolve()
    portfolio = validate_portfolio(
        _read(portfolio_path), expected_sha256=freeze["source_portfolio_sha256"],
        source_path=portfolio_path,
    )
    claims = freeze["paper_claims"]
    if not isinstance(claims, list) or not claims or any(not isinstance(x, str) or not x.strip() for x in claims):
        raise RuntimeError("committed freeze has an invalid claim set")
    tables = build_tables(
        portfolio=portfolio, claims=claims,
        portfolio_sha256=freeze["source_portfolio_sha256"],
    )
    branch = resolve_manuscript_branch(portfolio, branch_contract_value)
    branch.update({
        "branch_contract": str(branch_contract),
        "branch_contract_sha256": file_sha256(branch_contract),
        "branch_contract_git_commit": contract_commit,
        "branch_contract_source_sha256": contract_source_sha256,
        "source_portfolio_sha256": freeze["source_portfolio_sha256"],
        "freeze_receipt_sha256": file_sha256(freeze_receipt),
        "paper_claims_sha256": freeze["paper_claims_sha256"],
    })
    tables[BRANCH_FILE] = json.dumps(branch, ensure_ascii=False, indent=2) + "\n"
    if set(tables) != set(OUTPUT_FILES):
        raise RuntimeError("manuscript table set is incomplete")
    output = Path(output).resolve()
    for name in OUTPUT_FILES:
        _immutable_text(output / name, tables[name])
    receipt = {
        "schema": SCHEMA,
        "status": STATUS,
        "primary_epoch": 200,
        "sustained_epochs": [150, 175, 200],
        "source_portfolio": str(portfolio_path),
        "source_portfolio_sha256": freeze["source_portfolio_sha256"],
        "freeze_receipt": str(freeze_receipt),
        "freeze_receipt_sha256": file_sha256(freeze_receipt),
        "freeze_receipt_git_commit": freeze_commit,
        "paper_claims_sha256": freeze["paper_claims_sha256"],
        "manuscript_branch": branch,
        "exporter_git_commit": script_commit,
        "exporter_source_sha256": script_sha256,
        "outputs": {
            name: {"path": str((output / name).resolve()), "sha256": file_sha256(output / name)}
            for name in OUTPUT_FILES
        },
        "performance_values_read": True,
        "best_checkpoint_selection": False,
        "metric_values_used_for_training_or_scheduling": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }
    receipt_path = output / "MANUSCRIPT_TABLES_RECEIPT.json"
    text = json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    _immutable_text(receipt_path, text)
    return {**receipt, "receipt": str(receipt_path), "receipt_sha256": file_sha256(receipt_path)}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--freeze-receipt", type=Path, required=True)
    value.add_argument("--branch-contract", type=Path, default=CANONICAL_BRANCH_CONTRACT)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(run(
        freeze_receipt=args.freeze_receipt, branch_contract=args.branch_contract,
        output=args.output,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
