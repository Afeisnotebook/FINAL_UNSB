"""Render deterministic post-freeze manuscript figures from frozen tables.

The renderer consumes only ``MANUSCRIPT_TABLES_RECEIPT.json`` and the table
files hash-bound by that receipt.  It cannot train, schedule, choose a best
checkpoint, or authorize confirmation access.  The two SVG outputs show the
pre-registered e150/e175/e200 macro-PSNR deltas and the e200 six-domain PSNR
deltas without ranking or hiding a failed method.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

from research.paper_aio.protocol import ROOT, file_sha256, portable_source_sha256
from research.paper_aio.runtime_relation import PASS_STATUSES


TABLE_RECEIPT_SCHEMA = "final-unsb-paper-manuscript-table-receipt-v2"
TABLE_RECEIPT_STATUS = "COMPLETE_FROZEN_E200_MANUSCRIPT_TABLES"
SCHEMA = "final-unsb-paper-manuscript-figure-receipt-v1"
STATUS = "COMPLETE_FROZEN_E200_MANUSCRIPT_FIGURES"
TRAJECTORY_FILE = "ALGORITHM_SUSTAINED_PSNR.svg"
DOMAIN_FILE = "ALGORITHM_E200_DOMAIN_PSNR.svg"
OUTPUT_FILES = (TRAJECTORY_FILE, DOMAIN_FILE)
SOURCE_TABLES = ("ALGORITHM_SUSTAINED.csv", "ALGORITHM_DOMAIN_DELTAS.csv")
EPOCHS = (150, 175, 200)
METHOD_COLORS = {
    "proposal": "#4C78A8",
    "stcgr": "#F58518",
    "amtnc": "#54A24B",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _committed_script_identity() -> tuple[str, str]:
    path = Path(__file__).resolve()
    relative = path.relative_to(ROOT.resolve()).as_posix()
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--", relative], cwd=ROOT, text=True,
    ).strip()
    if status:
        raise RuntimeError("manuscript figure renderer has uncommitted changes")
    commit = subprocess.check_output(
        ["git", "log", "-1", "--format=%H", "--", relative], cwd=ROOT,
        text=True,
    ).strip()
    if len(commit) != 40:
        raise RuntimeError("manuscript figure renderer has no committed Git identity")
    committed = subprocess.check_output(
        ["git", "show", f"{commit}:{relative}"], cwd=ROOT,
    ).replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    committed_sha256 = hashlib.sha256(committed).hexdigest()
    if committed_sha256 != portable_source_sha256(path):
        raise RuntimeError("working manuscript figure renderer differs from its Git blob")
    return commit, committed_sha256


def _finite(value: str, *, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid numeric figure field: {label}") from error
    if not math.isfinite(result):
        raise RuntimeError(f"non-finite figure field: {label}")
    return result


def _integer(value: str, *, label: str) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"invalid integer figure field: {label}") from error
    return result


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"frozen figure source table is empty: {path.name}")
    return rows


def validate_table_receipt(
    value: dict[str, Any], *, receipt_path: Path,
) -> dict[str, Path]:
    if (
        value.get("schema") != TABLE_RECEIPT_SCHEMA
        or value.get("status") != TABLE_RECEIPT_STATUS
        or int(value.get("primary_epoch", -1)) != 200
        or value.get("sustained_epochs") != list(EPOCHS)
        or value.get("performance_values_read") is not True
        or value.get("best_checkpoint_selection") is not False
        or value.get("metric_values_used_for_training_or_scheduling") is not False
        or value.get("cross_non_equivalent_runtime_delta") is not False
        or value.get("paired_metric_control") is not False
        or value.get("confirmation_authorized") is not False
        or value.get("confirmation20_opened") is not False
    ):
        raise RuntimeError("manuscript table receipt is incomplete or unsafe")
    outputs = value.get("outputs")
    if not isinstance(outputs, dict):
        raise RuntimeError("manuscript table receipt lacks outputs")
    resolved: dict[str, Path] = {}
    for name in SOURCE_TABLES:
        entry = outputs.get(name)
        if not isinstance(entry, dict):
            raise RuntimeError(f"manuscript table receipt lacks figure source: {name}")
        path = Path(str(entry.get("path", ""))).resolve()
        digest = entry.get("sha256")
        if not path.is_file() or not isinstance(digest, str) or len(digest) != 64:
            raise RuntimeError(f"invalid frozen figure source: {name}")
        if file_sha256(path) != digest:
            raise RuntimeError(f"frozen figure source hash changed: {name}")
        resolved[name] = path
    if not Path(receipt_path).is_file():
        raise RuntimeError("manuscript table receipt is absent")
    return resolved


def _trajectory_data(rows: list[dict[str, str]]) -> dict[str, list[dict[str, Any]]]:
    required = {
        "method_id", "algorithm_id", "matched_plain", "epoch",
        "macro_psnr_delta", "crn_exact", "runtime_relation_status",
    }
    if not required.issubset(rows[0]):
        raise RuntimeError("sustained table lacks required figure columns")
    methods: dict[str, list[dict[str, Any]]] = {}
    seen: set[tuple[str, int]] = set()
    for row in rows:
        method = row["method_id"].strip()
        epoch = _integer(row["epoch"], label=f"{method}.epoch")
        key = (method, epoch)
        if (
            not method or key in seen or not row["matched_plain"].strip()
            or row["crn_exact"].strip().lower() != "true"
            or row["runtime_relation_status"].strip() not in PASS_STATUSES
        ):
            raise RuntimeError(f"unsafe or duplicate sustained row: {method}.e{epoch}")
        seen.add(key)
        methods.setdefault(method, []).append({
            "epoch": epoch,
            "psnr_delta": _finite(
                row["macro_psnr_delta"], label=f"{method}.e{epoch}.psnr_delta",
            ),
        })
    if not methods:
        raise RuntimeError("sustained table has no algorithms")
    for method, values in methods.items():
        values.sort(key=lambda item: item["epoch"])
        if tuple(item["epoch"] for item in values) != EPOCHS:
            raise RuntimeError(f"algorithm lacks exact late-three figure rows: {method}")
    return dict(sorted(methods.items()))


def _domain_data(
    rows: list[dict[str, str]], *, methods: set[str],
) -> tuple[list[str], dict[str, dict[str, float]]]:
    required = {"method_id", "epoch", "domain", "psnr_delta", "matched_plain"}
    if not required.issubset(rows[0]):
        raise RuntimeError("domain table lacks required figure columns")
    values: dict[str, dict[str, float]] = {method: {} for method in methods}
    all_domains: set[str] = set()
    seen_e200: set[tuple[str, str]] = set()
    observed_methods: set[str] = set()
    for row in rows:
        method = row["method_id"].strip()
        epoch = _integer(row["epoch"], label=f"{method}.domain_epoch")
        if method not in methods:
            raise RuntimeError(f"domain table contains an unknown method: {method}")
        observed_methods.add(method)
        if epoch != 200:
            continue
        domain = row["domain"].strip()
        key = (method, domain)
        if not domain or not row["matched_plain"].strip() or key in seen_e200:
            raise RuntimeError(f"unsafe or duplicate e200 domain row: {method}.{domain}")
        seen_e200.add(key)
        all_domains.add(domain)
        values[method][domain] = _finite(
            row["psnr_delta"], label=f"{method}.e200.{domain}.psnr_delta",
        )
    if observed_methods != methods:
        raise RuntimeError("domain and sustained method sets differ")
    domains = sorted(all_domains)
    if len(domains) != 6:
        raise RuntimeError("e200 figure requires exactly six domains")
    for method in methods:
        if set(values[method]) != set(domains):
            raise RuntimeError(f"algorithm lacks six e200 domain rows: {method}")
    return domains, values


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _method_color(method: str, index: int) -> str:
    fallback = ("#B279A2", "#E45756", "#72B7B2", "#9D755D", "#BAB0AC")
    return METHOD_COLORS.get(method, fallback[index % len(fallback)])


def render_trajectory_svg(methods: dict[str, list[dict[str, Any]]]) -> str:
    width, height = 960, 560
    left, right, top, bottom = 100, 40, 62, 92
    plot_w, plot_h = width - left - right, height - top - bottom
    all_y = [0.0] + [row["psnr_delta"] for rows in methods.values() for row in rows]
    low, high = min(all_y), max(all_y)
    span = high - low
    pad = 0.12 * span if span > 0 else 0.1
    low, high = low - pad, high + pad

    def x(epoch: int) -> float:
        return left + (epoch - EPOCHS[0]) / (EPOCHS[-1] - EPOCHS[0]) * plot_w

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Late-horizon macro PSNR delta</title>',
        '<desc id="desc">Matched macro PSNR deltas at fixed data epochs 150, 175 and 200. No best checkpoint selection.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="100" y="31" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#222">Late-horizon matched PSNR</text>',
        '<text x="100" y="51" font-family="Arial,sans-serif" font-size="12" fill="#555">Fixed e150 / e175 / e200; higher is better; zero means matched plain</text>',
    ]
    for index in range(6):
        value = low + index * (high - low) / 5
        yy = y(value)
        lines.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{left + plot_w}" y2="{yy:.2f}" stroke="#e6e6e6" stroke-width="1"/>')
        lines.append(f'<text x="{left - 12}" y="{yy + 4:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="12" fill="#555">{value:+.3f}</text>')
    zero_y = y(0.0)
    lines.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{left + plot_w}" y2="{zero_y:.2f}" stroke="#333" stroke-width="1.5" stroke-dasharray="6 5"/>')
    for epoch in EPOCHS:
        xx = x(epoch)
        lines.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{top + plot_h}" stroke="#f1f1f1" stroke-width="1"/>')
        lines.append(f'<text x="{xx:.2f}" y="{top + plot_h + 28}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13" fill="#333">e{epoch}</text>')
    for index, (method, rows) in enumerate(methods.items()):
        color = _method_color(method, index)
        points = " ".join(f'{x(row["epoch"]):.2f},{y(row["psnr_delta"]):.2f}' for row in rows)
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"/>')
        for row in rows:
            xx, yy = x(row["epoch"]), y(row["psnr_delta"])
            lines.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5" fill="#fff" stroke="{color}" stroke-width="3"/>')
        legend_x = left + index * 205
        legend_y = height - 27
        lines.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        lines.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-family="Arial,sans-serif" font-size="13" fill="#222">{_escape(method)}</text>')
    lines.extend([
        f'<text x="24" y="{top + plot_h / 2:.2f}" transform="rotate(-90 24 {top + plot_h / 2:.2f})" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#333">Macro PSNR delta (dB)</text>',
        f'<text x="{left + plot_w / 2:.2f}" y="{height - 54}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="#333">Data epoch</text>',
        '</svg>',
    ])
    return "\n".join(lines) + "\n"


def _heat_color(value: float, maximum: float) -> str:
    ratio = min(1.0, abs(value) / maximum) if maximum > 0 else 0.0
    target = (55, 126, 184) if value >= 0 else (214, 39, 40)
    rgb = tuple(round(247 + ratio * (component - 247)) for component in target)
    return "#" + "".join(f"{component:02x}" for component in rgb)


def render_domain_svg(
    domains: list[str], values: dict[str, dict[str, float]],
) -> str:
    methods = sorted(values)
    width = 1040
    left, top, cell_w, cell_h = 190, 112, 128, 82
    right, bottom = 72, 92
    height = top + len(methods) * cell_h + bottom
    maximum = max(abs(value) for row in values.values() for value in row.values())
    if maximum == 0:
        maximum = 1.0
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">E200 six-domain PSNR delta</title>',
        '<desc id="desc">Matched per-domain PSNR deltas at the fixed e200 checkpoint. Blue is positive and red is negative.</desc>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="56" y="34" font-family="Arial,sans-serif" font-size="21" font-weight="700" fill="#222">E200 domain coverage</text>',
        '<text x="56" y="57" font-family="Arial,sans-serif" font-size="12" fill="#555">Matched PSNR delta (dB); blue positive, red negative; fixed checkpoint only</text>',
    ]
    for column, domain in enumerate(domains):
        xx = left + column * cell_w + cell_w / 2
        lines.append(f'<text x="{xx:.2f}" y="{top - 18}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" fill="#333">{_escape(domain)}</text>')
    for row_index, method in enumerate(methods):
        yy = top + row_index * cell_h
        lines.append(f'<text x="{left - 18}" y="{yy + cell_h / 2 + 5:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="14" font-weight="700" fill="#222">{_escape(method)}</text>')
        for column, domain in enumerate(domains):
            value = values[method][domain]
            xx = left + column * cell_w
            fill = _heat_color(value, maximum)
            text_fill = "#ffffff" if abs(value) / maximum > 0.62 else "#222222"
            lines.append(f'<rect x="{xx + 2}" y="{yy + 2}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{fill}"/>')
            lines.append(f'<text x="{xx + cell_w / 2:.2f}" y="{yy + cell_h / 2 + 5:.2f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="14" fill="{text_fill}">{value:+.3f}</text>')
    legend_y = height - 36
    legend_x = left
    for index, value in enumerate((-maximum, -maximum / 2, 0.0, maximum / 2, maximum)):
        xx = legend_x + index * 116
        lines.append(f'<rect x="{xx}" y="{legend_y - 13}" width="28" height="18" fill="{_heat_color(value, maximum)}" stroke="#ddd"/>')
        lines.append(f'<text x="{xx + 34}" y="{legend_y + 1}" font-family="Arial,sans-serif" font-size="11" fill="#444">{value:+.3f}</text>')
    lines.append('</svg>')
    return "\n".join(lines) + "\n"


def build_figures(*, sustained: Path, domains: Path) -> dict[str, str]:
    trajectory = _trajectory_data(_csv_rows(sustained))
    domain_names, domain_values = _domain_data(
        _csv_rows(domains), methods=set(trajectory),
    )
    return {
        TRAJECTORY_FILE: render_trajectory_svg(trajectory),
        DOMAIN_FILE: render_domain_svg(domain_names, domain_values),
    }


def _immutable_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != value:
            raise RuntimeError(f"frozen manuscript figure differs: {path}")
        return
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(value, encoding="utf-8", newline="")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(*, table_receipt: Path, output: Path) -> dict[str, Any]:
    script_commit, script_sha256 = _committed_script_identity()
    table_receipt = Path(table_receipt).resolve()
    table_receipt_value = _read_json(table_receipt)
    sources = validate_table_receipt(
        table_receipt_value, receipt_path=table_receipt,
    )
    figures = build_figures(
        sustained=sources["ALGORITHM_SUSTAINED.csv"],
        domains=sources["ALGORITHM_DOMAIN_DELTAS.csv"],
    )
    output = Path(output).resolve()
    for name in OUTPUT_FILES:
        _immutable_text(output / name, figures[name])
    receipt = {
        "schema": SCHEMA,
        "status": STATUS,
        "primary_epoch": 200,
        "sustained_epochs": list(EPOCHS),
        "table_receipt": str(table_receipt),
        "table_receipt_sha256": file_sha256(table_receipt),
        "source_tables": {
            name: {
                "path": str(sources[name]),
                "sha256": file_sha256(sources[name]),
            }
            for name in SOURCE_TABLES
        },
        "renderer_git_commit": script_commit,
        "renderer_source_sha256": script_sha256,
        "outputs": {
            name: {
                "path": str((output / name).resolve()),
                "sha256": file_sha256(output / name),
            }
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
    receipt_path = output / "MANUSCRIPT_FIGURES_RECEIPT.json"
    _immutable_text(receipt_path, json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    return {
        **receipt,
        "receipt": str(receipt_path),
        "receipt_sha256": file_sha256(receipt_path),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--table-receipt", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main() -> int:
    args = parser().parse_args()
    print(json.dumps(run(
        table_receipt=args.table_receipt, output=args.output,
    ), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
