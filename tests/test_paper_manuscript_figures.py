from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import pytest

from operations import paper_aio_manuscript_figures as figures
from research.paper_aio.protocol import file_sha256


METHODS = ("proposal", "amtnc", "stcgr")
DOMAINS = ("d1", "d2", "d3", "d4", "d5", "d6")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(stream.getvalue(), encoding="utf-8")


def _sources(tmp_path: Path) -> tuple[Path, Path]:
    sustained = tmp_path / "ALGORITHM_SUSTAINED.csv"
    domain = tmp_path / "ALGORITHM_DOMAIN_DELTAS.csv"
    _write_csv(sustained, [
        {
            "method_id": method,
            "algorithm_id": f"algorithm-{method}",
            "matched_plain": "host/plain",
            "epoch": epoch,
            "macro_psnr_delta": index * 0.1 + (epoch - 175) / 1000,
            "macro_ssim_delta": 0.01,
            "macro_lpips_delta": -0.01,
            "candidate_macro_psnr": 20.1,
            "plain_macro_psnr": 20.0,
            "positive_domains": 4,
            "worst_domain_delta": -0.1,
            "crn_exact": True,
            "runtime_relation_status": "PASS_EXACT_CROSS_HOST_RUNTIME_RELATION",
        }
        for index, method in enumerate(METHODS)
        for epoch in figures.EPOCHS
    ])
    _write_csv(domain, [
        {
            "method_id": method,
            "algorithm_id": f"algorithm-{method}",
            "matched_plain": "host/plain",
            "epoch": epoch,
            "domain": name,
            "psnr_delta": (column - 2.5) / 10,
            "ssim_delta": 0.01,
            "lpips_delta": -0.01,
        }
        for method in METHODS
        for epoch in figures.EPOCHS
        for column, name in enumerate(DOMAINS)
    ])
    return sustained, domain


def _receipt(path: Path, sustained: Path, domain: Path) -> dict:
    path.write_text("{}\n", encoding="utf-8")
    return {
        "schema": figures.TABLE_RECEIPT_SCHEMA,
        "status": figures.TABLE_RECEIPT_STATUS,
        "primary_epoch": 200,
        "sustained_epochs": [150, 175, 200],
        "outputs": {
            sustained.name: {"path": str(sustained), "sha256": file_sha256(sustained)},
            domain.name: {"path": str(domain), "sha256": file_sha256(domain)},
        },
        "performance_values_read": True,
        "best_checkpoint_selection": False,
        "metric_values_used_for_training_or_scheduling": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    }


def test_builds_deterministic_late_trajectory_and_domain_svgs(tmp_path: Path) -> None:
    sustained, domain = _sources(tmp_path)
    result = figures.build_figures(sustained=sustained, domains=domain)
    assert set(result) == set(figures.OUTPUT_FILES)
    trajectory = result[figures.TRAJECTORY_FILE]
    heatmap = result[figures.DOMAIN_FILE]
    assert trajectory.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert "e150" in trajectory and "e175" in trajectory and "e200" in trajectory
    assert all(method in trajectory for method in METHODS)
    assert "No best checkpoint selection" in trajectory
    assert heatmap.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert all(domain_name in heatmap for domain_name in DOMAINS)
    assert all(method in heatmap for method in METHODS)
    assert result == figures.build_figures(sustained=sustained, domains=domain)


def test_rejects_missing_late_epoch_or_domain(tmp_path: Path) -> None:
    sustained, domain = _sources(tmp_path)
    rows = list(csv.DictReader(sustained.read_text(encoding="utf-8").splitlines()))
    _write_csv(sustained, [
        row for row in rows
        if not (row["method_id"] == "proposal" and row["epoch"] == "175")
    ])
    with pytest.raises(RuntimeError, match="exact late-three"):
        figures.build_figures(sustained=sustained, domains=domain)

    sustained, domain = _sources(tmp_path)
    rows = list(csv.DictReader(domain.read_text(encoding="utf-8").splitlines()))
    _write_csv(domain, [
        row for row in rows
        if not (
            row["method_id"] == "proposal" and row["epoch"] == "200"
            and row["domain"] == "d6"
        )
    ])
    with pytest.raises(RuntimeError, match="six e200 domain"):
        figures.build_figures(sustained=sustained, domains=domain)


def test_rejects_non_exact_runtime_relation(tmp_path: Path) -> None:
    sustained, domain = _sources(tmp_path)
    rows = list(csv.DictReader(sustained.read_text(encoding="utf-8").splitlines()))
    rows[0]["runtime_relation_status"] = "WAITING_FOR_EXACT_RUNTIME_RELATION"
    _write_csv(sustained, rows)
    with pytest.raises(RuntimeError, match="unsafe or duplicate"):
        figures.build_figures(sustained=sustained, domains=domain)


def test_receipt_rejects_drift_and_forbidden_selection(tmp_path: Path) -> None:
    sustained, domain = _sources(tmp_path)
    receipt_path = tmp_path / "MANUSCRIPT_TABLES_RECEIPT.json"
    receipt = _receipt(receipt_path, sustained, domain)
    resolved = figures.validate_table_receipt(receipt, receipt_path=receipt_path)
    assert resolved[sustained.name] == sustained.resolve()

    sustained.write_text(sustained.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash changed"):
        figures.validate_table_receipt(receipt, receipt_path=receipt_path)

    sustained, domain = _sources(tmp_path)
    receipt = _receipt(receipt_path, sustained, domain)
    receipt["best_checkpoint_selection"] = True
    with pytest.raises(RuntimeError, match="incomplete or unsafe"):
        figures.validate_table_receipt(receipt, receipt_path=receipt_path)


def test_immutable_figure_refuses_drift(tmp_path: Path) -> None:
    path = tmp_path / "figure.svg"
    figures._immutable_text(path, "a\n")
    figures._immutable_text(path, "a\n")
    with pytest.raises(RuntimeError, match="differs"):
        figures._immutable_text(path, "b\n")


def test_run_hash_binds_both_figures_without_opening_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    sustained, domain = _sources(tmp_path)
    receipt_path = tmp_path / "MANUSCRIPT_TABLES_RECEIPT.json"
    receipt = _receipt(receipt_path, sustained, domain)
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        figures, "_committed_script_identity", lambda: ("a" * 40, "b" * 64),
    )
    output = tmp_path / "figures"
    result = figures.run(table_receipt=receipt_path, output=output)
    assert result["status"] == figures.STATUS
    assert result["best_checkpoint_selection"] is False
    assert result["confirmation20_opened"] is False
    assert set(result["outputs"]) == set(figures.OUTPUT_FILES)
    for name, value in result["outputs"].items():
        assert Path(value["path"]).is_file()
        assert file_sha256(Path(value["path"])) == value["sha256"]
    assert Path(result["receipt"]).is_file()
