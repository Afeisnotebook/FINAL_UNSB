import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from operations import paper_aio_dclgan_portfolio_addendum_successor as addendum
from operations import paper_aio_dclgan_evaluation_successor as dcl_eval
from operations import paper_aio_final_delivery_successor as final
from research.paper_aio import runtime


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _base_portfolio() -> dict:
    return {
        "schema": final.PORTFOLIO_SCHEMA,
        "status": "COMPLETE_FULL_DATA_DISCOVERY_PORTFOLIO_AWAITING_CONFIRMATION_DECISION",
        "external_baselines": {"input": {}, "cut": {}, "cyclegan": {}},
        "complexity": {"plain": {}},
        "source_artifact_sha256": {"first_wave_results": "a" * 64},
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "metric_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "cross_non_equivalent_runtime_delta": False,
        "paired_metric_control": False,
        "confirmation20_opened": False,
    }


def _dclgan_result(tmp_path: Path) -> dict:
    receipts = []
    trajectory = []
    for epoch in dcl_eval.EPOCHS:
        receipt = _write(tmp_path / "receipts" / f"e{epoch}.json", {"epoch": epoch})
        metric = _write(tmp_path / "metrics" / f"e{epoch}.json", {"epoch": epoch})
        receipts.append({
            "epoch": epoch,
            "receipt": str(receipt.resolve()),
            "receipt_sha256": addendum.file_sha256(receipt),
            "metric": str(metric.resolve()),
            "metric_sha256": addendum.file_sha256(metric),
            "checkpoint_sha256": "c" * 64,
        })
        trajectory.append({"epoch": epoch, "macro_psnr": 20.0})
    return {
        "schema": dcl_eval.RESULT_SCHEMA,
        "status": "COMPLETE_FIXED_E200_EXTERNAL_BASELINE",
        "lane_id": "dclgan",
        "primary_epoch": 200,
        "fixed_epochs": list(dcl_eval.EPOCHS),
        "trajectory": trajectory,
        "terminal": trajectory[-1],
        "evaluation_receipts": receipts,
        "comparison_scope": "standalone_fixed_protocol_no_matched_delta_claim",
        "performance_values_used_for_training_or_scheduling": False,
        "best_checkpoint_selection": False,
        "paired_metric_control": False,
        "cross_non_equivalent_runtime_delta": False,
        "confirmation20_opened": False,
    }


def _complexity() -> dict:
    return {
        "lane_id": "dclgan",
        "checkpoint_sha256": "c" * 64,
        "environment": {},
        "parameters": {},
        "inference": {},
        "training_step": {},
        "flops": {"reported": False},
    }


def test_dclgan_optimizer_adapter_uses_one_unpaired_batch() -> None:
    class Stream:
        def __init__(self):
            self.calls = 0

        def next(self):
            self.calls += 1
            return {"A": self.calls, "B": -self.calls}

    class Model:
        def __init__(self):
            self.input = None
            self.optimized = 0

        def set_input(self, value):
            self.input = value

        def optimize_parameters(self):
            self.optimized += 1

    stream = Stream()
    model = Model()
    runtime.optimizer_step(model, SimpleNamespace(family="external", id="dclgan"), stream, stream)
    assert stream.calls == 1
    assert model.input == {"A": 1, "B": -1}
    assert model.optimized == 1


def test_ready_decisions_bind_completed_artifacts(tmp_path: Path) -> None:
    base_root = tmp_path / "base"
    portfolio = _write(base_root / "PAPER_ALGORITHM_PORTFOLIO.json", _base_portfolio())
    _write(base_root / "operations" / "FINAL_DELIVERY_STATE.json", {
        "schema": final.STATE_SCHEMA,
        "status": final.COMPLETE_STATUS,
        "portfolio": str(portfolio.resolve()),
        "portfolio_sha256": addendum.file_sha256(portfolio),
        "performance_values_in_control_state": False,
        "metric_values_used_for_training_or_scheduling": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "paper_claims_frozen": False,
        "confirmation_authorized": False,
        "confirmation20_opened": False,
    })
    dclgan_root = tmp_path / "dclgan"
    result = _write(dclgan_root / "DCLGAN_PAPER_RESULT.json", _dclgan_result(tmp_path))
    _write(dclgan_root / "operations" / "DCLGAN_EVALUATION_STATE.json", {
        "schema": dcl_eval.STATE_SCHEMA,
        "status": dcl_eval.COMPLETE_STATUS,
        "result": str(result.resolve()),
        "result_sha256": addendum.file_sha256(result),
        "performance_values_in_control_state": False,
        "paired_metric_control": False,
        "best_checkpoint_selection": False,
        "confirmation20_opened": False,
    })
    assert addendum.base_portfolio_decision(base_root) == "READY"
    assert addendum.dclgan_result_decision(dclgan_root) == "READY"
    result.write_text("{}", encoding="utf-8")
    assert addendum.dclgan_result_decision(dclgan_root) == "BLOCKED"


def test_augmented_portfolio_adds_dclgan_without_freezing_claims(tmp_path: Path) -> None:
    base = addendum.validate_base_portfolio(
        _write(tmp_path / "base.json", _base_portfolio())
    )
    dclgan = addendum.validate_dclgan_result(
        _write(tmp_path / "dclgan.json", _dclgan_result(tmp_path))
    )
    augmented, supplement = addendum.build_augmented_portfolio(
        base=base,
        dclgan=dclgan,
        complexity=_complexity(),
        source_hashes={"base_portfolio": "b" * 64},
    )
    assert augmented["external_baselines"]["dclgan"]["terminal"]["epoch"] == 200
    assert augmented["complexity"]["dclgan"]["lane_id"] == "dclgan"
    assert augmented["dclgan_is_nonblocking_post_core_addendum"] is True
    assert augmented["paper_claims_frozen"] is False
    assert supplement["core_paper_delivery_was_not_blocked"] is True
    assert supplement["confirmation20_opened"] is False


def test_dclgan_result_rejects_best_checkpoint_selection(tmp_path: Path) -> None:
    value = _dclgan_result(tmp_path)
    value["best_checkpoint_selection"] = True
    with pytest.raises(RuntimeError, match="invalid fixed DCLGAN"):
        addendum.validate_dclgan_result(_write(tmp_path / "bad.json", value))
