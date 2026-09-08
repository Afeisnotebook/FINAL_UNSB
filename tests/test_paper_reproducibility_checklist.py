from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "PAPER_REPRODUCIBILITY_CHECKLIST_CONTRACT.json"
CHECKLIST = ROOT / "research" / "paper_aio" / "REPRODUCIBILITY_CHECKLIST_EN.md"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_reproducibility_checklist_and_authorities_are_hash_bound() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    bindings = [contract["checklist"], *contract["authorities"].values()]
    for binding in bindings:
        path = ROOT / binding["path"]
        assert path.is_file()
        assert _sha256(path) == binding["sha256"]


def test_reproducibility_checklist_matches_frozen_protocol() -> None:
    protocol = json.loads(
        (ROOT / "configs" / "PAPER_AIO_UNPAIRED_V1.json").read_text(
            encoding="utf-8"
        )
    )
    text = CHECKLIST.read_text(encoding="utf-8")
    training = protocol["training"]
    common = protocol["common"]
    unsb = protocol["unsb"]
    evaluation = protocol["evaluation"]

    assert "PRE-RESULT / NO EMPIRICAL CLAIM" in text
    assert protocol["manifest"]["sha256"] in text
    assert f"{training['steps_per_data_epoch']:,}" in text
    assert f"{training['target_updates']:,}" in text
    assert f"{training['target_data_epochs']} constant" in text
    assert f"seed {protocol['seed']}" in text
    assert f"{common['crop_size']}-by-{common['crop_size']}" in text
    assert f"\\tau={unsb['tau']}" in text
    assert f"temperature {unsb['nce_T']}" in text
    assert f"{unsb['num_patches']} patches" in text
    assert f"{evaluation['trajectory_discovery_per_domain']} discovery images per domain" in text
    assert f"{evaluation['terminal_discovery_per_domain']} discovery images per domain" in text
    assert "e150/e175/e200" in text
    assert "No best-checkpoint result is allowed" in text


def test_reproducibility_checklist_covers_release_and_scientific_boundaries() -> None:
    text = CHECKLIST.read_text(encoding="utf-8")
    required = [
        "Git commit",
        "official_image_proportional_unpaired",
        "full-state checkpoints",
        "runtime-relation registry",
        "single seed",
        "source-bound",
        "confirmation20",
        "DDSB remains `REPRODUCTION_INCOMPLETE`",
        "deleted interpreter inode",
        "Secrets, SSH passwords, and private host keys must never appear",
    ]
    for phrase in required:
        assert phrase in text

    for stage in ("preflight", "materialize", "resume-gate", "authorize", "train", "evaluate"):
        assert f"--stage {stage}" in text

    assert re.search(r"[+-]\d+(?:\.\d+)?\s*dB", text, re.IGNORECASE) is None
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert all(value is False for value in contract["hard_boundaries"].values())


def test_reproducibility_checklist_is_registered_in_delivery_authorities() -> None:
    matrix = json.loads(
        (ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json").read_text(
            encoding="utf-8"
        )
    )
    portfolio = json.loads(
        (ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json").read_text(
            encoding="utf-8"
        )
    )
    project = json.loads((ROOT / "PROJECT_STATE.json").read_text(encoding="utf-8"))
    matrix_item = next(
        item
        for item in matrix["nonblocking_extensions"]
        if item["id"] == "pre_result_reproducibility_checklist"
    )
    portfolio_item = portfolio["pre_result_reproducibility_checklist"]
    project_item = project["paper_aio_20260902"][
        "pre_result_reproducibility_checklist"
    ]

    for item in (matrix_item, portfolio_item, project_item):
        assert item["document"] == str(CHECKLIST.relative_to(ROOT)).replace("\\", "/")
        assert item["contract"] == str(CONTRACT.relative_to(ROOT)).replace("\\", "/")
        assert item["training_queue_changed"] is False
        assert item["performance_values_read"] is False
        assert item["confirmation20_opened"] is False
