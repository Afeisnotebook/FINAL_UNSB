import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_JSON = (
    ROOT / "PROJECT_STATE.json",
    ROOT / "configs" / "FULL_DATA_METHOD_PORTFOLIO.json",
    ROOT / "configs" / "PAPER_DELIVERY_COMPLETION_MATRIX.json",
)


def _reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


@pytest.mark.parametrize("path", AUTHORITY_JSON, ids=lambda path: path.name)
def test_authority_json_has_unique_keys(path: Path) -> None:
    json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
