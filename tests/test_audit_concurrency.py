from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from research.local_route1.causal_audit import append_unique_rows


ROOT = Path(__file__).resolve().parents[1]


def _rows(group: int, count: int = 8) -> list[dict]:
    return [{
        "row_id": f"row-{group}-{index}",
        "probe": "hj",
        "data_epoch": group * 20 + index,
        "source_state": "plain",
        "operator_mode": "registered",
        "branch_regime": "continuous_intervention",
        "horizon": 1,
    } for index in range(count)]


def test_threaded_atlas_merge_is_lossless(tmp_path):
    path = tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda group: append_unique_rows(path, _rows(group)), range(8)))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 64
    assert len({row["row_id"] for row in rows}) == 64


def test_process_atlas_merge_is_lossless(tmp_path):
    path = tmp_path / "audit" / "LONG_REVERSAL_ATLAS.jsonl"
    code = """
import sys
from pathlib import Path
from research.local_route1.causal_audit import append_unique_rows
group = int(sys.argv[2])
append_unique_rows(Path(sys.argv[1]), [{
    'row_id': f'process-{group}-{index}', 'probe': 'hnek',
    'data_epoch': group * 20 + index, 'source_state': 'plain',
    'operator_mode': 'registered', 'branch_regime': 'continuous_intervention',
    'horizon': 1,
} for index in range(12)])
"""
    processes = [
        subprocess.Popen([sys.executable, "-c", code, str(path), str(group)], cwd=ROOT)
        for group in range(2)
    ]
    assert [process.wait(timeout=60) for process in processes] == [0, 0]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 24
    assert len({row["row_id"] for row in rows}) == 24
