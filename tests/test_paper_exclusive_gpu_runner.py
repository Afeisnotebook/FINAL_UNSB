import os
from pathlib import Path

from operations.paper_aio_exclusive_gpu_runner import acquire_lock, file_sha256


def test_file_hash_is_content_bound(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"a")
    first = file_sha256(path)
    path.write_bytes(b"b")
    assert file_sha256(path) != first


def test_exclusive_lock_rejects_second_holder(tmp_path: Path) -> None:
    path = tmp_path / "gpu.lock"
    path.write_text("0", encoding="utf-8")
    with path.open("a+", encoding="utf-8") as first:
        acquire_lock(first)
        with path.open("a+", encoding="utf-8") as second:
            try:
                acquire_lock(second)
            except OSError:
                pass
            else:
                raise AssertionError("second GPU lock holder was accepted")


def test_exclusive_lock_covers_legacy_append_offset(tmp_path: Path) -> None:
    if os.name != "nt":
        return
    import msvcrt

    path = tmp_path / "gpu.lock"
    path.write_text("0", encoding="utf-8")
    with path.open("a+", encoding="utf-8") as first:
        acquire_lock(first)
        with path.open("a+", encoding="utf-8") as legacy:
            try:
                msvcrt.locking(legacy.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                pass
            else:
                raise AssertionError("legacy append-offset lock was accepted")
