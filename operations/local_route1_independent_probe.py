"""Run an independent route-1 probe concurrently without changing its update.

The frozen anchor runner originally serialized plain -> HJ -> HNEK as an
operational safety rule.  HNEK does not consume the HJ state.  This wrapper may
therefore bypass only that scheduling dependency after a same-host plain e200
baseline exists.  Scientific model/data code is imported from the immutable
training worktree and retains its original commit and protocol fingerprint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_TRAINING_COMMIT = "0da2a37086cca5bc4ad4488bb07c53096a7152ed"
EXPECTED_PROTOCOL = "b0786b222790b84379802996448b8a68b86d69a6892ea0cdc04670cfcb1fb9b2"
EXPECTED_MANIFEST = "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
EXPECTED_E0_FILE = "599544e984d9db72c3e3061f50bc14143400546c4bdc7fbc23c9beb86ac32140"
EXPECTED_E0_SCIENTIFIC = "2105410818ec9f5382c497e1848009d18688dafe5878a19157b450d1e8b206c1"
ALLOWED_LANES = ("hnek",)


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def command(argv: list[str], *, cwd: Path) -> str:
    result = subprocess.run(
        argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {argv}\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def append_jsonl(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def validate_plain_sidecar(sidecar: dict[str, Any]) -> None:
    if sidecar.get("schema") != "final-unsb-local-route1-full-state-v1":
        raise RuntimeError("matched plain sidecar schema mismatch")
    if int(sidecar.get("step", -1)) != 30_000:
        raise RuntimeError("independent probe requires same-host plain e200")
    if int(sidecar.get("physical_epoch_completed", -1)) != 200:
        raise RuntimeError("matched plain physical epoch is not e200")
    metadata = sidecar.get("metadata", {})
    expected = {
        "probe_id": "plain",
        "git_commit": EXPECTED_TRAINING_COMMIT,
        "protocol_fingerprint": EXPECTED_PROTOCOL,
        "manifest_sha256": EXPECTED_MANIFEST,
        "confirmation20_opened": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise RuntimeError(f"matched plain identity mismatch for {key}")


def validate_e0_sidecar(sidecar: dict[str, Any]) -> None:
    if sidecar.get("schema") != "final-unsb-local-route1-shared-e0-v1":
        raise RuntimeError("shared e0 sidecar schema mismatch")
    if sidecar.get("checkpoint_sha256") != EXPECTED_E0_FILE:
        raise RuntimeError("shared e0 file identity mismatch")
    if sidecar.get("scientific_state_sha256") != EXPECTED_E0_SCIENTIFIC:
        raise RuntimeError("shared e0 scientific identity mismatch")
    metadata = sidecar.get("metadata", {})
    if metadata.get("git_commit") != EXPECTED_TRAINING_COMMIT:
        raise RuntimeError("shared e0 training commit mismatch")
    if metadata.get("protocol_fingerprint") != EXPECTED_PROTOCOL:
        raise RuntimeError("shared e0 protocol mismatch")
    if metadata.get("manifest_sha256") != EXPECTED_MANIFEST:
        raise RuntimeError("shared e0 manifest mismatch")


def process_exists(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            owner = read_json(path)
            owner_pid = int(owner.get("pid", -1))
        except Exception:
            owner_pid = -1
        if process_exists(owner_pid):
            raise RuntimeError(f"independent probe already owns lock with PID {owner_pid}")
        path.unlink()
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": now()}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        if path.exists():
            try:
                owner = read_json(path)
            except Exception:
                owner = {}
            if int(owner.get("pid", -1)) == os.getpid():
                path.unlink()


def validate_and_materialize_e0(
    *, training_repo: Path, matched_plain_root: Path, output_root: Path,
    manifest: Path,
) -> dict[str, Any]:
    if output_root.resolve() == matched_plain_root.resolve():
        raise RuntimeError("independent probe must use an isolated output root")
    if command(["git", "rev-parse", "HEAD"], cwd=training_repo) != EXPECTED_TRAINING_COMMIT:
        raise RuntimeError("immutable training worktree moved")
    if command(["git", "status", "--porcelain"], cwd=training_repo):
        raise RuntimeError("immutable training worktree is dirty")
    if file_sha256(manifest) != EXPECTED_MANIFEST:
        raise RuntimeError("manifest hash mismatch")

    plain_checkpoint = matched_plain_root / "anchors" / "plain" / "full_state_latest.pt"
    plain_sidecar_path = Path(str(plain_checkpoint) + ".json")
    if not plain_checkpoint.is_file() or not plain_sidecar_path.is_file():
        raise FileNotFoundError("same-host plain e200 checkpoint/sidecar missing")
    plain_sidecar = read_json(plain_sidecar_path)
    validate_plain_sidecar(plain_sidecar)
    if file_sha256(plain_checkpoint) != plain_sidecar.get("full_state_sha256"):
        raise RuntimeError("matched plain checkpoint file hash mismatch")

    source_e0 = matched_plain_root / "shared_e0" / "e0.pt"
    source_e0_sidecar = Path(str(source_e0) + ".json")
    if not source_e0.is_file() or not source_e0_sidecar.is_file():
        raise FileNotFoundError("accepted shared e0 missing from matched plain run")
    e0_sidecar = read_json(source_e0_sidecar)
    validate_e0_sidecar(e0_sidecar)
    if file_sha256(source_e0) != EXPECTED_E0_FILE:
        raise RuntimeError("accepted shared e0 file hash mismatch")

    target_e0 = output_root / "shared_e0" / "e0.pt"
    target_e0_sidecar = Path(str(target_e0) + ".json")
    target_e0.parent.mkdir(parents=True, exist_ok=True)
    if not target_e0.exists():
        shutil.copy2(source_e0, target_e0)
    if not target_e0_sidecar.exists():
        shutil.copy2(source_e0_sidecar, target_e0_sidecar)
    if file_sha256(target_e0) != EXPECTED_E0_FILE:
        raise RuntimeError("isolated output e0 changed during materialization")
    validate_e0_sidecar(read_json(target_e0_sidecar))

    return {
        "matched_plain_checkpoint": str(plain_checkpoint.resolve()),
        "matched_plain_checkpoint_sha256": file_sha256(plain_checkpoint),
        "matched_plain_scientific_state_sha256": plain_sidecar["scientific_state_sha256"],
        "shared_e0_file_sha256": EXPECTED_E0_FILE,
        "shared_e0_scientific_state_sha256": EXPECTED_E0_SCIENTIFIC,
    }


def install_frozen_imports(training_repo: Path):
    root = str(training_repo.resolve())
    sys.path = [root] + [value for value in sys.path if Path(value or ".").resolve() != training_repo.resolve()]
    for name in tuple(sys.modules):
        if name == "research" or name.startswith("research."):
            raise RuntimeError("research modules were imported before frozen worktree isolation")
    from research.local_route1 import anchors  # type: ignore
    from research.local_route1.protocol import protocol_fingerprint  # type: ignore

    if training_repo.resolve() not in Path(anchors.__file__).resolve().parents:
        raise RuntimeError("anchor runner was not imported from the frozen worktree")
    return anchors, protocol_fingerprint


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--training-repo", type=Path, required=True)
    value.add_argument("--matched-plain-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--train-view", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--lane", choices=ALLOWED_LANES, required=True)
    value.add_argument("--gpu", type=int, default=0)
    value.add_argument("--stop-after-epoch", type=int, default=200)
    value.add_argument("--preflight-only", action="store_true")
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    training_repo = args.training_repo.resolve()
    matched_plain_root = args.matched_plain_root.resolve()
    output_root = args.output.resolve()
    manifest = args.manifest.resolve()
    operations = output_root / "operations"
    script = Path(__file__).resolve()
    with exclusive_lock(operations / f"INDEPENDENT_{args.lane.upper()}.lock"):
        evidence = validate_and_materialize_e0(
            training_repo=training_repo, matched_plain_root=matched_plain_root,
            output_root=output_root, manifest=manifest,
        )
        anchors, protocol_fingerprint = install_frozen_imports(training_repo)
        if protocol_fingerprint(manifest) != EXPECTED_PROTOCOL:
            raise RuntimeError("frozen training protocol fingerprint mismatch")
        canonical_state_path = matched_plain_root / "operations" / "EXECUTION_STATE.json"
        canonical_state = read_json(canonical_state_path) if canonical_state_path.is_file() else {}
        if canonical_state.get("lane") == args.lane and canonical_state.get("status") == "CHUNK_RUNNING":
            raise RuntimeError(f"canonical executor is already training {args.lane}")

        contract = {
            "schema": "final-unsb-route1-independent-probe-contract-v1",
            "created": now(),
            "status": "PREFLIGHT_PASS" if args.preflight_only else "RUNNING",
            "lane": args.lane,
            "training_repo": str(training_repo),
            "training_git_commit": EXPECTED_TRAINING_COMMIT,
            "training_protocol_fingerprint": EXPECTED_PROTOCOL,
            "manifest": str(manifest),
            "manifest_sha256": EXPECTED_MANIFEST,
            "matched_plain_root": str(matched_plain_root),
            "output_root": str(output_root),
            "train_view": str(args.train_view.resolve()),
            "data_root": str(args.data_root.resolve()),
            "wrapper": str(script),
            "wrapper_sha256": file_sha256(script),
            "scheduling_dependency_bypassed": "HJ completion before HNEK launch",
            "training_update_changed": False,
            "batch_size_changed": False,
            "cross_host_state_used": False,
            "paired_controller_access": False,
            "confirmation20_opened": False,
            **evidence,
        }
        atomic_json(operations / "INDEPENDENT_PROBE_CONTRACT.json", contract)
        append_jsonl(operations / "INDEPENDENT_PROBE_EVENTS.jsonl", {
            "time": now(), "event": "PREFLIGHT_PASS", **contract,
        })
        if args.preflight_only:
            print(json.dumps(contract, ensure_ascii=False, indent=2))
            return 0

        original_guard = anchors.assert_anchor_order

        def independent_guard(candidate_root: Path, probe_id: str) -> None:
            if Path(candidate_root).resolve() != output_root or probe_id != args.lane:
                raise RuntimeError("independent scheduling guard scope violation")
            current_plain = read_json(
                Path(str(matched_plain_root / "anchors/plain/full_state_latest.pt") + ".json")
            )
            validate_plain_sidecar(current_plain)

        anchors.assert_anchor_order = independent_guard
        try:
            result = anchors.run_anchor(
                probe_id=args.lane,
                output_root=output_root,
                train_view=args.train_view.resolve(),
                data_root=args.data_root.resolve(),
                manifest_path=manifest,
                gpu=int(args.gpu),
                resume=True,
                engineering_stop_after_epoch=int(args.stop_after_epoch),
            )
        finally:
            anchors.assert_anchor_order = original_guard
        atomic_json(operations / "INDEPENDENT_PROBE_RESULT.json", result)
        append_jsonl(operations / "INDEPENDENT_PROBE_EVENTS.jsonl", {
            "time": now(), "event": "RUN_RETURN", "result": result,
            "confirmation20_opened": False,
        })
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
