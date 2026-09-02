"""Frozen identities and public lane contracts for the full-data paper stage."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "PAPER_AIO_UNPAIRED_V1.json"
FULL_STATE_SCHEMA = "final-unsb-paper-aio-full-state-v1"
E0_SCHEMA = "final-unsb-paper-aio-common-e0-v1"
EVALUATION_SCHEMA = "final-unsb-paper-aio-evaluation-v1"
EXPECTED_MANIFEST_SHA256 = (
    "02c01df580b882763fb0ff28dbdeac4b3729deb8bb772005f26f3e7bc2e36744"
)
FROZEN_EVALUATION_BUNDLE_FINGERPRINT = (
    "68f53a8e9d6fdafd750956d16fbd537aed6e727e081b1db6d0b62258e09b4e41"
)
REQUIRED_FIRST_WAVE_TRAINED = ("plain", "proposal", "cut", "cyclegan")
REQUIRED_PAPER_TABLE = ("input", *REQUIRED_FIRST_WAVE_TRAINED)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_source_sha256(path: Path) -> str:
    data = Path(path).read_bytes()
    if b"\x00" not in data:
        data = data.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(data).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNCOMMITTED"


def load_protocol() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def evaluation_bundle_fingerprint(protocol: dict | None = None) -> str:
    protocol = protocol or load_protocol()
    return str(protocol["evaluation"]["bundle_seed_fingerprint"])


@dataclass(frozen=True)
class LaneSpec:
    id: str
    backend: str
    family: str
    model: str
    role: str
    method: dict
    first_wave: bool = True

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "backend": self.backend,
            "family": self.family,
            "model": self.model,
            "role": self.role,
            "method": self.method,
            "first_wave": self.first_wave,
        }


def lane_spec(lane_id: str, protocol: dict | None = None) -> LaneSpec:
    protocol = protocol or load_protocol()
    matches = [row for row in protocol["lanes"] if row["id"] == lane_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate paper lane: {lane_id}")
    row = matches[0]
    return LaneSpec(
        id=row["id"], backend=row["backend"], family=row["family"],
        model=row["model"], role=row["role"], method=dict(row.get("method", {})),
        first_wave=bool(row.get("first_wave", True)),
    )


def steps_per_epoch(protocol: dict | None = None) -> int:
    protocol = protocol or load_protocol()
    return int(protocol["training"]["steps_per_data_epoch"])


def epoch_to_step(epoch: int, protocol: dict | None = None) -> int:
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    return int(epoch) * steps_per_epoch(protocol)


def step_to_epoch(zero_based_step: int, protocol: dict | None = None) -> int:
    if int(zero_based_step) < 0:
        raise ValueError("step must be non-negative")
    return 1 + int(zero_based_step) // steps_per_epoch(protocol)


def milestone_steps(protocol: dict | None = None) -> list[int]:
    protocol = protocol or load_protocol()
    return [
        epoch_to_step(epoch, protocol)
        for epoch in protocol["training"]["milestone_epochs"]
    ]


def _fingerprinted_files() -> list[Path]:
    exact = [
        CONFIG_PATH,
        ROOT / "configs" / "PAPER_AIO_MATCHED_RUNTIME_RELATIONS.json",
        ROOT / "production" / "metrics.py",
    ]
    # A receipt must become stale after *any* transition/evaluation source
    # changes, including shared network, option and data code used by an
    # external baseline. Metadata-only edits outside these roots do not
    # invalidate an expensive scientific run.
    roots = [
        ROOT / "research" / "paper_aio",
        ROOT / "src" / "models",
        ROOT / "src" / "data",
        ROOT / "src" / "options",
        ROOT / "src" / "util",
    ]
    found = {path.resolve() for path in exact if path.is_file()}
    for directory in roots:
        if directory.is_dir():
            found.update(path.resolve() for path in directory.rglob("*.py"))
    return sorted(found, key=lambda path: path.as_posix())


def protocol_fingerprint(manifest_path: Path | None = None) -> str:
    protocol = load_protocol()
    manifest_path = manifest_path or ROOT / protocol["manifest"]["repo_path"]
    payload = {
        "config": protocol,
        "manifest_sha256": file_sha256(manifest_path),
        "sources": [
            (path.relative_to(ROOT).as_posix(), portable_source_sha256(path))
            for path in _fingerprinted_files()
        ],
        "full_state_schema": FULL_STATE_SCHEMA,
        "evaluation_schema": EVALUATION_SCHEMA,
    }
    return object_sha256(payload)


def validate_protocol(protocol: dict | None = None) -> list[str]:
    protocol = protocol or load_protocol()
    errors: list[str] = []
    training = protocol.get("training", {})
    common = protocol.get("common", {})
    if protocol.get("status") != "ACTIVE_FULL_DATA_PAPER_RESEARCH":
        errors.append("paper protocol is not active")
    if protocol.get("manifest", {}).get("sha256") != EXPECTED_MANIFEST_SHA256:
        errors.append("full manifest identity changed")
    if steps_per_epoch(protocol) != 8553:
        errors.append("paper data epoch must contain exactly 8553 updates")
    if int(training.get("target_data_epochs", -1)) != 200:
        errors.append("paper target must be exactly 200 data epochs")
    if int(training.get("target_updates", -1)) != 1_710_600:
        errors.append("paper target must be exactly 1710600 updates")
    if epoch_to_step(200, protocol) != int(training.get("target_updates", -1)):
        errors.append("updates/data-epoch conversion is inconsistent")
    if int(common.get("batch_size", -1)) != 1:
        errors.append("scientific batch size must remain one")
    if common.get("load_size") != 128 or common.get("crop_size") != 128:
        errors.append("controlled paper resolution must remain 128")
    if not bool(training.get("confirmation20_locked")):
        errors.append("confirmation20 must remain locked")
    if training.get("sampling_measure") != "official_image_proportional_unpaired":
        errors.append("main paper sampling measure changed")
    if evaluation_bundle_fingerprint(protocol) != FROZEN_EVALUATION_BUNDLE_FINGERPRINT:
        errors.append("paper common-random-number bundle identity changed")
    lane_ids = [row.get("id") for row in protocol.get("lanes", [])]
    required = {"plain", "proposal", "hjcgr", "amtnc", "cyclegan", "cut", "ddsb"}
    if set(lane_ids) != required or len(lane_ids) != len(required):
        errors.append("paper lane set changed")
    if lane_spec("proposal", protocol).method != {
        "route1_ablation_enable": True,
        "pcrsmg_ablation_role": "proposal_only",
    }:
        errors.append("Proposal-only frozen semantics changed")
    if lane_spec("cut", protocol).backend != "internal":
        errors.append("pinned CUT full-state adapter is not active")
    if lane_spec("ddsb", protocol).backend != "external_locked":
        errors.append("DDSB must fail closed before its formula/source audit")
    return errors
