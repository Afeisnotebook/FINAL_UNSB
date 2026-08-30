"""Frozen protocol, identity and schedule validation for local route 1."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "LOCAL_ROUTE1_PROBES.json"
EVALUATION_SCHEMA = "local-route1-discovery70-crn-single-rollout-v1"
FULL_STATE_SCHEMA = "final-unsb-local-route1-full-state-v1"
EXPECTED_MANIFEST_SHA256 = (
    "1a66cf71420ebb996abce23eecb7e555a6d9d93a39b6b8c3fc17dbf0ead42b7b"
)


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


def semantic_source_sha256(path: Path) -> str:
    """Hash source after normalizing line endings and trailing blank lines."""
    text = Path(path).read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256((text.rstrip() + "\n").encode("utf-8")).hexdigest()


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


@dataclass(frozen=True)
class ProbeSpec:
    id: str
    contract_id: str
    model: str
    role: str
    method: dict
    historical_fact: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "contract_id": self.contract_id,
            "model": self.model,
            "role": self.role,
            "method": self.method,
            "historical_fact": self.historical_fact,
        }


def probe_spec(probe_id: str, protocol: dict | None = None) -> ProbeSpec:
    protocol = protocol or load_protocol()
    matches = [row for row in protocol["anchor_probes"] if row.get("lane") == probe_id]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate route-1 probe: {probe_id}")
    row = matches[0]
    return ProbeSpec(
        id=row["lane"], contract_id=row["id"], model=row["model"], role=row["role"],
        method=dict(row.get("method", {})),
        historical_fact=row.get("historical_fact"),
    )


def steps_per_epoch(protocol: dict | None = None) -> int:
    protocol = protocol or load_protocol()
    view = protocol["local_view"]
    return int(view["domains"]) * int(view["train_per_domain"])


def epoch_to_step(epoch: int, protocol: dict | None = None) -> int:
    if int(epoch) < 0:
        raise ValueError("epoch must be non-negative")
    return int(epoch) * steps_per_epoch(protocol)


def step_to_physical_epoch(zero_based_step: int, protocol: dict | None = None) -> int:
    if int(zero_based_step) < 0:
        raise ValueError("step must be non-negative")
    return 1 + int(zero_based_step) // steps_per_epoch(protocol)


def milestone_steps(protocol: dict | None = None) -> list[int]:
    protocol = protocol or load_protocol()
    return [epoch_to_step(epoch, protocol) for epoch in protocol["local_view"]["trajectory_epochs"]]


def dt_lambda_for_physical_epoch(epoch: int, protocol: dict | None = None) -> float:
    """Exact historical DT lambda with physical e21 mapped to active age 1."""
    method = probe_spec("dt", protocol).method
    age = max(0, int(epoch) - 20)
    base = float(method["dtcov_lambda"])
    ramp_start = int(method["dtcov_ramp_start_epoch"])
    ramp_end = int(method["dtcov_ramp_end_epoch"])
    decay_start = int(method["dtcov_decay_start_epoch"])
    decay_end = int(method["dtcov_decay_end_epoch"])
    if age < ramp_start:
        ramp = 0.0
    elif age <= ramp_end:
        ramp = float(age - ramp_start + 1) / float(max(1, ramp_end - ramp_start + 1))
    else:
        ramp = 1.0
    if age <= decay_start:
        decay = 1.0
    elif age >= decay_end:
        decay = 0.0
    else:
        progress = float(age - decay_start) / float(decay_end - decay_start)
        decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * max(0.0, min(1.0, ramp * decay))


def _fingerprinted_files(source_root: Path = ROOT) -> list[Path]:
    source_root = Path(source_root).resolve()
    exact = [
        source_root / "configs" / "LOCAL_ROUTE1_PROBES.json",
        source_root / "LOCAL_ROUTE1_RESEARCH_CONTRACT_CN.md",
        source_root / "ACTIVE_LOCAL_ROUTE1_PLAN_CN.md",
        source_root / "production" / "metrics.py",
    ]
    roots = [
        source_root / "research" / "local_route1",
        source_root / "src" / "data",
        source_root / "src" / "options",
        source_root / "src" / "models" / "dtcov",
        source_root / "src" / "models" / "hj",
        source_root / "src" / "models" / "hnek",
    ]
    exact += [
        source_root / "src" / "models" / "sb_model.py",
        source_root / "src" / "models" / "dtcov_model.py",
        source_root / "src" / "models" / "hj_model.py",
        source_root / "src" / "models" / "hnek_search_model.py",
        source_root / "src" / "models" / "base_model.py",
        source_root / "src" / "models" / "networks.py",
        source_root / "src" / "models" / "patchnce.py",
    ]
    found = {path.resolve() for path in exact if path.is_file()}
    for directory in roots:
        if directory.is_dir():
            found.update(path.resolve() for path in directory.rglob("*.py"))
    return sorted(found, key=lambda path: path.as_posix())


def protocol_fingerprint(
    manifest_path: Path | None = None, *, source_root: Path = ROOT,
) -> str:
    source_root = Path(source_root).resolve()
    protocol_path = source_root / "configs" / "LOCAL_ROUTE1_PROBES.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = manifest_path or source_root / protocol["manifest"]["repo_path"]
    rows = [
        (path.relative_to(source_root).as_posix(), portable_source_sha256(path))
        for path in _fingerprinted_files(source_root)
    ]
    payload = {
        "source": rows,
        "manifest_sha256": file_sha256(manifest_path),
        "evaluation_schema": EVALUATION_SCHEMA,
        "full_state_schema": FULL_STATE_SCHEMA,
    }
    return object_sha256(payload)


def validate_protocol(protocol: dict | None = None) -> list[str]:
    protocol = protocol or load_protocol()
    errors: list[str] = []
    view = protocol.get("local_view", {})
    common = protocol.get("common", {})
    if protocol.get("status") != "ACTIVE_LOCAL_RESEARCH":
        errors.append("route-1 protocol is not active")
    if steps_per_epoch(protocol) != 150:
        errors.append("small25 must contain exactly 150 updates per data epoch")
    if int(view.get("target_updates_per_lane", -1)) != 30000:
        errors.append("target must be exactly 30000 updates")
    if int(view.get("target_epochs", -1)) != 200:
        errors.append("target must be exactly 200 data epochs")
    if epoch_to_step(int(view.get("target_epochs", 0)), protocol) != int(view.get("target_updates_per_lane", -1)):
        errors.append("updates/data-epoch conversion is inconsistent")
    required_common = {
        "lr": 0.0001, "lambda_GAN": 1.0, "lambda_SB": 1.0,
        "lambda_NCE": 1.0, "tau": 0.01, "num_timesteps": 5,
        "n_epochs": 200, "n_epochs_decay": 0, "batch_size": 1,
        "no_flip": True,
    }
    for key, expected in required_common.items():
        if common.get(key) != expected:
            errors.append(f"common.{key} must be {expected!r}")
    if not bool(view.get("confirmation20_locked")):
        errors.append("confirmation20 must remain locked")
    ids = [row.get("id") for row in protocol.get("anchor_probes", [])]
    lanes = [row.get("lane") for row in protocol.get("anchor_probes", [])]
    if ids != ["P0_PLAIN_LONG", "P1_HJ_CONTINUOUS_LONG", "P2_HNEK_LONG", "P3_DT_LONG"]:
        errors.append("authoritative probe contract ids changed")
    if lanes != ["plain", "hj", "hnek", "dt"]:
        errors.append("CLI probe order must be plain -> hj -> hnek -> dt")
    if int(probe_spec("hj", protocol).method.get("hj_start_epoch", -1)) != 5:
        errors.append("HJ must activate at physical epoch 5")
    if int(probe_spec("hj", protocol).method.get("hj_search_start_step", 0)) != -1:
        errors.append("HJ must not use total-step-relative activation")
    if int(probe_spec("dt", protocol).method.get("dtcov_search_start_step", 0)) != -1:
        errors.append("DT must use physical epoch age, not a search-step override")
    forbidden = "|".join(str(item).lower() for item in protocol.get("not_current_tasks", []))
    for token in (
        "four-server", "cross-host", "full-data", "finite handoff",
        "paired metric controller", "best checkpoint",
    ):
        if token not in forbidden:
            errors.append(f"anti-drift exclusion is missing: {token}")
    return errors
