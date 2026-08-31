"""Executable long-run gate for an evidence-derived route-1 candidate.

The generic runner does not assume a particular algorithm shape.  Every
candidate registers a source-bound gate hook that must execute its mathematical
and engineering checks and return the canonical evidence report validated here.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .candidates import GATE_SCHEMA, CandidateRegistration, load_candidate_registration
from .protocol import ROOT, file_sha256, git_commit, load_protocol
from .runtime import write_json


REQUIRED_CHECKS = (
    "mathematical_invariants",
    "zero_intervention_identity",
    "resume_exact",
    "cross_state_counterfactual",
    "target_blind_observable",
    "micro_engineering",
    "base_unsb_semantics_preserved",
    "shared_e0_load_exact",
)


@dataclass(frozen=True)
class CandidateGateContext:
    output_root: Path
    train_view: Path
    data_root: Path
    manifest_path: Path
    gpu: int
    registration: CandidateRegistration


def _validate_gate_report(report: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise RuntimeError("candidate gate hook must return a JSON-compatible object")
    checks = report.get("checks")
    if not isinstance(checks, dict):
        raise RuntimeError("candidate gate hook did not return checks")
    for name in REQUIRED_CHECKS:
        if checks.get(name) is not True:
            raise RuntimeError(f"candidate executable gate failed {name}")
    invariants = report.get("mathematical_invariant_evidence")
    if not isinstance(invariants, list) or not invariants:
        raise RuntimeError("candidate gate requires executable mathematical invariant evidence")
    for row in invariants:
        if not isinstance(row, dict) or row.get("status") != "PASS":
            raise RuntimeError("every mathematical invariant must have executable PASS evidence")
        if not row.get("name") or not row.get("observed"):
            raise RuntimeError("mathematical invariant evidence requires name and observation")
    zero = report.get("zero_intervention_evidence", {})
    if not zero.get("candidate_state_sha256") or (
        zero.get("candidate_state_sha256") != zero.get("plain_state_sha256")
    ):
        raise RuntimeError("zero intervention is not exactly identical to plain")
    resume = report.get("resume_evidence", {})
    if not resume.get("continuous_state_sha256") or (
        resume.get("continuous_state_sha256") != resume.get("resumed_state_sha256")
    ):
        raise RuntimeError("candidate full-state resume is not exact")
    cross_state = report.get("cross_state_evidence", {})
    epochs = {int(value) for value in cross_state.get("data_epochs", [])}
    if not ({20, 100, 200} <= epochs):
        raise RuntimeError("candidate cross-state gate must exercise early/middle/late states")
    if cross_state.get("all_parent_state_hashes_preserved") is not True:
        raise RuntimeError("candidate counterfactual gate polluted a parent state")
    observable = report.get("target_blind_evidence", {})
    if observable.get("paired_fields_observed") not in ([], ()):
        raise RuntimeError("candidate observable includes paired fields")
    if observable.get("paired_target_available") is not False:
        raise RuntimeError("candidate gate does not prove target inaccessibility")
    micro = report.get("micro_engineering_evidence", {})
    updates = int(micro.get("updates", 0))
    if not 400 <= updates <= 800:
        raise RuntimeError("candidate engineering micro run must be 400--800 updates")
    if micro.get("finite") is not True or micro.get("paired_metric_used_for_promotion") is not False:
        raise RuntimeError("candidate micro run is not a pure engineering gate")
    if report.get("paired_metric_used_for_promotion") is not False:
        raise RuntimeError("paired metrics cannot promote a candidate through gates")
    if report.get("paired_controller_access") is not False:
        raise RuntimeError("candidate gate permits paired controller access")
    if report.get("confirmation20_opened") is not False:
        raise RuntimeError("confirmation20 must remain locked")
    return report


def _load_gate_hook(registration: CandidateRegistration):
    hook = registration.implementation.get("gate_hook")
    if not isinstance(hook, dict) or not hook.get("module") or not hook.get("callable"):
        raise RuntimeError("candidate implementation requires a source-bound gate_hook")
    module = importlib.import_module(str(hook["module"]))
    module_path = Path(inspect.getfile(module)).resolve()
    registered = {
        (ROOT / row["path"]).resolve()
        for row in registration.implementation["source_files"]
    }
    if module_path not in registered:
        raise RuntimeError("candidate gate hook module is not in frozen source_files")
    function = getattr(module, str(hook["callable"]), None)
    if not callable(function):
        raise RuntimeError("candidate gate hook callable is missing")
    return function, module_path


def run_candidate_gate(
    *, output_root: Path, candidate_id: str, train_view: Path, data_root: Path,
    manifest_path: Path, gpu: int,
) -> dict[str, Any]:
    output_root = Path(output_root).resolve()
    registration = load_candidate_registration(output_root, candidate_id, require_gate=False)
    manifest_path = Path(manifest_path).resolve()
    if file_sha256(manifest_path) != str(load_protocol()["manifest"]["sha256"]):
        raise RuntimeError("candidate gate manifest differs from the frozen route-1 manifest")
    function, module_path = _load_gate_hook(registration)
    context = CandidateGateContext(
        output_root=output_root,
        train_view=Path(train_view).resolve(),
        data_root=Path(data_root).resolve(),
        manifest_path=manifest_path,
        gpu=int(gpu),
        registration=registration,
    )
    report = _validate_gate_report(function(context))
    if registration.spec.model in (
        "route1_pcrsmg", "route1_amtnc", "route1_pcnr",
    ):
        expected_schedule = (
            ["DE_VIEW", "D_COMMIT", "E_COMMIT", "GF_VIEW", "GF_COMMIT"]
            if registration.spec.model == "route1_pcnr" else
            ["DE_BUNDLE", "D_COMMIT", "E_COMMIT", "GF_BUNDLE", "GF_COMMIT"]
        )
        player_evidence = report.get("player_conditional_execution_evidence")
        if not isinstance(player_evidence, dict):
            raise RuntimeError("replicated player-conditional gate did not prove execution")
        if (
            player_evidence.get("all_de_and_gf_counts_equal_updates") is not True
            or player_evidence.get("all_bundle_serials_equal_twice_updates") is not True
            or player_evidence.get("expected_schedule") != expected_schedule
        ):
            raise RuntimeError("replicated player-bundle gate evidence is invalid")
    if registration.spec.model in ("route1_pcammcrb", "route1_pcrfammcrb"):
        from models.route1.pcammcrb import (
            EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE,
        )
        from models.route1.pcnr import EXPECTED_PCNR_SCHEDULE

        repaired = registration.spec.model == "route1_pcrfammcrb"
        parent = str(registration.spec.method.get(
            "pcrfammcrb_sampling_parent" if repaired else "pcammcrb_sampling_parent",
            "pcnr",
        ))
        expected = list(
            EXPECTED_PCNR_SCHEDULE
            if parent == "pcnr" else EXPECTED_PCRSMG_PROPOSAL_BARRIER_SCHEDULE
        )
        player_evidence = report.get("player_conditional_execution_evidence")
        if not isinstance(player_evidence, dict) or (
            player_evidence.get("sampling_parent") != parent
            or player_evidence.get("expected_schedule") != expected
            or player_evidence.get("all_sampling_and_barrier_counts_equal_updates") is not True
            or (
                repaired and player_evidence.get("barrier_operator") !=
                "residual_feasible_adam_metric_without_absolute_margin"
            )
        ):
            raise RuntimeError("PC-AMMCRB player/barrier execution evidence is invalid")
        compatibility = report.get("component_compatibility_evidence")
        if not isinstance(compatibility, dict) or (
            compatibility.get("data_epochs") != [20, 100, 200]
            or compatibility.get("branch_updates") != [1, 8, 32]
            or compatibility.get("all_parent_state_hashes_preserved") is not True
            or float(compatibility.get("minimum_observed_component_correction_cosine", -2.0)) < -0.2
        ):
            raise RuntimeError("PC-AMMCRB preregistered component compatibility gate failed")
    result = {
        "schema": GATE_SCHEMA,
        "status": "PASS_LONG_RUN",
        "candidate_id": candidate_id,
        "algorithm_fingerprint": registration.algorithm_fingerprint,
        "candidate_fingerprint": registration.candidate_fingerprint,
        "gate_git_commit": git_commit(),
        "gate_hook_module": str(module_path),
        "gate_hook_sha256": file_sha256(module_path),
        "checks": {
            **{name: True for name in REQUIRED_CHECKS},
            **{
                name: True for name, passed in report.get("checks", {}).items()
                if passed is True and name not in REQUIRED_CHECKS
            },
        },
        "evidence": report,
        "paired_metric_used_for_promotion": False,
        "paired_controller_access": False,
        "confirmation20_opened": False,
    }
    path = output_root / "derive" / "gates" / f"{candidate_id}.json"
    if path.is_file():
        existing = __import__("json").loads(path.read_text(encoding="utf-8"))
        if existing != result:
            raise RuntimeError("candidate gate already exists with non-identical evidence")
        return existing
    write_json(path, result)
    return result
