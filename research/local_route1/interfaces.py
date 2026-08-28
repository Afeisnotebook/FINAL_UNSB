"""Research interfaces that make target-blindness and parent-state isolation explicit."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import torch

from .runtime import full_state_hash


FORBIDDEN_OBSERVATION_TOKENS = (
    "paired", "psnr", "ssim", "lpips", "target_image", "ground_truth",
    "discovery", "confirmation",
)


@dataclass(frozen=True)
class StateObservation:
    step: int
    physical_epoch: float
    gradient: dict[str, Any] = field(default_factory=dict)
    bridge: dict[str, Any] = field(default_factory=dict)
    game_balance: dict[str, Any] = field(default_factory=dict)
    sampling: dict[str, Any] = field(default_factory=dict)
    method_internal: dict[str, Any] = field(default_factory=dict)

    def validate_target_blind(self) -> None:
        def visit(value: Any, path: str) -> None:
            lowered = path.lower()
            if any(token in lowered for token in FORBIDDEN_OBSERVATION_TOKENS):
                raise ValueError(f"paired/held-out field rejected from observable: {path}")
            if isinstance(value, dict):
                for key, child in value.items():
                    visit(child, f"{path}.{key}")
            elif isinstance(value, (list, tuple)):
                for index, child in enumerate(value):
                    visit(child, f"{path}[{index}]")

        visit(asdict(self), "StateObservation")


@dataclass(frozen=True)
class InterventionProposal:
    candidate_id: str
    operator_type: str
    payload: dict[str, Any]
    identity_condition: str
    changes_training_target: bool
    unbiased_estimator: bool | None


class ConstraintProjector:
    """Minimal target-blind gradient operators used by derived candidates."""

    @staticmethod
    def zero_like(correction: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(correction)

    @staticmethod
    def one_sided(correction: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        dot = torch.sum(correction * reference)
        return correction if float(dot.detach()) >= 0.0 else torch.zeros_like(correction)

    @staticmethod
    def remove_conflicting_component(
        correction: torch.Tensor, reference: torch.Tensor, eps: float = 1e-12,
    ) -> torch.Tensor:
        dot = torch.sum(correction * reference)
        if float(dot.detach()) >= 0.0:
            return correction
        scale = dot / torch.sum(reference.square()).clamp_min(eps)
        return correction - scale * reference

    @staticmethod
    def trust_region(correction: torch.Tensor, radius: float, eps: float = 1e-12) -> torch.Tensor:
        norm = torch.linalg.vector_norm(correction)
        scale = min(1.0, float(radius) / max(float(norm.detach()), eps))
        return correction * scale


class CounterfactualAuditor:
    """Run a branch on a deep copy and prove the supplied parent state is unchanged."""

    def run(self, parent_state: dict, branch: Callable[[dict], Any]) -> tuple[Any, str]:
        before = full_state_hash(parent_state)
        result = branch(copy.deepcopy(parent_state))
        after = full_state_hash(parent_state)
        if before != after:
            raise RuntimeError("counterfactual branch polluted parent full state")
        return result, before


class HypothesisLedger:
    """Append-only logical ledger; disk persistence is owned by the stage runner."""

    def __init__(self, records: list[dict] | None = None):
        self.records = list(records or [])

    def append(self, record: dict) -> None:
        if "candidate_id" not in record or "parent_evidence" not in record:
            raise ValueError("hypothesis record requires candidate_id and parent_evidence")
        if any(row.get("candidate_id") == record["candidate_id"] for row in self.records):
            raise ValueError(f"duplicate candidate id: {record['candidate_id']}")
        self.records.append(copy.deepcopy(record))

    def to_dict(self) -> dict:
        return {"schema": "local-route1-hypothesis-ledger-v1", "records": self.records}
