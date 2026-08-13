"""Verification result dataclasses and JSON helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from projects.safe_policy_optimisation.utils.io import write_json


class VerificationStatus(str, Enum):
    """Allowed verifier outcomes."""

    VERIFIED = "VERIFIED"
    FALSIFIED = "FALSIFIED"
    UNKNOWN = "UNKNOWN"


@dataclass
class VerificationReport:
    """Machine-readable barrier verification report."""

    status: VerificationStatus
    environment: str
    policy_hash: str
    barrier_hash: str
    safety_threshold: float
    initial_set: list[list[float]]
    alpha: float
    margins: dict[str, float]
    boxes_processed: int
    maximum_depth: int
    counterexample: dict[str, Any] | None = None
    unresolved_boxes: list[dict[str, Any]] | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    def save(self, path: Path) -> None:
        write_json(path, self.to_dict())
