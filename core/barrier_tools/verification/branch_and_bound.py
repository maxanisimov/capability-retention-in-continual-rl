"""Interval branch-and-bound verifier for barrier certificates."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Literal

import torch

from barrier_tools.barriers.base import BarrierFunction
from barrier_tools.dynamics.base import DiscreteTimeDynamics
from barrier_tools.policies.base import VerifiedPolicy
from barrier_tools.sets.boxes import Box
from barrier_tools.verification.report import VerificationReport, VerificationStatus

ConditionName = Literal["initial", "unsafe", "invariance"]


@dataclass(frozen=True)
class BarrierSpecification:
    """Continuous-domain safety specification for barrier verification."""

    initial_boxes: list[Box]
    unsafe_boxes: list[Box]
    invariant_boxes: list[Box]
    safety_threshold: float = -1.0
    alpha: float = 0.1
    eps_init: float = 1e-5
    eps_unsafe: float = 1e-5
    eps_inv: float = 1e-6


@dataclass(frozen=True)
class VerifierConfig:
    """Resource and numerical settings for branch-and-bound."""

    max_depth: int = 18
    max_boxes: int = 50_000
    min_width: float = 1e-4
    timeout_seconds: float | None = None
    sample_counterexamples: bool = True


@dataclass
class _WorkItem:
    box: Box
    depth: int
    condition: ConditionName


@dataclass
class _CheckResult:
    proved: bool
    margin: float
    counterexample: dict[str, object] | None = None
    skip: bool = False


def tensor_hash(tensors: list[torch.Tensor]) -> str:
    """Return a stable hash over tensor contents."""

    digest = hashlib.sha256()
    for tensor in tensors:
        arr = tensor.detach().cpu().contiguous().numpy()
        digest.update(arr.tobytes())
    return digest.hexdigest()


def barrier_hash(barrier: BarrierFunction) -> str:
    """Best-effort hash for a barrier object."""

    if hasattr(barrier, "parameters"):
        return tensor_hash([p.detach() for p in barrier.parameters()])  # type: ignore[attr-defined]
    values = [torch.as_tensor(v).flatten() for v in vars(barrier).values() if isinstance(v, (int, float, torch.Tensor))]
    return tensor_hash(values) if values else "unavailable"


def policy_hash(policy: VerifiedPolicy) -> str:
    """Best-effort hash for a policy object."""

    if hasattr(policy, "parameters"):
        params = [p.detach() for p in policy.parameters()]  # type: ignore[attr-defined]
        if params:
            return tensor_hash(params)
    if hasattr(policy, "action"):
        return hashlib.sha256(str(getattr(policy, "action")).encode("utf-8")).hexdigest()
    return "unavailable"


class BranchAndBoundVerifier:
    """Verify barrier conditions over two-dimensional boxes."""

    def __init__(
        self,
        dynamics: DiscreteTimeDynamics,
        *,
        config: VerifierConfig | None = None,
        environment: str = "MountainCarContinuous-v0",
    ) -> None:
        self.dynamics = dynamics
        self.config = config or VerifierConfig()
        self.environment = environment

    def verify(
        self,
        barrier: BarrierFunction,
        policy: VerifiedPolicy,
        spec: BarrierSpecification,
    ) -> VerificationReport:
        """Verify all barrier conditions and return a report."""

        start = time.monotonic()
        margins = {"initial": float("inf"), "unsafe": float("inf"), "invariance": float("inf")}
        boxes_processed = 0
        maximum_depth = 0
        unresolved: list[dict[str, object]] = []

        work: list[_WorkItem] = []
        work.extend(_WorkItem(box, 0, "initial") for box in spec.initial_boxes)
        work.extend(_WorkItem(box, 0, "unsafe") for box in spec.unsafe_boxes)
        work.extend(_WorkItem(box, 0, "invariance") for box in spec.invariant_boxes)

        while work:
            if boxes_processed >= self.config.max_boxes:
                unresolved.extend(self._serialize_item(item) for item in work)
                return self._report(
                    VerificationStatus.UNKNOWN,
                    spec,
                    barrier,
                    policy,
                    margins,
                    boxes_processed,
                    maximum_depth,
                    unresolved,
                    reason="max_boxes reached",
                )
            if self.config.timeout_seconds is not None and time.monotonic() - start > self.config.timeout_seconds:
                unresolved.extend(self._serialize_item(item) for item in work)
                return self._report(
                    VerificationStatus.UNKNOWN,
                    spec,
                    barrier,
                    policy,
                    margins,
                    boxes_processed,
                    maximum_depth,
                    unresolved,
                    reason="timeout reached",
                )

            item = work.pop()
            boxes_processed += 1
            maximum_depth = max(maximum_depth, item.depth)
            result = self._check_item(item, barrier, policy, spec)
            if result.counterexample is not None:
                return self._report(
                    VerificationStatus.FALSIFIED,
                    spec,
                    barrier,
                    policy,
                    margins,
                    boxes_processed,
                    maximum_depth,
                    unresolved,
                    counterexample=result.counterexample,
                    reason=f"{item.condition} violation",
                )
            if result.skip:
                continue
            margins[item.condition] = min(margins[item.condition], result.margin)
            if result.proved:
                continue

            if item.depth >= self.config.max_depth or item.box.max_width <= self.config.min_width:
                unresolved.append(self._serialize_item(item))
                continue
            left, right = item.box.split()
            work.append(_WorkItem(left, item.depth + 1, item.condition))
            work.append(_WorkItem(right, item.depth + 1, item.condition))

        if unresolved:
            return self._report(
                VerificationStatus.UNKNOWN,
                spec,
                barrier,
                policy,
                margins,
                boxes_processed,
                maximum_depth,
                unresolved,
                reason="unresolved boxes remain",
            )

        finite_margins = {
            key: (0.0 if value == float("inf") else float(value)) for key, value in margins.items()
        }
        if any(value <= 0.0 for value in finite_margins.values()):
            return self._report(
                VerificationStatus.UNKNOWN,
                spec,
                barrier,
                policy,
                finite_margins,
                boxes_processed,
                maximum_depth,
                unresolved,
                reason="non-positive certified margin",
            )
        return self._report(
            VerificationStatus.VERIFIED,
            spec,
            barrier,
            policy,
            finite_margins,
            boxes_processed,
            maximum_depth,
            unresolved,
        )

    def _check_item(
        self,
        item: _WorkItem,
        barrier: BarrierFunction,
        policy: VerifiedPolicy,
        spec: BarrierSpecification,
    ) -> _CheckResult:
        lower, upper = item.box.as_batch()
        h_l, h_u = barrier.interval(lower, upper)
        if item.condition == "initial":
            margin = float((h_l - spec.eps_init).min().item())
            if margin > 0:
                return _CheckResult(True, margin)
            cex = self._find_point_violation(item.box, barrier, policy, spec, "initial")
            return _CheckResult(False, margin, cex)

        if item.condition == "unsafe":
            margin = float((-spec.eps_unsafe - h_u).min().item())
            if margin > 0:
                return _CheckResult(True, margin)
            cex = self._find_point_violation(item.box, barrier, policy, spec, "unsafe")
            return _CheckResult(False, margin, cex)

        if float(h_u.max().item()) < 0.0:
            return _CheckResult(True, float("inf"), skip=True)
        if float(h_l.min().item()) < 0.0:
            cex = self._find_point_violation(item.box, barrier, policy, spec, "invariance")
            return _CheckResult(False, float("-inf"), cex)

        action_l, action_u = policy.interval(lower, upper)
        next_l, next_u = self.dynamics.interval_step(lower, upper, action_l, action_u)
        h_next_l, _h_next_u = barrier.interval(next_l, next_u)
        beta = 1.0 - float(spec.alpha)
        margin = float((h_next_l - beta * h_u - spec.eps_inv).min().item())
        if margin > 0:
            return _CheckResult(True, margin)
        cex = self._find_point_violation(item.box, barrier, policy, spec, "invariance")
        return _CheckResult(False, margin, cex)

    def _find_point_violation(
        self,
        box: Box,
        barrier: BarrierFunction,
        policy: VerifiedPolicy,
        spec: BarrierSpecification,
        condition: ConditionName,
    ) -> dict[str, object] | None:
        points = [box.center, *box.corners()]
        if self.config.sample_counterexamples:
            generator = torch.Generator().manual_seed(0)
            points.extend(list(box.sample(16, generator=generator)))
        states = torch.stack([point.float() for point in points], dim=0)
        h = barrier.value(states)

        if condition == "initial":
            mask = h < spec.eps_init
            if bool(mask.any().item()):
                idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
                return self._point_payload(condition, states[idx], h[idx])
            return None

        if condition == "unsafe":
            mask = h > -spec.eps_unsafe
            if bool(mask.any().item()):
                idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
                return self._point_payload(condition, states[idx], h[idx])
            return None

        actions = policy.forward(states)
        next_states = self.dynamics.step(states, actions)
        h_next = barrier.value(next_states)
        beta = 1.0 - float(spec.alpha)
        mask = (h >= 0.0) & (h_next < beta * h + spec.eps_inv)
        if bool(mask.any().item()):
            idx = int(torch.nonzero(mask, as_tuple=False)[0].item())
            return self._point_payload(
                condition,
                states[idx],
                h[idx],
                action=actions[idx],
                next_state=next_states[idx],
                h_next=h_next[idx],
            )
        return None

    @staticmethod
    def _point_payload(
        category: ConditionName,
        state: torch.Tensor,
        h_value: torch.Tensor,
        *,
        action: torch.Tensor | None = None,
        next_state: torch.Tensor | None = None,
        h_next: torch.Tensor | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "category": category,
            "state": [float(v) for v in state.detach().cpu().tolist()],
            "h": float(h_value.detach().cpu().item()),
        }
        if action is not None:
            payload["action"] = [float(v) for v in action.detach().cpu().reshape(-1).tolist()]
        if next_state is not None:
            payload["next_state"] = [float(v) for v in next_state.detach().cpu().tolist()]
        if h_next is not None:
            payload["h_next"] = float(h_next.detach().cpu().item())
        return payload

    @staticmethod
    def _serialize_item(item: _WorkItem) -> dict[str, object]:
        return {
            "condition": item.condition,
            "depth": int(item.depth),
            "lower": [float(v) for v in item.box.lower.tolist()],
            "upper": [float(v) for v in item.box.upper.tolist()],
        }

    def _report(
        self,
        status: VerificationStatus,
        spec: BarrierSpecification,
        barrier: BarrierFunction,
        policy: VerifiedPolicy,
        margins: dict[str, float],
        boxes_processed: int,
        maximum_depth: int,
        unresolved: list[dict[str, object]],
        *,
        counterexample: dict[str, object] | None = None,
        reason: str | None = None,
    ) -> VerificationReport:
        initial = spec.initial_boxes[0]
        return VerificationReport(
            status=status,
            environment=self.environment,
            policy_hash=policy_hash(policy),
            barrier_hash=barrier_hash(barrier),
            safety_threshold=float(spec.safety_threshold),
            initial_set=[
                [float(initial.lower[0].item()), float(initial.upper[0].item())],
                [float(initial.lower[1].item()), float(initial.upper[1].item())],
            ],
            alpha=float(spec.alpha),
            margins={key: float(value) for key, value in margins.items()},
            boxes_processed=int(boxes_processed),
            maximum_depth=int(maximum_depth),
            counterexample=counterexample,
            unresolved_boxes=unresolved or None,
            reason=reason,
        )
