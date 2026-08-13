"""Counterexample-guided barrier synthesis loop."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from barrier_tools.dynamics.base import DiscreteTimeDynamics
from barrier_tools.policies.base import VerifiedPolicy
from barrier_tools.synthesis.learner import LearnerConfig, train_barrier
from barrier_tools.verification.branch_and_bound import (
    BarrierSpecification,
    BranchAndBoundVerifier,
)
from barrier_tools.verification.report import VerificationReport, VerificationStatus


@dataclass(frozen=True)
class CegisConfig:
    """Settings for the CEGIS loop."""

    iterations: int = 3


def run_cegis(
    policy: VerifiedPolicy,
    dynamics: DiscreteTimeDynamics,
    spec: BarrierSpecification,
    verifier: BranchAndBoundVerifier,
    learner_config: LearnerConfig,
    cegis_config: CegisConfig,
) -> tuple[object, VerificationReport]:
    """Train, verify, add counterexamples, and repeat."""

    counterexamples: list[torch.Tensor] = []
    barrier = None
    report: VerificationReport | None = None
    for iteration in range(int(cegis_config.iterations)):
        config = LearnerConfig(**{**learner_config.__dict__, "seed": learner_config.seed + iteration})
        barrier = train_barrier(
            policy,
            dynamics,
            spec,
            config,
            counterexamples=counterexamples,
        )
        report = verifier.verify(barrier, policy, spec)
        if report.status != VerificationStatus.FALSIFIED or report.counterexample is None:
            return barrier, report
        state = torch.tensor(report.counterexample["state"], dtype=torch.float32)
        counterexamples.append(state)

    if barrier is None or report is None:
        raise RuntimeError("CEGIS loop did not run.")
    return barrier, report
