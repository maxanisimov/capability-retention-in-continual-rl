"""Polynomial barrier learner."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from barrier_tools.barriers.polynomial import PolynomialBarrier
from barrier_tools.dynamics.base import DiscreteTimeDynamics
from barrier_tools.policies.base import VerifiedPolicy
from barrier_tools.sets.boxes import Box
from barrier_tools.synthesis.losses import (
    initial_loss,
    invariance_loss,
    nonempty_regularizer,
    unsafe_loss,
)
from barrier_tools.verification.branch_and_bound import BarrierSpecification


@dataclass(frozen=True)
class LearnerConfig:
    """Settings for sampled polynomial barrier optimisation."""

    degree: int = 4
    use_energy_base: bool = True
    epochs: int = 1_000
    batch_size: int = 512
    learning_rate: float = 1e-2
    samples_initial: int = 512
    samples_unsafe: int = 512
    samples_domain: int = 2_048
    boundary_noise: float = 0.02
    seed: int = 0
    weight_initial: float = 1.0
    weight_unsafe: float = 1.0
    weight_invariance: float = 1.0
    weight_nonempty: float = 0.1


def _sample_from_boxes(
    boxes: list[Box],
    n: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    if not boxes:
        raise ValueError("Cannot sample from an empty box list.")
    per_box = max(1, int(n) // len(boxes))
    samples = [box.sample(per_box, generator=generator) for box in boxes]
    out = torch.cat(samples, dim=0)
    if out.shape[0] < n:
        out = torch.cat([out, boxes[0].sample(n - out.shape[0], generator=generator)], dim=0)
    return out[:n]


def train_barrier(
    policy: VerifiedPolicy,
    dynamics: DiscreteTimeDynamics,
    spec: BarrierSpecification,
    config: LearnerConfig,
    *,
    counterexamples: list[torch.Tensor] | None = None,
) -> PolynomialBarrier:
    """Train and return a polynomial barrier candidate."""

    generator = torch.Generator().manual_seed(int(config.seed))
    barrier = PolynomialBarrier(
        degree=config.degree,
        use_energy_base=config.use_energy_base,
        p_safe=spec.safety_threshold,
    )
    optimizer = torch.optim.Adam(barrier.parameters(), lr=float(config.learning_rate))
    counterexample_tensor = None
    if counterexamples:
        counterexample_tensor = torch.stack([item.float() for item in counterexamples], dim=0)

    for _epoch in range(int(config.epochs)):
        initial_states = _sample_from_boxes(spec.initial_boxes, config.samples_initial, generator=generator)
        unsafe_states = _sample_from_boxes(spec.unsafe_boxes, config.samples_unsafe, generator=generator)
        domain_states = _sample_from_boxes(spec.invariant_boxes, config.samples_domain, generator=generator)
        if counterexample_tensor is not None:
            jitter = torch.randn(
                counterexample_tensor.shape,
                generator=generator,
                dtype=counterexample_tensor.dtype,
            ) * float(config.boundary_noise)
            domain_states = torch.cat([domain_states, counterexample_tensor, counterexample_tensor + jitter], dim=0)

        loss = (
            float(config.weight_initial) * initial_loss(barrier, initial_states, spec.eps_init)
            + float(config.weight_unsafe) * unsafe_loss(barrier, unsafe_states, spec.eps_unsafe)
            + float(config.weight_invariance)
            * invariance_loss(
                barrier,
                policy,
                dynamics,
                domain_states,
                alpha=spec.alpha,
                eps_inv=spec.eps_inv,
            )
            + float(config.weight_nonempty) * nonempty_regularizer(barrier, initial_states)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return barrier
