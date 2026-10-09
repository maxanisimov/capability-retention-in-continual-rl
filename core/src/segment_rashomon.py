"""Segment safe-parameter-region search: the longest certified step to a proposal.

Where :mod:`src.zonotope_rashomon` *learns* a low-rank zonotope around a model,
this module certifies the one-dimensional set PSPO actually needs: the line
segment from the last safe parameters ``theta_k`` towards the proposed update
``theta_k + delta``.  That segment is represented exactly (no geometric
over-approximation) as a rank-one zonotope

    Z(a, b) = {c + z g : z in [-1, 1]},
    c = theta_k + (a + b)/2 * delta,
    g = (b - a)/2 * delta,

so ``Z(a, b) = {theta_k + t delta : t in [a, b]}``.  Writing ``C(a, b)`` for the
hard certificate on one sub-segment and, for a ``K``-part partition of
``[0, alpha]``, ``C_K(alpha) = prod_i C(t_{i-1}, t_i)``, this module solves

    alpha_hat = max {alpha in [0, 1] : C_K(alpha) = 1}.

``C_K`` is monotone (shrinking the coefficient interval shrinks every
concretised range, hence every ReLU/Tanh relaxation), so the feasible set is the
interval ``[0, alpha_hat]`` and bisection converges to it.  Raising ``K`` only
tightens ``C_K``, so splitting is a sound refinement -- and because the union of
the sub-segments *is* ``Z(0, alpha)``, the region returned to the caller stays a
single rank-one zonotope.

The search evaluates only the hard certificate, never a differentiable
surrogate, so it needs no gradients and no softmax-temperature calibration.

See ``projects/safe_policy_optimisation/docs/methodology/pspo_segment_lid.tex``
for the full statement and proofs.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

import torch
from torch.utils.data import DataLoader

from provably_safe_policy_optimisation.regions import (
    ZonotopeRegion,
    flatten_tensors,
    unflatten_like,
)
from src.IntervalTensor import IntervalTensor
from src.rashomon_spec import RashomonCertificate
from src.verification import verify
from src.verification.verify import bound_forward_pass


@dataclasses.dataclass
class SegmentRashomonResult:
    """Result of a segment safe-region search.

    ``regions`` holds the single certified rank-one zonotope, or is empty when
    no positive step certified (the caller then reverts).  The remaining fields
    mirror :class:`src.rashomon_spec.RashomonResult` closely enough that the
    PSPO budget accounting and early-stop bookkeeping work unchanged.

    ``certificates`` carries the hard certificate only: this path never computes
    a differentiable surrogate, so ``min_surrogate`` repeats ``min_hard_acc``.
    """

    regions: list[ZonotopeRegion]
    certificates: list[list[RashomonCertificate]]
    alpha: float
    splits_used: int
    iterations_run: int = 0
    target_contained_and_certified: bool = False
    temperatures: dict[int | None, float] = dataclasses.field(default_factory=dict)
    surrogate: str = "none"
    resolved_surrogate: str = "none"


def _unpack_batch(
    batch: tuple[torch.Tensor, ...], *, has_input_intervals: bool
) -> tuple[IntervalTensor, torch.Tensor]:
    """Split one certificate batch into (input bounds, multi-hot actions)."""

    if has_input_intervals:
        if len(batch) != 3:
            raise ValueError(
                "Interval certificate datasets must yield (X_l, X_u, multi_hot_actions)."
            )
        x_l, x_u, targets = batch
        return IntervalTensor(x_l, x_u), targets
    if len(batch) != 2:
        raise ValueError(
            "Point certificate datasets must yield (inputs, multi_hot_actions)."
        )
    inputs, targets = batch
    return IntervalTensor(inputs), targets


def _set_flat_params(model: torch.nn.Module, flat: torch.Tensor) -> None:
    """Copy a flat parameter vector into ``model`` in ``parameters()`` order."""

    params = list(model.parameters())
    for param, value in zip(params, unflatten_like(flat, params)):
        param.data.copy_(value)


def certify_sub_segment(
    model: torch.nn.Sequential,
    theta_k_flat: torch.Tensor,
    delta_flat: torch.Tensor,
    a: float,
    b: float,
    batches: list[tuple[torch.Tensor, ...]],
    *,
    multi_label_mode: Literal["any", "all"] = "all",
    has_input_intervals: bool = False,
) -> bool:
    """Evaluate ``C(a, b)``: is every policy on the sub-segment greedy-safe?

    Mutates ``model``'s parameters to the sub-segment centre; the caller is
    responsible for restoring them (see :func:`compute_segment_rashomon_set`).
    Returns ``False`` as soon as one batch fails, so a step that is far too long
    costs a single batch rather than a pass over the whole certificate set.
    """

    centre = theta_k_flat + 0.5 * (a + b) * delta_flat
    _set_flat_params(model, centre)
    generators = (0.5 * (b - a) * delta_flat).unsqueeze(0)
    coefficients = IntervalTensor(
        -torch.ones(1, device=generators.device, dtype=generators.dtype),
        torch.ones(1, device=generators.device, dtype=generators.dtype),
    )

    for batch in batches:
        inputs, targets = _unpack_batch(batch, has_input_intervals=has_input_intervals)
        logits = bound_forward_pass(
            model,
            generators,
            coefficients,
            inputs,
            use_zonotopes=True,
        )
        hard = verify.bound_multi_label_accuracy(
            logits,
            targets,
            lower=True,
            aggregation="min",
            mode=multi_label_mode,
        )
        if float(hard.item()) < 1.0:
            return False
    return True


def certify_segment(
    model: torch.nn.Sequential,
    theta_k_flat: torch.Tensor,
    delta_flat: torch.Tensor,
    alpha: float,
    splits: int,
    batches: list[tuple[torch.Tensor, ...]],
    *,
    multi_label_mode: Literal["any", "all"] = "all",
    has_input_intervals: bool = False,
) -> bool:
    """Evaluate ``C_K(alpha)`` over an equal ``splits``-part partition."""

    if splits <= 0:
        raise ValueError(f"splits must be positive, got {splits}.")
    for index in range(int(splits)):
        a = alpha * index / splits
        b = alpha * (index + 1) / splits
        if not certify_sub_segment(
            model,
            theta_k_flat,
            delta_flat,
            a,
            b,
            batches,
            multi_label_mode=multi_label_mode,
            has_input_intervals=has_input_intervals,
        ):
            return False
    return True


def compute_segment_rashomon_set(
    model: torch.nn.Sequential,
    dataset: torch.utils.data.Dataset,
    *,
    delta: list[torch.Tensor],
    tolerance: float = 1e-3,
    splits: int = 4,
    max_splits: int = 8,
    batch_size: int = 500,
    multi_label_mode: Literal["any", "all"] = "all",
    has_input_intervals: bool = False,
) -> SegmentRashomonResult:
    """Find the longest certified step from ``model``'s parameters along ``delta``.

    ``model``'s current parameters are ``theta_k``; ``delta`` is the proposed
    update in ``model.parameters()`` order.  Returns the certified rank-one
    zonotope ``Z(0, alpha_hat)``, or an empty region list when no positive step
    certifies.  ``model`` is left holding ``theta_k`` again on every path.

    The bisection tests midpoints, so a step that certifies only after
    refinement is returned to within ``tolerance`` rather than exactly; the
    ``alpha = 1`` fast path is retried after each refinement so an update that
    is safe outright is still accepted exactly.

    ``splits`` defaults to 4 from a measured sweep on FrozenLake: against the
    true maximal step ``alpha_star`` (found by dense search), the certified
    ``alpha_hat`` reaches 0.91 of it at K=1, 0.97 at K=4 and 0.98 at K=8, with
    cost linear in K and flat beyond K=8.
    """

    if not 0.0 < float(tolerance) < 1.0:
        raise ValueError(f"tolerance must lie in (0, 1), got {tolerance}.")
    if int(splits) <= 0:
        raise ValueError(f"splits must be positive, got {splits}.")
    if int(max_splits) < int(splits):
        raise ValueError(
            f"max_splits ({max_splits}) must be at least splits ({splits})."
        )

    params = list(model.parameters())
    if len(delta) != len(params):
        raise ValueError(
            "delta must provide one tensor per model parameter: "
            f"expected={len(params)}, got={len(delta)}."
        )
    for index, (step, param) in enumerate(zip(delta, params)):
        if tuple(step.shape) != tuple(param.shape):
            raise ValueError(
                f"delta shape mismatch at parameter {index}: "
                f"expected={tuple(param.shape)}, got={tuple(step.shape)}."
            )

    device = params[0].device
    dtype = params[0].dtype
    theta_k_flat = flatten_tensors([param.detach() for param in params]).to(
        device=device, dtype=dtype
    )
    delta_flat = flatten_tensors([step.detach() for step in delta]).to(
        device=device, dtype=dtype
    )
    param_shapes = [tuple(param.shape) for param in params]

    def build_result(
        alpha: float, splits_used: int, evaluations: int, certified: bool
    ) -> SegmentRashomonResult:
        regions: list[ZonotopeRegion] = []
        certificates: list[list[RashomonCertificate]] = []
        if certified and alpha > 0.0:
            half = 0.5 * alpha
            centre = theta_k_flat + half * delta_flat
            regions.append(
                ZonotopeRegion(
                    center_params=[
                        tensor.detach().cpu().clone()
                        for tensor in unflatten_like(centre, params)
                    ],
                    generators=(half * delta_flat).unsqueeze(0).detach().cpu().clone(),
                    coefficient_l=-torch.ones(1, dtype=dtype),
                    coefficient_u=torch.ones(1, dtype=dtype),
                    param_shapes=param_shapes,
                )
            )
            certificates.append(
                [RashomonCertificate(group=None, min_surrogate=1.0, min_hard_acc=1.0)]
            )
        return SegmentRashomonResult(
            regions=regions,
            certificates=certificates,
            alpha=float(alpha) if certified else 0.0,
            splits_used=int(splits_used),
            iterations_run=int(evaluations),
            target_contained_and_certified=bool(certified and alpha >= 1.0),
        )

    # The certificate set is small enough to hold once; materialising it avoids
    # re-shuffling and re-collating on every one of the ~log2(1/tolerance) passes.
    batches = [
        tuple(tensor.to(device=device) for tensor in batch)
        for batch in DataLoader(dataset, batch_size=int(batch_size), shuffle=False)
    ]

    evaluations = 0
    current_splits = int(splits)
    lower = 0.0
    try:
        with torch.no_grad():
            while True:
                evaluations += 1
                if certify_segment(
                    model,
                    theta_k_flat,
                    delta_flat,
                    1.0,
                    current_splits,
                    batches,
                    multi_label_mode=multi_label_mode,
                    has_input_intervals=has_input_intervals,
                ):
                    return build_result(1.0, current_splits, evaluations, True)

                lower, upper = 0.0, 1.0
                while upper - lower > float(tolerance):
                    alpha = 0.5 * (lower + upper)
                    evaluations += 1
                    if certify_segment(
                        model,
                        theta_k_flat,
                        delta_flat,
                        alpha,
                        current_splits,
                        batches,
                        multi_label_mode=multi_label_mode,
                        has_input_intervals=has_input_intervals,
                    ):
                        lower = alpha
                    else:
                        upper = alpha

                if lower > 0.0 or current_splits >= int(max_splits):
                    break
                current_splits = min(2 * current_splits, int(max_splits))
    finally:
        _set_flat_params(model, theta_k_flat)

    return build_result(lower, current_splits, evaluations, lower > 0.0)


def select_certified_segment(
    result: SegmentRashomonResult,
) -> tuple[ZonotopeRegion, int] | None:
    """Select the certified segment, or ``None`` when the search failed closed."""

    for index, certificates in enumerate(result.certificates):
        worst = min(
            (certificate.min_hard_acc for certificate in certificates),
            default=float("-inf"),
        )
        if worst >= 1.0:
            return result.regions[index], int(index)
    return None


def segment_endpoint(region: ZonotopeRegion) -> list[torch.Tensor]:
    """The far endpoint ``theta_k + alpha_hat delta`` of a certified segment."""

    centre = flatten_tensors([tensor.detach() for tensor in region.center_params])
    generator = region.generators.detach().reshape(-1).to(
        device=centre.device, dtype=centre.dtype
    )
    endpoint = centre + generator
    return unflatten_like(endpoint, list(region.center_params))
