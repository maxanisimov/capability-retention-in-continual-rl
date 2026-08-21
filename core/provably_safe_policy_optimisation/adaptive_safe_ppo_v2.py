"""Adaptive provably-safe PPO v2: recompute safe regions after active projection.

``AdaptiveSafePPOV2`` starts from the same safe BC base policy as
``AdaptiveSafePPO`` but uses certified parameter regions to constrain PPO actor
updates. In the default ``gradient_step`` mode, a live certified region is
attached to the actor optimizer and each PPO gradient step is projected into
the current region. In ``train_phase`` mode, PPO first completes one full
``train()`` phase unconstrained; v2 then computes one certified region around
the previous last-safe policy and projects the phase-end candidate once.

With directional growth, the region is grown in the direction of the proposed
actor update. In ``gradient_step`` mode, the initial region is delayed until
the first nonzero actor proposal supplies a direction; a zero-width region at
the verified base policy protects the policy until that first region has been
certified. In ``train_phase`` mode, that first direction is the full phase
delta from the last-safe policy to the phase-end candidate.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Mapping

import torch as th

from provably_safe_policy_optimisation.adaptive_safe_ppo import (
    AdaptiveSafePPO,
    directional_objective_weights_from_update_deltas,
)
from provably_safe_policy_optimisation.projection import ProjectionResult
from provably_safe_policy_optimisation.regions import (
    OrthotopeRegion,
    SafeParameterRegion,
    project_to_region_union,
)

RegionUpdateMode = Literal["union", "replace"]
BudgetMode = Literal["per_computation", "total"]


def _projection_result_dict(result: ProjectionResult | None) -> dict[str, float | int] | None:
    if result is None:
        return None
    return {
        "n_projected": int(result.n_projected),
        "n_boundary": int(result.n_boundary),
        "selected_set_index": int(result.selected_set_index),
        "displacement_l2": float(result.displacement_l2),
        "displacement_linf": float(result.displacement_linf),
    }


def _directional_masks_from_update_deltas(
    deltas: list[th.Tensor],
) -> tuple[list[th.Tensor], list[th.Tensor], dict[str, int]]:
    """Convert proposed parameter updates into one-sided freeze masks.

    Positive updates grow only the upper bound, negative updates grow only the
    lower bound, and exact zeros freeze both sides.
    """

    if not deltas:
        raise ValueError("Directional Rashomon growth requires at least one update tensor.")

    param_l_mask: list[th.Tensor] = []
    param_u_mask: list[th.Tensor] = []
    counts = {"positive": 0, "negative": 0, "zero": 0}
    for index, delta in enumerate(deltas):
        if not isinstance(delta, th.Tensor):
            raise TypeError(
                f"Proposed update delta {index} must be a torch.Tensor, "
                f"got {type(delta).__name__}."
            )
        if not bool(th.isfinite(delta).all()):
            raise ValueError(f"Proposed update delta {index} contains non-finite values.")
        detached = delta.detach()
        positive = detached > 0
        negative = detached < 0
        zero = detached == 0
        param_l_mask.append(positive | zero)
        param_u_mask.append(negative | zero)
        counts["positive"] += int(positive.sum().item())
        counts["negative"] += int(negative.sum().item())
        counts["zero"] += int(zero.sum().item())
    return param_l_mask, param_u_mask, counts


def _orthotope_contains_params(
    region: SafeParameterRegion,
    params: list[th.Tensor],
) -> bool:
    """Check exact containment without mutating the live policy."""

    if not isinstance(region, OrthotopeRegion) or len(params) != len(region.lower):
        return False
    return all(
        target.shape == lower.shape
        and bool(((lower <= target) & (target <= upper)).all())
        for target, lower, upper in zip(params, region.lower, region.upper)
    )


class AdaptiveSafePPOV2(AdaptiveSafePPO):
    """PPO with shielded exploration and projection-triggered region updates."""

    def __init__(
        self,
        *args: Any,
        base_policy_state_dict: Mapping[str, th.Tensor] | None = None,
        region_update_mode: RegionUpdateMode = "replace",
        rashomon_budget_mode: BudgetMode = "per_computation",
        rashomon_total_iters: int | None = None,
        rashomon_initial_n_iters: int | None = None,
        rashomon_recompute_n_iters: int | None = None,
        rashomon_max_region_computations: int | None = None,
        directional_rashomon_growth: bool = True,
        stop_when_proposal_contained: bool = True,
        adaptive_frequency: int = 1,
        compute_region_once: bool = False,
        **kwargs: Any,
    ) -> None:
        if region_update_mode not in ("union", "replace"):
            raise ValueError(
                "region_update_mode must be either 'union' or 'replace', "
                f"got {region_update_mode!r}.",
            )
        if rashomon_budget_mode not in ("per_computation", "total"):
            raise ValueError(
                "rashomon_budget_mode must be either 'per_computation' or 'total', "
                f"got {rashomon_budget_mode!r}.",
            )
        requested_granularity = kwargs.pop("adaptive_granularity", "gradient_step")
        if requested_granularity not in ("gradient_step", "train_phase"):
            raise ValueError(
                "adaptive_granularity must be 'gradient_step' or 'train_phase', "
                f"got {requested_granularity!r}.",
            )
        requested_strategy = kwargs.pop("unsafe_update_strategy", "rashomon_project")
        if requested_strategy != "rashomon_project":
            raise ValueError("AdaptiveSafePPOV2 always uses projection onto certified regions.")
        if directional_rashomon_growth and kwargs.get("safe_region_shape", "orthotope") != "orthotope":
            raise ValueError("Directional Rashomon growth requires safe_region_shape='orthotope'.")
        if stop_when_proposal_contained and kwargs.get("safe_region_shape", "orthotope") != "orthotope":
            raise ValueError(
                "Proposal-containment stopping requires safe_region_shape='orthotope'."
            )
        if int(adaptive_frequency) <= 0:
            raise ValueError(f"adaptive_frequency must be positive, got {adaptive_frequency}.")
        if compute_region_once and requested_granularity != "gradient_step":
            raise ValueError("compute_region_once requires gradient-step projection.")
        if compute_region_once and directional_rashomon_growth:
            raise ValueError("compute_region_once requires non-directional region growth.")

        self._region_update_mode: RegionUpdateMode = region_update_mode
        self._directional_rashomon_growth = bool(directional_rashomon_growth)
        self._stop_when_proposal_contained = bool(stop_when_proposal_contained)
        self._compute_region_once = bool(compute_region_once)
        self._rashomon_budget_mode: BudgetMode = rashomon_budget_mode
        self._rashomon_total_iters = (
            int(rashomon_total_iters) if rashomon_total_iters is not None else None
        )
        default_n_iters = int(kwargs.get("rashomon_n_iters", 100))
        default_region_n_iters = (
            int(self._rashomon_total_iters)
            if self._rashomon_budget_mode == "total" and self._rashomon_total_iters is not None
            else default_n_iters
        )
        self._rashomon_initial_n_iters = (
            int(rashomon_initial_n_iters)
            if rashomon_initial_n_iters is not None
            else default_region_n_iters
        )
        self._rashomon_recompute_n_iters = (
            int(rashomon_recompute_n_iters)
            if rashomon_recompute_n_iters is not None
            else self._rashomon_initial_n_iters
        )
        self._rashomon_max_region_computations = (
            int(rashomon_max_region_computations)
            if rashomon_max_region_computations is not None
            else None
        )
        if self._rashomon_initial_n_iters <= 0:
            raise ValueError(
                "rashomon_initial_n_iters must be positive, got "
                f"{self._rashomon_initial_n_iters}."
            )
        if self._rashomon_recompute_n_iters <= 0:
            raise ValueError(
                "rashomon_recompute_n_iters must be positive, got "
                f"{self._rashomon_recompute_n_iters}."
            )
        if self._rashomon_budget_mode == "total":
            if self._rashomon_total_iters is None:
                raise ValueError(
                    "rashomon_total_iters is required when rashomon_budget_mode='total'."
                )
            if self._rashomon_max_region_computations is None:
                raise ValueError(
                    "rashomon_max_region_computations is required when "
                    "rashomon_budget_mode='total'."
                )
            if self._rashomon_total_iters <= 0:
                raise ValueError(
                    f"rashomon_total_iters must be positive, got {self._rashomon_total_iters}."
                )
            if self._rashomon_max_region_computations <= 0:
                raise ValueError(
                    "rashomon_max_region_computations must be positive, got "
                    f"{self._rashomon_max_region_computations}."
                )
            if self._rashomon_initial_n_iters > self._rashomon_total_iters:
                raise ValueError(
                    "rashomon_initial_n_iters cannot exceed rashomon_total_iters."
                )
            if self._rashomon_recompute_n_iters > self._rashomon_total_iters:
                raise ValueError(
                    "rashomon_recompute_n_iters cannot exceed rashomon_total_iters."
                )
            kwargs["rashomon_n_iters"] = max(
                1,
                min(
                    max(self._rashomon_initial_n_iters, self._rashomon_recompute_n_iters),
                    self._rashomon_total_iters,
                ),
            )
            kwargs["rashomon_checkpoint"] = int(
                kwargs.get("rashomon_checkpoint") or max(1, kwargs["rashomon_n_iters"] // 10)
            )
        self._rashomon_iters_spent = 0
        self._initial_region_iters_spent = 0
        self._region_recompute_iters_spent = 0
        self._region_recompute_budget_exhaustions = 0
        self._active_regions: list[SafeParameterRegion] = []
        self._selected_checkpoint_indices: list[int] = []
        self._initial_region_computations = 0
        self._initial_region_failures = 0
        self._projection_triggers = 0
        self._region_recomputations = 0
        self._region_recompute_failures = 0
        self._directional_region_recomputations = 0
        self._directional_initial_region_computations = 0
        self._directional_growth_failures = 0
        self._proposal_containment_early_stops = 0
        self._proposed_updates_accepted_after_growth = 0
        self._rashomon_iterations_saved_by_containment = 0
        self._phase_region_computations = 0
        self._phase_projections = 0
        self._phase_exact_candidate_accepts = 0
        self._phase_projection_failures = 0
        self._last_direction_counts: dict[str, int] | None = None
        self._last_optimizer_projection_result: ProjectionResult | None = None
        self._last_phase_projection_result: ProjectionResult | None = None
        self._directional_initial_region_pending = self._directional_rashomon_growth

        super().__init__(
            *args,
            base_policy_state_dict=base_policy_state_dict,
            adaptive_granularity=requested_granularity,
            adaptive_frequency=int(adaptive_frequency),
            unsafe_update_strategy="rashomon_project",
            directional_rashomon_growth=directional_rashomon_growth,
            stop_when_proposal_contained=stop_when_proposal_contained,
            **kwargs,
        )

        if getattr(self, "policy", None) is not None and requested_granularity == "gradient_step":
            if self._directional_initial_region_pending:
                self._install_initial_safe_point()
            else:
                self._install_initial_region()

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "_active_regions",
            "_last_optimizer_projection_result",
            "_last_phase_projection_result",
        ]

    def _install_initial_region(self) -> None:
        self._initial_region_computations += 1
        region = self._compute_rashomon_around_last_safe(
            self._rashomon_initial_n_iters,
            region_kind="initial",
        )
        if region is None:
            self._initial_region_failures += 1
            raise ValueError("AdaptiveSafePPOV2 could not compute an initial certified region.")
        self._set_active_region(region, initial=True)
        self._attach_active_region()

    def _install_initial_safe_point(self) -> None:
        """Protect the base policy until the first actor proposal gives a direction."""

        point = [param.detach().clone() for param in self._live_actor_params]
        self._active_regions = [
            OrthotopeRegion(
                lower=[param.clone() for param in point],
                upper=[param.clone() for param in point],
            )
        ]
        self._attach_active_region()

    def _set_active_region(
        self,
        region_with_index: tuple[SafeParameterRegion, int],
        *,
        initial: bool,
    ) -> None:
        region, checkpoint_index = region_with_index
        if initial or self._region_update_mode == "replace":
            self._active_regions = [region]
        else:
            self._active_regions.append(region)
        self._selected_checkpoint_indices.append(int(checkpoint_index))

    def _attach_active_region(self) -> None:
        if self._adaptive_granularity == "train_phase":
            return
        self.policy.optimizer.set_regions(
            self._active_regions,
            params=self._live_actor_params,
            project_on_set=False,
        )

    def _rashomon_iters_remaining(self) -> int | None:
        if self._rashomon_budget_mode != "total":
            return None
        assert self._rashomon_total_iters is not None
        return max(0, int(self._rashomon_total_iters) - int(self._rashomon_iters_spent))

    def _compute_rashomon_around_last_safe(
        self,
        requested_n_iters: int | None = None,
        *,
        region_kind: str = "generic",
        param_l_mask: list[th.Tensor] | None = None,
        param_u_mask: list[th.Tensor] | None = None,
        param_objective_weights: list[th.Tensor] | None = None,
        stop_target_params: list[th.Tensor] | None = None,
    ) -> tuple[SafeParameterRegion, int] | None:
        n_iters = int(requested_n_iters or self._rashomon_n_iters)
        if n_iters <= 0:
            raise ValueError(f"requested_n_iters must be positive, got {n_iters}.")

        if self._rashomon_budget_mode != "total":
            return self._compute_rashomon_with_n_iters(
                n_iters,
                region_kind=region_kind,
                param_l_mask=param_l_mask,
                param_u_mask=param_u_mask,
                param_objective_weights=param_objective_weights,
                stop_target_params=stop_target_params,
            )

        remaining = self._rashomon_iters_remaining()
        if remaining is None or remaining <= 0:
            return None

        return self._compute_rashomon_with_n_iters(
            min(n_iters, int(remaining)),
            region_kind=region_kind,
            param_l_mask=param_l_mask,
            param_u_mask=param_u_mask,
            param_objective_weights=param_objective_weights,
            stop_target_params=stop_target_params,
        )

    def _compute_rashomon_with_n_iters(
        self,
        n_iters: int,
        *,
        region_kind: str,
        param_l_mask: list[th.Tensor] | None = None,
        param_u_mask: list[th.Tensor] | None = None,
        param_objective_weights: list[th.Tensor] | None = None,
        stop_target_params: list[th.Tensor] | None = None,
    ) -> tuple[SafeParameterRegion, int] | None:
        original_n_iters = self._rashomon_n_iters
        original_checkpoint = self._rashomon_checkpoint
        self._rashomon_n_iters = int(n_iters)
        self._rashomon_checkpoint = max(1, min(int(original_checkpoint), n_iters))
        computations_before = self._n_rashomon_computations
        try:
            region = super()._compute_rashomon_around_last_safe(
                param_l_mask=param_l_mask,
                param_u_mask=param_u_mask,
                param_objective_weights=param_objective_weights,
                stop_target_params=stop_target_params,
            )
        finally:
            self._rashomon_n_iters = original_n_iters
            self._rashomon_checkpoint = original_checkpoint

        if self._n_rashomon_computations > computations_before:
            iterations_run = max(0, min(n_iters, int(self._last_rashomon_iterations_run)))
            self._rashomon_iters_spent += iterations_run
            if region_kind == "initial":
                self._initial_region_iters_spent += iterations_run
            elif region_kind == "recompute":
                self._region_recompute_iters_spent += iterations_run
            if self._last_rashomon_target_contained_and_certified:
                self._proposal_containment_early_stops += 1
                self._rashomon_iterations_saved_by_containment += n_iters - iterations_run
        return region

    def _on_gradient_step(self) -> None:
        if self._adaptive_granularity != "gradient_step":
            return
        optimizer_result = getattr(self.policy.optimizer, "last_projection_result", None)
        self._last_optimizer_projection_result = optimizer_result
        prior_projection_active = bool(
            optimizer_result is not None and int(optimizer_result.n_projected) > 0
        )
        if self._compute_region_once:
            if optimizer_result is not None and int(optimizer_result.n_projected) > 0:
                self._projection_triggers += 1
            self._snapshot_last_safe()
            return

        building_initial_region = self._directional_initial_region_pending
        proposed_params = getattr(self.policy.optimizer, "last_proposed_params", None)
        if proposed_params is None or len(proposed_params) != len(self._live_actor_params):
            warnings.warn(
                "PSPO adaptive could not retain the proposed actor parameters; "
                "reverting to the previous safe policy.",
                stacklevel=2,
            )
            self._copy_live_actor_params_from(self._last_safe_params)
            if building_initial_region:
                self._initial_region_failures += 1
            else:
                self._region_recompute_failures += 1
            if self._directional_rashomon_growth:
                self._directional_growth_failures += 1
            return
        if prior_projection_active:
            self._projection_triggers += 1
        for index, (target, snapshot) in enumerate(
            zip(proposed_params, self._last_safe_params)
        ):
            if target.shape != snapshot.shape or not bool(th.isfinite(target).all()):
                warnings.warn(
                    f"PSPO adaptive received invalid proposed actor parameters at index {index}; "
                    "reverting to the previous safe policy.",
                    stacklevel=2,
                )
                self._copy_live_actor_params_from(self._last_safe_params)
                if building_initial_region:
                    self._initial_region_failures += 1
                else:
                    self._region_recompute_failures += 1
                return

        param_l_mask = None
        param_u_mask = None
        param_objective_weights = None
        if self._directional_rashomon_growth:
            self._last_direction_counts = None
            try:
                deltas = [
                    target - snapshot
                    for target, snapshot in zip(proposed_params, self._last_safe_params)
                ]
                param_l_mask, param_u_mask, counts = _directional_masks_from_update_deltas(deltas)
                param_objective_weights = (
                    directional_objective_weights_from_update_deltas(deltas)
                )
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    "PSPO adaptive could not construct directional safe-region masks "
                    f"({exc}); reverting to the previous safe policy.",
                    stacklevel=2,
                )
                self._copy_live_actor_params_from(self._last_safe_params)
                if building_initial_region:
                    self._initial_region_failures += 1
                else:
                    self._region_recompute_failures += 1
                self._directional_growth_failures += 1
                return
            self._last_direction_counts = counts
        remaining = self._rashomon_iters_remaining()
        if remaining is not None and remaining <= 0:
            self._copy_live_actor_params_from(self._last_safe_params)
            if building_initial_region:
                self._initial_region_failures += 1
            else:
                self._region_recompute_failures += 1
            self._region_recompute_budget_exhaustions += 1
            return
        if building_initial_region:
            self._initial_region_computations += 1
            self._directional_initial_region_computations += 1
        region = self._compute_rashomon_around_last_safe(
            (
                self._rashomon_initial_n_iters
                if building_initial_region
                else self._rashomon_recompute_n_iters
            ),
            region_kind="initial" if building_initial_region else "recompute",
            param_l_mask=param_l_mask,
            param_u_mask=param_u_mask,
            param_objective_weights=param_objective_weights,
            stop_target_params=(proposed_params if self._stop_when_proposal_contained else None),
        )
        if region is None:
            phase = (
                "directional initial-region computation"
                if building_initial_region
                else "projection-triggered region recomputation"
            )
            warnings.warn(
                f"PSPO adaptive {phase} failed; reverting to the previous safe policy.",
                stacklevel=2,
            )
            self._copy_live_actor_params_from(self._last_safe_params)
            if building_initial_region:
                self._initial_region_failures += 1
            else:
                self._region_recompute_failures += 1
            return

        self._set_active_region(region, initial=building_initial_region)
        self._attach_active_region()
        if building_initial_region:
            self._directional_initial_region_pending = False
        else:
            self._region_recomputations += 1
        self._copy_live_actor_params_from(proposed_params)
        if _orthotope_contains_params(region[0], proposed_params):
            self._proposed_updates_accepted_after_growth += 1
        else:
            distance_norm = getattr(self.policy.optimizer, "_distance_norm", "l2")
            with th.no_grad():
                result = project_to_region_union(
                    self._live_actor_params,
                    self._active_regions,
                    distance_norm=distance_norm,
                )
            if not prior_projection_active:
                self._projection_triggers += 1
            self._n_projections += 1
            self._last_projection_result = result
            self._last_optimizer_projection_result = result
        self._snapshot_last_safe()
        if self._directional_rashomon_growth and not building_initial_region:
            self._directional_region_recomputations += 1

    def _copy_live_actor_params_from(self, params: list[th.Tensor]) -> None:
        with th.no_grad():
            for live, source in zip(self._live_actor_params, params):
                live.data.copy_(source.to(device=live.device, dtype=live.dtype))

    def _phase_candidate_params(self) -> list[th.Tensor]:
        candidate = [param.detach().clone() for param in self._live_actor_params]
        if len(candidate) != len(self._last_safe_params):
            raise ValueError(
                "phase-end candidate has the wrong number of actor parameter tensors "
                f"({len(candidate)} != {len(self._last_safe_params)})."
            )
        for index, (target, snapshot) in enumerate(zip(candidate, self._last_safe_params)):
            if target.shape != snapshot.shape:
                raise ValueError(
                    f"phase-end candidate tensor {index} has shape {tuple(target.shape)}, "
                    f"expected {tuple(snapshot.shape)}."
                )
            if not bool(th.isfinite(target).all()):
                raise ValueError(f"phase-end candidate tensor {index} contains non-finite values.")
        return candidate

    def _on_train_phase_end(self) -> None:
        candidate_params = self._phase_candidate_params()
        building_initial_region = (
            len(self._active_regions) == 0 or self._directional_initial_region_pending
        )
        param_l_mask = None
        param_u_mask = None
        param_objective_weights = None
        proposed_params = candidate_params if self._stop_when_proposal_contained else None
        if self._directional_rashomon_growth:
            self._last_direction_counts = None
            deltas = [
                candidate.detach() - snapshot.detach()
                for candidate, snapshot in zip(candidate_params, self._last_safe_params)
            ]
            try:
                param_l_mask, param_u_mask, counts = _directional_masks_from_update_deltas(deltas)
                param_objective_weights = (
                    directional_objective_weights_from_update_deltas(deltas)
                )
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    "AdaptiveSafePPOV2 could not construct train-phase directional "
                    f"Rashomon masks ({exc}); reverting to the previous certified policy.",
                    stacklevel=2,
                )
                self._copy_live_actor_params_from(self._last_safe_params)
                if building_initial_region:
                    self._initial_region_failures += 1
                else:
                    self._region_recompute_failures += 1
                self._directional_growth_failures += 1
                self._phase_projection_failures += 1
                return
            self._last_direction_counts = counts

        remaining = self._rashomon_iters_remaining()
        if remaining is not None and remaining <= 0:
            self._copy_live_actor_params_from(self._last_safe_params)
            if building_initial_region:
                self._initial_region_failures += 1
            else:
                self._region_recompute_failures += 1
            self._region_recompute_budget_exhaustions += 1
            self._phase_projection_failures += 1
            return

        self._phase_region_computations += 1
        if building_initial_region:
            self._initial_region_computations += 1
            if self._directional_rashomon_growth:
                self._directional_initial_region_computations += 1

        region = self._compute_rashomon_around_last_safe(
            (
                self._rashomon_initial_n_iters
                if building_initial_region
                else self._rashomon_recompute_n_iters
            ),
            region_kind="initial" if building_initial_region else "recompute",
            param_l_mask=param_l_mask,
            param_u_mask=param_u_mask,
            param_objective_weights=param_objective_weights,
            stop_target_params=proposed_params,
        )
        if region is None:
            phase = "initial-region computation" if building_initial_region else "region recomputation"
            warnings.warn(
                f"AdaptiveSafePPOV2 train-phase {phase} failed; reverting to the "
                "previous certified policy.",
                stacklevel=2,
            )
            self._copy_live_actor_params_from(self._last_safe_params)
            if building_initial_region:
                self._initial_region_failures += 1
            else:
                self._region_recompute_failures += 1
            self._phase_projection_failures += 1
            return

        self._set_active_region(region, initial=building_initial_region)
        self._attach_active_region()
        if building_initial_region:
            self._directional_initial_region_pending = False
        else:
            self._region_recomputations += 1
        if self._directional_rashomon_growth and not building_initial_region:
            self._directional_region_recomputations += 1

        if (
            proposed_params is not None
            and self._last_rashomon_target_contained_and_certified
            and _orthotope_contains_params(region[0], proposed_params)
        ):
            self._copy_live_actor_params_from(proposed_params)
            self._snapshot_last_safe()
            self._proposed_updates_accepted_after_growth += 1
            self._phase_exact_candidate_accepts += 1
            return

        distance_norm = getattr(self.policy.optimizer, "_distance_norm", "l2")
        with th.no_grad():
            result = project_to_region_union(
                self._live_actor_params,
                self._active_regions,
                distance_norm=distance_norm,
            )
        self._n_projections += 1
        self._phase_projections += 1
        self._last_projection_result = result
        self._last_phase_projection_result = result
        self._snapshot_last_safe()

    def train(self, *args: Any, **kwargs: Any) -> Any:
        if self._adaptive_granularity != "train_phase":
            return super().train(*args, **kwargs)

        self._ensure_adaptive_state()
        result = super(AdaptiveSafePPO, self).train(*args, **kwargs)
        self._train_phases_since_enforcement += 1
        self._pending_adaptive_update = True
        if self._train_phases_since_enforcement < self._adaptive_frequency:
            return result
        try:
            self._on_train_phase_end()
        except ValueError as exc:
            warnings.warn(
                f"AdaptiveSafePPOV2 train-phase projection failed ({exc}); reverting "
                "to the previous certified policy.",
                stacklevel=2,
            )
            self._copy_live_actor_params_from(self._last_safe_params)
            self._phase_projection_failures += 1
        self._train_phases_since_enforcement = 0
        self._pending_adaptive_update = False
        return result

    def finalize_adaptive_update(self) -> None:
        """Project a pending aggregate rollout update before save/evaluation."""

        self._ensure_adaptive_state()
        if self._adaptive_granularity != "train_phase" or not self._pending_adaptive_update:
            return
        try:
            self._on_train_phase_end()
        except ValueError as exc:
            warnings.warn(
                f"PSPO adaptive final projection failed ({exc}); reverting to the "
                "previous certified policy.",
                stacklevel=2,
            )
            self._copy_live_actor_params_from(self._last_safe_params)
            self._phase_projection_failures += 1
        self._train_phases_since_enforcement = 0
        self._pending_adaptive_update = False
        self._final_flushes += 1

    def adaptive_diagnostics(self) -> dict[str, Any]:
        base = super().adaptive_diagnostics()
        base.update(
            {
                "compute_region_once": self._compute_region_once,
                "region_update_mode": self._region_update_mode,
                "directional_rashomon_growth": self._directional_rashomon_growth,
                "stop_when_proposal_contained": self._stop_when_proposal_contained,
                "rashomon_budget_mode": self._rashomon_budget_mode,
                "rashomon_total_iters": self._rashomon_total_iters,
                "rashomon_initial_n_iters": int(self._rashomon_initial_n_iters),
                "rashomon_recompute_n_iters": int(self._rashomon_recompute_n_iters),
                "rashomon_iters_spent": int(self._rashomon_iters_spent),
                "initial_region_iters_spent": int(self._initial_region_iters_spent),
                "region_recompute_iters_spent": int(self._region_recompute_iters_spent),
                "rashomon_iters_remaining": self._rashomon_iters_remaining(),
                "rashomon_max_region_computations": self._rashomon_max_region_computations,
                "initial_region_computations": int(self._initial_region_computations),
                "initial_region_failures": int(self._initial_region_failures),
                "projection_triggers": int(self._projection_triggers),
                "region_recomputations": int(self._region_recomputations),
                "region_recompute_failures": int(self._region_recompute_failures),
                "directional_region_recomputations": int(
                    self._directional_region_recomputations
                ),
                "directional_initial_region_pending": bool(
                    self._directional_initial_region_pending
                ),
                "directional_initial_region_computations": int(
                    self._directional_initial_region_computations
                ),
                "directional_growth_failures": int(self._directional_growth_failures),
                "proposal_containment_early_stops": int(
                    self._proposal_containment_early_stops
                ),
                "proposed_updates_accepted_after_growth": int(
                    self._proposed_updates_accepted_after_growth
                ),
                "rashomon_iterations_saved_by_containment": int(
                    self._rashomon_iterations_saved_by_containment
                ),
                "phase_region_computations": int(self._phase_region_computations),
                "phase_projections": int(self._phase_projections),
                "phase_exact_candidate_accepts": int(self._phase_exact_candidate_accepts),
                "phase_projection_failures": int(self._phase_projection_failures),
                "last_rashomon_iterations_run": int(self._last_rashomon_iterations_run),
                "last_rashomon_target_contained_and_certified": bool(
                    self._last_rashomon_target_contained_and_certified
                ),
                "last_direction_counts": (
                    None if self._last_direction_counts is None else dict(self._last_direction_counts)
                ),
                "region_recompute_budget_exhaustions": int(
                    self._region_recompute_budget_exhaustions
                ),
                "current_region_count": int(len(self._active_regions)),
                "selected_checkpoint_indices": list(self._selected_checkpoint_indices),
                "last_optimizer_projection": _projection_result_dict(
                    self._last_optimizer_projection_result
                ),
                "last_phase_projection": _projection_result_dict(
                    self._last_phase_projection_result
                ),
            }
        )
        return base

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> "AdaptiveSafePPOV2":
        model = super().load(*args, **kwargs)
        warnings.warn(
            "AdaptiveSafePPOV2 loaded without persisted active regions; construct a fresh "
            "model from the safe base policy to resume v2 training with certified regions.",
            stacklevel=2,
        )
        return model
