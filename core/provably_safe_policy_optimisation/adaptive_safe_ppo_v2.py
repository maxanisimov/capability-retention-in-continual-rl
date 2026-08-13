"""Adaptive provably-safe PPO v2: recompute safe regions after active projection.

``AdaptiveSafePPOV2`` starts from the same safe BC base policy as
``AdaptiveSafePPO`` but keeps a live certified parameter region attached to the
actor optimizer. Each PPO gradient step is projected into the current region.
Only when that projection clamps at least one actor parameter do we compute a new
certified Rashomon box around the projected policy.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal, Mapping

import torch as th

from provably_safe_policy_optimisation.adaptive_safe_ppo import AdaptiveSafePPO
from provably_safe_policy_optimisation.regions import SafeParameterRegion
from provably_safe_policy_optimisation.projection import ProjectionResult

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


class AdaptiveSafePPOV2(AdaptiveSafePPO):
    """PPO with shielded exploration and projection-triggered region updates."""

    def __init__(
        self,
        *args: Any,
        base_policy_state_dict: Mapping[str, th.Tensor] | None = None,
        region_update_mode: RegionUpdateMode = "union",
        rashomon_budget_mode: BudgetMode = "per_computation",
        rashomon_total_iters: int | None = None,
        rashomon_initial_n_iters: int | None = None,
        rashomon_recompute_n_iters: int | None = None,
        rashomon_max_region_computations: int | None = None,
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
        if requested_granularity != "gradient_step":
            raise ValueError("AdaptiveSafePPOV2 treats every optimizer step as one policy update.")
        requested_strategy = kwargs.pop("unsafe_update_strategy", "rashomon_project")
        if requested_strategy != "rashomon_project":
            raise ValueError("AdaptiveSafePPOV2 always uses projection onto certified regions.")

        self._region_update_mode: RegionUpdateMode = region_update_mode
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
        self._last_optimizer_projection_result: ProjectionResult | None = None

        super().__init__(
            *args,
            base_policy_state_dict=base_policy_state_dict,
            adaptive_granularity="gradient_step",
            unsafe_update_strategy="rashomon_project",
            **kwargs,
        )

        if getattr(self, "policy", None) is not None:
            self._install_initial_region()

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "_active_regions",
            "_last_optimizer_projection_result",
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
    ) -> tuple[SafeParameterRegion, int] | None:
        n_iters = int(requested_n_iters or self._rashomon_n_iters)
        if n_iters <= 0:
            raise ValueError(f"requested_n_iters must be positive, got {n_iters}.")

        if self._rashomon_budget_mode != "total":
            return self._compute_rashomon_with_n_iters(n_iters, region_kind=region_kind)

        remaining = self._rashomon_iters_remaining()
        if remaining is None or remaining <= 0:
            return None

        return self._compute_rashomon_with_n_iters(
            min(n_iters, int(remaining)),
            region_kind=region_kind,
        )

    def _compute_rashomon_with_n_iters(
        self,
        n_iters: int,
        *,
        region_kind: str,
    ) -> tuple[SafeParameterRegion, int] | None:
        original_n_iters = self._rashomon_n_iters
        original_checkpoint = self._rashomon_checkpoint
        self._rashomon_n_iters = int(n_iters)
        self._rashomon_checkpoint = max(1, min(int(original_checkpoint), n_iters))
        computations_before = self._n_rashomon_computations
        try:
            region = super()._compute_rashomon_around_last_safe()
        finally:
            self._rashomon_n_iters = original_n_iters
            self._rashomon_checkpoint = original_checkpoint

        if self._n_rashomon_computations > computations_before:
            self._rashomon_iters_spent += n_iters
            if region_kind == "initial":
                self._initial_region_iters_spent += n_iters
            elif region_kind == "recompute":
                self._region_recompute_iters_spent += n_iters
        return region

    def _on_gradient_step(self) -> None:
        result = getattr(self.policy.optimizer, "last_projection_result", None)
        self._last_optimizer_projection_result = result
        if result is None or int(result.n_projected) <= 0:
            return

        self._projection_triggers += 1
        self._snapshot_last_safe()
        remaining = self._rashomon_iters_remaining()
        if remaining is not None and remaining <= 0:
            self._region_recompute_failures += 1
            self._region_recompute_budget_exhaustions += 1
            return
        region = self._compute_rashomon_around_last_safe(
            self._rashomon_recompute_n_iters,
            region_kind="recompute",
        )
        if region is None:
            warnings.warn(
                "AdaptiveSafePPOV2 projection-triggered region recomputation failed; "
                "continuing with the previous certified region.",
                stacklevel=2,
            )
            self._region_recompute_failures += 1
            return

        self._set_active_region(region, initial=False)
        self._attach_active_region()
        self._region_recomputations += 1

    def adaptive_diagnostics(self) -> dict[str, Any]:
        base = super().adaptive_diagnostics()
        base.update(
            {
                "version": "v2",
                "region_update_mode": self._region_update_mode,
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
                "region_recompute_budget_exhaustions": int(
                    self._region_recompute_budget_exhaustions
                ),
                "current_region_count": int(len(self._active_regions)),
                "selected_checkpoint_indices": list(self._selected_checkpoint_indices),
                "last_optimizer_projection": _projection_result_dict(
                    self._last_optimizer_projection_result
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
