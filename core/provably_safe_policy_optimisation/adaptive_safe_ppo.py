"""Adaptive provably-safe PPO: verify each update, project only when needed.

:class:`AdaptiveSafePPO` keeps :class:`~provably_safe_policy_optimisation.
provably_safe_ppo.ProvablySafePPO`'s shielded exploration but replaces the
*static* precomputed Rashomon bounds with an adaptive accept/project scheme
over policy iterates:

* Iterate ``k=0`` is a user-supplied **safe base policy** (greedy action is
  shield-safe in every state), loaded into the PPO actor at construction.
* After each policy update, the candidate iterate ``k+1`` is **verified** with
  an exact greedy check: the argmax of the policy logits must be shield-safe in
  every state with at least one safe action (exact for ``Discrete``
  observation spaces, where the one-hot state enumeration is exhaustive).
* If the candidate is safe it is **accepted** — no fallback work is done.
* If it is unsafe, ``unsafe_update_strategy`` selects the fallback:

  - ``"rashomon_project"`` (default): a Rashomon set is computed **around
    iterate k** (the last safe iterate) with a fixed ``rashomon_n_iters``
    budget, and the candidate is projected onto the largest IBP-certified box
    -- projected gradient descent onto a certified-safe parameter region. The
    projection is safe by construction (every policy inside a certified box
    picks only shield-safe greedy actions), so it becomes iterate ``k+1``
    without re-verification. If no certified box is found, the candidate is
    reverted to iterate ``k``.
  - ``"none"``: monitor-only ablation. The candidate is verified and the
    outcome recorded, but nothing is projected or reverted, so training is
    exactly PPO from a BC-initialised policy with shielded exploration. Use it
    to measure what the safety correction is actually buying:
    ``safe_update_fraction`` reports the share of updates that stayed safe by
    themselves, and the usual evaluation reports the resulting safety and
    return. **This mode provides no safety guarantee.**

  A third strategy, ``"line_search"`` (bisection line search toward the
  candidate), was removed after a bug was found in it. ``rashomon_project``
  is now the only certified correction behaviour.

``adaptive_granularity`` selects what counts as one policy update:
``"gradient_step"`` (default) treats every optimizer step inside ``train()``
as a candidate (via :attr:`ProjectedAdam.post_step_hook`);``"train_phase"``
treats one full PPO update phase as a candidate.

Unlike :class:`ProjectedPPO`, no static parameter bounds are attached — the
optimizer stays plain Adam and one-shot projections are applied directly.

Usage
-----
>>> from provably_safe_policy_optimisation import AdaptiveSafePPO
>>> model = AdaptiveSafePPO(
...     "MlpPolicy", env, shield=mask,
...     base_policy_state_dict=safe_actor_state_dict,  # PPO policy param names
...     rashomon_n_iters=100,
... )
>>> model.learn(total_timesteps=100_000)
>>> model.adaptive_diagnostics()

Save/Load notes
---------------
The live/frozen actor views, verification caches, and the last-safe snapshot
are not persisted. After ``AdaptiveSafePPO.load(...)`` they are rebuilt lazily
on the next ``train()`` (or explicitly via :meth:`mark_current_policy_safe`),
which re-verifies the loaded policy to re-establish the safety invariant.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Literal, Mapping

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from torch.utils.data import TensorDataset

from provably_safe_policy_optimisation.policy_introspection import (
    extract_feature_actor_parameters_and_network,
)
from provably_safe_policy_optimisation.projected_ppo import (
    projection_target_parameter_names,
)
from provably_safe_policy_optimisation.projection import (
    ProjectionResult,
)
from provably_safe_policy_optimisation.regions import (
    OrthotopeRegion,
    SafeParameterRegion,
    project_to_region_union,
)
from provably_safe_policy_optimisation.provably_safe_ppo import ProvablySafePPO
from provably_safe_policy_optimisation.safe_init import _greedy_safe_rate

AdaptiveGranularity = Literal["gradient_step", "train_phase"]
UnsafeUpdateStrategy = Literal["rashomon_project", "none"]


def directional_masks_from_update_deltas(
    deltas: list[th.Tensor],
) -> tuple[list[th.Tensor], list[th.Tensor], dict[str, int]]:
    """Freeze each parameter bound opposite to the proposed update direction."""

    if not deltas:
        raise ValueError("Directional safe-region growth requires at least one update tensor.")
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


def directional_objective_weights_from_update_deltas(
    deltas: list[th.Tensor],
) -> list[th.Tensor]:
    """Return ``abs(delta)`` weights aligned with actor parameters."""

    if not deltas:
        raise ValueError("Directional safe-region growth requires at least one update tensor.")
    weights = []
    for index, delta in enumerate(deltas):
        if not isinstance(delta, th.Tensor):
            raise TypeError(
                f"Proposed update delta {index} must be a torch.Tensor, "
                f"got {type(delta).__name__}."
            )
        if not bool(th.isfinite(delta).all()):
            raise ValueError(f"Proposed update delta {index} contains non-finite values.")
        weights.append(delta.detach().abs().clone())
    return weights


def orthotope_contains_params(
    region: SafeParameterRegion,
    params: list[th.Tensor],
) -> bool:
    """Return whether an orthotope contains an aligned parameter list."""

    if not isinstance(region, OrthotopeRegion) or len(params) != len(region.lower):
        return False
    return all(
        target.shape == lower.shape
        and bool(((lower <= target) & (target <= upper)).all())
        for target, lower, upper in zip(params, region.lower, region.upper)
    )


def shield_safe_behaviour_dataset(
    mask: Any, state_to_features: Any = None
) -> tuple[TensorDataset, np.ndarray]:
    """Build the shield's safe-behaviour dataset for verification / calibration.

    Returns ``(dataset, safe_state_ids)`` where the dataset yields
    ``(state_representation, multi_hot_safe_actions)`` float32 rows, one per
    state with at least one safe action (states without any safe action are
    excluded, mirroring the offline ``compute_shield_rashomon_set`` stage).

    ``state_to_features`` selects the representation, matching the policy input:
    ``None`` gives one-hot state ids; a callable gives the env's decoded feature
    vectors (features mode), so the exact greedy verifier feeds the same feature
    input the live policy consumes.
    """
    mask_arr = np.asarray(mask, dtype=np.float32)
    if mask_arr.ndim != 2:
        raise ValueError(f"Expected shield mask shape (n_states, n_actions), got {mask_arr.shape}.")
    n_states = int(mask_arr.shape[0])
    safe_state_ids = np.flatnonzero(mask_arr.sum(axis=1) > 0)
    if safe_state_ids.size == 0:
        raise ValueError("Shield contains no states with at least one safe action.")
    if state_to_features is None:
        states = F.one_hot(
            th.as_tensor(safe_state_ids, dtype=th.long), num_classes=n_states
        ).to(th.float32)
    else:
        feature_rows = np.stack([np.asarray(state_to_features(int(s))) for s in safe_state_ids])
        states = th.as_tensor(feature_rows, dtype=th.float32)
    actions = th.as_tensor(mask_arr[safe_state_ids], dtype=th.float32)
    return TensorDataset(states, actions), safe_state_ids


def calibrate_inverse_temperature(
    logits: th.Tensor,
    masks: th.Tensor,
    *,
    start: int = 1,
    cap: int = 1000,
    multi_label_mode: Literal["any", "all"] = "any",
    surrogate: Literal["auto", "probability", "logsumexp"] = "auto",
) -> int:
    """First integer inverse temperature whose per-state surrogate is feasible.

    Same calibration rule as the offline ``compute_shield_rashomon_set`` stage:
    evaluate the selected Rashomon surrogate independently for every state. In
    probability-based ``"any"`` mode, each state's threshold is
    ``n_valid / (1 + n_valid)`` using that state's own number of valid actions.
    Raises ``ValueError`` if no inverse temperature in ``[start, cap]`` works.
    """
    if start > cap:
        raise ValueError(f"start must be <= cap; got start={start}, cap={cap}.")
    if multi_label_mode not in ("any", "all"):
        raise ValueError(
            "multi_label_mode must be either 'any' or 'all', "
            f"got {multi_label_mode!r}."
        )
    from src.IntervalTensor import IntervalTensor
    from src.verification import verify

    resolved_surrogate = verify.resolve_surrogate_form(multi_label_mode, surrogate)
    masks = masks.to(dtype=logits.dtype)
    if not bool(masks.bool().any(dim=1).all().item()):
        raise ValueError("Safe-action masks contain no valid actions.")
    min_margin = float("-inf")
    point_logits = IntervalTensor(logits, logits)
    with th.no_grad():
        for inverse_temp in range(int(start), int(cap) + 1):
            margins = verify.bound_multi_label_accuracy_margin(
                point_logits,
                masks,
                tau=1.0 / float(inverse_temp),
                lower=True,
                aggregation="none",
                mode=multi_label_mode,
                surrogate=surrogate,
            )
            min_margin = float(margins.min().item())
            feasible = (
                min_margin >= 0.0
                if resolved_surrogate == "probability" and multi_label_mode == "any"
                else min_margin > 0.0
            )
            if feasible:
                return int(inverse_temp)
    raise ValueError(
        "Could not calibrate inverse temperature for the Rashomon surrogate "
        f"({resolved_surrogate}): min_state_specific_margin={min_margin:.6f}."
    )


def _run_rashomon_engine(
    model: th.nn.Sequential,
    dataset: TensorDataset,
    *,
    n_iters: int,
    checkpoint: int,
    batch_size: int,
    certificate_samples: int,
    inverse_temp: int,
    seed: int,
    multi_label_mode: str = "any",
    surrogate: str = "auto",
    param_l_mask: list[th.Tensor] | None = None,
    param_u_mask: list[th.Tensor] | None = None,
    param_objective_weights: list[th.Tensor] | None = None,
    stop_target_params: list[th.Tensor] | None = None,
) -> Any:
    """Run the IBP Rashomon-set engine around ``model``'s current parameters.

    Module-level indirection so tests can monkeypatch the expensive Cooper
    loop. The import is lazy to keep the ``cooper`` dependency optional.
    """
    from src.interval_utils import compute_rashomon_set

    return compute_rashomon_set(
        model,
        dataset,
        1.0,
        batch_size=int(batch_size),
        certificate_samples=int(certificate_samples),
        n_iters=int(n_iters),
        checkpoint=int(checkpoint),
        temperatures={None: 1.0 / float(inverse_temp)},
        seed=int(seed),
        multi_label_mode=multi_label_mode,
        surrogate=surrogate,
        param_l_mask=param_l_mask,
        param_u_mask=param_u_mask,
        param_objective_weights=param_objective_weights,
        stop_target_params=stop_target_params,
    )


def _run_zonotope_rashomon_engine(
    model: th.nn.Sequential,
    dataset: TensorDataset,
    *,
    n_iters: int,
    checkpoint: int,
    batch_size: int,
    certificate_samples: int,
    inverse_temp: int,
    seed: int,
    zonotope_rank: int | None,
    multi_label_mode: str = "any",
    surrogate: str = "auto",
) -> Any:
    """Run the zonotope safe-region engine around ``model``'s parameters."""

    from src.zonotope_rashomon import compute_zonotope_rashomon_set

    return compute_zonotope_rashomon_set(
        model,
        dataset,
        rank=zonotope_rank,
        n_iters=int(n_iters),
        checkpoint=int(checkpoint),
        batch_size=int(batch_size),
        certificate_samples=int(certificate_samples),
        inverse_temp=int(inverse_temp),
        seed=int(seed),
        multi_label_mode=multi_label_mode,  # type: ignore[arg-type]
        surrogate=surrogate,  # type: ignore[arg-type]
    )


def select_certified_box(
    result: Any,
) -> tuple[list[th.Tensor], list[th.Tensor], int] | None:
    """Select the last fully-certified box of a ``RashomonResult``.

    A checkpoint qualifies when its worst per-group ``min_hard_acc`` reaches
    1.0 — i.e. every policy inside the box is IBP-certified to pick only
    shield-safe greedy actions on the certificate batch. Returns detached
    clones ``(param_bounds_l, param_bounds_u, checkpoint_index)``, or ``None``
    when no checkpoint is fully certified (caller must fall back).
    """
    cert_values = [
        min((certificate.min_hard_acc for certificate in certificates), default=float("-inf"))
        for certificates in result.certificates
    ]
    valid_indices = [idx for idx, value in enumerate(cert_values) if value >= 1.0]
    if not valid_indices:
        return None
    selected_idx = valid_indices[-1]
    bounded_model = result.bounded_models[selected_idx]
    param_bounds_l = [param.detach().clone() for param in bounded_model.param_l]
    param_bounds_u = [param.detach().clone() for param in bounded_model.param_u]
    return param_bounds_l, param_bounds_u, int(selected_idx)


def select_certified_region(
    result: Any,
    *,
    safe_region_shape: Literal["orthotope", "zonotope"],
) -> tuple[SafeParameterRegion, int] | None:
    """Select the last fully certified safe region."""

    if safe_region_shape == "orthotope":
        selected = select_certified_box(result)
        if selected is None:
            return None
        lower, upper, idx = selected
        return OrthotopeRegion(lower=lower, upper=upper), idx
    from src.zonotope_rashomon import select_certified_zonotope

    return select_certified_zonotope(result)


class AdaptiveSafePPO(ProvablySafePPO):
    """PPO with shielded exploration and adaptive verify-then-project updates."""

    def __init__(
        self,
        *args: Any,
        base_policy_state_dict: Mapping[str, th.Tensor] | None = None,
        adaptive_granularity: AdaptiveGranularity = "gradient_step",
        adaptive_frequency: int = 1,
        unsafe_update_strategy: UnsafeUpdateStrategy = "rashomon_project",
        rashomon_n_iters: int = 100,
        rashomon_checkpoint: int | None = None,
        rashomon_batch_size: int = 500,
        rashomon_certificate_samples: int | None = None,
        rashomon_inverse_temperature: int | None = None,
        rashomon_multi_label_mode: Literal["any", "all"] = "all",
        rashomon_surrogate: Literal["auto", "probability", "logsumexp"] = "logsumexp",
        safe_region_shape: Literal["orthotope", "zonotope"] = "orthotope",
        zonotope_rank: int | None = None,
        rashomon_seed: int | None = None,
        directional_rashomon_growth: bool = True,
        stop_when_proposal_contained: bool = True,
        verify_base_policy: bool = True,
        state_to_features: Any = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        base_policy_state_dict:
            Safe base policy (iterate ``k=0``): tensors keyed by PPO policy
            parameter names, covering exactly the actor projection target (see
            :func:`projection_target_parameter_names`). ``None`` keeps the
            random initialisation — then pass ``verify_base_policy=False`` and
            establish safety yourself (e.g. ``pretrain_on_shield()`` followed
            by :meth:`mark_current_policy_safe`).
        adaptive_granularity:
            ``"gradient_step"`` (default): every optimizer step is a candidate
            iterate. ``"train_phase"``: one candidate per PPO update phase.
        unsafe_update_strategy:
            Fallback applied when a candidate fails verification.
            ``"rashomon_project"`` (default): on-demand Rashomon set around
            the last safe iterate, then projection onto its certified box --
            projected gradient descent onto a certified-safe parameter
            region. ``"none"``: monitor-only ablation, see above.
        rashomon_n_iters:
            Optimization budget of each on-demand Rashomon-set computation.
            Only used with ``"rashomon_project"``.
        rashomon_checkpoint:
            Checkpoint cadence inside the engine; ``None`` defaults to
            ``max(1, rashomon_n_iters // 10)`` so small budgets still yield
            certified candidate boxes.
        rashomon_certificate_samples:
            Certificate batch size. ``None`` (default) certifies over *all*
            states with a safe action, keeping the box certificate exhaustive
            (required for the projected iterate to be exactly safe).
        rashomon_inverse_temperature:
            Fixed integer inverse temperature for the engine's surrogate.
            ``None`` (default) recalibrates against iterate ``k`` before each
            Rashomon computation (softmax sharpness drifts during training).
        rashomon_multi_label_mode:
            ``"any"`` preserves the historical admissible-set certificate
            where at least one safe action beats every unsafe action.
            ``"all"`` requires every safe action logit to beat every unsafe
            action logit inside certified Rashomon boxes.
        rashomon_surrogate:
            ``"auto"`` preserves the historical per-mode formula;
            ``"logsumexp"`` uses the LSE margin for either mode.
        rashomon_seed:
            Seed for the engine's dataloaders; defaults to the model seed.
        verify_base_policy:
            Verify the (possibly loaded) initial policy is greedy-safe and
            raise ``ValueError`` otherwise.
        """
        # Forward feature map (state id -> feature vector) for the exact verifier
        # when the observation space is a feature Box; None means one-hot states.
        self._state_to_features = state_to_features
        if adaptive_granularity not in ("gradient_step", "train_phase"):
            raise ValueError(
                "adaptive_granularity must be 'gradient_step' or 'train_phase', "
                f"got {adaptive_granularity!r}.",
            )
        if int(adaptive_frequency) <= 0:
            raise ValueError(
                f"adaptive_frequency must be positive; got {adaptive_frequency}."
            )
        if unsafe_update_strategy not in ("rashomon_project", "none"):
            raise ValueError(
                "unsafe_update_strategy must be 'rashomon_project' or 'none', got "
                f"{unsafe_update_strategy!r}.",
            )
        if int(rashomon_n_iters) <= 0:
            raise ValueError(f"rashomon_n_iters must be positive; got {rashomon_n_iters}.")
        if rashomon_multi_label_mode not in ("any", "all"):
            raise ValueError(
                "rashomon_multi_label_mode must be either 'any' or 'all', "
                f"got {rashomon_multi_label_mode!r}.",
            )
        if rashomon_surrogate not in ("auto", "probability", "logsumexp"):
            raise ValueError(
                "rashomon_surrogate must be 'auto', 'probability', or 'logsumexp', "
                f"got {rashomon_surrogate!r}.",
            )
        if safe_region_shape not in ("orthotope", "zonotope"):
            raise ValueError(
                "safe_region_shape must be either 'orthotope' or 'zonotope', "
                f"got {safe_region_shape!r}.",
            )
        if zonotope_rank is not None and int(zonotope_rank) <= 0:
            raise ValueError(f"zonotope_rank must be positive when set, got {zonotope_rank}.")
        if directional_rashomon_growth and safe_region_shape != "orthotope":
            raise ValueError("Directional safe-region growth requires an orthotope region.")
        if stop_when_proposal_contained and safe_region_shape != "orthotope":
            raise ValueError("Proposal-containment stopping requires an orthotope region.")

        self._adaptive_granularity: AdaptiveGranularity = adaptive_granularity
        self._adaptive_frequency = int(adaptive_frequency)
        self._train_phases_since_enforcement = 0
        self._pending_adaptive_update = False
        self._final_flushes = 0
        self._unsafe_update_strategy: UnsafeUpdateStrategy = unsafe_update_strategy
        self._rashomon_n_iters = int(rashomon_n_iters)
        self._rashomon_checkpoint = (
            int(rashomon_checkpoint)
            if rashomon_checkpoint is not None
            else max(1, int(rashomon_n_iters) // 10)
        )
        self._rashomon_batch_size = int(rashomon_batch_size)
        self._rashomon_certificate_samples = (
            int(rashomon_certificate_samples) if rashomon_certificate_samples is not None else None
        )
        self._rashomon_inverse_temperature = (
            int(rashomon_inverse_temperature) if rashomon_inverse_temperature is not None else None
        )
        self._rashomon_multi_label_mode: Literal["any", "all"] = rashomon_multi_label_mode
        self._rashomon_surrogate: Literal[
            "auto", "probability", "logsumexp"
        ] = rashomon_surrogate
        from src.verification import verify

        self._rashomon_resolved_surrogate = verify.resolve_surrogate_form(
            rashomon_multi_label_mode, rashomon_surrogate
        )
        self._safe_region_shape: Literal["orthotope", "zonotope"] = safe_region_shape
        self._zonotope_rank = int(zonotope_rank) if zonotope_rank is not None else None
        self._rashomon_seed = rashomon_seed
        self._directional_rashomon_growth = bool(directional_rashomon_growth)
        self._stop_when_proposal_contained = bool(stop_when_proposal_contained)
        self._last_direction_counts: dict[str, int] | None = None
        self._directional_growth_failures = 0
        # Diagnostics counters (plain scalars: persisted by SB3 save/load).
        self._n_verifications = 0
        self._n_verified_safe = 0
        self._n_verified_unsafe = 0
        self._n_accepted_without_rashomon = 0
        self._n_rashomon_computations = 0
        self._n_projections = 0
        self._n_fallback_reverts = 0
        self._n_accepted_unsafe = 0
        self._rashomon_wall_time_s = 0.0
        self._last_selected_checkpoint_index: int | None = None
        self._last_projection_result: ProjectionResult | None = None
        self._last_rashomon_iterations_run = 0
        self._last_rashomon_target_contained_and_certified = False

        super().__init__(*args, **kwargs)

        # On SB3's load path (_init_setup_model=False) the policy does not
        # exist yet; adaptive state is rebuilt lazily by _ensure_adaptive_state.
        if getattr(self, "policy", None) is not None:
            self._init_adaptive_state(base_policy_state_dict, verify_base_policy)
        elif base_policy_state_dict is not None:
            raise ValueError(
                "base_policy_state_dict cannot be applied with _init_setup_model=False "
                "(the policy has not been built).",
            )

    # ------------------------------------------------------------------ setup

    def _init_adaptive_state(
        self,
        base_policy_state_dict: Mapping[str, th.Tensor] | None,
        verify_base_policy: bool,
    ) -> None:
        if self._shield is None:
            raise ValueError("AdaptiveSafePPO requires a shield.")
        # The exact greedy verifier enumerates every state. With a Discrete
        # observation it feeds one-hot rows; with a feature Box it feeds each
        # state's feature vector, which requires state_to_features.
        if isinstance(self.observation_space, spaces.Box):
            if self._state_to_features is None:
                raise ValueError(
                    "AdaptiveSafePPO with a Box (feature) observation space needs "
                    "state_to_features to enumerate states for the exact verifier.",
                )
            n_states = int(self._shield.n_states)
        elif isinstance(self.observation_space, spaces.Discrete):
            n_states = int(self.observation_space.n)
            if self._shield.n_states != n_states:
                raise ValueError(
                    f"Shield n_states ({self._shield.n_states}) does not match the "
                    f"observation space ({n_states}).",
                )
        else:
            raise ValueError(
                "AdaptiveSafePPO requires a Discrete or Box observation space; got "
                f"{type(self.observation_space).__name__}.",
            )

        if base_policy_state_dict is not None:
            self._load_base_policy_state_dict(base_policy_state_dict)

        # Live view: Sequential sharing the actor's parameters (projection /
        # snapshot target). Frozen view: an isolated deep copy handed to the
        # Rashomon engine, reloaded from the last-safe snapshot per call. Both
        # expose parameters in the same order, so engine bounds align with the
        # live parameters without any name mapping.
        _, self._live_actor_seq = extract_feature_actor_parameters_and_network(
            self, copy_modules=False
        )
        self._live_actor_params = list(self._live_actor_seq.parameters())
        _, self._frozen_actor_seq = extract_feature_actor_parameters_and_network(
            self, copy_modules=True
        )
        self._frozen_actor_params = list(self._frozen_actor_seq.parameters())

        # Shield dataset for the engine + cached exact-verification tensors.
        self._rashomon_dataset, self._safe_state_ids = shield_safe_behaviour_dataset(
            self._shield.mask, self._state_to_features
        )
        self._dataset_states = self._rashomon_dataset.tensors[0]
        self._dataset_actions = self._rashomon_dataset.tensors[1]
        if self._state_to_features is None:
            # Discrete: obs_to_tensor one-hots the integer ids.
            verify_obs, _ = self.policy.obs_to_tensor(self._safe_state_ids)
        else:
            # Features: the dataset already holds each state's feature vector;
            # feed it straight through as the verifier's observation batch.
            verify_obs, _ = self.policy.obs_to_tensor(self._dataset_states.cpu().numpy())
        self._verify_obs = verify_obs
        self._verify_mask = th.as_tensor(
            self._shield.mask[self._safe_state_ids], dtype=th.bool, device=self.device
        )

        if verify_base_policy:
            rate = self._greedy_safe_rate_now()
            if rate < 1.0:
                raise ValueError(
                    "Initial policy does not satisfy the configured hard shield invariant "
                    f"(safe rate {rate:.6f} < 1.0). Provide a safe "
                    "base_policy_state_dict, or pass verify_base_policy=False and "
                    "establish safety via pretrain_on_shield() + "
                    "mark_current_policy_safe().",
                )
        self._snapshot_last_safe()

        if self._adaptive_granularity == "gradient_step":
            self.policy.optimizer.post_step_hook = self._on_gradient_step

    def _ensure_adaptive_state(self) -> None:
        """Rebuild the non-persisted adaptive state after ``load()``.

        Re-verifies the loaded policy to re-establish the iterate-k invariant;
        raises ``ValueError`` if the loaded policy is not greedy-safe.
        """
        if getattr(self, "_live_actor_seq", None) is not None:
            return
        self._init_adaptive_state(base_policy_state_dict=None, verify_base_policy=True)

    def _load_base_policy_state_dict(self, state_dict: Mapping[str, th.Tensor]) -> None:
        target_names = projection_target_parameter_names(self)
        missing = sorted(set(target_names) - set(state_dict))
        extra = sorted(set(state_dict) - set(target_names))
        if missing or extra:
            raise ValueError(
                "base_policy_state_dict must cover exactly the actor projection "
                f"target parameters. Missing: {missing}. Unexpected: {extra}.",
            )
        name_to_param = dict(self.policy.named_parameters())
        with th.no_grad():
            for name in target_names:
                param = name_to_param[name]
                tensor = th.as_tensor(state_dict[name])
                if tuple(tensor.shape) != tuple(param.shape):
                    raise ValueError(
                        f"base_policy_state_dict shape mismatch for {name!r}: "
                        f"expected {tuple(param.shape)}, got {tuple(tensor.shape)}.",
                    )
                param.data.copy_(tensor.to(device=param.device, dtype=param.dtype))

    def _excluded_save_params(self) -> list[str]:
        return [
            *super()._excluded_save_params(),
            "_live_actor_seq",
            "_live_actor_params",
            "_frozen_actor_seq",
            "_frozen_actor_params",
            "_last_safe_params",
            "_rashomon_dataset",
            "_safe_state_ids",
            "_dataset_states",
            "_dataset_actions",
            "_verify_obs",
            "_verify_mask",
        ]

    # ------------------------------------------------------ safety invariant

    @th.no_grad()
    def _greedy_safe_rate_now(self) -> float:
        """Fraction of states satisfying the configured hard safety invariant."""
        logits = self.policy.get_distribution(self._verify_obs).distribution.logits
        if self._rashomon_multi_label_mode == "all":
            safe_mask = self._verify_mask
            unsafe_mask = ~safe_mask
            has_unsafe = unsafe_mask.any(dim=1)
            least_safe = logits.masked_fill(~safe_mask, float("inf")).min(dim=1).values
            greatest_unsafe = logits.masked_fill(
                ~unsafe_mask, float("-inf")
            ).max(dim=1).values
            safe = (~has_unsafe) | (least_safe > greatest_unsafe)
            return float(safe.float().mean().item())
        return _greedy_safe_rate(logits, self._verify_mask)

    def _verify_greedy_safe(self) -> bool:
        """Exact greedy safety check of the current policy, with diagnostics."""
        rate = self._greedy_safe_rate_now()
        safe = rate >= 1.0
        self._n_verifications += 1
        if safe:
            self._n_verified_safe += 1
        else:
            self._n_verified_unsafe += 1
        return safe

    @th.no_grad()
    def _snapshot_last_safe(self) -> None:
        self._last_safe_params = [p.detach().clone() for p in self._live_actor_params]

    def mark_current_policy_safe(self, *, verify: bool = True) -> None:
        """Re-baseline iterate ``k`` on the current parameters.

        Call after changing the policy weights outside the adaptive loop (e.g.
        after ``pretrain_on_shield()`` or ``load()``). With ``verify=True``
        (default) the current policy must pass the exact greedy check.
        """
        self._ensure_adaptive_state()
        if verify:
            rate = self._greedy_safe_rate_now()
            if rate < 1.0:
                raise ValueError(
                    "Current policy does not satisfy the configured hard shield invariant "
                    f"(safe rate {rate:.6f} < 1.0); "
                    "refusing to mark it as the safe iterate.",
                )
        self._snapshot_last_safe()

    # ------------------------------------------------------ adaptive updates

    def _compute_rashomon_around_last_safe(
        self,
        *,
        param_l_mask: list[th.Tensor] | None = None,
        param_u_mask: list[th.Tensor] | None = None,
        param_objective_weights: list[th.Tensor] | None = None,
        stop_target_params: list[th.Tensor] | None = None,
    ) -> tuple[SafeParameterRegion, int] | None:
        """Compute a certified safe parameter region around iterate ``k``."""
        if self._safe_region_shape != "orthotope" and (
            param_l_mask is not None
            or param_u_mask is not None
            or param_objective_weights is not None
            or stop_target_params is not None
        ):
            raise ValueError(
                "Directional masks, magnitude weighting, and proposal-containment "
                "stopping require an orthotope safe region."
            )
        with th.no_grad():
            for frozen, snapshot in zip(self._frozen_actor_params, self._last_safe_params):
                frozen.data.copy_(snapshot)

        if self._rashomon_inverse_temperature is not None:
            inverse_temp = self._rashomon_inverse_temperature
        else:
            try:
                with th.no_grad():
                    logits = self._frozen_actor_seq(
                        self._dataset_states.to(next(self._frozen_actor_seq.parameters()).device)
                    )
                inverse_temp = calibrate_inverse_temperature(
                    logits,
                    self._dataset_actions.to(logits.device),
                    multi_label_mode=self._rashomon_multi_label_mode,
                    surrogate=self._rashomon_surrogate,
                )
            except ValueError as exc:
                warnings.warn(
                    f"Inverse-temperature calibration failed ({exc}); falling back to "
                    "reverting the candidate to the last safe iterate.",
                    stacklevel=2,
                )
                return None

        certificate_samples = (
            self._rashomon_certificate_samples
            if self._rashomon_certificate_samples is not None
            else len(self._rashomon_dataset)
        )
        seed = self._rashomon_seed
        if seed is None:
            seed = self.seed if self.seed is not None else 0

        started = time.perf_counter()
        # Gradient-step mode invokes this from inside ProjectedAdam.step's
        # @torch.no_grad(); the engine needs autograd for its Cooper loop.
        with th.enable_grad():
            if self._safe_region_shape == "orthotope":
                result = _run_rashomon_engine(
                    self._frozen_actor_seq,
                    self._rashomon_dataset,
                    n_iters=self._rashomon_n_iters,
                    checkpoint=self._rashomon_checkpoint,
                    batch_size=self._rashomon_batch_size,
                    certificate_samples=certificate_samples,
                    inverse_temp=inverse_temp,
                    seed=int(seed),
                    multi_label_mode=self._rashomon_multi_label_mode,
                    surrogate=self._rashomon_surrogate,
                    param_l_mask=param_l_mask,
                    param_u_mask=param_u_mask,
                    param_objective_weights=param_objective_weights,
                    stop_target_params=stop_target_params,
                )
            else:
                result = _run_zonotope_rashomon_engine(
                    self._frozen_actor_seq,
                    self._rashomon_dataset,
                    n_iters=self._rashomon_n_iters,
                    checkpoint=self._rashomon_checkpoint,
                    batch_size=self._rashomon_batch_size,
                    certificate_samples=certificate_samples,
                    inverse_temp=inverse_temp,
                    seed=int(seed),
                    zonotope_rank=self._zonotope_rank,
                    multi_label_mode=self._rashomon_multi_label_mode,
                    surrogate=self._rashomon_surrogate,
                )
        self._rashomon_wall_time_s += time.perf_counter() - started
        self._n_rashomon_computations += 1
        self._last_rashomon_iterations_run = int(
            getattr(result, "iterations_run", self._rashomon_n_iters)
        )
        self._last_rashomon_target_contained_and_certified = bool(
            getattr(result, "target_contained_and_certified", False)
        )

        selected = select_certified_region(result, safe_region_shape=self._safe_region_shape)
        if selected is not None:
            self._last_selected_checkpoint_index = selected[1]
        return selected

    def _accept_or_project_candidate(self) -> None:
        """The iterate k -> k+1 transition: verify, else project."""
        if self._verify_greedy_safe():
            self._n_accepted_without_rashomon += 1
            self._snapshot_last_safe()
            return

        if self._unsafe_update_strategy == "none":
            # Monitor-only ablation: the candidate is recorded as unsafe and
            # kept as-is. Nothing is projected or reverted, so the run measures
            # what a BC-initialised, shield-explored policy does *without* the
            # safety correction -- the fraction of updates that happen to stay
            # safe is candidates_verified_safe / verifications_run. The
            # last-safe snapshot is deliberately not advanced onto an unsafe
            # iterate, though nothing consumes it in this mode.
            self._n_accepted_unsafe += 1
            return

        candidate_params = [param.detach().clone() for param in self._live_actor_params]
        param_l_mask = None
        param_u_mask = None
        param_objective_weights = None
        stop_target_params = None
        if self._directional_rashomon_growth:
            try:
                deltas = [
                    candidate - snapshot
                    for candidate, snapshot in zip(candidate_params, self._last_safe_params)
                ]
                param_l_mask, param_u_mask, counts = directional_masks_from_update_deltas(
                    deltas
                )
                param_objective_weights = (
                    directional_objective_weights_from_update_deltas(deltas)
                )
                self._last_direction_counts = counts
                if self._stop_when_proposal_contained:
                    stop_target_params = candidate_params
            except (TypeError, ValueError) as exc:
                warnings.warn(
                    f"Could not construct directional safe-region masks ({exc}); "
                    "reverting to the last safe policy.",
                    stacklevel=2,
                )
                self._directional_growth_failures += 1
                with th.no_grad():
                    for param, snapshot in zip(
                        self._live_actor_params, self._last_safe_params
                    ):
                        param.data.copy_(snapshot)
                self._n_fallback_reverts += 1
                return

        region_with_index = self._compute_rashomon_around_last_safe(
            param_l_mask=param_l_mask,
            param_u_mask=param_u_mask,
            param_objective_weights=param_objective_weights,
            stop_target_params=stop_target_params,
        )
        if region_with_index is None:
            # Degenerate projection: revert to iterate k (snapshot unchanged).
            with th.no_grad():
                for param, snapshot in zip(self._live_actor_params, self._last_safe_params):
                    param.data.copy_(snapshot)
            self._n_fallback_reverts += 1
            return

        region, _ = region_with_index
        if stop_target_params is not None and orthotope_contains_params(
            region, stop_target_params
        ):
            with th.no_grad():
                for param, target in zip(self._live_actor_params, stop_target_params):
                    param.data.copy_(target)
            self._snapshot_last_safe()
            return
        distance_norm = getattr(self.policy.optimizer, "_distance_norm", "l2")
        with th.no_grad():
            result = project_to_region_union(
                self._live_actor_params, [region], distance_norm=distance_norm
            )
        self._n_projections += 1
        self._last_projection_result = result
        # The box is IBP-certified: every policy inside it is greedy-safe, so
        # the projected candidate is iterate k+1 by construction.
        self._snapshot_last_safe()

    def _on_gradient_step(self) -> None:
        self._accept_or_project_candidate()

    def train(self, *args: Any, **kwargs: Any) -> Any:
        self._ensure_adaptive_state()
        result = super().train(*args, **kwargs)
        if self._adaptive_granularity == "train_phase":
            self._train_phases_since_enforcement += 1
            self._pending_adaptive_update = True
            if self._train_phases_since_enforcement >= self._adaptive_frequency:
                self._accept_or_project_candidate()
                self._train_phases_since_enforcement = 0
                self._pending_adaptive_update = False
        return result

    def finalize_adaptive_update(self) -> None:
        """Enforce a pending aggregate train-phase update before save/evaluation."""

        self._ensure_adaptive_state()
        if self._adaptive_granularity != "train_phase" or not self._pending_adaptive_update:
            return
        self._accept_or_project_candidate()
        self._train_phases_since_enforcement = 0
        self._pending_adaptive_update = False
        self._final_flushes += 1

    # ------------------------------------------------------------ diagnostics

    def adaptive_diagnostics(self) -> dict[str, Any]:
        """Cumulative diagnostics of the adaptive verify-then-project scheme."""
        last_projection = None
        if self._last_projection_result is not None:
            last_projection = {
                "n_projected": int(self._last_projection_result.n_projected),
                "n_boundary": int(self._last_projection_result.n_boundary),
                "displacement_l2": float(self._last_projection_result.displacement_l2),
                "displacement_linf": float(self._last_projection_result.displacement_linf),
            }
        return {
            "granularity": self._adaptive_granularity,
            "frequency": int(self._adaptive_frequency),
            "train_phases_since_enforcement": int(
                self._train_phases_since_enforcement
            ),
            "pending_adaptive_update": bool(self._pending_adaptive_update),
            "final_flushes": int(self._final_flushes),
            "unsafe_update_strategy": self._unsafe_update_strategy,
            "rashomon_n_iters": int(self._rashomon_n_iters),
            "rashomon_checkpoint": int(self._rashomon_checkpoint),
            "rashomon_multi_label_mode": self._rashomon_multi_label_mode,
            "rashomon_surrogate": self._rashomon_surrogate,
            "rashomon_resolved_surrogate": self._rashomon_resolved_surrogate,
            "directional_rashomon_growth": self._directional_rashomon_growth,
            "stop_when_proposal_contained": self._stop_when_proposal_contained,
            "directional_growth_failures": int(self._directional_growth_failures),
            "last_direction_counts": (
                None
                if self._last_direction_counts is None
                else dict(self._last_direction_counts)
            ),
            "safe_region_shape": self._safe_region_shape,
            "zonotope_rank": self._zonotope_rank,
            "verifications_run": int(self._n_verifications),
            "candidates_verified_safe": int(self._n_verified_safe),
            "candidates_verified_unsafe": int(self._n_verified_unsafe),
            "accepted_without_rashomon": int(self._n_accepted_without_rashomon),
            "rashomon_computations": int(self._n_rashomon_computations),
            "projections_applied": int(self._n_projections),
            "fallback_reverts": int(self._n_fallback_reverts),
            "accepted_unsafe": int(self._n_accepted_unsafe),
            "safe_update_fraction": (
                float(self._n_verified_safe / self._n_verifications)
                if self._n_verifications else 0.0
            ),
            "rashomon_wall_time_total_s": float(self._rashomon_wall_time_s),
            "last_selected_checkpoint_index": self._last_selected_checkpoint_index,
            "last_projection": last_projection,
        }

    @classmethod
    def load(cls, *args: Any, **kwargs: Any) -> "AdaptiveSafePPO":
        model = super().load(*args, **kwargs)
        warnings.warn(
            "AdaptiveSafePPO loaded: the adaptive state (last-safe snapshot, actor "
            "views) is rebuilt on the next train() by re-verifying the loaded policy. "
            "Call mark_current_policy_safe() to re-establish it explicitly.",
            stacklevel=2,
        )
        return model
