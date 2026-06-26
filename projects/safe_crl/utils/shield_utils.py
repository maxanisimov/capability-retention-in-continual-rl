"""Safety shield synthesis utilities for tabular Gymnasium environments.

Scope: this module performs exact tabular value iteration. Both
``_almost_sure_safe_set`` (deterministic shields) and
``_eventual_unsafe_risk_value_iteration`` (probabilistic shields, the safety
critic used to derive ``action_risk``) require a fully enumerated transition
matrix ``P(s'|s,a)`` of shape ``(n_states, n_states, n_actions)`` up front and
perform exact Bellman backups over it (``O(|S|^2 * |A|)`` per iteration) -
there is no environment-rollout/sampling code path. This is by design: every
environment in ``projects.safe_crl.utils.masa_tabular_envs`` has a small, finite,
enumerable state space, so exact VI is both tractable and gives certified
(non-approximate) risk bounds.

It does **not** generalise to continuous-state or image-observation
("non-tabular") environments without replacing exact VI with a fundamentally
different algorithm - e.g. fitted/neural value iteration over sampled
transitions with a learned critic - which would also give up the exact
convergence guarantee in favour of empirical generalisation. Extending shield
synthesis to non-tabular environments is tracked as separate future work, not
in scope for this module.
"""

from __future__ import annotations

import copy
import types
from dataclasses import dataclass
from typing import Any, Callable, Literal

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError:  # pragma: no cover - exercised only without RL extras
    gym = None
    spaces = None


ShieldType = Literal["deterministic", "probabilistic"]
TransitionMatrixFn = Callable[[Any], np.ndarray]
LabelFn = Callable[[Any], set[str]]
CostFn = Callable[[set[str]], float]


@dataclass(frozen=True)
class ShieldSynthesisInfo:
    """Extra artefacts produced while synthesising a shield."""

    successor_states_matrix: np.ndarray
    probabilities: np.ndarray
    winning_states: np.ndarray
    safe_states: np.ndarray
    state_risk: np.ndarray | None = None
    action_risk: np.ndarray | None = None
    vi_steps: int | None = None
    vi_residual: float | None = None


def synthesise_shield(
    env: Any,
    transition_matrix_fn: TransitionMatrixFn,
    label_fn: LabelFn,
    cost_fn: CostFn,
    *,
    shield_type: ShieldType = "deterministic",
    risk_threshold: float = 0.0,
    theta: float = 1e-10,
    max_vi_steps: int = 1000,
    unsafe_cost_threshold: float = 0.5,
    use_masa_helper: bool = True,
    copy_env: bool = True,
    start_state: int | None = None,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, ShieldSynthesisInfo]:
    """Synthesise a safety shield for a tabular Gymnasium environment.

    Args:
        env: Environment with a discrete action space. The observation space can
            be discrete or inferable from the supplied transition matrix.
        transition_matrix_fn: Returns a transition matrix with shape
            ``(n_states, n_states, n_actions)``, where
            ``P[s_next, s, a] = Pr(s_next | s, a)``. The function is called with
            ``env.unwrapped`` when the Gymnasium API is available.
        label_fn: Maps a tabular observation/state id to a set of labels.
        cost_fn: Maps labels to a safety cost. States with cost greater than or
            equal to ``unsafe_cost_threshold`` are treated as unsafe.
        shield_type: ``"deterministic"`` returns the almost-sure shield. An
            action is allowed only if every possible successor remains in the
            winning set. ``"probabilistic"`` estimates eventual unsafe
            reachability risk by value iteration and allows actions with
            conservative risk at most ``risk_threshold``.
        risk_threshold: Maximum eventual unsafe reachability risk allowed for a
            probabilistic shield.
        theta: Value-iteration convergence tolerance for probabilistic shields.
        max_vi_steps: Maximum value-iteration steps for probabilistic shields.
        unsafe_cost_threshold: Cost threshold separating safe and unsafe states.
        use_masa_helper: Prefer MASA's ``build_successor_states_matrix`` when
            MASA is installed and the default unsafe threshold is used.
        copy_env: Deep-copy the environment before adding the MASA-style
            transition-matrix API.
        start_state: Optional start state to expose on the copied environment.
        return_info: Return shield synthesis artefacts alongside the mask.

    Returns:
        A binary matrix of shape ``(n_states, n_actions)``. Entry ``[s, a]`` is
        ``1`` when action ``a`` is allowed in state ``s`` and ``0`` otherwise.
        If ``return_info=True``, returns ``(shield, info)``.
    """

    _require_gymnasium()
    if shield_type not in {"deterministic", "probabilistic"}:
        raise ValueError("shield_type must be either 'deterministic' or 'probabilistic'.")
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError(f"risk_threshold must be in [0, 1], got {risk_threshold}.")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}.")
    if max_vi_steps <= 0:
        raise ValueError(f"max_vi_steps must be positive, got {max_vi_steps}.")
    if not np.isfinite(unsafe_cost_threshold):
        raise ValueError(f"unsafe_cost_threshold must be finite, got {unsafe_cost_threshold}.")

    output = _build_successor_states_matrix(
        env,
        transition_matrix_fn,
        label_fn,
        cost_fn,
        unsafe_cost_threshold=unsafe_cost_threshold,
        use_masa_helper=use_masa_helper,
        copy_env=copy_env,
        start_state=start_state,
    )
    successor_states_matrix, probabilities, _max_successors, resolved_label_fn, resolved_cost_fn, winning_set = output
    n_states = successor_states_matrix.shape[1]

    winning_states = np.array(sorted(int(s) for s in winning_set), dtype=np.int64)
    safe_states = _safe_states(resolved_label_fn, resolved_cost_fn, n_states, unsafe_cost_threshold)

    if shield_type == "deterministic":
        shield = _deterministic_shield(successor_states_matrix, probabilities, winning_states)
        info = ShieldSynthesisInfo(
            successor_states_matrix=successor_states_matrix,
            probabilities=probabilities,
            winning_states=winning_states,
            safe_states=safe_states,
        )
    else:
        state_risk, action_risk, steps, residual = _eventual_unsafe_risk_value_iteration(
            successor_states_matrix,
            probabilities,
            resolved_label_fn,
            resolved_cost_fn,
            winning_states,
            theta=theta,
            max_steps=max_vi_steps,
            unsafe_cost_threshold=unsafe_cost_threshold,
        )
        shield = (action_risk <= risk_threshold + theta).astype(int)
        shield[np.setdiff1d(np.arange(n_states), safe_states), :] = 0
        info = ShieldSynthesisInfo(
            successor_states_matrix=successor_states_matrix,
            probabilities=probabilities,
            winning_states=winning_states,
            safe_states=safe_states,
            state_risk=state_risk,
            action_risk=action_risk,
            vi_steps=steps,
            vi_residual=residual,
        )

    if return_info:
        return shield, info
    return shield


def synthesise_deterministic_shield(
    env: Any,
    transition_matrix_fn: TransitionMatrixFn,
    label_fn: LabelFn,
    cost_fn: CostFn,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, ShieldSynthesisInfo]:
    """Convenience wrapper for almost-sure deterministic shield synthesis."""

    return synthesise_shield(
        env,
        transition_matrix_fn,
        label_fn,
        cost_fn,
        shield_type="deterministic",
        **kwargs,
    )


def synthesise_probabilistic_shield(
    env: Any,
    transition_matrix_fn: TransitionMatrixFn,
    label_fn: LabelFn,
    cost_fn: CostFn,
    *,
    risk_threshold: float,
    **kwargs: Any,
) -> np.ndarray | tuple[np.ndarray, ShieldSynthesisInfo]:
    """Convenience wrapper for risk-thresholded probabilistic shield synthesis."""

    return synthesise_shield(
        env,
        transition_matrix_fn,
        label_fn,
        cost_fn,
        shield_type="probabilistic",
        risk_threshold=risk_threshold,
        **kwargs,
    )


def synthesise_shield_from_successor_dict(
    env: Any,
    label_fn: LabelFn,
    cost_fn: CostFn,
    *,
    shield_type: ShieldType = "deterministic",
    risk_threshold: float = 0.0,
    theta: float = 1e-10,
    max_vi_steps: int = 1000,
    unsafe_cost_threshold: float = 0.5,
    return_info: bool = False,
) -> np.ndarray | tuple[np.ndarray, ShieldSynthesisInfo]:
    """Synthesise a shield from an env's sparse successor-states dict.

    Equivalent to :func:`synthesise_shield` but consumes the sparse
    ``(successor_states, transition_probs)`` representation exposed by
    ``TabularEnv.get_successor_states_dict()`` instead of a dense
    ``(n_states, n_states, n_actions)`` transition matrix. This is required for the
    larger tabular envs (e.g. Pacman), whose dense matrix would be ``O(|S|^2)`` and is
    therefore never materialised; the exact value iteration itself only needs the sparse
    successor support, so this path is both correct and memory-efficient.

    ``successor_states[s]`` is the list of successor state ids reachable from ``s`` (its
    support across all actions); ``transition_probs[(s, a)]`` is the probability vector over
    that support for action ``a``.
    """

    _require_gymnasium()
    if shield_type not in {"deterministic", "probabilistic"}:
        raise ValueError("shield_type must be either 'deterministic' or 'probabilistic'.")
    if not 0.0 <= risk_threshold <= 1.0:
        raise ValueError(f"risk_threshold must be in [0, 1], got {risk_threshold}.")

    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    if not isinstance(unwrapped.action_space, spaces.Discrete):
        raise TypeError(
            "Shield synthesis only supports discrete action spaces, got "
            f"{type(unwrapped.action_space).__name__}.",
        )
    dict_result = (
        unwrapped.get_successor_states_dict()
        if hasattr(unwrapped, "get_successor_states_dict")
        else None
    )
    if not dict_result:
        raise ValueError(
            f"{type(unwrapped).__name__} does not expose a successor-states dict via "
            "get_successor_states_dict().",
        )
    successors, transition_probs = dict_result

    n_actions = int(unwrapped.action_space.n)
    if isinstance(unwrapped.observation_space, spaces.Discrete):
        n_states = int(unwrapped.observation_space.n)
    else:
        n_states = int(getattr(unwrapped, "_n_states"))

    successor_states_matrix, probabilities = _successor_arrays_from_dict(
        successors, transition_probs, n_states=n_states, n_actions=n_actions,
    )

    safe_states = _safe_states(label_fn, cost_fn, n_states, unsafe_cost_threshold)
    winning_set = _almost_sure_safe_set(
        successor_states_matrix, probabilities, set(safe_states.tolist()),
    )
    winning_states = np.array(sorted(int(s) for s in winning_set), dtype=np.int64)

    if shield_type == "deterministic":
        shield = _deterministic_shield(successor_states_matrix, probabilities, winning_states)
        info = ShieldSynthesisInfo(
            successor_states_matrix=successor_states_matrix,
            probabilities=probabilities,
            winning_states=winning_states,
            safe_states=safe_states,
        )
    else:
        state_risk, action_risk, steps, residual = _eventual_unsafe_risk_value_iteration(
            successor_states_matrix,
            probabilities,
            label_fn,
            cost_fn,
            winning_states,
            theta=theta,
            max_steps=max_vi_steps,
            unsafe_cost_threshold=unsafe_cost_threshold,
        )
        shield = (action_risk <= risk_threshold + theta).astype(int)
        shield[np.setdiff1d(np.arange(n_states), safe_states), :] = 0
        info = ShieldSynthesisInfo(
            successor_states_matrix=successor_states_matrix,
            probabilities=probabilities,
            winning_states=winning_states,
            safe_states=safe_states,
            state_risk=state_risk,
            action_risk=action_risk,
            vi_steps=steps,
            vi_residual=residual,
        )

    if return_info:
        return shield, info
    return shield


def _successor_arrays_from_dict(
    successors: Any,
    transition_probs: Any,
    *,
    n_states: int,
    n_actions: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the sparse ``(successor_states_matrix, probabilities)`` arrays used by value
    iteration from an env's successor-states dict."""
    supports = [list(successors.get(state, [])) for state in range(n_states)]
    max_successors = max((len(support) for support in supports), default=0)
    if max_successors == 0:
        raise ValueError("successor-states dict does not contain any reachable successors.")

    successor_states_matrix = -np.ones((max_successors, n_states), dtype=np.int64)
    probabilities = np.zeros((max_successors, n_states, n_actions), dtype=np.float64)
    for state in range(n_states):
        support = supports[state]
        k = len(support)
        if k == 0:
            continue
        successor_states_matrix[:k, state] = np.asarray(support, dtype=np.int64)
        for action in range(n_actions):
            probs = transition_probs.get((state, action))
            if probs is None:
                continue
            probs = np.asarray(probs, dtype=np.float64)
            probabilities[: probs.shape[0], state, action] = probs
    return successor_states_matrix, probabilities


def _require_gymnasium() -> None:
    if gym is None or spaces is None:
        raise ModuleNotFoundError("gymnasium is required to synthesise shields.")


def _build_successor_states_matrix(
    env: Any,
    transition_matrix_fn: TransitionMatrixFn,
    label_fn: LabelFn,
    cost_fn: CostFn,
    *,
    unsafe_cost_threshold: float,
    use_masa_helper: bool,
    copy_env: bool,
    start_state: int | None,
) -> tuple[np.ndarray, np.ndarray, int, LabelFn, CostFn, list[int]]:
    if use_masa_helper and unsafe_cost_threshold == 0.5:
        try:
            from masa.prob_shield.helpers import build_successor_states_matrix
        except ModuleNotFoundError:
            pass
        else:
            masa_env = _prepare_env_for_masa(env, transition_matrix_fn, copy_env, start_state)
            return build_successor_states_matrix(
                env=masa_env,
                label_fn=label_fn,
                cost_fn=cost_fn,
            )

    return _build_successor_states_matrix_local(
        env,
        transition_matrix_fn,
        label_fn,
        cost_fn,
        unsafe_cost_threshold=unsafe_cost_threshold,
    )


def _prepare_env_for_masa(
    env: Any,
    transition_matrix_fn: TransitionMatrixFn,
    copy_env: bool,
    start_state: int | None,
) -> Any:
    masa_env = copy.deepcopy(env) if copy_env else env
    unwrapped = masa_env.unwrapped
    transition_matrix = np.asarray(transition_matrix_fn(unwrapped), dtype=np.float64)

    if hasattr(unwrapped, "_transition_matrix"):
        unwrapped._transition_matrix = transition_matrix  # noqa: SLF001
    if start_state is not None:
        unwrapped._start_state = int(start_state)  # noqa: SLF001

    try:
        unwrapped.has_transition_matrix = True
    except AttributeError:
        pass

    try:
        unwrapped.has_successor_states_dict = False
    except AttributeError:
        pass

    unwrapped.get_transition_matrix = types.MethodType(  # type: ignore[method-assign]
        lambda self: transition_matrix,
        unwrapped,
    )
    if not hasattr(unwrapped, "get_successor_states_dict"):
        unwrapped.get_successor_states_dict = types.MethodType(  # type: ignore[method-assign]
            lambda self: None,
            unwrapped,
        )
    return masa_env


def _build_successor_states_matrix_local(
    env: Any,
    transition_matrix_fn: TransitionMatrixFn,
    label_fn: LabelFn,
    cost_fn: CostFn,
    *,
    unsafe_cost_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, LabelFn, CostFn, list[int]]:
    transition_matrix = np.asarray(transition_matrix_fn(env.unwrapped), dtype=np.float64)
    _validate_transition_matrix(env, transition_matrix)

    n_states, _, n_actions = transition_matrix.shape
    reachable = np.any(transition_matrix > 0.0, axis=2)
    successor_counts = np.count_nonzero(reachable, axis=0)
    max_successors = int(successor_counts.max(initial=0))
    if max_successors == 0:
        raise ValueError("transition_matrix does not contain any reachable successor states.")

    successor_states_matrix = -np.ones((max_successors, n_states), dtype=np.int64)
    probabilities = np.zeros((max_successors, n_states, n_actions), dtype=np.float64)

    for state in range(n_states):
        successors = np.nonzero(reachable[:, state])[0]
        k = len(successors)
        successor_states_matrix[:k, state] = successors
        probabilities[:k, state, :] = transition_matrix[successors, state, :]

    safe_states = set(_safe_states(label_fn, cost_fn, n_states, unsafe_cost_threshold).tolist())
    winning_set = _almost_sure_safe_set(successor_states_matrix, probabilities, safe_states)
    return successor_states_matrix, probabilities, max_successors, label_fn, cost_fn, winning_set


def _validate_transition_matrix(env: Any, transition_matrix: np.ndarray) -> None:
    if not isinstance(env.action_space, spaces.Discrete):
        raise TypeError(
            "Shield synthesis only supports discrete action spaces, got "
            f"{type(env.action_space).__name__}.",
        )
    if transition_matrix.ndim != 3:
        raise ValueError(
            "transition_matrix must have shape (n_states, n_states, n_actions), "
            f"got {transition_matrix.shape}.",
        )
    n_states, n_states_2, n_actions = transition_matrix.shape
    if n_states != n_states_2:
        raise ValueError(
            "transition_matrix must be square on its state axes, got "
            f"{transition_matrix.shape}.",
        )
    if int(env.action_space.n) != n_actions:
        raise ValueError(
            f"transition_matrix has {n_actions} actions, but env.action_space has "
            f"{env.action_space.n}.",
        )
    if isinstance(env.observation_space, spaces.Discrete) and int(env.observation_space.n) != n_states:
        raise ValueError(
            f"transition_matrix has {n_states} states, but env.observation_space has "
            f"{env.observation_space.n}.",
        )
    if np.any(transition_matrix < -1e-12):
        raise ValueError("transition_matrix contains negative probabilities.")

    column_sums = transition_matrix.sum(axis=0)
    if not np.allclose(column_sums, 1.0):
        raise ValueError("transition_matrix probabilities must sum to 1 for every state-action pair.")


def _safe_states(
    label_fn: LabelFn,
    cost_fn: CostFn,
    n_states: int,
    unsafe_cost_threshold: float,
) -> np.ndarray:
    return np.array(
        [state for state in range(n_states) if cost_fn(label_fn(state)) < unsafe_cost_threshold],
        dtype=np.int64,
    )


def _almost_sure_safe_set(
    successor_states_matrix: np.ndarray,
    probabilities: np.ndarray,
    safe_states: set[int],
) -> list[int]:
    n_actions = probabilities.shape[2]
    winning_set = set(safe_states)
    changed = True

    while changed:
        changed = False
        to_remove = []
        for state in list(winning_set):
            has_winning_action = False
            successors = successor_states_matrix[:, state]
            for action in range(n_actions):
                support = _action_support(successors, probabilities[:, state, action])
                if support.size > 0 and all(int(next_state) in winning_set for next_state in support):
                    has_winning_action = True
                    break
            if not has_winning_action:
                to_remove.append(state)
        if to_remove:
            winning_set.difference_update(to_remove)
            changed = True

    return sorted(winning_set)


def _deterministic_shield(
    successor_states_matrix: np.ndarray,
    probabilities: np.ndarray,
    winning_states: np.ndarray,
) -> np.ndarray:
    n_states = successor_states_matrix.shape[1]
    n_actions = probabilities.shape[2]
    winning_set = set(winning_states.tolist())
    shield = np.zeros((n_states, n_actions), dtype=int)

    for state in winning_set:
        successors = successor_states_matrix[:, state]
        for action in range(n_actions):
            support = _action_support(successors, probabilities[:, state, action])
            if support.size > 0 and all(int(next_state) in winning_set for next_state in support):
                shield[state, action] = 1
    return shield


def _action_support(successors: np.ndarray, transition_probs: np.ndarray) -> np.ndarray:
    return successors[(transition_probs > 0.0) & (successors != -1)]


def _eventual_unsafe_risk_value_iteration(
    successor_states_matrix: np.ndarray,
    probabilities: np.ndarray,
    label_fn: LabelFn,
    cost_fn: CostFn,
    winning_states: np.ndarray,
    *,
    theta: float,
    max_steps: int,
    unsafe_cost_threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    n_states = successor_states_matrix.shape[1]
    unsafe_mask = np.array(
        [cost_fn(label_fn(state)) >= unsafe_cost_threshold for state in range(n_states)],
        dtype=bool,
    )
    winning_mask = np.zeros(n_states, dtype=bool)
    winning_mask[winning_states] = True
    absorbing_mask = unsafe_mask | winning_mask

    risk_lower = unsafe_mask.astype(np.float64)
    risk_lower[winning_mask] = 0.0
    risk_upper = np.ones(n_states, dtype=np.float64)
    risk_upper[winning_mask] = 0.0
    risk_upper[unsafe_mask] = 1.0
    successor_indices = np.where(successor_states_matrix == -1, 0, successor_states_matrix)

    residual = np.inf
    steps = 0
    for steps in range(1, max_steps + 1):
        next_lower = risk_lower[successor_indices]
        next_upper = risk_upper[successor_indices]
        expected_lower = np.sum(next_lower[..., None] * probabilities, axis=0)
        expected_upper = np.sum(next_upper[..., None] * probabilities, axis=0)

        updated_lower = risk_lower.copy()
        updated_upper = risk_upper.copy()
        updated_lower[~absorbing_mask] = np.min(expected_lower[~absorbing_mask], axis=1)
        updated_upper[~absorbing_mask] = np.min(expected_upper[~absorbing_mask], axis=1)
        updated_lower[unsafe_mask] = 1.0
        updated_upper[unsafe_mask] = 1.0
        updated_lower[winning_mask] = 0.0
        updated_upper[winning_mask] = 0.0

        risk_lower = np.clip(updated_lower, 0.0, 1.0)
        risk_upper = np.clip(updated_upper, 0.0, 1.0)
        residual = float(np.max(np.abs(risk_upper - risk_lower)))
        if residual <= theta:
            break

    action_risk = np.sum(risk_upper[successor_indices][..., None] * probabilities, axis=0)
    action_risk[unsafe_mask, :] = 1.0
    return risk_upper, np.clip(action_risk, 0.0, 1.0), steps, residual


synthesize_shield = synthesise_shield
synthesize_deterministic_shield = synthesise_deterministic_shield
synthesize_probabilistic_shield = synthesise_probabilistic_shield
