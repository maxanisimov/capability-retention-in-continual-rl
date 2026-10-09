"""Sparse-model FrozenLake adapter and exact finite-state safety analysis.

Gymnasium's native environment stores sparse transition lists. Unlike the local
MASA adapter, this implementation never constructs an S-by-A-by-S dense array.
Use the module-qualified Gymnasium id so subprocesses register it automatically.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv

ENV_NAME = "PSPOFrozenLake-v0"
ENV_ID = f"projects.safe_policy_optimisation.utils.frozen_lake_experiment:{ENV_NAME}"


def structured_layout(size: int = 128) -> list[str]:
    """Four separated hole islands, wide cross corridors and safe perimeter.

    Scaling uses the same fractional geometry. No random map changes on reset.
    """
    if size < 16:
        raise ValueError("Use size >= 16 for this structured layout.")
    grid = np.full((size, size), "F", dtype="U1")
    bands = [(size // 8, 5 * size // 16), (11 * size // 16, 7 * size // 8)]
    for row_start, row_end in bands:
        for col_start, col_end in bands:
            grid[row_start:row_end, col_start:col_end] = "H"
    grid[0, 0], grid[-1, -1] = "S", "G"
    return ["".join(row) for row in grid]


class SparseFrozenLake(FrozenLakeEnv):
    """Native slippery FrozenLake plus PSPO state/cost accessors."""

    def __init__(
        self, *, layout_path: str | None = None, desc: list[str] | None = None,
        size: int = 128, observation_mode: str = "index", is_slippery: bool = True,
        success_rate: float = 0.8, step_penalty: float = 0.001,
        render_mode: str | None = None,
    ) -> None:
        if observation_mode != "index":
            raise ValueError("This experiment uses discrete state ids / one-hot policies.")
        if desc is not None and layout_path is not None:
            raise ValueError("Specify either desc or layout_path, not both.")
        if layout_path is not None:
            desc = Path(layout_path).read_text().splitlines()
        if desc is None:
            desc = structured_layout(size)
        if step_penalty < 0 or not 0 < success_rate <= 1:
            raise ValueError("Invalid step penalty or transition success rate.")
        super().__init__(
            desc=desc, is_slippery=is_slippery, success_rate=success_rate,
            reward_schedule=(1.0, 0.0, -float(step_penalty)), render_mode=render_mode,
        )
        # Charge every nonterminal-origin step, including reaching the goal/hole.
        # Return = goal indicator - step_penalty * episode length.
        for state, actions in self.P.items():
            if self.desc.flat[state] in (b"H", b"G"):
                continue
            for action, transitions in actions.items():
                actions[action] = [
                    (p, nxt, float(self.desc.flat[nxt] == b"G") - step_penalty, done)
                    for p, nxt, _reward, done in transitions
                ]
        self._n_states = int(self.observation_space.n)
        self._n_actions = int(self.action_space.n)
        self._observation_mode = "index"
        self._step_penalty = float(step_penalty)

    @staticmethod
    def make_obs_to_state():
        def decode(obs: Any) -> np.ndarray:
            values = obs.detach().cpu().numpy() if hasattr(obs, "detach") else np.asarray(obs)
            return values.reshape(-1).astype(np.int64)
        return decode

    def label_fn(self, state: int) -> set[str]:
        cell = self.desc.flat[int(state)]
        return {"hole" if cell == b"H" else "goal" if cell == b"G" else "frozen"}

    @staticmethod
    def cost_fn(labels: set[str]) -> float:
        return float("hole" in labels)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = super().step(action)
        info.update({
            "cost": self.cost_fn(self.label_fn(int(obs))),
            "success": bool(self.desc.flat[int(obs)] == b"G"),
            "is_success": bool(self.desc.flat[int(obs)] == b"G"),
        })
        return obs, reward, terminated, truncated, info


def transition_arrays(env: SparseFrozenLake) -> tuple[np.ndarray, np.ndarray]:
    """Pad sparse outcome lists to S x A x K, retaining all positive outcomes."""
    width = max(len(outcomes) for actions in env.P.values() for outcomes in actions.values())
    successors = np.zeros((env._n_states, env._n_actions, width), dtype=np.int64)
    probabilities = np.zeros(successors.shape, dtype=np.float64)
    for state, actions in env.P.items():
        for action, outcomes in actions.items():
            for index, (probability, nxt, _reward, _done) in enumerate(outcomes):
                successors[state, action, index] = nxt
                probabilities[state, action, index] = probability
    return successors, probabilities


def synthesise_shield(
    env: SparseFrozenLake,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Greatest safety-winning set, treating every positive outcome adversarially."""
    successors, probabilities = transition_arrays(env)
    positive = probabilities > 0
    winning = np.asarray(env.desc.reshape(-1) != b"H")
    while True:
        mask = winning[:, None] & np.all(~positive | winning[successors], axis=2)
        updated = winning & mask.any(axis=1)
        if np.array_equal(updated, winning):
            break
        winning = updated
    return mask, winning, successors, probabilities


def goal_distances(
    mask: np.ndarray, successors: np.ndarray, probabilities: np.ndarray, goal: int,
) -> np.ndarray:
    """Distances to goal in the union graph of allowed positive-probability edges."""
    predecessors: list[set[int]] = [set() for _ in range(len(mask))]
    for state, action in zip(*np.nonzero(mask)):
        for nxt in successors[state, action][probabilities[state, action] > 0]:
            predecessors[int(nxt)].add(int(state))
    distance = np.full(len(mask), -1, dtype=np.int64)
    distance[goal] = 0
    queue = deque([goal])
    while queue:
        nxt = queue.popleft()
        for state in predecessors[nxt]:
            if distance[state] < 0:
                distance[state] = distance[nxt] + 1
                queue.append(state)
    return distance


def proper_goal_policies(
    mask: np.ndarray, winning: np.ndarray, successors: np.ndarray,
    probabilities: np.ndarray, goal: int,
) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Construct four witnesses with an exact almost-sure reachability proof.

    Every selected action stays inside W and has a positive-probability successor
    with strictly smaller integer goal distance. Therefore no non-goal closed
    recurrent class can exist: its minimum-distance state would have an outgoing
    edge to a smaller-distance state outside that class. This is an unbounded
    reachability proof, not a finite-horizon success guarantee.
    """
    distance = goal_distances(mask, successors, probabilities, goal)
    if not np.all(distance[winning] >= 0):
        raise ValueError("Not every safety-winning state can reach the goal.")
    proper = mask & np.any(
        (probabilities > 0) & (distance[successors] < distance[:, None, None]), axis=2
    )
    proper[goal] = mask[goal]
    if not np.all(proper[winning].any(axis=1)):
        raise ValueError("Missing a safe distance-progress action.")
    expected_distance = (probabilities * distance[successors]).sum(axis=2)
    score = np.where(proper, expected_distance, np.inf)
    best = score.min(axis=1)
    policies = []
    for index, order in enumerate(((1, 2, 0, 3), (2, 1, 3, 0), (0, 1, 2, 3), (3, 2, 1, 0))):
        policy = np.zeros(len(mask), dtype=np.int64)
        rng = np.random.default_rng(1000 + index)
        for state in np.flatnonzero(winning):
            # Prefer lower expected distance; orders produce distinct tie choices.
            candidates = [a for a in order if score[state, a] <= best[state] + 1e-10]
            policy[state] = candidates[0] if index < 2 else int(rng.choice(candidates))
        if not np.all(proper[np.flatnonzero(winning), policy[winning]]):
            raise AssertionError("Invalid goal-reaching witness.")
        policies.append(policy)
    differences = [
        int(np.count_nonzero((a != b) & winning))
        for index, a in enumerate(policies) for b in policies[index + 1:]
    ]
    if min(differences) == 0:
        raise ValueError("Goal-reaching witnesses are not distinct.")
    branch_states = int(np.count_nonzero(winning & (proper.sum(axis=1) >= 2)))
    diagnostics = {
        "proof": "positive-probability strict distance progress rules out non-goal closed recurrent classes",
        "almost_sure_goal_reachability": True,
        "finite_horizon_guarantee": False,
        "witness_policy_count": len(policies),
        "pairwise_policy_differing_winning_states": differences,
        "states_with_multiple_proven_progress_actions": branch_states,
        "distinct_deterministic_proper_policy_count_lower_bound": f"2^{branch_states}",
    }
    return distance, policies, diagnostics


if ENV_NAME not in gym.registry:
    gym.register(ENV_NAME, entry_point=SparseFrozenLake)
