"""
Highway Parking Utilities for Safe Continual Learning Experiments
=================================================================

This module provides all helper functions for the Highway Parking SafeAdapt demo:

- **Reproducibility**: ``set_all_seeds``.
- **Config helpers**: ``_convert_spot_lists`` for YAML→Python tuple conversion.
- **Environment setup**: Gymnasium wrappers for flat observation encoding
  (``ParkingObservationWrapper``) and safety flags (``SafetyWrapperHighway``),
  plus a convenience factory ``make_parking_env``.
- **Safety-dataset creation**: two dataset builders
  (``create_safe_optimal_policy_dataset``, ``create_safe_training_dataset``)
  and the aggregator ``build_safety_datasets``.
  *Note*: "Sufficient Safety Data" is **not** available for continuous state
  spaces — only the two rollout/training-based variants are provided.
- **Safety-actor training**: ``train_safety_actor`` trains a reference model
  whose parameters define the centre of the certified safe region.
  Single-label cross-entropy only (multi-label is not applicable here).
- **Evaluation & verification**: ``evaluate_policy`` collects rollout metrics;
  ``verify_safety_accuracy`` checks the SafeAdapt actor against the safety
  dataset and prints diagnostics.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# Reproducibility
# =============================================================================

def set_all_seeds(seed: int) -> None:
    """Set all random seeds (stdlib, numpy, torch) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# Config helpers
# =============================================================================

def _convert_spot_lists(spots: list[list] | None) -> list[tuple] | None:
    """Convert YAML lists to Python tuples for lane indices.

    YAML serialises tuples as lists, but highway-env lane indices
    must be tuples for correct dict-key lookups.
    """
    if spots is None:
        return None
    return [tuple(s) for s in spots]


# =============================================================================
# Gymnasium wrappers
# =============================================================================

class ParkingObservationWrapper(gym.ObservationWrapper):
    """Convert Dict observation to a flat array, optionally with a task indicator.

    The base highway-env parking environment returns an OrderedDict with keys
    ``'observation'``, ``'achieved_goal'``, ``'desired_goal'``.  This wrapper:

    * Concatenates ``observation`` (6) + ``desired_goal`` (6) = 12 dims
      (when ``use_goal=True``), or just ``observation`` (6) otherwise.
    * When *task_num* is not ``None``, appends a scalar task indicator (+1).

    Set ``task_num=None`` to keep the original dimensionality (useful when
    loading pre-trained checkpoints that were not trained with a task indicator).
    """

    def __init__(self, env: gym.Env, task_num: int | None = None, use_goal: bool = True):
        super().__init__(env)
        self.use_goal = use_goal
        self.task_num = task_num
        base_dim = 12 if use_goal else 6
        obs_dim = base_dim + (1 if task_num is not None else 0)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

    def observation(self, obs):
        if self.use_goal:
            flat = np.concatenate([
                obs["observation"].flatten(),
                obs["desired_goal"].flatten(),
            ]).astype(np.float32)
        else:
            flat = obs["observation"].flatten().astype(np.float32)
        if self.task_num is not None:
            flat = np.append(flat, np.float32(self.task_num))
        return flat


class SafetyWrapperHighway(gym.Wrapper):
    """Add ``info['safe']`` — *True* when the vehicle has **not** crashed."""

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        info["safe"] = not info.get("crashed", False)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["safe"] = not info.get("crashed", False)
        return obs, reward, terminated, truncated, info


class DiscretizeActionWrapper(gym.ActionWrapper):
    """Convert a 2-D continuous ``Box`` action space into a flat ``Discrete`` grid.

    The grid is ``n_bins_accel × n_bins_steer``.  A discrete action ``a`` is
    decoded as:

        accel_idx = a // n_bins_steer
        steer_idx = a %  n_bins_steer

    Both axes are linearly spaced in [-1, 1].
    """

    def __init__(self, env: gym.Env, n_bins_accel: int = 5, n_bins_steer: int = 5):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box), (
            "DiscretizeActionWrapper requires a Box action space"
        )
        assert env.action_space.shape == (2,), (
            f"Expected 2-D action space, got {env.action_space.shape}"
        )
        self.accel_values = np.linspace(-1.0, 1.0, n_bins_accel).astype(np.float32)
        self.steer_values = np.linspace(-1.0, 1.0, n_bins_steer).astype(np.float32)
        self.n_accel = n_bins_accel
        self.n_steer = n_bins_steer
        self.action_space = gym.spaces.Discrete(n_bins_accel * n_bins_steer)

    def action(self, act: int) -> np.ndarray:
        accel_idx = act // self.n_steer
        steer_idx = act % self.n_steer
        return np.array(
            [self.accel_values[accel_idx], self.steer_values[steer_idx]],
            dtype=np.float32,
        )


# =============================================================================
# Environment factory
# =============================================================================

def make_parking_env(
    env_base_config: dict,
    task_config: dict,
    task_num: int,
    use_goal: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a ``custom-parking-v0`` env with Observation + Safety wrappers.

    The caller must ensure that ``highway_env`` and
    ``utils.custom_envs.CustomParkingEnv`` have already been imported so that
    the ``custom-parking-v0`` environment is registered.

    Args:
        env_base_config: Shared environment settings (observation, action,
            reward, simulation, rendering, etc.).
        task_config: Task-specific settings (``goal_spots``,
            ``parked_vehicles_spots``, ``vehicles_count``).
        task_num: Integer task indicator appended to the observation.
        use_goal: Whether to include the desired goal in the observation.
        render_mode: Gymnasium render mode (``None``, ``'human'``,
            ``'rgb_array'``).

    Returns:
        A wrapped Gymnasium environment.
    """
    config = {**env_base_config, **task_config}

    # Convert YAML lists → tuples for lane indices
    if "goal_spots" in config:
        config["goal_spots"] = _convert_spot_lists(config["goal_spots"])
    if "parked_vehicles_spots" in config:
        config["parked_vehicles_spots"] = _convert_spot_lists(
            config["parked_vehicles_spots"],
        )

    # Convert steering_range from degrees if stored that way in YAML
    if "steering_range_deg" in config:
        config["steering_range"] = np.deg2rad(config.pop("steering_range_deg"))

    env = gym.make("custom-parking-v0", render_mode=render_mode, config=config)
    env = ParkingObservationWrapper(env, task_num=task_num, use_goal=use_goal)
    env = SafetyWrapperHighway(env)
    return env


def make_discrete_parking_env(
    env_config: dict,
    task_num: int | None = None,
    n_bins_accel: int = 5,
    n_bins_steer: int = 5,
    use_goal: bool = True,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a parking env with **discretised** actions.

    Unlike :func:`make_parking_env`, which takes split ``env_base_config`` +
    ``task_config`` dicts (designed for the YAML-based pipeline), this
    convenience function accepts a single *flat* config dict — suitable for
    scripts that define the full config inline or load pre-trained checkpoints.

    Wrapping order: ``custom-parking-v0`` → ``ParkingObservationWrapper`` →
    ``DiscretizeActionWrapper`` → ``SafetyWrapperHighway``.

    Args:
        env_config: Complete highway-env config dict.
        task_num: Task indicator appended to the observation.
            Pass ``None`` to omit (keeps 12-dim obs for checkpoint compat).
        n_bins_accel: Number of acceleration bins.
        n_bins_steer: Number of steering bins.
        use_goal: Include the desired-goal in observations.
        render_mode: Gymnasium render mode.

    Returns:
        Wrapped Gymnasium environment with ``Discrete(n_bins_accel * n_bins_steer)``
        action space.
    """
    config = dict(env_config)  # shallow copy

    # Convert YAML lists → tuples where needed
    if "goal_spots" in config:
        config["goal_spots"] = _convert_spot_lists(config["goal_spots"])
    if "parked_vehicles_spots" in config:
        config["parked_vehicles_spots"] = _convert_spot_lists(
            config["parked_vehicles_spots"],
        )
    if "steering_range_deg" in config:
        config["steering_range"] = np.deg2rad(config.pop("steering_range_deg"))

    env = gym.make("custom-parking-v0", render_mode=render_mode, config=config)
    env = ParkingObservationWrapper(env, task_num=task_num, use_goal=use_goal)
    env = DiscretizeActionWrapper(env, n_bins_accel=n_bins_accel, n_bins_steer=n_bins_steer)
    env = SafetyWrapperHighway(env)
    return env


# =============================================================================
# Safety-dataset helpers
# =============================================================================

def create_safe_optimal_policy_dataset(
    env: gym.Env,
    actor: torch.nn.Module,
    num_rollouts: int,
    deterministic: bool = True,
    seed: int = 42,
) -> torch.utils.data.TensorDataset:
    """Collect ``(state, action)`` pairs by rolling out the *actor* in *env*.

    Returns:
        ``TensorDataset(states, actions)`` with ``states`` float32 and
        ``actions`` int64.
    """
    print("\nCreating 'Safe Optimal Policy Data' dataset...")
    states, actions = [], []
    for ep in range(num_rollouts):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            with torch.no_grad():
                logits = actor(torch.FloatTensor(obs).unsqueeze(0))
                if deterministic:
                    action = torch.argmax(logits, dim=1).item()
                else:
                    action = torch.distributions.Categorical(
                        logits=logits,
                    ).sample().item()
            states.append(obs.copy() if isinstance(obs, np.ndarray) else obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(np.array(states)),
        torch.LongTensor(actions),
    )
    print(f"  Collected {len(ds)} state-action pairs from {num_rollouts} rollouts")
    return ds


def create_safe_training_dataset(
    training_data: dict,
) -> torch.utils.data.TensorDataset:
    """Filter unique safe ``(state, action)`` pairs from PPO training data.

    For continuous state spaces, observations are rounded to 6 decimal places
    before deduplication to collapse near-identical transitions.

    Args:
        training_data: Dict with keys ``'states'``, ``'actions'``, ``'safe'``
            as returned by ``ppo_train(..., return_training_data=True)``.

    Returns:
        ``TensorDataset(states, actions)``.
    """
    print("\nCreating 'Safe Training Data' dataset...")
    states = torch.FloatTensor(training_data["states"])
    actions = torch.LongTensor(training_data["actions"])
    safe_idx = np.where(training_data["safe"] == 1.0)[0]
    states_safe = states[safe_idx]
    actions_safe = actions[safe_idx]

    # De-duplicate (round continuous states to avoid float noise)
    df_s = pd.DataFrame(np.round(states_safe.numpy(), decimals=6))
    df_s.columns = [f"f{i}" for i in range(df_s.shape[1])]
    df_a = pd.DataFrame(actions_safe.numpy(), columns=["action"])
    df = pd.concat([df_s, df_a], axis=1).drop_duplicates().reset_index(drop=True)

    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(df.drop(columns=["action"]).values),
        torch.LongTensor(df["action"].values),
    )
    print(f"  Filtered {len(ds)} unique safe state-action pairs from training data")
    return ds


def build_safety_datasets(
    env: gym.Env,
    actor: torch.nn.Module,
    training_data: dict,
    num_rollouts: int = 100,
    seed: int = 42,
) -> dict[str, torch.utils.data.TensorDataset]:
    """Build all applicable safety-dataset variants and return them as a dict.

    For continuous state spaces only two variants are applicable
    (no "Sufficient Safety Data"):

    Keys: ``'Safe Optimal Policy Data'``, ``'Safe Training Data'``.
    """
    ds_optimal = create_safe_optimal_policy_dataset(
        env=env, actor=actor, num_rollouts=num_rollouts, seed=seed,
    )
    ds_training = create_safe_training_dataset(training_data)

    return {
        "Safe Optimal Policy Data": ds_optimal,
        "Safe Training Data": ds_training,
    }


# =============================================================================
# Safety-actor training
# =============================================================================

def train_safety_actor(
    base_actor: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    lr: float = 1e-3,
    epochs: int = 1000,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[torch.nn.Module, float]:
    """Train a *safety reference model* on the safety dataset.

    Single-label cross-entropy only (multi-label is not applicable for
    continuous state spaces).

    The safety actor is initialised as a deep copy of *base_actor* and trained
    until it reaches perfect accuracy or *epochs* is exhausted.

    Args:
        base_actor: The actor to clone and fine-tune.
        dataset: ``TensorDataset(states, actions)`` — single-label.
        lr: Learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size (clamped to dataset length).
        device: ``'cpu'`` or ``'cuda'``.
        verbose: Print progress every 100 epochs.

    Returns:
        ``(safety_actor, final_accuracy)``
    """
    safety_actor = copy.deepcopy(base_actor).to(device)
    batch_size = min(batch_size, len(dataset))
    optimizer = torch.optim.Adam(safety_actor.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
    )

    if verbose:
        print("\n--- Training safety actor ---")
        print(f"  Dataset size: {len(dataset)}")
        print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch size: {batch_size}")

    for epoch in range(epochs):
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            logits = safety_actor(batch_states)

            loss = F.cross_entropy(logits, batch_actions)
            epoch_correct += (logits.argmax(dim=1) == batch_actions).sum().item()
            epoch_total += batch_states.shape[0]
            epoch_loss += loss.item() * batch_states.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0 or acc == 1.0):
            print(
                f"  Epoch {epoch+1:4d}/{epochs}"
                f"  loss={epoch_loss / epoch_total:.4f}  acc={acc:.4f}"
            )
        if acc == 1.0:
            if verbose:
                print("  Perfect accuracy reached — stopping early.")
            break

    # --- final accuracy on full dataset ---
    with torch.no_grad():
        all_states = dataset.tensors[0].to(device)
        all_actions = dataset.tensors[1].to(device)
        all_logits = safety_actor(all_states)
        all_preds = all_logits.argmax(dim=1)
        final_acc = (all_preds == all_actions).float().mean().item()

    if verbose:
        print(f"\n  Safety actor final accuracy: {final_acc:.4f}")
        print("--- Safety actor training complete ---\n")

    return safety_actor, final_acc


# =============================================================================
# Policy evaluation
# =============================================================================

def evaluate_policy(
    env: gym.Env,
    actor: torch.nn.Module,
    num_episodes: int = 100,
) -> dict[str, float]:
    """Roll out *actor* in *env* for *num_episodes* and return metrics.

    Returns:
        Dict with keys ``avg_reward``, ``avg_success``, ``avg_safety_success``,
        ``avg_steps``.
    """
    total_reward = 0.0
    total_success = 0
    total_safety_success = 0
    total_steps = 0

    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        had_crash = False

        while not done:
            with torch.no_grad():
                logits = actor(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(logits, dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            if not info.get("safe", True):
                had_crash = True

        total_reward += ep_reward
        total_success += int(info.get("is_success", False))
        total_safety_success += int(not had_crash)
        total_steps += ep_steps

    n = num_episodes
    return {
        "avg_reward": total_reward / n,
        "avg_success": total_success / n,
        "avg_safety_success": total_safety_success / n,
        "avg_steps": total_steps / n,
    }


# =============================================================================
# Safety-certificate verification
# =============================================================================

def verify_safety_accuracy(
    actor: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    min_acc_limit: float = 1.0,
    verbose: bool = True,
) -> tuple[float, bool]:
    """Check the SafeAdapt actor's accuracy on the safety dataset.

    Single-label only (no multi-label for continuous state spaces).

    Args:
        actor: The trained SafeAdapt actor.
        dataset: Safety ``TensorDataset(states, actions)``.
        min_acc_limit: Required accuracy threshold.
        verbose: Print diagnostics.

    Returns:
        ``(accuracy, passed)`` where *passed* is ``accuracy >= min_acc_limit``.
    """
    with torch.no_grad():
        preds = actor(dataset.tensors[0]).argmax(dim=1)

    accuracy = (preds == dataset.tensors[1]).float().mean().item()
    passed = accuracy >= min_acc_limit

    if verbose:
        status = "PASSED" if passed else "FAILED"
        print(f"\nSafety-certificate verification: {status}")
        print(f"  Accuracy: {accuracy:.4f}  (required >= {min_acc_limit:.4f})")

    if verbose and not passed:
        wrong = torch.where(preds != dataset.tensors[1])[0]
        for i in wrong[:10]:
            print(
                f"  sample {i.item()}: predicted {preds[i].item()}, "
                f"target {dataset.tensors[1][i].item()}"
            )
        if len(wrong) > 10:
            print(f"  ... and {len(wrong) - 10} more failures")
        print(f"  Total incorrect: {len(wrong)} / {len(preds)}")

    return accuracy, passed
