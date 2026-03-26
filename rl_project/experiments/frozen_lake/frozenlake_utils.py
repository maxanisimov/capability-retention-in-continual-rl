"""
FrozenLake Utilities for Safe Continual Learning Experiments
=============================================================

This module provides all helper functions for the FrozenLake SafeAdapt demo:

- **Environment setup**: Gymnasium wrappers for one-hot encoding and safety flags,
  plus a convenience factory ``make_frozenlake_env``.
- **State encoding / decoding**: ``one_hot_encode_state``, ``observation_to_position``,
  ``position_to_observation``.
- **Safety-dataset creation**: three dataset builders
  (``create_safe_optimal_policy_dataset``, ``create_safe_training_dataset``,
  ``generate_sufficient_safe_state_action_dataset``) and the helper
  ``get_all_unsafe_state_action_pairs``.
- **Safety-actor training**: ``train_safety_actor`` trains a reference model
  whose parameters define the centre of the certified safe region.
- **Evaluation & verification**: ``evaluate_policy`` collects rollout metrics;
  ``verify_safety_accuracy`` checks the SafeAdapt actor against the safety
  dataset and prints diagnostics.
- **Reproducibility**: ``set_all_seeds``.
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
from torch.utils.data import TensorDataset

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
# State encoding helpers
# =============================================================================

def one_hot_encode_state(
    state: int,
    num_states: int,
    task_num: int | float = 0,
) -> np.ndarray:
    """Convert a discrete state to a one-hot vector with a task indicator appended.

    Returns:
        np.ndarray of shape ``(num_states + 1,)`` with dtype float32.
    """
    encoded = np.zeros(num_states, dtype=np.float32)
    encoded[state] = 1.0
    encoded = np.append(encoded, float(task_num))
    return encoded


def observation_to_position(observation: np.ndarray | torch.Tensor) -> int:
    """Convert a one-hot observation (with task indicator) to a flat grid index.

    The last element of *observation* is the task indicator and is ignored.
    """
    if isinstance(observation, torch.Tensor):
        return int(torch.argmax(observation[:-1]).item())
    return int(np.argmax(observation[:-1]))


def position_to_observation(
    position: int,
    num_states: int,
    task_num: int | float = 0,
) -> np.ndarray:
    """Convert a flat grid index to a one-hot observation with task indicator."""
    return one_hot_encode_state(position, num_states, task_num)


# =============================================================================
# Gymnasium wrappers
# =============================================================================

class OneHotWrapper(gym.ObservationWrapper):
    """Wrap FrozenLake's discrete state into a one-hot vector + task indicator."""

    def __init__(self, env: gym.Env, task_num: int):
        super().__init__(env)
        n = env.observation_space.n
        low = np.zeros(n + 1, dtype=np.float32)
        high = np.ones(n + 1, dtype=np.float32)
        high[-1] = np.inf  # task indicator is unbounded above
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.task_num = task_num

    def observation(self, obs: int) -> np.ndarray:
        return one_hot_encode_state(obs, self.env.observation_space.n, self.task_num)


class SafetyFlagWrapper(gym.Wrapper):
    """Add ``info['safe']`` — *True* when the current cell is **not** a hole."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.desc = env.unwrapped.desc
        self.nrow, self.ncol = self.desc.shape

    def _is_safe_state(self, state: int) -> bool:
        row = state // self.ncol
        col = state % self.ncol
        cell = self.desc[row, col]
        cell = cell.decode("utf-8") if isinstance(cell, bytes) else cell
        return cell != "H"

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        state = np.argmax(obs[:-1]) if isinstance(obs, np.ndarray) else obs
        info["safe"] = self._is_safe_state(state)
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        state = np.argmax(obs[:-1]) if isinstance(obs, np.ndarray) else obs
        info["safe"] = self._is_safe_state(state)
        return obs, reward, terminated, truncated, info


# =============================================================================
# Environment factory
# =============================================================================

def make_frozenlake_env(
    env_map: list[str],
    task_num: int,
    is_slippery: bool = False,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a FrozenLake-v1 env wrapped with OneHot + SafetyFlag.

    Args:
        env_map: List of strings describing the map (e.g. ``["SFFF", ...]``).
        task_num: Integer task indicator appended to the observation.
        is_slippery: Whether transitions are stochastic.
        render_mode: Gymnasium render mode (``None``, ``'human'``, ``'rgb_array'``).

    Returns:
        A wrapped Gymnasium environment.
    """
    env = gym.make(
        "FrozenLake-v1",
        desc=env_map,
        is_slippery=is_slippery,
        render_mode=render_mode,
    )
    env = OneHotWrapper(env, task_num=task_num)
    env = SafetyFlagWrapper(env)
    return env


# =============================================================================
# Safety-dataset helpers
# =============================================================================
def create_frozenlake_safety_rashomon_dataset(env, task_flag: float = 0.0):
    """
    Create a TensorDataset containing only safety-critical states and their safe actions.

    - X: one-hot observations (optionally with final task-flag dimension)
    - Y: multi-hot vectors of length n_actions, with 1s for safe actions
    """
    desc = env.unwrapped.desc
    grid = [
        "".join(ch.decode() if isinstance(ch, (bytes, bytearray)) else str(ch) for ch in row)
        for row in desc
    ]
    nrows, ncols = len(grid), len(grid[0])
    n_states = nrows * ncols
    n_actions = 4  # FrozenLake: Left, Down, Right, Up

    # Infer observation size from env wrapper
    if hasattr(env.observation_space, "shape") and len(env.observation_space.shape) > 0:
        obs_dim_local = int(env.observation_space.shape[0])
    elif hasattr(env.observation_space, "n"):
        obs_dim_local = int(env.observation_space.n)
    else:
        raise ValueError("Cannot infer observation dimension from env.observation_space.")

    if obs_dim_local not in (n_states, n_states + 1):
        raise ValueError(f"Unsupported obs_dim={obs_dim_local}. Expected {n_states} or {n_states + 1}.")

    def state_to_rc(s: int):
        return s // ncols, s % ncols

    def rc_to_state(r: int, c: int):
        return r * ncols + c

    action_deltas = {
        0: (0, -1),  # Left
        1: (1, 0),   # Down
        2: (0, 1),   # Right
        3: (-1, 0),  # Up
    }

    # Identify hole states
    hole_states = set()
    for r in range(nrows):
        for c in range(ncols):
            if grid[r][c] == "H":
                hole_states.add(rc_to_state(r, c))

    obs_list = []
    label_list = []

    for s in range(n_states):
        r, c = state_to_rc(s)
        cell = grid[r][c]

        # Skip terminal/non-traversable states
        if cell in ("H", "G"):
            continue

        safe_actions = []
        for a, (dr, dc) in action_deltas.items():
            nr, nc = r + dr, c + dc
            hits_wall = (nr < 0 or nr >= nrows or nc < 0 or nc >= ncols)

            if hits_wall:
                # In FrozenLake, wall-hit keeps agent in place (safe)
                safe_actions.append(a)
            else:
                ns = rc_to_state(nr, nc)
                if ns not in hole_states:
                    safe_actions.append(a)

        # Keep only safety-critical states (at least one unsafe action exists)
        if len(safe_actions) == n_actions:
            continue

        obs = np.zeros(obs_dim_local, dtype=np.float32)
        obs[s] = 1.0
        if obs_dim_local == n_states + 1:
            obs[-1] = float(task_flag)

        multi_hot = np.zeros(n_actions, dtype=np.float32)
        for a in safe_actions:
            multi_hot[a] = 1.0
        obs_list.append(obs)
        label_list.append(multi_hot)

    if len(obs_list) == 0:
        raise RuntimeError("No safety-critical states found; dataset is empty.")

    obs_tensor = torch.tensor(np.asarray(obs_list), dtype=torch.float32)
    label_tensor = torch.tensor(np.asarray(label_list), dtype=torch.float32)
    return TensorDataset(obs_tensor, label_tensor)


def get_all_unsafe_state_action_pairs(
    env_map: list[str],
    task_num: int,
    state_repr: str = "observation",
) -> list[tuple]:
    """Return every ``(state, action)`` pair that leads into a hole.

    For deterministic (non-slippery) FrozenLake only.

    Args:
        env_map: List of strings describing the map.
        task_num: Task indicator for the one-hot encoding.
        state_repr: ``'observation'`` → one-hot ndarray, ``'position'`` → int.

    Returns:
        List of ``(state, action)`` tuples.
    """
    assert state_repr in ("observation", "position")
    desc = np.array([list(row) for row in env_map])
    nrow, ncol = desc.shape
    num_states = nrow * ncol
    unsafe_pairs: list[tuple] = []

    for s in range(num_states):
        row, col = s // ncol, s % ncol
        if desc[row, col] in ("H", "G"):
            continue

        if state_repr == "observation":
            state: Any = np.zeros(num_states + 1, dtype=np.float32)
            state[s] = 1.0
            state[-1] = float(task_num)
        else:
            state = s

        deltas = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # L D R U
        for action, (dr, dc) in deltas.items():
            nr = max(0, min(row + dr, nrow - 1))
            nc = max(0, min(col + dc, ncol - 1))
            if desc[nr, nc] == "H":
                if state_repr == "observation":
                    unsafe_pairs.append((state.copy(), action))
                else:
                    unsafe_pairs.append((state, action))

    return unsafe_pairs


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
                    action = torch.distributions.Categorical(logits=logits).sample().item()
            states.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(states), torch.LongTensor(actions)
    )
    print(f"  Collected {len(ds)} state-action pairs from {num_rollouts} rollouts")
    return ds


def create_safe_training_dataset(
    training_data: dict,
) -> torch.utils.data.TensorDataset:
    """Filter unique safe ``(state, action)`` pairs from PPO training data.

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

    # de-duplicate
    df_s = pd.DataFrame(states_safe.numpy())
    df_s.columns = [f"f{i}" for i in range(df_s.shape[1])]
    df_a = pd.DataFrame(actions_safe.numpy(), columns=["action"])
    df = pd.concat([df_s, df_a], axis=1).drop_duplicates().reset_index(drop=True)

    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(df.drop(columns=["action"]).values),
        torch.LongTensor(df["action"].values),
    )
    print(f"  Filtered {len(ds)} unique safe state-action pairs from training data")
    return ds


def generate_sufficient_safe_state_action_dataset(
    unsafe_state_action_pairs: list[tuple],
    env: gym.Env,
) -> torch.utils.data.TensorDataset:
    """Complement of unsafe actions → multi-label safe dataset.

    For each state that has at least one unsafe neighbour, the *safe* actions
    are all actions **not** in the unsafe set.

    Returns:
        ``TensorDataset(states, safe_actions)`` where ``safe_actions`` is
        padded with ``-1`` to ``n_actions`` columns.
    """
    print("\nCreating 'Sufficient Safety Data' dataset...")
    all_actions = set(range(env.action_space.n))

    unsafe_by_state: dict[tuple, set[int]] = {}
    for state, action in unsafe_state_action_pairs:
        key = tuple(state)
        unsafe_by_state.setdefault(key, set()).add(action)

    safe_by_state = {k: all_actions - v for k, v in unsafe_by_state.items()}

    states_t = torch.FloatTensor(list(safe_by_state.keys()))
    max_a = env.action_space.n
    padded = [
        list(sa) + [-1] * (max_a - len(sa)) for sa in safe_by_state.values()
    ]
    actions_t = torch.LongTensor(padded)

    ds = torch.utils.data.TensorDataset(states_t, actions_t)
    print(f"  Generated safe actions for {len(ds)} states with unsafe neighbours")
    return ds


def build_all_safety_datasets(
    env: gym.Env,
    actor: torch.nn.Module,
    env_map: list[str],
    task_num: int,
    training_data: dict,
    num_rollouts: int = 10,
    seed: int = 42,
) -> dict[str, torch.utils.data.TensorDataset]:
    """Build all three safety-dataset variants and return them as a dict.

    Keys: ``'Safe Optimal Policy Data'``, ``'Safe Training Data'``,
    ``'Sufficient Safety Data'``.
    """
    ds_optimal = create_safe_optimal_policy_dataset(
        env=env, actor=actor, num_rollouts=num_rollouts, seed=seed,
    )
    ds_training = create_safe_training_dataset(training_data)
    unsafe_pairs = get_all_unsafe_state_action_pairs(env_map=env_map, task_num=task_num)
    ds_sufficient = generate_sufficient_safe_state_action_dataset(unsafe_pairs, env)

    return {
        "Safe Optimal Policy Data": ds_optimal,
        "Safe Training Data": ds_training,
        "Sufficient Safety Data": ds_sufficient,
    }


# =============================================================================
# Safety-actor training
# =============================================================================

def margin_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Cross-entropy with a margin penalty to encourage large logit gaps.

    Adds a hinge term that penalises samples where the correct-class logit
    is not at least *margin* above the highest competing logit.  This makes
    the safety actor easier to certify with interval bound propagation.

    Args:
        logits: Raw network outputs, shape ``(B, C)``.
        targets: Ground-truth class indices, shape ``(B,)``.
        margin: Required gap between correct-class logit and runner-up.

    Returns:
        Scalar loss (CE + mean hinge violation).
    """
    ce = F.cross_entropy(logits, targets)
    correct_logits = logits.gather(1, targets.unsqueeze(1))  # (B, 1)
    # Mask out the target class so we take max over non-target logits
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, targets.unsqueeze(1), False)
    max_other = logits.masked_fill(~mask, -1e9).max(dim=1, keepdim=True).values
    margin_violation = torch.clamp(margin - (correct_logits - max_other), min=0)
    return ce + margin_violation.mean()


def _multi_label_margin_cross_entropy(
    logits: torch.Tensor,
    target_dist: torch.Tensor,
    valid_mask_any: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Soft CE + margin penalty for the multi-label case.

    The margin is enforced between the *best valid-action logit* and the
    *best invalid-action logit* for each sample.

    Args:
        logits: Raw network outputs, shape ``(B, C)``.
        target_dist: Normalised target distribution, shape ``(B, C)``.
        valid_mask_any: Boolean mask of shape ``(B, C)`` — True for valid actions.
        margin: Required gap.

    Returns:
        Scalar loss.
    """
    log_probs = F.log_softmax(logits, dim=1)
    ce = -(target_dist * log_probs).sum(dim=1).mean()

    # Best valid-action logit per sample
    valid_logits = logits.masked_fill(~valid_mask_any, -1e9)
    best_valid = valid_logits.max(dim=1).values  # (B,)

    # Best invalid-action logit per sample
    invalid_logits = logits.masked_fill(valid_mask_any, -1e9)
    best_invalid = invalid_logits.max(dim=1).values  # (B,)

    margin_violation = torch.clamp(margin - (best_valid - best_invalid), min=0)
    return ce + margin_violation.mean()


def train_safety_actor(
    base_actor: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    multi_label: bool,
    lr: float = 1e-3,
    epochs: int = 500,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
    use_margin_loss: bool = False,
    margin: float = 2.0,
) -> tuple[torch.nn.Module, float]:
    """Train a *safety reference model* on the safety dataset.

    The safety actor is initialised as a deep copy of *base_actor* and trained
    with (soft) cross-entropy until it reaches perfect accuracy or
    *epochs* is exhausted.

    Args:
        base_actor: The actor to clone and fine-tune.
        dataset: ``TensorDataset(states, actions)``.  For multi-label datasets
            the actions tensor is a multi-hot float tensor of shape
            ``(N, n_actions)`` with 1 for valid actions and 0 otherwise.
        multi_label: Whether the dataset contains multiple valid actions per
            state.
        lr: Learning rate.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size (clamped to dataset length).
        device: ``'cpu'`` or ``'cuda'``.
        verbose: Print progress every 100 epochs.
        use_margin_loss: If ``True``, add a hinge-margin penalty that
            encourages the correct-class logit to exceed all others by at
            least *margin*.  This produces larger logit gaps that are easier
            to certify with interval bound propagation.
        margin: Required logit gap when ``use_margin_loss=True``.

    Returns:
        ``(safety_actor, final_accuracy)``
    """
    safety_actor = copy.deepcopy(base_actor).to(device)
    batch_size = min(batch_size, len(dataset))
    optimizer = torch.optim.Adam(safety_actor.parameters(), lr=lr)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if verbose:
        print("\n--- Training safety actor ---")
        print(f"  Multi-label: {multi_label}  |  Dataset size: {len(dataset)}")
        print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch size: {batch_size}")
        if use_margin_loss:
            print(f"  Margin loss: enabled  |  Margin: {margin}")

    for epoch in range(epochs):
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            logits = safety_actor(batch_states)

            if multi_label:
                target_dist = batch_actions / batch_actions.sum(dim=1, keepdim=True).clamp(min=1e-8)

                if use_margin_loss:
                    valid_action_mask = batch_actions.bool()
                    loss = _multi_label_margin_cross_entropy(
                        logits, target_dist, valid_action_mask, margin=margin,
                    )
                else:
                    log_probs = F.log_softmax(logits, dim=1)
                    loss = -(target_dist * log_probs).sum(dim=1).mean()

                predicted = logits.argmax(dim=1)
                epoch_correct += batch_actions[torch.arange(batch_actions.size(0)), predicted].sum().item()
            else:
                if use_margin_loss:
                    loss = margin_cross_entropy(logits, batch_actions, margin=margin)
                else:
                    loss = F.cross_entropy(logits, batch_actions)
                epoch_correct += (logits.argmax(dim=1) == batch_actions).sum().item()

            epoch_total += batch_states.shape[0]
            epoch_loss += loss.item() * batch_states.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        acc = epoch_correct / epoch_total if epoch_total > 0 else 0.0
        if verbose and ((epoch + 1) % 100 == 0 or epoch == 0 or acc == 1.0):
            print(f"  Epoch {epoch+1:4d}/{epochs}  loss={epoch_loss/epoch_total:.4f}  acc={acc:.4f}")
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
        if multi_label:
            final_acc = all_actions[torch.arange(all_actions.size(0)), all_preds].float().mean().item()
        else:
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
        Dict with keys ``avg_total_reward``, ``avg_success``, ``avg_safety_success``,
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
        fell_in_hole = False

        while not done:
            with torch.no_grad():
                logits = actor(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(logits, dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            if not info.get("safe", True):
                fell_in_hole = True

        total_reward += ep_reward
        total_success += int(reward > 0)
        total_safety_success += int(not fell_in_hole)
        total_steps += ep_steps

    n = num_episodes
    return {
        "avg_total_reward": total_reward / n,
        "avg_success": total_success / n,
        "avg_safety_success": total_safety_success / n,
        "avg_steps": total_steps / n,
    }


# =============================================================================
# Safety-certificate verification
# =============================================================================

def verify_safety_posthoc(
    actor: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    multi_label: bool,
    min_safety_limit: float = 1.0,
    env_map: list[str] | None = None,
    aggregation: str = 'min', # 'min' or 'mean'
    verbose: bool = True,
) -> tuple[float, bool]:
    """Check the SafeAdapt actor's safety in the dataset of safety-critical states.

    Args:
        actor: The trained SafeAdapt actor.
        dataset: Safety ``TensorDataset(states, actions)``.
        multi_label: Whether *dataset* uses multi-label targets.
        min_safety_limit: Required safety threshold.
        env_map: Used only for printing grid positions of failures.
        aggregation: Method to aggregate the safety - 'min' implies taking the min acorss states, 'mean' implies taking the mean across states.
        verbose: Print diagnostics.

    Returns:
        ``(accuracy, passed)`` where *passed* is ``accuracy >= min_safety_limit``.
    """
    with torch.no_grad():
        preds = actor(dataset.tensors[0]).argmax(dim=1)

    if multi_label:
        valid_mask = dataset.tensors[1].bool()  # multi-hot (N, n_actions)
        correct = valid_mask[torch.arange(len(preds)), preds]
        safe_per_state = correct.float()
    else:
        safe_per_state = (preds == dataset.tensors[1]).float()
        
    if aggregation == 'min':
        accuracy = safe_per_state.min().item()
    else:
        accuracy = safe_per_state.mean().item()

    passed = safe_per_state >= min_safety_limit

    if verbose:
        status = "PASSED" if passed else "FAILED"
        print(f"\nSafety-certificate verification: {status}")
        print(f"  Accuracy: {accuracy:.4f}  (required >= {min_safety_limit:.4f})")

    if verbose and not passed and env_map is not None:
        nrow, ncol = len(env_map), len(env_map[0])

        def _pos(state_vec: torch.Tensor) -> tuple[int, int]:
            idx = int(torch.argmax(state_vec[:-1]).item())
            return idx // ncol, idx % ncol

        if multi_label:
            wrong = torch.where(~correct)[0]
            for i in wrong:
                r, c = _pos(dataset.tensors[0][i])
                p = preds[i].item()
                va = torch.where(valid_mask[i])[0].tolist()
                print(f"  ({r},{c}): predicted {p}, valid {va}")
        else:
            wrong = torch.where(preds != dataset.tensors[1])[0]
            for i in wrong:
                r, c = _pos(dataset.tensors[0][i])
                print(
                    f"  ({r},{c}): predicted {preds[i].item()}, "
                    f"target {dataset.tensors[1][i].item()}"
                )
        print(f"  Total incorrect: {len(wrong)} / {len(preds)}")

    return accuracy, passed


# =============================================================================
# Convenience: extract safe position-action pairs for plotting
# =============================================================================

def extract_position_action_pairs(
    dataset: torch.utils.data.TensorDataset,
) -> list[tuple[int, int]]:
    """Convert a multi-label safety dataset to ``(position, action)`` pairs.

    Useful for passing to ``plot_state_action_pairs``.
    """
    pairs: list[tuple[int, int]] = []
    states = dataset.tensors[0]
    actions = dataset.tensors[1]
    for i in range(states.shape[0]):
        pos = observation_to_position(states[i])
        for a in actions[i]:
            if a.item() != -1:
                pairs.append((pos, a.item()))
    return pairs
