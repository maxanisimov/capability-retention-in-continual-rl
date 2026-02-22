"""
Taxi Utilities for Safe Continual Learning Experiments
======================================================

This module provides all helper functions for the Taxi SafeAdapt demo:

- **Environment setup**: Gymnasium wrappers for one-hot observation encoding,
  safety flags (illegal pickup/dropoff detection) and initial-state
  distribution control, plus a convenience factory ``make_taxi_env``.
- **State encoding / decoding**: ``one_hot_encode_state``,
  ``observation_to_state``, ``decode_taxi_state``, ``encode_taxi_state``.
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


# =============================================================================
# Taxi-v3 constants
# =============================================================================

#: Named pickup/dropoff locations in (row, col) order.
TAXI_LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
LOC_NAMES = {0: "R", 1: "G", 2: "Y", 3: "B", 4: "in_taxi"}
ACTION_NAMES = {
    0: "South", 1: "North", 2: "East", 3: "West",
    4: "Pickup", 5: "Dropoff",
}

NUM_ROWS, NUM_COLS = 5, 5
NUM_PASS_LOCS = 5       # 0-3 = at a location, 4 = in taxi
NUM_DEST = 4
NUM_TAXI_STATES = NUM_ROWS * NUM_COLS * NUM_PASS_LOCS * NUM_DEST  # 500
NUM_ACTIONS = 6


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

def encode_taxi_state(
    taxi_row: int, taxi_col: int, pass_loc: int, dest_idx: int,
) -> int:
    """Encode taxi state components into a single state index (Taxi-v3 layout)."""
    return (
        ((taxi_row * NUM_COLS + taxi_col) * NUM_PASS_LOCS + pass_loc)
        * NUM_DEST + dest_idx
    )


def decode_taxi_state(state: int) -> tuple[int, int, int, int]:
    """Decode a Taxi-v3 state index into ``(taxi_row, taxi_col, pass_loc, dest_idx)``."""
    dest_idx = state % NUM_DEST
    state //= NUM_DEST
    pass_loc = state % NUM_PASS_LOCS
    state //= NUM_PASS_LOCS
    taxi_col = state % NUM_COLS
    taxi_row = state // NUM_COLS
    return taxi_row, taxi_col, pass_loc, dest_idx


def one_hot_encode_state(
    state: int,
    num_states: int = NUM_TAXI_STATES,
    task_num: int | float = 0,
) -> np.ndarray:
    """Convert a discrete state to a one-hot vector with a task indicator appended.

    Returns:
        np.ndarray of shape ``(num_states + 1,)`` with dtype float32.
    """
    encoded = np.zeros(num_states + 1, dtype=np.float32)
    encoded[state] = 1.0
    encoded[-1] = np.float32(task_num)
    return encoded


def observation_to_state(observation: np.ndarray | torch.Tensor) -> int:
    """Convert a one-hot observation (with task indicator) to a discrete state index.

    The last element of *observation* is the task indicator and is ignored.
    """
    if isinstance(observation, torch.Tensor):
        return int(torch.argmax(observation[:-1]).item())
    return int(np.argmax(observation[:-1]))


# =============================================================================
# Gymnasium wrappers
# =============================================================================

class OneHotWrapper(gym.ObservationWrapper):
    """Wrap Taxi's discrete state into a one-hot vector + task indicator."""

    def __init__(self, env: gym.Env, task_num: int):
        super().__init__(env)
        n = env.observation_space.n  # type: ignore[union-attr]
        low = np.zeros(n + 1, dtype=np.float32)
        high = np.ones(n + 1, dtype=np.float32)
        high[-1] = np.inf  # task indicator is unbounded above
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.task_num = task_num

    def observation(self, obs: int) -> np.ndarray:
        return one_hot_encode_state(
            obs, self.env.observation_space.n, self.task_num,  # type: ignore[union-attr]
        )


class TaxiSafetyWrapper(gym.Wrapper):
    """Add ``info['safe']`` — *True* when the last action was **not** an
    illegal pickup/dropoff (i.e. did **not** receive a -10 penalty).
    """

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        info["safe"] = True  # initial state is always safe
        return obs, info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info["safe"] = not (action in (4, 5) and reward == -10)
        return obs, reward, terminated, truncated, info


class FixedRouteTaxiWrapper(gym.Wrapper):
    """Constrain initial passenger and destination locations.

    On each ``reset`` the wrapper overrides the sampled initial state so that
    the passenger starts at one of *passenger_locs* and the destination is one
    of *dest_locs* (with passenger != destination).  The taxi position is
    uniformly random across the 5x5 grid.

    Args:
        env: Base Taxi-v3 environment (must have a discrete state ``s``).
        passenger_locs: Allowed passenger-start location indices (0-3).
        dest_locs: Allowed destination location indices (0-3).
    """

    def __init__(
        self,
        env: gym.Env,
        passenger_locs: list[int],
        dest_locs: list[int],
    ):
        super().__init__(env)
        self.passenger_locs = passenger_locs
        self.dest_locs = dest_locs

    def reset(self, *, seed: int | None = None, **kwargs: Any):
        # Let the base env handle seeding and internal bookkeeping.
        obs, info = self.env.reset(seed=seed, **kwargs)

        rng = self.env.unwrapped.np_random
        taxi_row = int(rng.integers(0, NUM_ROWS))
        taxi_col = int(rng.integers(0, NUM_COLS))
        pass_loc = int(rng.choice(self.passenger_locs))
        dest_candidates = [d for d in self.dest_locs if d != pass_loc]
        if not dest_candidates:
            # Fallback when sets overlap and only one element remains.
            dest_candidates = [d for d in range(NUM_DEST) if d != pass_loc]
        dest_idx = int(rng.choice(dest_candidates))

        state = encode_taxi_state(taxi_row, taxi_col, pass_loc, dest_idx)
        self.env.unwrapped.s = state
        return int(state), info


# =============================================================================
# Environment factory
# =============================================================================

def make_taxi_env(
    task_num: int,
    passenger_locs: list[int] | None = None,
    dest_locs: list[int] | None = None,
    render_mode: str | None = None,
) -> gym.Env:
    """Create a Taxi-v3 env wrapped with FixedRoute + OneHot + Safety wrappers.

    Args:
        task_num: Integer task indicator appended to the observation.
        passenger_locs: If given (together with *dest_locs*), wrap with
            :class:`FixedRouteTaxiWrapper` to constrain initial states.
        dest_locs: See *passenger_locs*.
        render_mode: Gymnasium render mode (``None``, ``'human'``, ``'rgb_array'``).

    Returns:
        A wrapped Gymnasium environment.
    """
    env = gym.make("Taxi-v3", render_mode=render_mode)
    if passenger_locs is not None and dest_locs is not None:
        env = FixedRouteTaxiWrapper(
            env, passenger_locs=passenger_locs, dest_locs=dest_locs,
        )
    env = OneHotWrapper(env, task_num=task_num)
    env = TaxiSafetyWrapper(env)
    return env


# =============================================================================
# Safety-dataset helpers
# =============================================================================

def get_all_unsafe_state_action_pairs(
    task_num: int = 0,
    passenger_locs: list[int] | None = None,
    dest_locs: list[int] | None = None,
    state_repr: str = "observation",
) -> list[tuple]:
    """Return every ``(state, action)`` pair that is an illegal pickup/dropoff.

    For every reachable state in the task, actions 4 (Pickup) and 5 (Dropoff)
    are checked against the Taxi-v3 rules:

    * **Pickup** is illegal when the passenger is already in the taxi
      (``pass_loc == 4``) **or** the taxi is not at the passenger's location.
    * **Dropoff** is illegal when the passenger is **not** in the taxi
      (``pass_loc != 4``) **or** the taxi is not at any of the four special
      locations (R, G, Y, B).

    If *passenger_locs* and *dest_locs* are provided, only states reachable
    under that route constraint are enumerated.  Otherwise all 500 states are
    checked.

    Args:
        task_num: Task indicator for the one-hot encoding.
        passenger_locs: Allowed initial passenger locations (for filtering).
        dest_locs: Allowed destination locations (for filtering).
        state_repr: ``'observation'`` → one-hot ndarray, ``'state_index'`` → int.

    Returns:
        List of ``(state, action)`` tuples.
    """
    assert state_repr in ("observation", "state_index")

    # Determine reachable states
    reachable: list[int] = []
    if passenger_locs is not None and dest_locs is not None:
        for row in range(NUM_ROWS):
            for col in range(NUM_COLS):
                for dest in dest_locs:
                    # Before pickup: passenger at one of the allowed locations
                    for ploc in passenger_locs:
                        if ploc != dest:
                            reachable.append(
                                encode_taxi_state(row, col, ploc, dest),
                            )
                    # After pickup: passenger in taxi
                    reachable.append(encode_taxi_state(row, col, 4, dest))
    else:
        reachable = list(range(NUM_TAXI_STATES))

    unsafe_pairs: list[tuple] = []
    for s in reachable:
        taxi_row, taxi_col, pass_loc, dest_idx = decode_taxi_state(s)
        taxi_loc = (taxi_row, taxi_col)

        if state_repr == "observation":
            state_vec: Any = one_hot_encode_state(s, NUM_TAXI_STATES, task_num)
        else:
            state_vec = s

        # Pickup illegal when: pass already in taxi OR taxi not at passenger
        pickup_unsafe = (pass_loc == 4) or (taxi_loc != TAXI_LOCS[pass_loc])
        # Dropoff illegal when: pass not in taxi OR taxi not at any location
        dropoff_unsafe = (pass_loc != 4) or (taxi_loc not in TAXI_LOCS)

        if pickup_unsafe:
            if state_repr == "observation":
                unsafe_pairs.append((state_vec.copy(), 4))
            else:
                unsafe_pairs.append((state_vec, 4))

        if dropoff_unsafe:
            if state_repr == "observation":
                unsafe_pairs.append((state_vec.copy(), 5))
            else:
                unsafe_pairs.append((state_vec, 5))

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
                    action = torch.distributions.Categorical(
                        logits=logits,
                    ).sample().item()
            states.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(states), torch.LongTensor(actions),
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
    """Complement of unsafe actions -> multi-label safe dataset.

    For each state that has at least one unsafe action, the *safe* actions
    are all actions **not** in the unsafe set.

    Returns:
        ``TensorDataset(states, safe_actions)`` where ``safe_actions`` is
        padded with ``-1`` to ``NUM_ACTIONS`` columns.
    """
    print("\nCreating 'Sufficient Safety Data' dataset...")
    all_actions = set(range(NUM_ACTIONS))

    unsafe_by_state: dict[tuple, set[int]] = {}
    for state, action in unsafe_state_action_pairs:
        key = tuple(state)
        unsafe_by_state.setdefault(key, set()).add(action)

    safe_by_state = {k: all_actions - v for k, v in unsafe_by_state.items()}

    states_t = torch.FloatTensor(list(safe_by_state.keys()))
    padded = [
        list(sa) + [-1] * (NUM_ACTIONS - len(sa))
        for sa in safe_by_state.values()
    ]
    actions_t = torch.LongTensor(padded)

    ds = torch.utils.data.TensorDataset(states_t, actions_t)
    print(f"  Generated safe actions for {len(ds)} states with unsafe actions")
    return ds


def build_all_safety_datasets(
    env: gym.Env,
    actor: torch.nn.Module,
    task_num: int,
    training_data: dict,
    passenger_locs: list[int] | None = None,
    dest_locs: list[int] | None = None,
    num_rollouts: int = 100,
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
    unsafe_pairs = get_all_unsafe_state_action_pairs(
        task_num=task_num,
        passenger_locs=passenger_locs,
        dest_locs=dest_locs,
    )
    ds_sufficient = generate_sufficient_safe_state_action_dataset(unsafe_pairs, env)

    return {
        "Safe Optimal Policy Data": ds_optimal,
        "Safe Training Data": ds_training,
        "Sufficient Safety Data": ds_sufficient,
    }


# =============================================================================
# Safety-actor training
# =============================================================================

def train_safety_actor(
    base_actor: torch.nn.Module,
    dataset: torch.utils.data.TensorDataset,
    multi_label: bool,
    lr: float = 1e-3,
    epochs: int = 1000,
    batch_size: int = 64,
    device: str = "cpu",
    verbose: bool = True,
) -> tuple[torch.nn.Module, float]:
    """Train a *safety reference model* on the safety dataset.

    The safety actor is initialised as a deep copy of *base_actor* and trained
    with (soft) cross-entropy until it reaches perfect accuracy or *epochs* is
    exhausted.

    Args:
        base_actor: The actor to clone and fine-tune.
        dataset: ``TensorDataset(states, actions)``.  For multi-label datasets
            the actions tensor has shape ``(N, max_actions)`` and is padded
            with ``-1``.
        multi_label: Whether the dataset contains multiple valid actions per
            state.
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
        print(f"  Multi-label: {multi_label}  |  Dataset size: {len(dataset)}")
        print(f"  Epochs: {epochs}  |  LR: {lr}  |  Batch size: {batch_size}")

    for epoch in range(epochs):
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for batch_states, batch_actions in loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)
            logits = safety_actor(batch_states)

            if multi_label:
                valid_mask = batch_actions != -1
                target_dist = torch.zeros_like(logits)
                for i in range(batch_actions.shape[1]):
                    col = batch_actions[:, i].clamp(min=0).unsqueeze(1)
                    col_valid = valid_mask[:, i].float().unsqueeze(1)
                    target_dist.scatter_(1, col, col_valid)
                target_dist = target_dist / target_dist.sum(
                    dim=1, keepdim=True,
                ).clamp(min=1e-8)
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(target_dist * log_probs).sum(dim=1).mean()

                predicted = logits.argmax(dim=1)
                pred_exp = predicted.unsqueeze(1).expand_as(batch_actions)
                epoch_correct += (
                    ((pred_exp == batch_actions) & valid_mask).any(dim=1).sum().item()
                )
            else:
                loss = F.cross_entropy(logits, batch_actions)
                epoch_correct += (
                    (logits.argmax(dim=1) == batch_actions).sum().item()
                )

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
        if multi_label:
            vm = all_actions != -1
            pe = all_preds.unsqueeze(1).expand_as(all_actions)
            final_acc = (
                ((pe == all_actions) & vm).any(dim=1).float().mean().item()
            )
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
        had_unsafe_action = False
        ep_terminated = False

        while not done:
            with torch.no_grad():
                logits = actor(torch.FloatTensor(obs).unsqueeze(0))
                action = torch.argmax(logits, dim=1).item()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_reward += reward
            ep_steps += 1
            if not info.get("safe", True):
                had_unsafe_action = True
            if terminated:
                ep_terminated = True

        total_reward += ep_reward
        # In Taxi-v3 terminated=True only on successful delivery
        total_success += int(ep_terminated)
        total_safety_success += int(not had_unsafe_action)
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
    multi_label: bool,
    min_acc_limit: float = 1.0,
    verbose: bool = True,
) -> tuple[float, bool]:
    """Check the SafeAdapt actor's accuracy on the safety dataset.

    Args:
        actor: The trained SafeAdapt actor.
        dataset: Safety ``TensorDataset(states, actions)``.
        multi_label: Whether *dataset* uses multi-label targets.
        min_acc_limit: Required accuracy threshold.
        verbose: Print diagnostics.

    Returns:
        ``(accuracy, passed)`` where *passed* is ``accuracy >= min_acc_limit``.
    """
    with torch.no_grad():
        preds = actor(dataset.tensors[0]).argmax(dim=1)

    if multi_label:
        valid_actions = dataset.tensors[1]
        valid_mask = valid_actions != -1
        preds_exp = preds.unsqueeze(1).expand_as(valid_actions)
        correct = ((preds_exp == valid_actions) & valid_mask).any(dim=1)
        accuracy = correct.float().mean().item()
    else:
        accuracy = (preds == dataset.tensors[1]).float().mean().item()

    passed = accuracy >= min_acc_limit

    if verbose:
        status = "PASSED" if passed else "FAILED"
        print(f"\nSafety-certificate verification: {status}")
        print(f"  Accuracy: {accuracy:.4f}  (required >= {min_acc_limit:.4f})")

    if verbose and not passed:
        if multi_label:
            wrong = torch.where(~correct)[0]
            for i in wrong[:10]:
                state_idx = observation_to_state(dataset.tensors[0][i])
                taxi_row, taxi_col, pass_loc, dest_idx = decode_taxi_state(
                    state_idx,
                )
                p = preds[i].item()
                va = valid_actions[i][valid_actions[i] != -1].tolist()
                print(
                    f"  state={state_idx}"
                    f" (taxi=({taxi_row},{taxi_col}),"
                    f" pass={LOC_NAMES[pass_loc]},"
                    f" dest={LOC_NAMES[dest_idx]}):"
                    f" predicted {ACTION_NAMES[p]},"
                    f" valid {[ACTION_NAMES[a] for a in va]}"
                )
            if len(wrong) > 10:
                print(f"  ... and {len(wrong) - 10} more failures")
        else:
            wrong = torch.where(preds != dataset.tensors[1])[0]
            for i in wrong[:10]:
                state_idx = observation_to_state(dataset.tensors[0][i])
                print(
                    f"  state={state_idx}:"
                    f" predicted {preds[i].item()},"
                    f" target {dataset.tensors[1][i].item()}"
                )
        print(f"  Total incorrect: {len(wrong)} / {len(preds)}")

    return accuracy, passed
