"""
Taxi Utilities for Safe Continual Learning Experiments
======================================================

This module provides all helper functions for the Taxi SafeAdapt demo:

- **Environment setup**: Gymnasium wrappers for decoded-vector observations,
  safety flags (illegal pickup/dropoff detection) and initial-state
  distribution control, plus a convenience factory ``make_taxi_env``.
- **State encoding / decoding**: ``vector_encode_state``,
  ``observation_to_state``, ``decode_taxi_state``, ``encode_taxi_state``.
- **Safety-dataset creation**: three dataset builders
  (``create_safe_optimal_policy_dataset``, ``create_safe_training_dataset``,
  ``generate_sufficient_safe_state_action_dataset``) and the helpers
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


def _validate_taxi_grid_location(
    taxi_loc: tuple[int, int],
    *,
    arg_name: str = "initial_taxi_loc",
) -> tuple[int, int]:
    """Validate and normalize a taxi grid location."""
    if not isinstance(taxi_loc, (tuple, list)) or len(taxi_loc) != 2:
        raise ValueError(f"{arg_name} must be a (row, col) pair, got {taxi_loc!r}")

    row_raw, col_raw = taxi_loc
    if not isinstance(row_raw, (int, np.integer)) or not isinstance(col_raw, (int, np.integer)):
        raise TypeError(
            f"{arg_name} row/col must be integers, got types "
            f"({type(row_raw).__name__}, {type(col_raw).__name__})"
        )

    row, col = int(row_raw), int(col_raw)
    if not (0 <= row < NUM_ROWS and 0 <= col < NUM_COLS):
        raise ValueError(
            f"{arg_name} must be within Taxi grid bounds "
            f"(row in [0, {NUM_ROWS - 1}], col in [0, {NUM_COLS - 1}]), "
            f"got ({row}, {col})"
        )
    return row, col


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


def decode_observation(obs: int) -> list[int]:
    """Decode Taxi-v3 integer observation into [taxi_row, taxi_col, passenger, destination]."""
    taxi_row = obs // 100
    taxi_col = (obs % 100) // 20
    passenger = (obs % 20) // 4
    destination = obs % 4
    return [taxi_row, taxi_col, passenger, destination]


def vector_encode_state(
    state: int,
    task_num: int | float = 0,  # kept for API compatibility
) -> np.ndarray:
    """Convert a discrete Taxi state index to decoded vector representation.

    The output format is ``[taxi_row, taxi_col, passenger_loc, destination]``.
    """
    taxi_row, taxi_col, pass_loc, dest_idx = decode_taxi_state(state)
    _ = task_num  # currently unused in the vector representation
    return np.array([taxi_row, taxi_col, pass_loc, dest_idx], dtype=np.float32)


def one_hot_encode_state(
    state: int,
    num_states: int = NUM_TAXI_STATES,
    task_num: int | float = 0,
) -> np.ndarray:
    """Backward-compatible alias to vector observation encoding.

    NOTE: despite its legacy name, this now returns the decoded 4D vector
    representation. ``num_states`` and ``task_num`` are accepted only for API
    compatibility with existing call sites.
    """
    _ = num_states
    return vector_encode_state(state, task_num=task_num)


def observation_to_state(observation: np.ndarray | torch.Tensor) -> int:
    """Convert an observation vector back to a discrete Taxi state index.

    Supports:
    - current decoded-vector format ``[taxi_row, taxi_col, passenger, destination]``
    - legacy one-hot format (for backward compatibility)
    """
    if isinstance(observation, torch.Tensor):
        obs = observation.detach().cpu().numpy()
    else:
        obs = np.asarray(observation)
    obs = obs.astype(np.float32).reshape(-1)

    # Legacy compatibility: one-hot state(+task) vectors.
    if obs.size >= NUM_TAXI_STATES:
        return int(np.argmax(obs[:NUM_TAXI_STATES]))

    if obs.size < 4:
        raise ValueError(
            f"Observation must have at least 4 values for decoded Taxi vector. Got shape={obs.shape}."
        )

    taxi_row = int(np.clip(round(float(obs[0])), 0, NUM_ROWS - 1))
    taxi_col = int(np.clip(round(float(obs[1])), 0, NUM_COLS - 1))
    pass_loc = int(np.clip(round(float(obs[2])), 0, NUM_PASS_LOCS - 1))
    dest_idx = int(np.clip(round(float(obs[3])), 0, NUM_DEST - 1))
    return encode_taxi_state(taxi_row, taxi_col, pass_loc, dest_idx)


# =============================================================================
# Gymnasium wrappers
# =============================================================================

class DecodedObservationWrapper(gym.ObservationWrapper):
    """Wrap Taxi's discrete state into decoded vector [row, col, passenger, destination]."""

    def __init__(self, env: gym.Env, task_num: int):
        super().__init__(env)
        _ = task_num  # kept for API compatibility
        self.observation_space = gym.spaces.Box(
            low=0,
            high=np.array([NUM_ROWS - 1, NUM_COLS - 1, NUM_PASS_LOCS - 1, NUM_DEST - 1], dtype=np.int32),
            shape=(4,),
            dtype=np.int32,
        )

    def observation(self, obs: int) -> np.ndarray:
        return np.asarray(decode_observation(int(obs)), dtype=np.int32)


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
        # Reward of -10 is given by Taxi-v3 for illegal pickup/dropoff actions.
        info["safe"] = not (action in (4, 5) and reward == -10)
        return obs, reward, terminated, truncated, info


class FixedRouteTaxiWrapper(gym.Wrapper):
    """Constrain initial passenger and destination locations.

    On each ``reset`` the wrapper overrides the sampled initial state so that
    the passenger starts at *passenger_loc* and the destination is
    *dest_loc*. The taxi position is fixed to *initial_taxi_loc*.

    Args:
        env: Base Taxi-v3 environment (must have a discrete state ``s``).
        passenger_loc: Fixed passenger-start location index (0-3).
        dest_loc: Fixed destination location index (0-3).
        initial_taxi_loc: Fixed initial taxi location ``(row, col)``.
    """

    def __init__(
        self,
        env: gym.Env,
        passenger_loc: int,
        dest_loc: int,
        initial_taxi_loc: tuple[int, int] = (2, 3),
    ):
        super().__init__(env)
        self.passenger_loc = int(passenger_loc)
        self.dest_loc = int(dest_loc)
        self.initial_taxi_loc = _validate_taxi_grid_location(initial_taxi_loc)
        if self.passenger_loc == self.dest_loc:
            raise ValueError(
                "passenger_loc and dest_loc must be different for Taxi-v3 route constraints."
            )
        if not (0 <= self.passenger_loc < NUM_DEST):
            raise ValueError(f"passenger_loc must be in [0, {NUM_DEST - 1}], got {self.passenger_loc}")
        if not (0 <= self.dest_loc < NUM_DEST):
            raise ValueError(f"dest_loc must be in [0, {NUM_DEST - 1}], got {self.dest_loc}")

    def reset(self, *, seed: int | None = None, **kwargs: Any):
        # Let the base env handle seeding and internal bookkeeping.
        _obs, info = self.env.reset(seed=seed, **kwargs)

        taxi_row, taxi_col = self.initial_taxi_loc
        pass_loc = int(self.passenger_loc)
        dest_idx = int(self.dest_loc)

        state = encode_taxi_state(taxi_row, taxi_col, pass_loc, dest_idx)
        self.env.unwrapped.s = state
        return int(state), info


# =============================================================================
# Environment factory
# =============================================================================

def make_taxi_env(
    task_num: int,
    passenger_loc: int | None = None,
    dest_loc: int | None = None,
    initial_taxi_loc: tuple[int, int] = (2, 3),
    render_mode: str | None = None,
    is_rainy: bool = False,
    fickle_passenger: bool = False,
) -> gym.Env:
    """Create a Taxi-v3 env wrapped with FixedRoute + decoded-observation + Safety wrappers.

    Args:
        task_num: Kept for API compatibility (not used in decoded-vector observations).
        passenger_loc: If given (together with *dest_loc*), wrap with
            :class:`FixedRouteTaxiWrapper` to constrain initial states.
        dest_loc: See *passenger_loc*.
        initial_taxi_loc: Fixed taxi start location ``(row, col)`` used when
            route constraints are enabled.
        render_mode: Gymnasium render mode (``None``, ``'human'``, ``'rgb_array'``).
        is_rainy: Taxi-v3 stochastic transition flag (False = deterministic).
        fickle_passenger: Taxi-v3 destination-switch flag.

    Returns:
        A wrapped Gymnasium environment.
    """
    env = gym.make(
        "Taxi-v3",
        render_mode=render_mode,
        is_rainy=is_rainy,
        fickle_passenger=fickle_passenger,
    )
    if passenger_loc is not None and dest_loc is not None:
        env = FixedRouteTaxiWrapper(
            env,
            passenger_loc=passenger_loc,
            dest_loc=dest_loc,
            initial_taxi_loc=initial_taxi_loc,
        )
    env = DecodedObservationWrapper(env, task_num=task_num)
    env = TaxiSafetyWrapper(env)
    return env


# =============================================================================
# Safety-dataset helpers
# =============================================================================

def get_safe_actions_for_state_index(state_index: int) -> list[int]:
    """Return the list of safe actions for a Taxi-v3 state index.

    Safety definition:
    - Movement actions (0..3) are always safe.
    - Pickup (4) is safe iff passenger is waiting at the taxi location.
    - Dropoff (5) is safe iff passenger is in the taxi and the taxi is at the
      current state's destination location.
    """
    taxi_row, taxi_col, pass_loc, dest_idx = decode_taxi_state(state_index)
    taxi_loc = (taxi_row, taxi_col)
    dest_loc = TAXI_LOCS[dest_idx]

    safe_actions = [0, 1, 2, 3]  # move actions are always safe

    pickup_safe = (pass_loc != 4) and (taxi_loc == TAXI_LOCS[pass_loc])
    dropoff_safe = (pass_loc == 4) and (taxi_loc == dest_loc)

    if pickup_safe:
        safe_actions.append(4)
    if dropoff_safe:
        safe_actions.append(5)

    return safe_actions


def _enumerate_candidate_states(
    passenger_loc: int | None = None,
    dest_loc: int | None = None,
) -> list[int]:
    """Enumerate task-relevant states under optional route constraints.

    Route constraints control initial states only, but during episodes the
    passenger may be dropped at any named location. Therefore, when constraints
    are provided, we still enumerate all passenger-location values (0..4) while
    restricting destination to `dest_loc`.
    """
    if passenger_loc is None or dest_loc is None:
        return list(range(NUM_TAXI_STATES))

    states: list[int] = []
    for row in range(NUM_ROWS):
        for col in range(NUM_COLS):
            for pass_loc in range(NUM_PASS_LOCS):
                states.append(encode_taxi_state(row, col, pass_loc, dest_loc))
    return states


def get_all_unsafe_state_action_pairs(
    task_num: int = 0,
    passenger_loc: int | None = None,
    dest_loc: int | None = None,
    state_repr: str = "observation",
) -> list[tuple]:
    """Return every ``(state, action)`` pair that is an illegal pickup/dropoff.

    For every reachable state in the task, actions 4 (Pickup) and 5 (Dropoff)
    are checked against the Taxi-v3 rules:

    * **Pickup** is illegal when the passenger is already in the taxi
      (``pass_loc == 4``) **or** the taxi is not at the passenger's location.
    * **Dropoff** is illegal when the passenger is **not** in the taxi
      (``pass_loc != 4``) **or** the taxi is not at the destination for that
      state.

    If *passenger_loc* and *dest_loc* are provided, only states reachable
    under that route constraint are enumerated.  Otherwise all 500 states are
    checked.

    Args:
        task_num: Unused placeholder kept for API compatibility.
        passenger_loc: Fixed initial passenger location (for filtering).
        dest_loc: Fixed destination location (for filtering).
        state_repr: ``'observation'`` → decoded-vector ndarray, ``'state_index'`` → int.

    Returns:
        List of ``(state, action)`` tuples.
    """
    assert state_repr in ("observation", "state_index")

    # Determine task-relevant candidate states.
    reachable = _enumerate_candidate_states(
        passenger_loc=passenger_loc, dest_loc=dest_loc,
    )

    unsafe_pairs: list[tuple] = []
    for s in reachable:
        safe_actions = set(get_safe_actions_for_state_index(s))
        unsafe_actions = [a for a in (4, 5) if a not in safe_actions]

        if state_repr == "observation":
            state_vec: Any = one_hot_encode_state(s, NUM_TAXI_STATES, task_num)
        else:
            state_vec = s

        for action in unsafe_actions:
            if state_repr == "observation":
                unsafe_pairs.append((state_vec.copy(), action))
            else:
                unsafe_pairs.append((state_vec, action))

    return unsafe_pairs


def create_taxi_safety_rashomon_dataset(
    task_num: int = 0,
    passenger_loc: int | None = None,
    dest_loc: int | None = None,
) -> TensorDataset:
    """Create a multi-label safety dataset for Taxi Rashomon certification.

    Safety labels follow Taxi legality:
    - actions 0..3 are always safe,
    - pickup (4) is safe only at passenger location when passenger not in taxi,
    - dropoff (5) is safe only when passenger is in taxi and taxi is at destination.

    Returns:
        TensorDataset(X, Y) where:
        - X: decoded observations, shape (N, 4)
        - Y: multi-hot safe-action vectors, shape (N, 6)
    """
    candidate_states = _enumerate_candidate_states(
        passenger_loc=passenger_loc, dest_loc=dest_loc,
    )

    obs_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    for s in candidate_states:
        safe_actions = get_safe_actions_for_state_index(s)
        if len(safe_actions) == NUM_ACTIONS:
            # Keep only safety-critical states.
            continue

        obs = one_hot_encode_state(s, NUM_TAXI_STATES, task_num)
        multi_hot = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for a in safe_actions:
            multi_hot[a] = 1.0

        obs_list.append(obs)
        label_list.append(multi_hot)

    if len(obs_list) == 0:
        raise RuntimeError("No safety-critical states found for Taxi dataset.")

    return TensorDataset(
        torch.tensor(np.asarray(obs_list), dtype=torch.float32),
        torch.tensor(np.asarray(label_list), dtype=torch.float32),
    )


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
        ``TensorDataset(states, safe_actions_multi_hot)`` where labels are
        multi-hot vectors of length ``NUM_ACTIONS``.
    """
    print("\nCreating 'Sufficient Safety Data' dataset...")
    all_actions = set(range(NUM_ACTIONS))

    unsafe_by_state: dict[tuple, set[int]] = {}
    for state, action in unsafe_state_action_pairs:
        key = tuple(state)
        unsafe_by_state.setdefault(key, set()).add(action)

    safe_by_state = {k: all_actions - v for k, v in unsafe_by_state.items()}

    states_t = torch.FloatTensor(list(safe_by_state.keys()))
    labels = []
    for safe_actions in safe_by_state.values():
        y = np.zeros(NUM_ACTIONS, dtype=np.float32)
        for action in safe_actions:
            y[action] = 1.0
        labels.append(y)
    actions_t = torch.FloatTensor(np.asarray(labels))

    ds = torch.utils.data.TensorDataset(states_t, actions_t)
    print(f"  Generated safe actions for {len(ds)} states with unsafe actions")
    return ds


def build_all_safety_datasets(
    env: gym.Env,
    actor: torch.nn.Module,
    task_num: int,
    training_data: dict,
    passenger_loc: int | None = None,
    dest_loc: int | None = None,
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
        passenger_loc=passenger_loc,
        dest_loc=dest_loc,
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
            the actions tensor is a multi-hot float tensor of shape
            ``(N, n_actions)`` with 1 for valid actions and 0 otherwise.
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
                target_dist = batch_actions / batch_actions.sum(dim=1, keepdim=True).clamp(min=1e-8)
                log_probs = F.log_softmax(logits, dim=1)
                loss = -(target_dist * log_probs).sum(dim=1).mean()

                predicted = logits.argmax(dim=1)
                epoch_correct += batch_actions[torch.arange(batch_actions.size(0)), predicted].sum().item()
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
        if valid_actions.ndim != 2:
            raise ValueError(
                "Multi-label dataset must provide 2D labels. "
                f"Got shape={tuple(valid_actions.shape)}."
            )

        # Multi-hot vectors (N, n_actions), 1 for safe action.
        idx = torch.arange(valid_actions.shape[0])
        accuracy = (valid_actions[idx, preds] > 0).float().mean().item()
        correct = valid_actions[idx, preds] > 0
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
                if torch.is_floating_point(valid_actions):
                    va = torch.where(valid_actions[i] > 0)[0].tolist()
                else:
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
