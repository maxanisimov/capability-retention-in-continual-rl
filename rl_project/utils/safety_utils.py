import gymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

def generate_sufficient_safe_state_action_dataset(
    unsafe_state_action_pairs: list[tuple], env: gymnasium.Env
) -> torch.utils.data.TensorDataset:
    """
    Generate a dataset of sufficient safe state-action pairs by computing the complement of unsafe actions for each state.
    This dataset can be used to compute a Rashomon set that enforces safety constraints without requiring an optimal policy demonstration.

    Args:
        unsafe_state_action_pairs: list of (state, action) tuples that are unsafe.
        env: The environment, needed to determine the action space for computing the complement set of safe actions.
    Returns:
        A torch dataset containing states and their corresponding multi-hot safe action masks
        (1 for valid actions, 0 for invalid actions).
    """
    all_actions = set(range(env.action_space.n)) # type: ignore
    # Group unsafe actions by state
    unsafe_actions_by_state: dict[tuple, set] = {}
    for state, action in unsafe_state_action_pairs:
        state_key = tuple(state)
        if state_key not in unsafe_actions_by_state:
            unsafe_actions_by_state[state_key] = set()
        unsafe_actions_by_state[state_key].add(action)
    # Compute complement (safe actions) for each state
    safe_actions_by_state = {
        state_key: all_actions - unsafe_actions
        for state_key, unsafe_actions in unsafe_actions_by_state.items()
    }
    sufficient_safe_states = torch.FloatTensor(list(safe_actions_by_state.keys()))

    n_actions = env.action_space.n # type: ignore
    multi_hot = torch.zeros(len(safe_actions_by_state), n_actions)
    for i, safe_actions in enumerate(safe_actions_by_state.values()):
        for a in safe_actions:
            multi_hot[i, a] = 1.0

    return torch.utils.data.TensorDataset(sufficient_safe_states, multi_hot)


def get_unique_safe_state_action_pairs(training_data: dict[str, np.ndarray | torch.Tensor]) -> torch.utils.data.TensorDataset:
    """
    Extract unique safe state-action pairs from the standard training data.

    Args:
        standard_training_data: A dictionary with keys 'states', 'actions', and 'safe'.
            'states' should be a numpy array or torch tensor of shape (N, state_dim).
            'actions' should be a numpy array or torch tensor of shape (N,).
            'safe' should be a numpy array or torch tensor of shape (N,), with 1.0 for safe and 0.0 for unsafe. 
    Returns:
        safe_training_data_torch_dataset: A torch.utils.data.TensorDataset containing unique safe state-action
     pairs.
    """

    states = torch.FloatTensor(training_data['states'])
    actions = torch.LongTensor(training_data['actions'])
    safety_flags = torch.FloatTensor(training_data['safe'])  # 1.0 for safe, 0.0 for unsafe
    # Filter only safe state-action pairs
    safe_indices = np.where(safety_flags == 1.0)[0]
    states = states[safe_indices]
    actions = actions[safe_indices]

    # Remove duplicate safe state-action pairs
    safe_states_df = pd.DataFrame(states.detach().numpy())
    safe_states_df.columns = [f'state_feature_{i}' for i in range(safe_states_df.shape[1])]
    actions_df = pd.DataFrame(actions.detach().numpy())
    actions_df.columns = ['action'] if actions_df.shape[1] == 1 else [f'action_{i}' for i in range(actions_df.shape[1])]
    action_cols = actions_df.columns.tolist()
    safe_state_action_pairs_df = pd.concat([safe_states_df, actions_df], axis=1)
    safe_state_action_pairs_df = safe_state_action_pairs_df.drop_duplicates(keep='first').reset_index(drop=True)
    # Convert back to torch dataset
    states = torch.FloatTensor(safe_state_action_pairs_df.drop(columns=action_cols).values)
    actions = torch.LongTensor(safe_state_action_pairs_df[action_cols].values)
    safe_training_data_torch_dataset = torch.utils.data.TensorDataset(states, actions)
    return safe_training_data_torch_dataset

def generate_safe_optimal_policy_data(
        env,
        safe_actor,
        num_episodes: int = 1,
        deterministic: bool = True,
        log_std: float | None = None,
    ) -> torch.utils.data.TensorDataset:
    """
    Generate safe optimal policy data by rolling out the safe_actor in the environment.

    Args:
        env: The Gymnasium environment to roll out in.
        safe_actor: The trained safe policy network (safety guaranteed when deterministic=True).
        num_episodes: Number of rollouts to collect data from.
        deterministic: Whether to use deterministic actions.
        log_std: Log standard deviation for continuous action spaces (if applicable).

    Returns:
        safe_optimal_policy_data_torch_dataset: A torch.utils.data.TensorDataset containing
         (states, actions) pairs from the safe optimal policy.
    """
    states = []
    actions = []

    continuouse_actions = isinstance(env.action_space, gymnasium.spaces.Box)

    # Collect multiple rollouts to get diverse state coverage
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            # Get action from safe_actor (the trained safe policy)
            with torch.no_grad():
                if continuouse_actions:
                    # Continuous actions: use mean from safe_actor
                    if deterministic:
                        action = safe_actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = safe_actor(obs_t)
                        std = np.exp(log_std).expand_as(mean) # type: ignore
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    # Clip to action space bounds
                    action = np.clip(action, env.action_space.low, env.action_space.high)
                else:
                    # Discrete actions: use argmax
                    if deterministic:
                        logits = safe_actor(obs_t)
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        logits = safe_actor(obs_t)
                        dist = torch.distributions.Categorical(logits=logits)
                        action = dist.sample().item()
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action) # type: ignore
            done = terminated or truncated
            
            # Record state-action pair (if it is safe)
            safety_flag = info.get('is_safe', 1.0)  # Default to safe if not provided
            if safety_flag == 1.0:
                states.append(obs)
                actions.append(action)

            obs = next_obs

    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)

    # Remove duplicate safe state-action pairs
    safe_states_df = pd.DataFrame(states.detach().numpy())
    safe_states_df.columns = [f'state_feature_{i}' for i in range(safe_states_df.shape[1])]
    actions_df = pd.DataFrame(actions.detach().numpy())
    actions_df.columns = ['action'] if actions_df.shape[1] == 1 else [f'action_{i}' for i in range(actions_df.shape[1])]
    action_cols = actions_df.columns.tolist()
    safe_state_action_pairs_df = pd.concat([safe_states_df, actions_df], axis=1)
    safe_state_action_pairs_df = safe_state_action_pairs_df.drop_duplicates(keep='first').reset_index(drop=True)
    # Convert back to torch dataset
    states = torch.FloatTensor(safe_state_action_pairs_df.drop(columns=action_cols).values)
    actions = torch.LongTensor(safe_state_action_pairs_df[action_cols].values)
    safe_state_action_torch_dataset = torch.utils.data.TensorDataset(states, actions)
    return safe_state_action_torch_dataset


# ═══════════════════════════════════════════════════════════════════════════════
# Safety Critic (SQRL-style)
# Learns Q_safe(s,a) = discounted probability of future failure
# Reference: Srinivasan et al., "Learning to be Safe: Deep RL with a Safety
#            Critic", arXiv:2010.14603
#
# Bellman backup:
#   Q_safe(s,a) = I(s) + (1 - I(s)) * gamma_safe * E_{s',a'}[Q_safe(s',a')]
#
# where I(s) = 1 if state is unsafe, 0 if safe.
# Actions are REJECTED when Q_safe(s,a) >= epsilon_safe.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class SafetyCriticConfig:
    """Configuration for safety critic training (SQRL-style)."""
    hidden_dim: int = 256
    n_hidden: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    n_iterations: int = 200_000      # number of fitted Q-iteration gradient steps
    tau: float = 0.005               # target network soft update rate
    gamma_safe: float = 0.9          # safety discount factor
    epsilon_safe: float = 0.15       # safety threshold: reject actions with Q_safe >= epsilon
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    log_interval: int = 10_000       # print loss every N steps
    # Default safe action (used as fallback when no action passes threshold)
    default_safe_action: Optional[int] = None
    # ── Supervised fast-path (used when gamma_safe == 0) ──
    epochs: int = 100                # max training epochs (supervised path)
    patience: int = 10               # early stopping patience (supervised path)
    val_fraction: float = 0.15       # validation fraction (supervised path)


class SafetyCriticNetwork(nn.Module):
    """
    Safety Q-network: maps state -> Q_safe(s, a) for each discrete action.

    Q_safe(s, a) estimates the discounted cumulative probability that the agent
    will reach an unsafe state in the future, starting from (s, a).

    Higher values = more dangerous.
    """

    def __init__(self, obs_dim: int, n_actions: int, hidden_dim: int = 256,
                 n_hidden: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        """Returns Q_safe values of shape (batch, n_actions)."""
        return self.net(s)


class ContinuousSafetyCriticNetwork(nn.Module):
    """
    Safety Q-network for continuous action spaces.

    Maps (state, action) -> scalar Q_safe(s, a).
    Input is the concatenation of state and action vectors.

    Q_safe(s, a) estimates the discounted cumulative probability that the agent
    will reach an unsafe state in the future, starting from (s, a).
    Higher values = more dangerous.
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256,
                 n_hidden: int = 2):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        layers: list[nn.Module] = []
        in_dim = obs_dim + action_dim
        for _ in range(n_hidden):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))  # single scalar output
        self.net = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Args:
            s: (batch, obs_dim) state tensor
            a: (batch, action_dim) action tensor
        Returns:
            Q_safe logit of shape (batch,)
        """
        sa = torch.cat([s, a], dim=-1)
        return self.net(sa).squeeze(-1)


class SafetyReplayBuffer:
    """
    Replay buffer storing (state, action, cost, next_state, done) for safety
    critic training.  Cost = 1 - safe_flag (1 = unsafe, 0 = safe).
    """

    def __init__(self, capacity: int, obs_dim: int, device: str = 'cpu'):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.costs = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)

    def add(self, state, action, cost, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.costs[self.ptr] = cost
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idxs], dtype=torch.int64, device=self.device),
            torch.tensor(self.costs[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device),
        )


def _build_safety_replay_buffer(
    training_data: dict[str, np.ndarray],
    actor: nn.Module,
    device: str = 'cpu',
) -> SafetyReplayBuffer:
    """
    Build a safety replay buffer from offline training data.

    Reconstructs (s, a, cost, s', done) transitions from the sequential
    training data.  cost = 1 - safe_flag  (unsafe=1, safe=0).

    For the next action a' needed in the Bellman target, we use the provided
    actor policy (the pre-trained safe policy).

    Args:
        training_data: dict with 'states', 'actions', 'safe', 'terminated', 'truncated'
        actor: the policy network used for next-action selection
        device: torch device string

    Returns:
        Filled SafetyReplayBuffer
    """
    states = np.array(training_data['states'], dtype=np.float32)
    actions = np.array(training_data['actions'], dtype=np.int64)
    safe_flags = np.array(training_data['safe'], dtype=np.float32)
    terminated = np.array(training_data['terminated'], dtype=np.float32)
    truncated = np.array(training_data['truncated'], dtype=np.float32)

    N, obs_dim = states.shape
    costs = 1.0 - safe_flags  # unsafe indicator: I(s)

    dones = np.logical_or(terminated > 0.5, truncated > 0.5).astype(np.float32)

    buf = SafetyReplayBuffer(capacity=N, obs_dim=obs_dim, device=device)

    for t in range(N - 1):
        if dones[t]:
            # Last step of episode: next_state is a reset state (states[t+1]
            # belongs to a new episode). For terminal transitions the Bellman
            # target is just cost(s), so next_state is unused — store zeros.
            buf.add(states[t], actions[t], costs[t], np.zeros(obs_dim, dtype=np.float32), 1.0)
        else:
            buf.add(states[t], actions[t], costs[t], states[t + 1], 0.0)

    # Handle last transition in the dataset (always terminal by convention)
    buf.add(states[-1], actions[-1], costs[-1], np.zeros(obs_dim, dtype=np.float32), 1.0)

    return buf


def _train_supervised_safety_critic(
    training_data: dict[str, np.ndarray],
    n_actions: int,
    obs_dim: int,
    cfg: SafetyCriticConfig,
    verbose: bool = True,
) -> tuple["SafetyCritic", dict]:
    """
    Fast-path: train safety critic via supervised learning (gamma_safe == 0).

    When gamma_safe = 0 the Bellman target is just the immediate cost c(s,a),
    so we can train directly with BCE on binary labels.  This is more sample-
    efficient than fitted Q-iteration because we use structured epochs over the
    full dataset, BCE loss (better calibrated for binary targets), and proper
    train/val split with early stopping.

    The network still outputs Q_safe(s, a) for all actions (same architecture).
    For each observed transition (s, a, cost), we supervise the a-th output.

    Args:
        training_data: dict with 'states', 'actions', 'safe'
        n_actions: number of discrete actions
        obs_dim: observation dimension
        cfg: training configuration
        verbose: print progress

    Returns:
        safety_critic, info
    """
    import time

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # ── Prepare data ──
    states = torch.tensor(
        np.array(training_data['states'], dtype=np.float32), device=device
    )
    actions = torch.tensor(
        np.array(training_data['actions'], dtype=np.int64), device=device
    )
    costs = 1.0 - torch.tensor(
        np.array(training_data['safe'], dtype=np.float32), device=device
    )  # 1 = unsafe, 0 = safe

    N = states.shape[0]
    n_unsafe = int(costs.sum().item())

    if verbose:
        print(f"Training supervised safety critic (gamma_safe=0, one-step):")
        print(f"  {N} transitions, {n_unsafe} unsafe ({n_unsafe/N:.2%})")
        print(f"  epochs={cfg.epochs}, patience={cfg.patience}, batch_size={cfg.batch_size}")

    # ── Train / Val split ──
    perm = torch.randperm(N, device=device)
    n_val = max(1, int(N * cfg.val_fraction))
    n_train = N - n_val

    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, a_train, y_train = states[train_idx], actions[train_idx], costs[train_idx]
    X_val, a_val, y_val = states[val_idx], actions[val_idx], costs[val_idx]

    if verbose:
        print(f"  Split: train={n_train}, val={n_val}")

    # ── Handle class imbalance ──
    n_pos = y_train.sum().clamp(min=1)  # unsafe count
    n_neg = (n_train - n_pos).clamp(min=1)
    pos_weight = (n_neg / n_pos).clamp(max=10.0)

    # ── Build model ──
    model = SafetyCriticNetwork(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_dim=cfg.hidden_dim, n_hidden=cfg.n_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # BCE on the taken action's logit
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(0))

    # ── Training loop ──
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_time = time.time()

    train_dataset = torch.utils.data.TensorDataset(X_train, a_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    )

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_s, batch_a, batch_y in train_loader:
            logits_all = model(batch_s)                        # (B, n_actions)
            logits_a = logits_all.gather(1, batch_a.unsqueeze(1)).squeeze(1)  # (B,)
            loss = criterion(logits_a, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val).gather(1, a_val.unsqueeze(1)).squeeze(1)
            val_loss = criterion(val_logits, y_val).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{cfg.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # ── Final statistics ──
    if verbose:
        with torch.no_grad():
            # Compute Q_safe via sigmoid (probability of being unsafe)
            all_q = torch.sigmoid(model(states)).cpu()
            safe_mask = all_q < cfg.epsilon_safe
            n_safe_per_state = safe_mask.sum(dim=1).float()
            print(f"\nSupervised safety critic trained.")
            print(f"  Avg safe actions per state: {n_safe_per_state.mean():.1f} / {n_actions}")
            print(f"  Min safe actions: {n_safe_per_state.min():.0f}")
            print(f"  States with >=1 safe action: "
                  f"{(n_safe_per_state > 0).float().mean():.2%}")

    # ── Wrap ──
    safety_critic = SafetyCritic(
        model=model,
        n_actions=n_actions,
        obs_dim=obs_dim,
        gamma_safe=0.0,
        epsilon_safe=cfg.epsilon_safe,
        default_safe_action=cfg.default_safe_action,
        device=device,
        use_sigmoid=True,  # supervised path outputs logits → need sigmoid
    )

    info = {
        'losses': train_losses,
        'val_losses': val_losses,
        'n_transitions': N,
        'gamma_safe': 0.0,
        'epsilon_safe': cfg.epsilon_safe,
        'best_epoch': len(train_losses) - patience_counter,
        'training_mode': 'supervised',
    }

    return safety_critic, info


def _train_supervised_continuous_safety_critic(
    training_data: dict[str, np.ndarray],
    action_dim: int,
    obs_dim: int,
    cfg: SafetyCriticConfig,
    verbose: bool = True,
) -> tuple["ContinuousSafetyCritic", dict]:
    """
    Supervised training of a continuous-action safety critic (gamma_safe == 0).

    When gamma_safe = 0 the target is just the immediate cost c(s,a), so we
    train with BCE on binary labels.  The network takes (s, a) and outputs
    a single logit for P(unsafe | s, a).

    Args:
        training_data: dict with 'states', 'actions', 'safe'
        action_dim: continuous action dimensionality
        obs_dim: observation dimension
        cfg: training configuration
        verbose: print progress

    Returns:
        safety_critic: ContinuousSafetyCritic, info dict
    """
    import time

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # ── Prepare data ──
    states = torch.tensor(
        np.array(training_data['states'], dtype=np.float32), device=device
    )
    actions = torch.tensor(
        np.array(training_data['actions'], dtype=np.float32), device=device
    )
    costs = 1.0 - torch.tensor(
        np.array(training_data['safe'], dtype=np.float32), device=device
    )  # 1 = unsafe, 0 = safe

    N = states.shape[0]
    n_unsafe = int(costs.sum().item())

    if verbose:
        print(f"Training supervised continuous safety critic (gamma_safe=0, one-step):")
        print(f"  {N} transitions, {n_unsafe} unsafe ({n_unsafe/N:.2%})")
        print(f"  obs_dim={obs_dim}, action_dim={action_dim}")
        print(f"  epochs={cfg.epochs}, patience={cfg.patience}, batch_size={cfg.batch_size}")

    # ── Train / Val split ──
    perm = torch.randperm(N, device=device)
    n_val = max(1, int(N * cfg.val_fraction))
    n_train = N - n_val

    train_idx, val_idx = perm[:n_train], perm[n_train:]
    X_train, a_train, y_train = states[train_idx], actions[train_idx], costs[train_idx]
    X_val, a_val, y_val = states[val_idx], actions[val_idx], costs[val_idx]

    if verbose:
        print(f"  Split: train={n_train}, val={n_val}")

    # ── Handle class imbalance ──
    n_pos = y_train.sum().clamp(min=1)  # unsafe count
    n_neg = (n_train - n_pos).clamp(min=1)
    pos_weight = (n_neg / n_pos).clamp(max=10.0)

    # ── Build model ──
    model = ContinuousSafetyCriticNetwork(
        obs_dim=obs_dim, action_dim=action_dim,
        hidden_dim=cfg.hidden_dim, n_hidden=cfg.n_hidden,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    # BCE on the scalar logit
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.unsqueeze(0))

    # ── Training loop ──
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    train_losses: list[float] = []
    val_losses: list[float] = []
    start_time = time.time()

    train_dataset = torch.utils.data.TensorDataset(X_train, a_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=False,
    )

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for batch_s, batch_a, batch_y in train_loader:
            logits = model(batch_s, batch_a)          # (B,)
            loss = criterion(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        train_loss = epoch_loss / max(n_batches, 1)

        # ── Validation ──
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val, a_val)
            val_loss = criterion(val_logits, y_val).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1}/{cfg.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | {elapsed:.1f}s")

        if patience_counter >= cfg.patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # ── Final statistics ──
    if verbose:
        with torch.no_grad():
            all_q = torch.sigmoid(model(states, actions)).cpu()
            safe_frac = (all_q < cfg.epsilon_safe).float().mean().item()
            print(f"\nSupervised continuous safety critic trained.")
            print(f"  Fraction of (s,a) pairs predicted safe: {safe_frac:.2%}")
            print(f"  Mean Q_safe: {all_q.mean():.4f}")
            print(f"  Max Q_safe:  {all_q.max():.4f}")

    # ── Wrap ──
    safety_critic = ContinuousSafetyCritic(
        model=model,
        action_dim=action_dim,
        obs_dim=obs_dim,
        gamma_safe=0.0,
        epsilon_safe=cfg.epsilon_safe,
        device=device,
        use_sigmoid=True,
    )

    info = {
        'losses': train_losses,
        'val_losses': val_losses,
        'n_transitions': N,
        'gamma_safe': 0.0,
        'epsilon_safe': cfg.epsilon_safe,
        'best_epoch': len(train_losses) - patience_counter,
        'training_mode': 'supervised_continuous',
    }

    return safety_critic, info


def train_safety_critic(
    training_data: dict[str, np.ndarray],
    n_actions: int,
    obs_dim: int,
    actor: nn.Module | None = None,
    cfg: SafetyCriticConfig | None = None,
    verbose: bool = True,
    continuous: bool = False,
) -> tuple["SafetyCritic | ContinuousSafetyCritic", dict]:
    """
    Train a safety critic from offline data.

    Dispatches between training modes:
      - continuous=True, gamma_safe == 0: supervised learning for continuous actions
      - continuous=False, gamma_safe == 0: supervised learning for discrete actions
      - continuous=False, gamma_safe > 0:  fitted Q-iteration (SQRL-style)

    When continuous=True, ``n_actions`` is interpreted as ``action_dim``
    (the dimensionality of the continuous action vector).

    The safety Q-function satisfies:
        Q_safe(s,a) = c(s,a) + (1 - c(s,a)) * gamma_safe * E_{a'~pi}[Q_safe(s', a')]

    where c(s,a) = 1 if the transition (s,a) leads to an unsafe next state.

    Args:
        training_data: dict with keys 'states', 'actions', 'safe', 'terminated', 'truncated'
        n_actions: number of discrete actions, or action dimensionality if continuous
        obs_dim: observation dimension
        actor: the pre-trained policy (required when gamma_safe > 0, unused for gamma_safe == 0)
        cfg: training configuration (uses defaults if None)
        verbose: print training progress
        continuous: if True, use continuous-action safety critic

    Returns:
        safety_critic: trained SafetyCritic or ContinuousSafetyCritic object
        info: dict with training metrics
    """
    import copy
    import time

    if cfg is None:
        cfg = SafetyCriticConfig()

    # ── Dispatch: continuous actions ──
    if continuous:
        if cfg.gamma_safe == 0.0:
            return _train_supervised_continuous_safety_critic(
                training_data, action_dim=n_actions, obs_dim=obs_dim, cfg=cfg, verbose=verbose,
            )
        else:
            raise NotImplementedError(
                "Bellman-based training for continuous actions is not yet implemented. "
                "Use gamma_safe=0.0 for the supervised path."
            )

    # ── Dispatch: supervised fast-path when gamma_safe == 0 ──
    if cfg.gamma_safe == 0.0:
        return _train_supervised_safety_critic(
            training_data, n_actions, obs_dim, cfg, verbose
        )

    if actor is None:
        raise ValueError("actor is required when gamma_safe > 0 (Bellman backups need a policy)")

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device)

    if verbose:
        print(f"Training SQRL safety critic (Bellman):")
        print(f"  gamma_safe={cfg.gamma_safe}, epsilon_safe={cfg.epsilon_safe}")
        print(f"  {cfg.n_iterations} gradient steps, batch_size={cfg.batch_size}")

    # ── Build replay buffer from offline data ──
    buf = _build_safety_replay_buffer(training_data, actor, device=str(device))
    if verbose:
        n_unsafe = (1.0 - np.array(training_data['safe'])).sum()
        print(f"  Replay buffer: {buf.size} transitions, {int(n_unsafe)} unsafe steps "
              f"({n_unsafe/buf.size:.2%})")

    # ── Build networks ──
    qf_safe = SafetyCriticNetwork(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_dim=cfg.hidden_dim, n_hidden=cfg.n_hidden,
    ).to(device)
    qf_safe_target = copy.deepcopy(qf_safe)

    actor = actor.to(device)  # type: ignore
    actor.eval()

    optimizer = torch.optim.Adam(qf_safe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # ── Fitted Q-Iteration ──
    losses = []
    start_time = time.time()

    for step in range(1, cfg.n_iterations + 1):
        states, actions, costs, next_states, dones = buf.sample(cfg.batch_size)

        # Current Q_safe values for the taken actions
        q_safe_all = qf_safe(states)                      # (B, n_actions)
        q_safe_a = q_safe_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # (B,)

        # Compute Bellman target
        with torch.no_grad():
            # Next actions from the policy
            next_logits = actor(next_states)
            next_probs = F.softmax(next_logits, dim=-1)    # (B, n_actions)

            # Target Q_safe values at next state
            q_safe_next = qf_safe_target(next_states)      # (B, n_actions)

            # Expected Q_safe under policy: sum_a' pi(a'|s') * Q_safe(s', a')
            expected_q_next = (next_probs * q_safe_next).sum(dim=-1)  # (B,)

            # Bellman target: I(s) + (1 - I(s)) * gamma_safe * E[Q_safe(s',a')]
            # For terminal states (done=1), the future term is 0
            target = costs + (1.0 - costs) * (1.0 - dones) * cfg.gamma_safe * expected_q_next

            # Clamp targets to [0, 1] — it's a probability
            target = target.clamp(0.0, 1.0)

        loss = F.mse_loss(q_safe_a, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Soft update target network
        with torch.no_grad():
            for p, p_tgt in zip(qf_safe.parameters(), qf_safe_target.parameters()):
                p_tgt.data.copy_(cfg.tau * p.data + (1.0 - cfg.tau) * p_tgt.data)

        losses.append(loss.item())

        if verbose and step % cfg.log_interval == 0:
            recent_loss = np.mean(losses[-cfg.log_interval:])
            elapsed = time.time() - start_time

            # Quick stats on Q_safe values
            with torch.no_grad():
                sample_states = torch.tensor(
                    training_data['states'][:1000], dtype=torch.float32, device=device
                )
                sample_q = qf_safe(sample_states)
                q_mean = sample_q.mean().item()
                q_max = sample_q.max().item()
                # Fraction of (state, action) pairs that would be rejected
                rejected_frac = (sample_q >= cfg.epsilon_safe).float().mean().item()

            print(f"  Step {step}/{cfg.n_iterations} | loss={recent_loss:.6f} | "
                  f"Q_safe mean={q_mean:.4f} max={q_max:.4f} | "
                  f"rejected={rejected_frac:.2%} | {elapsed:.1f}s")

    qf_safe.eval()

    # ── Final statistics ──
    if verbose:
        with torch.no_grad():
            all_states = torch.tensor(
                training_data['states'], dtype=torch.float32, device=device
            )
            # Process in chunks to avoid OOM
            chunk_size = 10_000
            all_q_vals = []
            for i in range(0, len(all_states), chunk_size):
                chunk = all_states[i:i+chunk_size]
                all_q_vals.append(qf_safe(chunk).cpu())
            all_q = torch.cat(all_q_vals, dim=0)

            safe_mask = all_q < cfg.epsilon_safe
            n_safe_per_state = safe_mask.sum(dim=1).float()
            print(f"\nSafety critic trained.")
            print(f"  Avg safe actions per state: {n_safe_per_state.mean():.1f} / {n_actions}")
            print(f"  Min safe actions: {n_safe_per_state.min():.0f}")
            print(f"  States with >=1 safe action: "
                  f"{(n_safe_per_state > 0).float().mean():.2%}")

    # ── Wrap ──
    safety_critic = SafetyCritic(
        model=qf_safe,
        n_actions=n_actions,
        obs_dim=obs_dim,
        gamma_safe=cfg.gamma_safe,
        epsilon_safe=cfg.epsilon_safe,
        default_safe_action=cfg.default_safe_action,
        device=device,
    )

    info = {
        'losses': losses,
        'n_transitions': buf.size,
        'gamma_safe': cfg.gamma_safe,
        'epsilon_safe': cfg.epsilon_safe,
        'training_mode': 'bellman',
    }

    return safety_critic, info


class SafetyCritic:
    """
    SQRL-style safety critic wrapper.

    Q_safe(s, a) estimates discounted probability of future failure.
    Higher values = more dangerous.
    Actions are REJECTED when Q_safe(s, a) >= epsilon_safe.

    Key methods:
        predict_q_safe(states) -> (batch, n_actions) failure probabilities
        get_safe_actions(state) -> list of safe action indices
        get_safe_action_mask(states) -> (batch, n_actions) boolean mask
        filter_policy(actor, state) -> safest high-reward action
    """

    def __init__(
        self,
        model: SafetyCriticNetwork,
        n_actions: int,
        obs_dim: int,
        gamma_safe: float = 0.9,
        epsilon_safe: float = 0.15,
        default_safe_action: Optional[int] = None,
        device: torch.device | str = 'cpu',
        use_sigmoid: bool = False,
    ):
        self.model = model
        self.n_actions = n_actions
        self.obs_dim = obs_dim
        self.gamma_safe = gamma_safe
        self.epsilon_safe = epsilon_safe
        self.default_safe_action = default_safe_action
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_sigmoid = use_sigmoid  # True for supervised path (logits → probs)
        self.model.eval()

    def predict_q_safe(self, states: torch.Tensor | np.ndarray) -> torch.Tensor:
        """
        Predict Q_safe(s, a) for all actions.

        Args:
            states: (batch, obs_dim) or (obs_dim,) state tensor/array
        Returns:
            q_safe: (batch, n_actions) — estimated failure probabilities
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        if states.dim() == 1:
            states = states.unsqueeze(0)
        states = states.to(self.device)
        with torch.no_grad():
            q = self.model(states)
            if self.use_sigmoid:
                q = torch.sigmoid(q)
            return q

    def get_safe_action_mask(
        self, states: torch.Tensor | np.ndarray, epsilon: float | None = None
    ) -> torch.Tensor:
        """
        Returns boolean mask: True where Q_safe(s, a) < epsilon (action is safe).

        Args:
            states: (batch, obs_dim) states
            epsilon: safety threshold; defaults to self.epsilon_safe
        Returns:
            mask: (batch, n_actions) boolean tensor — True = safe
        """
        if epsilon is None:
            epsilon = self.epsilon_safe
        q_safe = self.predict_q_safe(states)
        return q_safe < epsilon

    def get_safe_actions(
        self, state: torch.Tensor | np.ndarray, epsilon: float | None = None
    ) -> list[int]:
        """
        Returns list of safe action indices for a single state.

        Args:
            state: (obs_dim,) single state
            epsilon: safety threshold; defaults to self.epsilon_safe
        Returns:
            List of action indices where Q_safe < epsilon.
            Falls back to [default_safe_action] if no action is safe.
        """
        mask = self.get_safe_action_mask(state, epsilon).squeeze(0)
        safe = torch.where(mask)[0].tolist()
        if len(safe) == 0 and self.default_safe_action is not None:
            safe = [self.default_safe_action]
        return safe

    def filter_policy(
        self,
        actor: nn.Module,
        state: torch.Tensor | np.ndarray,
        epsilon: float | None = None,
        deterministic: bool = True,
    ) -> int:
        """
        Select an action from the actor policy, restricted to safe actions.

        Masks actions with Q_safe >= epsilon to -inf logits, then selects via
        argmax (deterministic) or categorical sampling (stochastic).
        If no action is safe, picks the action with LOWEST Q_safe (safest).

        Args:
            actor: policy network (state -> logits)
            state: (obs_dim,) single state
            epsilon: safety threshold
            deterministic: argmax vs categorical sampling
        Returns:
            action: int
        """
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(self.device)

        safe_mask = self.get_safe_action_mask(state, epsilon).squeeze(0)  # (n_actions,)

        with torch.no_grad():
            logits = actor(state).squeeze(0)  # (n_actions,)

        # If no action is safe, fall back to the least dangerous action
        if not safe_mask.any():
            q_safe = self.predict_q_safe(state).squeeze(0)
            return int(q_safe.argmin().item())

        # Mask unsafe actions
        masked_logits = logits.clone()
        masked_logits[~safe_mask] = float('-inf')

        if deterministic:
            return int(masked_logits.argmax().item())
        else:
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            return int(dist.sample().item())

    def evaluate_with_safety_filter(
        self,
        env: gymnasium.Env,
        actor: nn.Module,
        episodes: int = 100,
        epsilon: float | None = None,
        deterministic: bool = True,
        seed: int = 2025,
    ) -> tuple[float, float, float, float]:
        """
        Evaluate actor with the safety filter applied.

        Returns:
            avg_reward, std_reward, failure_rate, avg_safe_action_fraction
        """
        scores = []
        failures = 0
        safe_fractions = []

        for ep in range(episodes):
            obs, _ = env.reset(seed=seed * ep)
            done = False
            ep_reward = 0.0
            total_steps = 0
            safe_action_count = 0

            while not done:
                n_safe = len(self.get_safe_actions(obs, epsilon))
                safe_action_count += n_safe
                total_steps += 1

                action = self.filter_policy(actor, obs, epsilon, deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                done = terminated or truncated

                is_safe = info.get('safe', None)
                if is_safe is None:
                    cost = info.get('cost', 0)
                    is_safe = (cost == 0)
                if not is_safe:
                    failures += 1

            scores.append(ep_reward)
            if total_steps > 0:
                safe_fractions.append(safe_action_count / (total_steps * self.n_actions))

        avg_r = float(np.mean(scores))
        std_r = float(np.std(scores))
        failure_rate = failures / episodes
        avg_safe_frac = float(np.mean(safe_fractions)) if safe_fractions else 0.0

        return avg_r, std_r, failure_rate, avg_safe_frac

    def summary(self, states: torch.Tensor | np.ndarray, epsilon: float | None = None) -> dict:
        """
        Compute summary statistics of the safety critic over a set of states.

        Returns dict with:
            - avg_safe_actions: mean number of safe actions per state
            - min_safe_actions: min safe actions across states
            - max_safe_actions: max safe actions across states
            - frac_states_with_safe_action: fraction of states with >= 1 safe action
            - avg_q_safe: mean Q_safe value across all (state, action) pairs
            - max_q_safe: max Q_safe value
            - per_action_safe_rate: (n_actions,) fraction of states where each action is safe
        """
        if epsilon is None:
            epsilon = self.epsilon_safe
        q_safe = self.predict_q_safe(states)
        mask = q_safe < epsilon  # True = safe

        n_safe = mask.sum(dim=1).float()

        return {
            'avg_safe_actions': n_safe.mean().item(),
            'min_safe_actions': n_safe.min().item(),
            'max_safe_actions': n_safe.max().item(),
            'frac_states_with_safe_action': (n_safe > 0).float().mean().item(),
            'avg_q_safe': q_safe.mean().item(),
            'max_q_safe': q_safe.max().item(),
            'per_action_safe_rate': mask.float().mean(dim=0).cpu().numpy(),
        }

    def state_dict(self) -> dict:
        """Serialize for saving."""
        return {
            'model_state_dict': self.model.state_dict(),
            'n_actions': self.n_actions,
            'obs_dim': self.obs_dim,
            'gamma_safe': self.gamma_safe,
            'epsilon_safe': self.epsilon_safe,
            'default_safe_action': self.default_safe_action,
            'use_sigmoid': self.use_sigmoid,
        }

    @classmethod
    def load(cls, checkpoint: dict, device: str = 'cpu') -> "SafetyCritic":
        """Load from checkpoint dict."""
        model = SafetyCriticNetwork(
            obs_dim=checkpoint['obs_dim'],
            n_actions=checkpoint['n_actions'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return cls(
            model=model,
            n_actions=checkpoint['n_actions'],
            obs_dim=checkpoint['obs_dim'],
            gamma_safe=checkpoint['gamma_safe'],
            epsilon_safe=checkpoint['epsilon_safe'],
            default_safe_action=checkpoint['default_safe_action'],
            device=device,
            use_sigmoid=checkpoint.get('use_sigmoid', False),
        )


class ContinuousSafetyCritic:
    """
    Safety critic for continuous action spaces.

    Q_safe(s, a) estimates the probability that (s, a) leads to an unsafe
    outcome.  Higher values = more dangerous.
    An action is rejected when Q_safe(s, a) >= epsilon_safe.

    Key methods:
        predict_q_safe(states, actions) -> (batch,) failure probabilities
        is_safe(state, action) -> bool
    """

    def __init__(
        self,
        model: ContinuousSafetyCriticNetwork,
        action_dim: int,
        obs_dim: int,
        gamma_safe: float = 0.0,
        epsilon_safe: float = 0.15,
        device: torch.device | str = 'cpu',
        use_sigmoid: bool = True,
    ):
        self.model = model
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.gamma_safe = gamma_safe
        self.epsilon_safe = epsilon_safe
        self.device = torch.device(device) if isinstance(device, str) else device
        self.use_sigmoid = use_sigmoid
        self.model.eval()

    def predict_q_safe(
        self,
        states: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
    ) -> torch.Tensor:
        """
        Predict Q_safe(s, a) for continuous (state, action) pairs.

        Args:
            states: (batch, obs_dim) or (obs_dim,) state tensor/array
            actions: (batch, action_dim) or (action_dim,) action tensor/array
        Returns:
            q_safe: (batch,) — estimated failure probabilities
        """
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).float()
        if states.dim() == 1:
            states = states.unsqueeze(0)
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)
        states = states.to(self.device)
        actions = actions.to(self.device)
        with torch.no_grad():
            q = self.model(states, actions)
            if self.use_sigmoid:
                q = torch.sigmoid(q)
            return q

    def is_safe(
        self,
        state: torch.Tensor | np.ndarray,
        action: torch.Tensor | np.ndarray,
        epsilon: float | None = None,
    ) -> bool:
        """Check if a single (state, action) pair is safe."""
        if epsilon is None:
            epsilon = self.epsilon_safe
        q = self.predict_q_safe(state, action).item()
        return q < epsilon

    def summary(
        self,
        states: torch.Tensor | np.ndarray,
        actions: torch.Tensor | np.ndarray,
        epsilon: float | None = None,
    ) -> dict:
        """
        Compute summary statistics over a set of (state, action) pairs.
        """
        if epsilon is None:
            epsilon = self.epsilon_safe
        q_safe = self.predict_q_safe(states, actions)
        safe_mask = q_safe < epsilon

        return {
            'frac_safe': safe_mask.float().mean().item(),
            'avg_q_safe': q_safe.mean().item(),
            'max_q_safe': q_safe.max().item(),
            'min_q_safe': q_safe.min().item(),
        }

    def state_dict(self) -> dict:
        """Serialize for saving."""
        return {
            'model_state_dict': self.model.state_dict(),
            'action_dim': self.action_dim,
            'obs_dim': self.obs_dim,
            'gamma_safe': self.gamma_safe,
            'epsilon_safe': self.epsilon_safe,
            'use_sigmoid': self.use_sigmoid,
        }

    @classmethod
    def load(cls, checkpoint: dict, device: str = 'cpu') -> "ContinuousSafetyCritic":
        """Load from checkpoint dict."""
        model = ContinuousSafetyCriticNetwork(
            obs_dim=checkpoint['obs_dim'],
            action_dim=checkpoint['action_dim'],
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return cls(
            model=model,
            action_dim=checkpoint['action_dim'],
            obs_dim=checkpoint['obs_dim'],
            gamma_safe=checkpoint['gamma_safe'],
            epsilon_safe=checkpoint['epsilon_safe'],
            device=device,
            use_sigmoid=checkpoint.get('use_sigmoid', True),
        )