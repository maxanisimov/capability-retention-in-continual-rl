import torch
from dataclasses import dataclass
import random
import numpy as np
import gymnasium as gym
import time
import torch.nn.functional as F
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
from collections import deque
from collections.abc import Callable

#### --- Discrete SAC Utilities --- ####
@dataclass
class DiscreteSACConfig:
    seed: int = 42
    total_timesteps: int = 100_000
    eval_episodes: int = 1_000
    buffer_size: int = 1_000_000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005  # Target network soft update rate
    lr: float = 3e-4
    alpha: float = 0.2  # Initial entropy coefficient
    autotune_alpha: bool = True  # Whether to automatically tune entropy coefficient
    policy_frequency: int = 2  # How often to update policy (and alpha)
    target_network_frequency: int = 1  # How often to update target networks
    learning_starts: int = 5000  # How many steps before learning starts
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_envs: int = 1            # parallel environments (>1 requires a Callable env factory)
    compile_model: bool = False  # torch.compile the actor/critic forward passes (requires PyTorch >= 2.0)

class DiscreteReplayBuffer:
    """
    Replay buffer for discrete action spaces.
    Stores transitions (state, action, reward, next_state, done).
    """
    def __init__(self, obs_dim: int, capacity: int, device: str):
        self.capacity = capacity
        self.device = device
        self.size = 0
        self.ptr = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)  # Discrete actions
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add a transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        idxs = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.tensor(self.states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idxs], dtype=torch.int64, device=self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[idxs], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32, device=self.device),
        )

def make_actor_critic(
    obs_dim: int, n_actions: int,
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
):
    """
    Create actor and twin critic networks for Discrete SAC.

    Actor outputs logits for each discrete action.
    Critics output Q-values for ALL actions given a state.

    Args:
        obs_dim: Observation space dimension
        n_actions: Number of discrete actions
        actor_warm_start: Optional pre-trained actor network
        critic_warm_start: Optional pre-trained critic network (will be copied to both Q1 and Q2)

    Returns:
        actor: Actor network outputting n_actions logits
        qf1: First Q-network outputting n_actions Q-values
        qf2: Second Q-network outputting n_actions Q-values
    """
    if actor_warm_start is not None:
        actor = copy.deepcopy(actor_warm_start)
        assert isinstance(actor, torch.nn.Sequential), "Warm-start actor must be nn.Sequential"
        last = actor[-1]
        assert hasattr(last, "out_features"), "Last layer must expose out_features"
        assert last.out_features == n_actions, f"Actor output dim must be n_actions (got {last.out_features}, expected {n_actions})"
    else:
        # Actor network: outputs logits for each action
        actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )

    if critic_warm_start is not None:
        qf1 = copy.deepcopy(critic_warm_start)
        qf2 = copy.deepcopy(critic_warm_start)
        assert isinstance(qf1, torch.nn.Sequential), "Warm-start critic must be nn.Sequential"
        assert isinstance(qf2, torch.nn.Sequential), "Warm-start critic must be nn.Sequential"
        last = qf1[-1]
        assert hasattr(last, "out_features"), "Last layer must expose out_features"
        assert last.out_features == n_actions, f"Critic output dim must be n_actions (got {last.out_features}, expected {n_actions})"
    else:
        # Twin Q-networks: take state as input and output Q-value for each action
        qf1 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
        qf2 = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )

    return actor, qf1, qf2

def set_seed(env, seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
        env.action_space.seed(seed)

def evaluate(
        env: gym.Env, actor: nn.Sequential, episodes=10, seed: int = 2025,
        device: str = 'cpu', render_mode: str | None = None,
        deterministic: bool = True
    ):
    """
    Evaluate the actor policy over a number of episodes and return mean and std of rewards.

    Args:
        env: Gymnasium environment
        actor: Actor network
        episodes: Number of episodes to evaluate
        seed: Random seed
        device: Device to run on
        render_mode: Rendering mode (None or 'rgb_array')
        deterministic: Whether to use deterministic actions (argmax) or stochastic (sampled)
    """
    assert render_mode in (None, 'rgb_array')
    actor.eval()
    scores = []
    failures = 0

    for episode_num in range(episodes):
        obs, _ = env.reset(seed=seed*episode_num)
        episodic_reward = 0.0
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                if deterministic:
                    action = torch.argmax(logits, dim=-1).item()
                else:
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs=probs)
                    action = dist.sample().item()

            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == 'rgb_array':
                plt.imshow(env.render())  # type: ignore
                plt.axis('off')
                plt.show()
            # Safety tracking: check 'safe' flag (custom envs) or 'cost' (Safety Gymnasium)
            is_safe = info.get('safe', None)
            if is_safe is None:
                # For Safety Gymnasium: cost > 0 means unsafe
                cost = info.get('cost', 0)
                is_safe = (cost == 0)
            if not is_safe:
                failures += 1
            episodic_reward += reward  # type: ignore
            done = terminated or truncated
        scores.append(episodic_reward)
    actor.train()
    avg_total_reward = float(np.mean(scores))
    std_total_reward = float(np.std(scores))
    failure_rate = failures / episodes
    return avg_total_reward, std_total_reward, failure_rate

def discrete_sac_train(
    env: "gym.Env | Callable[[], gym.Env]",
    cfg: DiscreteSACConfig,
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    actor_param_bounds_l: list[torch.Tensor] | None = None,
    actor_param_bounds_u: list[torch.Tensor] | None = None,
    return_training_data: bool = False
):
    """
    Train a Discrete SAC agent. If actor_param_bounds_l and actor_param_bounds_u are provided,
    uses projected gradient descent to keep actor parameters within bounds.

    When cfg.num_envs > 1, *env* must be a Callable[[], gym.Env] factory (not an instance).
    When cfg.compile_model is True, actor and critic forward passes are compiled with
    torch.compile (requires PyTorch >= 2.0).

    Args:
        env: Gymnasium environment instance OR a no-arg factory callable.
            A callable is required when cfg.num_envs > 1.
        cfg: Discrete SAC configuration
        actor_warm_start: Optional pre-trained actor network
        critic_warm_start: Optional pre-trained critic network
        actor_param_bounds_l: Optional lower bounds for actor parameters (for PGD)
        actor_param_bounds_u: Optional upper bounds for actor parameters (for PGD)
        return_training_data: If True, returns state-action pairs collected during training

    Returns:
        If return_training_data is False: (actor, qf1, qf2)
        If return_training_data is True: (actor, qf1, qf2, training_data)
            where training_data is a dict containing:
                - 'states': numpy array of shape (N, obs_dim) containing all states visited
                - 'actions': numpy array of shape (N,) containing all actions taken
                - 'terminated': numpy array of shape (N,) containing all termination flags
                - 'truncated': numpy array of shape (N,) containing all truncation flags
                - 'safe': numpy array of shape (N,) containing all safety flags
    """
    num_envs = cfg.num_envs

    # ── Environment setup ───────────────────────────────────────────────────
    if num_envs > 1:
        assert callable(env), (
            "cfg.num_envs > 1 requires a Callable[[], gym.Env] factory, not a gym.Env instance"
        )
        env_factory = env
        training_env = gym.vector.SyncVectorEnv([env_factory] * num_envs)  # type: ignore
        eval_env = env_factory()   # single env used for periodic evaluation
        obs_dim = training_env.single_observation_space.shape[0]  # type: ignore
        n_actions = int(training_env.single_action_space.n)  # type: ignore
    else:
        if callable(env):
            env = env()
        training_env = env  # type: ignore
        eval_env = env      # type: ignore
        # Verify discrete action space
        assert isinstance(training_env.action_space, gym.spaces.Discrete), \
            "Discrete SAC requires discrete action space"
        obs_dim = training_env.observation_space.shape[0]  # type: ignore
        n_actions = int(training_env.action_space.n)  # type: ignore

    set_seed(training_env, cfg.seed)

    # Create networks
    actor, qf1, qf2 = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
    )

    # Create target networks
    qf1_target = copy.deepcopy(qf1)
    qf2_target = copy.deepcopy(qf2)

    device = torch.device(cfg.device)
    actor.to(device)
    qf1.to(device)
    qf2.to(device)
    qf1_target.to(device)
    qf2_target.to(device)

    # ── Optional torch.compile wrappers ─────────────────────────────────────
    # _actor/_qf* are used only for forward passes inside the training loop;
    # optimizer.step() and PGD always act on the raw (uncompiled) module.
    if cfg.compile_model:
        _actor = torch.compile(actor)          # type: ignore
        _qf1 = torch.compile(qf1)              # type: ignore
        _qf2 = torch.compile(qf2)              # type: ignore
        _qf1_target = torch.compile(qf1_target)  # type: ignore
        _qf2_target = torch.compile(qf2_target)  # type: ignore
    else:
        _actor, _qf1, _qf2 = actor, qf1, qf2
        _qf1_target, _qf2_target = qf1_target, qf2_target

    # Check if we have parameter bounds for projected gradient descent
    use_pgd = (actor_param_bounds_l is not None and actor_param_bounds_u is not None)
    print('Use PGD:', use_pgd)
    bounds_l = None
    bounds_u = None
    if use_pgd:
        bounds_l = [bound.to(device) for bound in actor_param_bounds_l]  # type: ignore
        bounds_u = [bound.to(device) for bound in actor_param_bounds_u]  # type: ignore
        print(f"Using projected gradient descent with parameter bounds")

        # Ensure initial parameters are within bounds
        with torch.no_grad():
            for param, lb, ub in zip(actor.parameters(), bounds_l, bounds_u):
                param.data.clamp_(lb, ub)

    # Create optimizers
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.lr)
    q_optimizer = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=cfg.lr)

    # Automatic entropy tuning
    if cfg.autotune_alpha:
        # For discrete actions, target entropy is typically -log(1/|A|) * scale
        # Common heuristic: use 0.98 * max_entropy where max_entropy = log(n_actions)
        target_entropy = -0.98 * np.log(1.0 / n_actions)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optimizer = torch.optim.Adam([log_alpha], lr=cfg.lr)
    else:
        alpha = cfg.alpha

    # Create replay buffer
    replay_buffer = DiscreteReplayBuffer(obs_dim, cfg.buffer_size, device)  # type: ignore

    # Training data tracking
    if return_training_data:
        training_data: dict = {
            'states': [],
            'actions': [],
            'terminated': [],
            'truncated': [],
            'safe': [],
        }

    # ── Initial reset ────────────────────────────────────────────────────────
    if num_envs > 1:
        obs, _ = training_env.reset(seed=cfg.seed)  # obs shape: (num_envs, obs_dim)
    else:
        obs, _ = training_env.reset(seed=cfg.seed)

    global_step = 0
    start_time = time.time()
    pgd_projections = 0  # Count total parameter projections for logging
    episode_rewards: deque = deque(maxlen=10)
    episode_reward: "np.ndarray | float"
    if num_envs > 1:
        episode_reward = np.zeros(num_envs, dtype=np.float32)
    else:
        episode_reward = 0.0

    while global_step < cfg.total_timesteps:
        # ── Select actions ─────────────────────────────────────────────────
        if global_step < cfg.learning_starts:
            if num_envs > 1:
                actions = np.array([training_env.single_action_space.sample() for _ in range(num_envs)])  # type: ignore
            else:
                actions = training_env.action_space.sample()  # type: ignore
        else:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
                if num_envs == 1:
                    obs_t = obs_t.unsqueeze(0)
                logits = _actor(obs_t)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs=probs)
                actions_t = dist.sample()
                if num_envs > 1:
                    actions = actions_t.cpu().numpy()
                else:
                    actions = actions_t.item()

        # ── Step environment ───────────────────────────────────────────────
        next_obs, rewards, terminateds, truncateds, infos = training_env.step(actions)
        global_step += num_envs

        if num_envs > 1:
            # VectorEnv: add one transition per sub-environment
            for i in range(num_envs):
                done_i = bool(terminateds[i]) or bool(truncateds[i])
                episode_reward[i] += float(rewards[i])  # type: ignore

                # Use final_observation for bootstrap when the env auto-resets
                fin_obs = infos.get('final_observation')
                if done_i and fin_obs is not None and fin_obs[i] is not None:
                    real_next_obs = fin_obs[i]
                else:
                    real_next_obs = next_obs[i]

                replay_buffer.add(obs[i], int(actions[i]), float(rewards[i]), real_next_obs, float(done_i))

                if return_training_data:
                    training_data['states'].append(obs[i].copy())
                    training_data['actions'].append(int(actions[i]))
                    training_data['terminated'].append(float(terminateds[i]))
                    training_data['truncated'].append(float(truncateds[i]))
                    # Safety info may live in final_info for done envs
                    fin_info = infos.get('final_info')
                    if done_i and fin_info is not None and fin_info[i] is not None:
                        step_info = fin_info[i]
                    else:
                        step_info = {k: (v[i] if isinstance(v, np.ndarray) else v)
                                     for k, v in infos.items()
                                     if k not in ('final_observation', 'final_info', '_final_observation', '_final_info')}
                    is_safe = step_info.get('safe', None)
                    if is_safe is None:
                        cost = step_info.get('cost', 0)
                        is_safe = 1.0 if cost == 0 else 0.0
                    training_data['safe'].append(float(is_safe))

                if done_i:
                    episode_rewards.append(float(episode_reward[i]))  # type: ignore
                    episode_reward[i] = 0.0  # type: ignore

            obs = next_obs  # shape (num_envs, obs_dim)

        else:
            # Single env
            terminated = bool(terminateds)
            truncated = bool(truncateds)
            done = terminated or truncated
            episode_reward += float(rewards)  # type: ignore

            replay_buffer.add(obs, actions, rewards, next_obs, float(done))

            if return_training_data:
                training_data['states'].append(obs.copy())  # type: ignore
                training_data['actions'].append(actions)  # type: ignore
                training_data['terminated'].append(float(terminated))  # type: ignore
                training_data['truncated'].append(float(truncated))  # type: ignore
                is_safe = infos.get('safe', None)
                if is_safe is None:
                    cost = infos.get('cost', 0)
                    is_safe = 1.0 if cost == 0 else 0.0
                training_data['safe'].append(float(is_safe))  # type: ignore

            obs = next_obs
            if done:
                episode_rewards.append(float(episode_reward))  # type: ignore
                episode_reward = 0.0
                obs, _ = training_env.reset()

        # ── Training updates ───────────────────────────────────────────────
        if global_step > cfg.learning_starts:
            # Sample from replay buffer
            states, act_batch, reward_batch, next_states, dones = replay_buffer.sample(cfg.batch_size)

            # Update Q-networks
            with torch.no_grad():
                next_logits = _actor(next_states)
                next_probs = F.softmax(next_logits, dim=-1)
                next_log_probs = F.log_softmax(next_logits, dim=-1)

                next_q1 = _qf1_target(next_states)
                next_q2 = _qf2_target(next_states)
                next_q = torch.min(next_q1, next_q2)

                next_v = (next_probs * (next_q - alpha * next_log_probs)).sum(dim=-1, keepdim=True)
                target_q = reward_batch + (1 - dones) * cfg.gamma * next_v

            current_q1 = _qf1(states)
            current_q2 = _qf2(states)
            current_q1_a = current_q1.gather(1, act_batch.unsqueeze(1))
            current_q2_a = current_q2.gather(1, act_batch.unsqueeze(1))

            q1_loss = F.mse_loss(current_q1_a, target_q)
            q2_loss = F.mse_loss(current_q2_a, target_q)
            q_loss = q1_loss + q2_loss

            q_optimizer.zero_grad()
            q_loss.backward()
            q_optimizer.step()

            # Update policy (and alpha) with delay
            if global_step % cfg.policy_frequency == 0:
                logits = _actor(states)
                probs = F.softmax(logits, dim=-1)
                log_probs = F.log_softmax(logits, dim=-1)

                with torch.no_grad():
                    q1_pi = _qf1(states)
                    q2_pi = _qf2(states)
                    min_q_pi = torch.min(q1_pi, q2_pi)

                inside_term = alpha * log_probs - min_q_pi
                actor_loss = (probs * inside_term).sum(dim=-1).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Projected gradient descent: clamp actor parameters to bounds
                if use_pgd:
                    with torch.no_grad():
                        for param, lb, ub in zip(actor.parameters(), bounds_l, bounds_u):  # type: ignore
                            violations = ((param.data < lb) | (param.data > ub)).sum().item()
                            if violations > 0:
                                pgd_projections += violations
                            param.data.clamp_(lb, ub)

                # Update alpha
                if cfg.autotune_alpha:
                    logits = _actor(states)
                    log_probs = F.log_softmax(logits, dim=-1)
                    probs = F.softmax(logits, dim=-1)

                    entropy = -(probs * log_probs).sum(dim=-1).mean()

                    alpha_loss = -log_alpha.exp() * (entropy - target_entropy)  # type: ignore
                    alpha_optimizer.zero_grad()  # type: ignore
                    alpha_loss.backward()
                    alpha_optimizer.step()  # type: ignore
                    alpha = log_alpha.exp().item()  # type: ignore

            # Update target networks
            if global_step % cfg.target_network_frequency == 0:
                with torch.no_grad():
                    for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                        target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)
                    for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                        target_param.data.copy_(cfg.tau * param.data + (1 - cfg.tau) * target_param.data)

        # ── Logging (milestone-based, works for any num_envs) ──────────────
        if global_step // 10000 > (global_step - num_envs) // 10000 and global_step > 0:
            mean_r, std_r, failure_rate = evaluate(env=eval_env, actor=actor, device=device, episodes=10)  # type: ignore
            # Reset training env after evaluation only when eval_env IS training_env
            if num_envs == 1:
                obs, _ = training_env.reset()
                episode_reward = 0.0
            elapsed = time.time() - start_time
            recent_reward = float(np.mean(episode_rewards)) if len(episode_rewards) > 0 else 0.0
            log_msg = (f"Steps={global_step} | evalR={mean_r:.1f}±{std_r:.1f}"
                       f" | recentR={recent_reward:.1f} | alpha={alpha:.3f} | elapsed={elapsed:.1f}s")
            log_msg += f" | failure_rate={failure_rate:.2f}"
            if use_pgd:
                log_msg += f" | PGD={pgd_projections}"
            print(log_msg)

    # ── Cleanup ─────────────────────────────────────────────────────────────
    training_env.close()

    if cfg.eval_episodes is not None and cfg.eval_episodes > 0:
        mean_r, std_r, failure_rate = evaluate(env=eval_env, actor=actor, device=device, episodes=cfg.eval_episodes)  # type: ignore
        final_msg = f"Final evaluation over {cfg.eval_episodes} episodes: mean_reward={mean_r:.2f}±{std_r:.2f}"
        final_msg += f" | failure_rate={failure_rate:.2f}"
        if use_pgd:
            final_msg += f" | Total PGD projections during training: {pgd_projections}"
        print(final_msg)

    if num_envs > 1:
        eval_env.close()

    # Return results
    if return_training_data:
        training_data['states'] = np.array(training_data['states'])  # type: ignore
        training_data['actions'] = np.array(training_data['actions'])  # type: ignore
        training_data['terminated'] = np.array(training_data['terminated'])  # type: ignore
        training_data['truncated'] = np.array(training_data['truncated'])  # type: ignore
        training_data['safe'] = np.array(training_data['safe'])  # type: ignore
        return actor, qf1, qf2, training_data  # type: ignore
    else:
        return actor, qf1, qf2
