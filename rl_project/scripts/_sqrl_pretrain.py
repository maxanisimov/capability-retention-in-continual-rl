# sqrl_pretrain.py
# SQRL Pre-training (Algorithm 1) — PyTorch, continuous + discrete obs/acts
# Based on: "Learning to be Safe: Deep RL with a Safety Critic" (arXiv:2010.14603)

from __future__ import annotations
from dataclasses import dataclass
import warnings
import time
from typing import Callable, Tuple, Dict, Optional
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from collections import deque, namedtuple
import random

# -----------------------------
# Utilities
# -----------------------------

def to_tensor(x, device, dtype=torch.float32):
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        if t.dtype == torch.float64:
            t = t.float()
        return t.to(device, dtype=dtype if t.dtype.is_floating_point else None)
    return torch.tensor(x, device=device, dtype=dtype)

def polyak_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(tau * sp.data)

def one_hot(indices: np.ndarray, depth: int) -> np.ndarray:
    indices = indices.astype(np.int64).ravel()
    eye = np.eye(depth, dtype=np.float32)
    return eye[indices]

# -----------------------------
# Networks
# -----------------------------

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=(256, 256), act=nn.ReLU, out_act=None):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        if out_act is not None:
            layers += [out_act()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -------- Continuous policy + critics (unchanged) --------

class GaussianTanhPolicy(nn.Module):
    """SAC-style policy for continuous actions."""
    def __init__(self, obs_dim, act_dim, hidden=(256, 256), log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.backbone = MLP(obs_dim, 2 * act_dim, hidden=hidden, act=nn.ReLU, out_act=None)
        self.log_std_bounds = log_std_bounds
        self.act_dim = act_dim

    def forward(self, obs):
        mu_logstd = self.backbone(obs)
        mu, log_std = torch.chunk(mu_logstd, 2, dim=-1)
        log_std = torch.tanh(log_std)
        low, high = self.log_std_bounds
        log_std = low + 0.5*(high - low)*(log_std + 1)
        std = torch.exp(log_std)
        return mu, std, log_std

    def sample(self, obs, deterministic=False):
        mu, std, log_std = self.forward(obs)
        if deterministic:
            z = mu
        else:
            eps = torch.randn_like(std)
            z = mu + std * eps
        a = torch.tanh(z)
        log_prob = None
        if not deterministic:
            log_prob = -0.5 * (((z - mu) / (std + 1e-8))**2 + 2*log_std + np.log(2*np.pi)).sum(dim=-1, keepdim=True)
            log_prob -= torch.log(1 - a.pow(2) + 1e-8).sum(dim=-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)

class QCritic(nn.Module):
    """Twin Q for continuous actions: Q(s,a)."""
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, 1, hidden=hidden, act=nn.ReLU)
        self.q2 = MLP(obs_dim + act_dim, 1, hidden=hidden, act=nn.ReLU)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

class SafetyCritic(nn.Module):
    """Safety Q for continuous: probability of eventual failure in (0,1)."""
    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.core = MLP(obs_dim + act_dim, 1, hidden=hidden, act=nn.ReLU)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.sigmoid(self.core(x))  # (B,1)

# -------- Discrete policy + critics --------

class CategoricalPolicy(nn.Sequential):
    """Categorical policy for discrete actions (subscriptable nn.Sequential)."""
    def __init__(self, obs_dim, n_actions, hidden=(256, 256), act_cls=nn.ReLU):
        layers = OrderedDict()
        last = obs_dim
        for i, h in enumerate(hidden):
            layers[f"fc{i}"] = nn.Linear(last, h)
            layers[f"act{i}"] = act_cls()
            last = h
        layers["head"] = nn.Linear(last, n_actions)  # logits layer

        super().__init__(layers)  # <-- makes it subscriptable
        self.n_actions = n_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Ensure 2D input (B, obs_dim), but don't force dtype/device changes here
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return super().forward(obs)  # returns logits (B, n_actions)

    # Keep the old API for compatibility
    def logits(self, obs: torch.Tensor) -> torch.Tensor:
        return self.forward(obs)

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        if deterministic:
            a = torch.argmax(logits, dim=-1)                       # (B,)
            logp = torch.log_softmax(logits, dim=-1)               # (B, nA)
            logp = logp.gather(1, a.unsqueeze(-1))                 # (B, 1)
            return a, logp, logits
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()                                          # (B,)
        logp = dist.log_prob(a).unsqueeze(-1)                      # (B, 1)
        return a, logp, logits

class QCriticDiscrete(nn.Module):
    """Twin Q for discrete actions: returns Q(s,·) for both heads."""
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.q1 = MLP(obs_dim, n_actions, hidden=hidden, act=nn.ReLU)
        self.q2 = MLP(obs_dim, n_actions, hidden=hidden, act=nn.ReLU)

    def forward(self, obs):
        return self.q1(obs), self.q2(obs)  # both (B, nA)

class SafetyCriticDiscrete(nn.Module):
    """Safety critic for discrete actions: per-action failure probabilities (0,1)."""
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        self.core = MLP(obs_dim, n_actions, hidden=hidden, act=nn.ReLU)
        self.sigmoid = nn.Sigmoid()

    def forward(self, obs):
        return self.sigmoid(self.core(obs))  # (B, nA)

# -----------------------------
# Replay buffers
# -----------------------------

Transition = namedtuple("Transition",
                        ["obs", "act", "rew", "next_obs", "done", "fail"])

class ReplayBuffer:
    def __init__(self, capacity=int(1e6)):
        self.capacity = capacity
        self.storage = deque(maxlen=capacity)

    def add(self, *args):
        self.storage.append(Transition(*args))

    def sample(self, batch_size) -> Transition:
        batch = random.sample(self.storage, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.storage)

# -----------------------------
# SQRL projection / rejection sampling
# -----------------------------

@torch.no_grad()
def project_action_continuous(obs: torch.Tensor,
                              base_policy: GaussianTanhPolicy,
                              safety_critic: SafetyCritic,
                              tau_threshold: float,
                              K_candidates: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Continuous Pi_safe: sample K from policy, pick feasible closest-to-τ; else min-risk."""
    B = obs.shape[0]
    obs_rep = obs.repeat_interleave(K_candidates, dim=0)
    a_cand, _, _ = base_policy.sample(obs_rep, deterministic=False)
    q = safety_critic(obs_rep, a_cand).reshape(B, K_candidates, 1)  # (B,K,1)
    feasible = (q <= tau_threshold)
    chosen, chosen_q = [], []
    for b in range(B):
        qb = q[b, :, 0]
        ab = a_cand[b*K_candidates:(b+1)*K_candidates]
        fb = feasible[b, :, 0]
        if fb.any():
            idx = torch.argmax(qb.masked_fill(~fb, -1.0))
        else:
            idx = torch.argmin(qb)
        chosen.append(ab[idx])
        chosen_q.append(qb[idx:idx+1])
    a_proj = torch.stack(chosen, dim=0)
    q_proj = torch.stack(chosen_q, dim=0)
    return a_proj, q_proj  # (B, act_dim), (B,1)

@torch.no_grad()
def project_action_discrete(obs: torch.Tensor,
                            safety_critic: SafetyCriticDiscrete,
                            tau_threshold: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Discrete Pi_safe: evaluate all actions. If any q<=τ, pick argmax(q | q<=τ)
    (closest to τ from below). Else pick argmin q.
    Returns (a_idx, q_selected) with shapes (B,), (B,1).
    """
    q = safety_critic(obs)  # (B, nA), failure probs
    feasible = (q <= tau_threshold)
    a_list, q_list = [], []
    for b in range(q.shape[0]):
        qb = q[b]
        fb = feasible[b]
        if fb.any():
            # choose feasible with highest q (closest to tau from below)
            masked = qb.clone()
            masked[~fb] = -1.0
            idx = torch.argmax(masked).item()
        else:
            idx = torch.argmin(qb).item()
        a_list.append(idx)
        q_list.append(qb[idx].unsqueeze(0))
    a_idx = torch.tensor(a_list, device=obs.device, dtype=torch.long)
    q_sel = torch.stack(q_list, dim=0)  # (B,1)
    return a_idx, q_sel

# -----------------------------
# Config
# -----------------------------

@dataclass
class SQRLPretrainConfig:
    # Env / training
    env_id: str = None
    env: gym.Env = None  # optionally pass an instance
    seed: int = 0
    total_env_steps: int = 200_000
    start_steps: int = 1_000
    episode_max_steps: Optional[int] = None
    # Replay
    replay_size: int = 1_000_000
    onpolicy_size: int = 20_000
    batch_size: int = 512
    # Discounts
    gamma: float = 0.99
    gamma_safety: float = 0.99
    # SAC
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_update_tau: float = 0.005
    target_update_period: int = 1
    actor_update_period: int = 1
    critic_update_steps_per_env_step: int = 1
    # Temperature (autotune)
    autotune_alpha: bool = True
    init_alpha: float = 0.2
    target_entropy_scale: float = 1.0
    # Safety critic
    safety_lr: float = 3e-4
    safety_update_steps_per_env_step: int = 1
    safety_target_update_tau: float = 0.01
    # Projection
    tau_threshold: float = 0.2
    K_candidates: int = 64
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Safety Critic Only Training
# -----------------------------

@dataclass
class SafetyCriticConfig:
    """Configuration for training only the safety critic given a fixed policy."""
    env: Optional[gym.Env] = None
    env_id: str = "CartPole-v1"
    device: str = "cpu"
    seed: int = 42
    
    # Training parameters
    total_env_steps: int = 50000
    batch_size: int = 256
    replay_size: int = 100000
    
    # Safety critic specific
    gamma_safety: float = 0.99
    safety_lr: float = 3e-4
    safety_target_update_tau: float = 0.005
    safety_update_steps_per_env_step: int = 1
    
    # Data collection
    exploration_noise: float = 0.1  # For continuous actions
    
    # Logging
    log_period: int = 1000

def train_safety_critic_only(
    policy: nn.Module,
    cfg: SafetyCriticConfig,
    failure_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], bool],
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Train only a safety critic given a fixed policy.
    
    Args:
        policy: Pre-trained policy (either CategoricalPolicy or GaussianTanhPolicy)
        cfg: Configuration for safety critic training
        failure_fn: Function to determine if a state transition is a failure
    
    Returns:
        (trained safety critic, logs)
    """
    # ---- Setup env
    if cfg.env is not None:
        env = cfg.env
        try:
            cfg.env_id = env.spec.id
            warnings.warn(f"Using env instance from cfg.env; cfg.env_id set to {cfg.env_id}")
        except Exception:
            warnings.warn("Using env instance from cfg.env.")
    else:
        env = gym.make(cfg.env_id)

    obs, info = env.reset(seed=cfg.seed)

    is_disc_obs = isinstance(env.observation_space, gym.spaces.Discrete)
    is_disc_act = isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = env.observation_space.n if is_disc_obs else int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n if is_disc_act else int(np.prod(env.action_space.shape))

    device = torch.device(cfg.device)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); random.seed(cfg.seed)

    # ---- Move policy to device and set to eval mode
    policy = policy.to(device)
    policy.eval()

    # ---- Encoders
    def encode_obs_batch(raw_batch) -> torch.Tensor:
        arr = np.array(raw_batch)
        if is_disc_obs:
            oh = one_hot(arr, obs_dim)  # (B, nS)
            return to_tensor(oh, device)
        else:
            arr = np.reshape(arr, (len(arr), obs_dim)).astype(np.float32)
            return to_tensor(arr, device)

    # ---- Build safety critic
    if is_disc_act:
        safety_critic = SafetyCriticDiscrete(obs_dim, act_dim).to(device)
        safety_critic_tgt = SafetyCriticDiscrete(obs_dim, act_dim).to(device)
    else:
        safety_critic = SafetyCritic(obs_dim, act_dim).to(device)
        safety_critic_tgt = SafetyCritic(obs_dim, act_dim).to(device)

    # Copy parameters to target network
    safety_critic_tgt.load_state_dict(safety_critic.state_dict())
    safety_critic_opt = optim.Adam(safety_critic.parameters(), lr=cfg.safety_lr)

    # ---- Replay buffer
    Transition = namedtuple('Transition', ['obs', 'act', 'rew', 'next_obs', 'done', 'fail'])
    
    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
        
        def add(self, obs, act, rew, next_obs, done, fail):
            self.buffer.append(Transition(obs, act, rew, next_obs, done, fail))
        
        def sample(self, batch_size):
            return Transition(*zip(*random.sample(self.buffer, batch_size)))
        
        def __len__(self):
            return len(self.buffer)

    replay = ReplayBuffer(cfg.replay_size)

    # ---- Logging
    logs = {
        "safety_critic_loss": [],
        "qsafe_mean": [],
        "episode_return": [],
        "episode_length": [],
        "failure_rate": [],
    }

    # ---- Safety critic update function
    def safety_update():
        if len(replay) < cfg.batch_size: 
            return torch.tensor(0.0, device=device)
        
        total_loss = 0.0
        for _ in range(cfg.safety_update_steps_per_env_step):
            batch = replay.sample(cfg.batch_size)
            b_obs = encode_obs_batch(batch.obs)
            b_next = encode_obs_batch(batch.next_obs)
            b_done = to_tensor(np.array(batch.done, dtype=np.float32).reshape(-1,1), device)
            b_fail = to_tensor(np.array(batch.fail, dtype=np.float32).reshape(-1,1), device)

            if is_disc_act:
                # For discrete actions, get next action from policy
                with torch.no_grad():
                    if isinstance(policy, CategoricalPolicy):
                        logits_next = policy(b_next)
                        a_next_idx = torch.argmax(logits_next, dim=-1)  # Greedy action selection
                    else:
                        a_next_idx, _, _ = policy.sample(b_next, deterministic=True)
                    
                    q_tgt_all = safety_critic_tgt(b_next)  # (B,nA)
                    q_tgt_next = q_tgt_all.gather(1, a_next_idx.unsqueeze(-1))  # (B,1)
                    y = b_fail + (1.0 - b_fail) * (1.0 - b_done) * cfg.gamma_safety * q_tgt_next

                q_all = safety_critic(b_obs)  # (B,nA)
                a_idx = to_tensor(np.array(batch.act, dtype=np.int64).reshape(-1,1), device, dtype=torch.long)
                q_pred = q_all.gather(1, a_idx)  # (B,1)
                
            else:
                # For continuous actions
                b_act = to_tensor(np.stack(batch.act), device)
                with torch.no_grad():
                    if hasattr(policy, 'sample'):
                        a_next, _, _ = policy.sample(b_next, deterministic=True)
                    else:
                        # Fallback: assume policy outputs mean action
                        a_next = policy(b_next)
                    
                    q_tgt_next = safety_critic_tgt(b_next, a_next)
                    y = b_fail + (1.0 - b_fail) * (1.0 - b_done) * cfg.gamma_safety * q_tgt_next

                q_pred = safety_critic(b_obs, b_act)

            loss = nn.functional.mse_loss(q_pred, y)
            total_loss += loss.item()
            
            safety_critic_opt.zero_grad()
            loss.backward()
            safety_critic_opt.step()
            
            # Update target network
            polyak_update(safety_critic_tgt, safety_critic, cfg.safety_target_update_tau)

        return total_loss / cfg.safety_update_steps_per_env_step

    # ---- Data collection and training loop
    ep_ret, ep_len, ep_fail = 0.0, 0, 0
    num_episodes = 0
    total_failures = 0
    total_steps = 0
    
    steps = 0
    while steps < cfg.total_env_steps:
        obs_t = encode_obs_batch([obs])  # (1, obs_dim)

        # ACTION SELECTION using the fixed policy
        with torch.no_grad():
            if is_disc_act:
                if isinstance(policy, CategoricalPolicy):
                    logits = policy(obs_t)
                    # Add exploration noise for data collection
                    if cfg.exploration_noise > 0 and np.random.random() < cfg.exploration_noise:
                        action = env.action_space.sample()
                    else:
                        action = int(torch.argmax(logits, dim=-1)[0].item())
                else:
                    action_idx, _, _ = policy.sample(obs_t, deterministic=False)
                    action = int(action_idx[0].item())
            else:
                if hasattr(policy, 'sample'):
                    action_tensor, _, _ = policy.sample(obs_t, deterministic=False)
                    action = action_tensor[0].cpu().numpy()
                    # Add exploration noise
                    if cfg.exploration_noise > 0:
                        noise = np.random.normal(0, cfg.exploration_noise, action.shape)
                        action = np.clip(action + noise, env.action_space.low, env.action_space.high)
                else:
                    action = policy(obs_t)[0].cpu().numpy()

        # STEP env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        c_fail = bool(failure_fn(obs, action, next_obs, info))
        if c_fail:
            done = True  # ensure terminal on failure
            total_failures += 1

        # Store in replay buffer
        replay.add(obs, action, reward, next_obs, float(done), float(c_fail))

        ep_ret += reward
        ep_len += 1
        ep_fail = max(ep_fail, c_fail)
        steps += 1
        total_steps += 1

        # Update safety critic
        safety_loss = safety_update()

        # Reset episode if done
        if done:
            logs["episode_return"].append(ep_ret)
            logs["episode_length"].append(ep_len)
            logs["failure_rate"].append(float(ep_fail))
            
            num_episodes += 1
            ep_ret, ep_len, ep_fail = 0.0, 0, 0
            obs, info = env.reset()
        else:
            obs = next_obs

        # Logging
        if steps % cfg.log_period == 0:
            if len(replay) >= cfg.batch_size:
                # Get current safety critic predictions for logging
                obs_batch = encode_obs_batch([obs])
                if is_disc_act:
                    q_mean = safety_critic(obs_batch).mean().item()
                else:
                    if hasattr(policy, 'sample'):
                        with torch.no_grad():
                            a_sample, _, _ = policy.sample(obs_batch, deterministic=True)
                            q_mean = safety_critic(obs_batch, a_sample).mean().item()
                    else:
                        with torch.no_grad():
                            a_sample = policy(obs_batch)
                            q_mean = safety_critic(obs_batch, a_sample).mean().item()
                
                logs["safety_critic_loss"].append(float(safety_loss))
                logs["qsafe_mean"].append(q_mean)
                
                failure_rate = total_failures / max(total_steps, 1)
                print(f"Step {steps}/{cfg.total_env_steps} | Episodes: {num_episodes} | "
                      f"Safety Loss: {safety_loss:.4f} | Q_safe: {q_mean:.4f} | "
                      f"Failure Rate: {failure_rate:.4f}")

    env.close()
    return safety_critic, logs

# -----------------------------
# Main pretraining routine (Algorithm 1)
# -----------------------------

def pretrain_sqrl(
    cfg: SQRLPretrainConfig,
    failure_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, dict], bool],
) -> Tuple[nn.Module, nn.Module, Dict[str, list]]:
    """
    Returns: (trained policy, trained safety critic, logs)
    Supports:
      - Box obs + Box act (continuous SAC)
      - Discrete obs (one-hot) + Box act
      - Box obs + Discrete act (discrete SAC)
      - Discrete obs + Discrete act
    """
    # ---- Setup env
    if cfg.env is not None:
        env = cfg.env
        try:
            cfg.env_id = env.spec.id
            warnings.warn(f"Using env instance from cfg.env; cfg.env_id set to {cfg.env_id}")
        except Exception:
            warnings.warn("Using env instance from cfg.env.")
    else:
        env = gym.make(cfg.env_id)

    obs, info = env.reset(seed=cfg.seed)

    is_disc_obs = isinstance(env.observation_space, gym.spaces.Discrete)
    is_disc_act = isinstance(env.action_space, gym.spaces.Discrete)

    obs_dim = env.observation_space.n if is_disc_obs else int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.n if is_disc_act else int(np.prod(env.action_space.shape))

    device = torch.device(cfg.device)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed); random.seed(cfg.seed)

    # ---- Encoders
    def encode_obs_batch(raw_batch) -> torch.Tensor:
        arr = np.array(raw_batch)
        if is_disc_obs:
            oh = one_hot(arr, obs_dim)  # (B, nS)
            return to_tensor(oh, device)
        else:
            arr = np.reshape(arr, (len(arr), obs_dim)).astype(np.float32)
            return to_tensor(arr, device)

    # ---- Build policy/critics (continuous vs discrete)
    if is_disc_act:
        actor = CategoricalPolicy(obs_dim, act_dim).to(device)
        critic = QCriticDiscrete(obs_dim, act_dim).to(device)
        critic_tgt = QCriticDiscrete(obs_dim, act_dim).to(device)
        critic_tgt.load_state_dict(critic.state_dict())
        qsafe = SafetyCriticDiscrete(obs_dim, act_dim).to(device)
        qsafe_tgt = SafetyCriticDiscrete(obs_dim, act_dim).to(device)
        qsafe_tgt.load_state_dict(qsafe.state_dict())
    else:
        actor = GaussianTanhPolicy(obs_dim, act_dim).to(device)
        critic = QCritic(obs_dim, act_dim).to(device)
        critic_tgt = QCritic(obs_dim, act_dim).to(device)
        critic_tgt.load_state_dict(critic.state_dict())
        qsafe = SafetyCritic(obs_dim, act_dim).to(device)
        qsafe_tgt = SafetyCritic(obs_dim, act_dim).to(device)
        qsafe_tgt.load_state_dict(qsafe.state_dict())

    actor_opt  = optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    qsafe_opt  = optim.Adam(qsafe.parameters(),  lr=cfg.safety_lr)

    # Temperature
    if cfg.autotune_alpha:
        log_alpha = torch.tensor(np.log(cfg.init_alpha), requires_grad=True, device=device)
        alpha_opt = optim.Adam([log_alpha], lr=cfg.alpha_lr)
        if is_disc_act:
            target_entropy = -cfg.target_entropy_scale * float(np.log(act_dim))
        else:
            target_entropy = -cfg.target_entropy_scale * float(act_dim)
        def alpha():
            return float(log_alpha.exp().item())
    else:
        log_alpha = None
        alpha_opt = None
        target_entropy = None
        alpha = lambda: cfg.init_alpha

    # ---- Replay buffers
    replay = ReplayBuffer(capacity=cfg.replay_size)
    onpolicy = ReplayBuffer(capacity=cfg.onpolicy_size)

    # ---- Logging
    logs = {k: [] for k in [
        "episode_return", "episode_len", "episode_fail",
        "safety_critic_loss", "actor_loss", "critic_loss", "alpha", "qsafe_mean"
    ]}

    # ---- SAC update helpers
    def sac_update():
        if len(replay) < cfg.batch_size: return
        for step_i in range(cfg.critic_update_steps_per_env_step):
            batch = replay.sample(cfg.batch_size)

            b_obs  = encode_obs_batch(batch.obs)
            b_next = encode_obs_batch(batch.next_obs)
            b_rew  = to_tensor(np.array(batch.rew,  dtype=np.float32).reshape(-1,1), device)
            b_done = to_tensor(np.array(batch.done, dtype=np.float32).reshape(-1,1), device)

            if is_disc_act:
                # ---- Discrete SAC
                # Targets
                with torch.no_grad():
                    logits_next = actor.logits(b_next)             # (B,nA)
                    logp_next   = torch.log_softmax(logits_next, dim=-1)  # (B,nA)
                    pi_next     = torch.softmax(logits_next, dim=-1)
                    q1t, q2t    = critic_tgt(b_next)              # (B,nA)
                    qmin_tgt    = torch.min(q1t, q2t)
                    v_next      = (pi_next * (qmin_tgt - alpha() * logp_next)).sum(dim=-1, keepdim=True)
                    y           = b_rew + cfg.gamma * (1.0 - b_done) * v_next

                # Critic loss on chosen actions
                a_idx = to_tensor(np.array(batch.act, dtype=np.int64).reshape(-1,1), device, dtype=torch.long)
                q1, q2 = critic(b_obs)  # (B,nA)
                q1_sel = q1.gather(1, a_idx)
                q2_sel = q2.gather(1, a_idx)
                critic_loss = nn.functional.mse_loss(q1_sel, y) + nn.functional.mse_loss(q2_sel, y)
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                # Actor update
                if step_i % cfg.actor_update_period == 0:
                    logits = actor.logits(b_obs)
                    logp   = torch.log_softmax(logits, dim=-1)
                    pi     = torch.softmax(logits, dim=-1)
                    q1p, q2p = critic(b_obs)
                    qmin_pi = torch.min(q1p, q2p)
                    # E_pi[alpha*logpi - Q]
                    actor_loss = (pi * (alpha() * logp - qmin_pi)).sum(dim=-1).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                    if cfg.autotune_alpha:
                        entropy = -(pi * logp).sum(dim=-1, keepdim=True)  # H[pi]
                        alpha_loss = -(log_alpha * (entropy.detach() - target_entropy)).mean()
                        alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                else:
                    actor_loss = torch.tensor(0.0, device=device)

            else:
                # ---- Continuous SAC (original)
                b_act = to_tensor(np.stack(batch.act), device)
                with torch.no_grad():
                    a_next, logp_next, _ = actor.sample(b_next, deterministic=False)
                    q1t, q2t = critic_tgt(b_next, a_next)
                    qmin_tgt = torch.min(q1t, q2t) - alpha() * logp_next
                    y = b_rew + cfg.gamma * (1.0 - b_done) * qmin_tgt

                q1, q2 = critic(b_obs, b_act)
                critic_loss = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
                critic_opt.zero_grad(); critic_loss.backward(); critic_opt.step()

                if step_i % cfg.actor_update_period == 0:
                    a_pi, logp_pi, _ = actor.sample(b_obs, deterministic=False)
                    q1_pi, q2_pi = critic(b_obs, a_pi)
                    qmin_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = (alpha() * logp_pi - qmin_pi).mean()
                    actor_opt.zero_grad(); actor_loss.backward(); actor_opt.step()

                    if cfg.autotune_alpha:
                        alpha_loss = -(log_alpha * (logp_pi + target_entropy).detach()).mean()
                        alpha_opt.zero_grad(); alpha_loss.backward(); alpha_opt.step()
                else:
                    actor_loss = torch.tensor(0.0, device=device)

            # Target nets
            if cfg.target_update_period == 1:
                polyak_update(critic_tgt, critic, cfg.target_update_tau)

        logs["critic_loss"].append(float(critic_loss.item()))
        logs["actor_loss"].append(float(actor_loss.item()))
        logs["alpha"].append(float(alpha()))

    def safety_update():
        if len(onpolicy) < cfg.batch_size: return
        for _ in range(cfg.safety_update_steps_per_env_step):
            batch = onpolicy.sample(cfg.batch_size)
            b_obs  = encode_obs_batch(batch.obs)
            b_next = encode_obs_batch(batch.next_obs)
            b_done = to_tensor(np.array(batch.done, dtype=np.float32).reshape(-1,1), device)
            b_fail = to_tensor(np.array(batch.fail, dtype=np.float32).reshape(-1,1), device)

            if is_disc_act:
                # projected next action index
                a_next_idx, _ = project_action_discrete(b_next, qsafe, cfg.tau_threshold)
                with torch.no_grad():
                    q_tgt_all = qsafe_tgt(b_next)  # (B,nA)
                    q_tgt_next = q_tgt_all.gather(1, a_next_idx.unsqueeze(-1))  # (B,1)
                    y = b_fail + (1.0 - b_fail) * (1.0 - b_done) * cfg.gamma_safety * q_tgt_next

                q_all = qsafe(b_obs)  # (B,nA)
                a_idx = to_tensor(np.array(batch.act, dtype=np.int64).reshape(-1,1), device, dtype=torch.long)
                q_pred = q_all.gather(1, a_idx)  # (B,1)
                loss = nn.functional.mse_loss(q_pred, y)
                qsafe_opt.zero_grad(); loss.backward(); qsafe_opt.step()
                polyak_update(qsafe_tgt, qsafe, cfg.safety_target_update_tau)

            else:
                b_act = to_tensor(np.stack(batch.act), device)
                with torch.no_grad():
                    a_next, _ = project_action_continuous(b_next, actor, qsafe, cfg.tau_threshold, cfg.K_candidates)
                    q_tgt_next = qsafe_tgt(b_next, a_next)
                    y = b_fail + (1.0 - b_fail) * (1.0 - b_done) * cfg.gamma_safety * q_tgt_next

                q_pred = qsafe(b_obs, b_act)
                loss = nn.functional.mse_loss(q_pred, y)
                qsafe_opt.zero_grad(); loss.backward(); qsafe_opt.step()
                polyak_update(qsafe_tgt, qsafe, cfg.safety_target_update_tau)

        logs["safety_critic_loss"].append(float(loss.item()))
        logs["qsafe_mean"].append(float(q_pred.mean().item()))

    # -----------------------------
    # Algorithm 1 loop
    # -----------------------------
    ep_ret, ep_len, ep_fail = 0.0, 0, 0
    steps = 0
    while steps < cfg.total_env_steps:
        obs_t = encode_obs_batch([obs])  # (1, obs_dim)

        # ACTION SELECTION with projection
        if is_disc_act:
            # Discrete: projection chooses among all actions
            a_proj_idx, q_risk = project_action_discrete(obs_t, qsafe, cfg.tau_threshold)
            action = int(a_proj_idx[0].item())
        else:
            # Continuous: sample then project
            if steps < cfg.start_steps:
                # still project for safety
                _a_uncon, _, _ = actor.sample(obs_t, deterministic=False)
            else:
                _a_uncon, _, _ = actor.sample(obs_t, deterministic=False)
            a_proj, q_risk = project_action_continuous(obs_t, actor, qsafe, cfg.tau_threshold, cfg.K_candidates)
            action = a_proj[0].cpu().numpy()

        # STEP env
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = bool(terminated or truncated)
        c_fail = bool(failure_fn(obs, action, next_obs, info))
        if c_fail:
            done = True  # ensure terminal on failure

        # Store
        replay.add(obs, action, reward, next_obs, float(done), float(c_fail))
        onpolicy.add(obs, action, reward, next_obs, float(done), float(c_fail))

        ep_ret += reward
        ep_len += 1
        ep_fail |= int(c_fail)
        steps += 1

        # Updates
        sac_update()
        safety_update()

        # Reset episode
        if done or (cfg.episode_max_steps and ep_len >= cfg.episode_max_steps):
            logs["episode_return"].append(ep_ret)
            logs["episode_len"].append(ep_len)
            logs["episode_fail"].append(ep_fail)
            obs, info = env.reset()
            ep_ret, ep_len, ep_fail = 0.0, 0, 0
        else:
            obs = next_obs

    env.close()
    return actor, qsafe, logs

# -----------------------------
# Example failure function and runner
# -----------------------------

def default_failure_fn(obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray, info: dict) -> bool:
    # Example: look for env-provided signal. You should override per env.
    return bool(info.get("is_failure", False))

if __name__ == "__main__":
    # Example: works with continuous or discrete envs.
    # Continuous smoke test:
    cfg = SQRLPretrainConfig(env_id="Pendulum-v1", total_env_steps=5_000, tau_threshold=0.2)
    t0 = time.time()
    policy, safety_critic, logs = pretrain_sqrl(cfg, failure_fn=default_failure_fn)
    print(f"[Continuous] steps: {len(logs['episode_return'])} episodes in {time.time()-t0:.1f}s")

    # Discrete smoke test (discrete actions; obs is Box for CartPole)
    cfg = SQRLPretrainConfig(env_id="CartPole-v1", total_env_steps=5_000, tau_threshold=0.2)
    t0 = time.time()
    policy, safety_critic, logs = pretrain_sqrl(cfg, failure_fn=default_failure_fn)
    print(f"[Discrete actions] steps: {len(logs['episode_return'])} episodes in {time.time()-t0:.1f}s")
