import argparse
from dataclasses import dataclass
import random
from typing import Tuple, List

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
import torch.optim as optim

import sys
import os
import torch

project_root = os.path.abspath('/Users/ma5923/Documents/_projects/CertifiedContinualLearning')
if project_root not in sys.path:
    sys.path.append(project_root)

from src.trainer import IntervalTrainer


# =========================
# Environment: Reach-Avoid Grid
# =========================
# Grid of shape (H, W). Start at S, goal G, lava cells L (terminal failure).
# Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
# Rewards: +1 at goal, 0 otherwise; -1 on lava (optional); episode terminates on goal/lava or after max_steps.
# Optional "slip" probability to inject stochasticity: with prob p, execute a random action instead.
class ReachAvoidGrid(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        height: int = 5,
        width: int = 5,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (4, 4),
        lava: List[Tuple[int, int]] = ((1, 2), (2, 2), (3, 2)),
        slip: float = 0.0,
        step_penalty: float = 0.0,       # tiny negative step cost can help exploration, keep small (e.g., -0.01)
        lava_penalty: float = 0.0,       # set to -1.0 if you want explicit negative on failure
        max_steps: int = 60,
        seed: int = 42,
    ):
        super().__init__()
        self.H, self.W = height, width
        self.start = start
        self.goal = goal
        self.lava = set(lava) if lava is not None else set()
        self.slip = slip
        self.step_penalty = step_penalty
        self.lava_penalty = lava_penalty
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.nS = self.H * self.W
        self.nA = 4
        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)

        self._pos = tuple(start)
        self._t = 0

    def _to_state(self, pos: Tuple[int, int]) -> int:
        r, c = pos
        return r * self.W + c

    def _in_bounds(self, r, c):
        return 0 <= r < self.H and 0 <= c < self.W

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._pos = tuple(self.start)
        self._t = 0
        return self._to_state(self._pos), {}

    def step(self, action: int):
        self._t += 1
        # slip to a random action with prob self.slip
        if self.rng.random() < self.slip:
            action = int(self.rng.integers(0, self.nA))

        r, c = self._pos
        if action == 0: nr, nc = r - 1, c
        elif action == 1: nr, nc = r, c + 1
        elif action == 2: nr, nc = r + 1, c
        else: nr, nc = r, c - 1

        if self._in_bounds(nr, nc):
            self._pos = (nr, nc)  # bumping into walls just keeps you in place otherwise

        terminated = False
        reward = self.step_penalty

        if self._pos == self.goal:
            reward = 1.0
            terminated = True
        elif self._pos in self.lava:
            reward = self.lava_penalty
            terminated = True

        truncated = (self._t >= self.max_steps)
        return self._to_state(self._pos), float(reward), terminated, truncated, {}

    # lightweight text render for debugging
    def render(self):
        grid = [["." for _ in range(self.W)] for __ in range(self.H)]
        for (lr, lc) in self.lava:
            grid[lr][lc] = "L"
        gr, gc = self.goal
        grid[gr][gc] = "G"
        sr, sc = self._pos
        grid[sr][sc] = "A"
        print("\n".join(" ".join(row) for row in grid), "\n")


# =========================
# Potential-based shaping (optional)
# =========================
def manhattan_phi(state_idx: int, W: int, goal_rc: Tuple[int, int]):
    r, c = divmod(state_idx, W)
    gr, gc = goal_rc
    return - (abs(r - gr) + abs(c - gc))  # more negative far away


# =========================
# Parameter bounds projection
# =========================
def _project_to_interval_bounds(policy: nn.Module, param_bounds_l: list, param_bounds_u: list, verbose=False):
    """
    Project policy parameters to stay within the computed interval bounds.
    
    Args:
        policy: The policy network to project
        param_bounds_l: List of lower bounds for each parameter tensor
        param_bounds_u: List of upper bounds for each parameter tensor
        verbose: Whether to print projection statistics
    """
    total_clipped = 0
    total_params = 0
    
    with torch.no_grad():
        for i, param in enumerate(policy.net.parameters()):
            if i < len(param_bounds_l) and i < len(param_bounds_u):
                # Get bounds for this parameter (flatten to match param shape)
                p_l = param_bounds_l[i].view(param.shape).to(param.device)
                p_u = param_bounds_u[i].view(param.shape).to(param.device)
                
                # Count violations before clipping
                if verbose:
                    violations = ((param.data < p_l) | (param.data > p_u)).sum().item()
                    total_clipped += violations
                    total_params += param.numel()
                
                # Clamp parameters to bounds
                param.data.clamp_(min=p_l, max=p_u)
    
    if verbose and total_params > 0:
        print(f"  Projected {total_clipped}/{total_params} parameters ({100*total_clipped/total_params:.2f}%)")


# =========================
# REINFORCE (episodic) with baseline + entropy
# =========================
class Policy(nn.Module):
    def __init__(self, nS: int, nA: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nS, hidden), nn.ReLU(),
            nn.Linear(hidden, nA)
        )

    def forward(self, s_idx: torch.Tensor):
        # s_idx: [B] long -> one-hot -> logits
        nS = self.net[0].in_features
        x = torch.nn.functional.one_hot(s_idx, num_classes=nS).float()
        logits = self.net(x)
        return torch.distributions.Categorical(logits=logits)


@dataclass
class TrainCfg:
    episodes: int = 4000
    gamma: float = 0.99
    lr: float = 3e-3
    entropy_coef: float = 0.01
    baseline_beta: float = 0.9  # running exponential mean of returns
    max_steps: int = 60
    shaping_beta: float = 0.0   # 0 disables shaping
    # ---- PGD controls ----
    pgd_eps: float | None = None     # e.g., 1.0 (set None or 0.0 to disable)
    pgd_norm: str = "l2"             # "l2" or "linf"
    pgd_project_every: int = 1       # project every k episodes
    param_clip: float | None = None  # optional box-clip after projection
    grad_clip: float | None = None   # optional global grad-norm clip (e.g., 1.0)


def train_reinforce(env: ReachAvoidGrid, cfg: TrainCfg, pretrained_policy = None, device="cpu", log_every=200, 
                   param_bounds_l=None, param_bounds_u=None):
    nS = env.observation_space.n
    nA = env.action_space.n
    if pretrained_policy is None:
        policy = Policy(nS, nA, hidden=64).to(device)
    else:
        policy = pretrained_policy.to(device)
    opt = optim.Adam(policy.parameters(), lr=cfg.lr)

    # --- PGD: choose reference center (pretrained or initial snapshot) ---
    if pretrained_policy is not None:
        center_params = _params_snapshot(pretrained_policy)  # center = provided policy
    else:
        center_params = _params_snapshot(policy)             # center = initial θ₀

    running_baseline = 0.0
    success_window = []

    # precompute goal coordinates for shaping
    goal_rc = env.goal
    W = env.W

    for ep in range(cfg.episodes):
        logps, rewards, entropies, states = [], [], [], []
        s, _ = env.reset()
        for t in range(cfg.max_steps):
            s_t = torch.tensor([s], dtype=torch.long, device=device)
            dist = policy(s_t)
            a = dist.sample()
            logps.append(dist.log_prob(a).squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
            states.append(s)

            s_next, r_env, terminated, truncated, _ = env.step(int(a.item()))

            # potential-based shaping F = gamma*phi(s') - phi(s)
            if cfg.shaping_beta != 0.0:
                phi_s = manhattan_phi(s, W, goal_rc)
                phi_sn = manhattan_phi(s_next, W, goal_rc)
                shaping = cfg.gamma * phi_sn - phi_s
                r = r_env + cfg.shaping_beta * shaping
            else:
                r = r_env

            rewards.append(float(r))
            s = s_next
            if terminated or truncated:
                break

        # discounted returns
        G = 0.0
        returns = []
        for r in reversed(rewards):
            G = r + cfg.gamma * G
            returns.append(G)
        returns.reverse()

        # baseline (running mean of returns)
        if returns:
            ep_return = float(sum(rewards))  # same as episodic return
            running_baseline = cfg.baseline_beta * running_baseline + (1 - cfg.baseline_beta) * (np.mean(returns))

        if returns:
            R = torch.tensor(returns, dtype=torch.float32, device=device)
            b = torch.tensor(running_baseline, dtype=torch.float32, device=device)
            adv = R - b

            logps_t = torch.stack(logps)
            ent_t = torch.stack(entropies)

            policy_loss = -(logps_t * adv).sum() - cfg.entropy_coef * ent_t.sum()
            loss = policy_loss

            opt.zero_grad()
            loss.backward()
            # optional gradient clipping (not part of PGD, but stabilizes REINFORCE)
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
            opt.step()
            opt.step()

            # ---- PGD projection step (parameter space) ----
            if param_bounds_l is not None and param_bounds_u is not None:
                # Use interval bounds from Rashomon set
                verbose_projection = (ep < 5 or (ep + 1) % log_every == 0)  # Show details for first few episodes
                _project_to_interval_bounds(policy, param_bounds_l, param_bounds_u, verbose=verbose_projection)
            elif cfg.pgd_eps is not None and cfg.pgd_eps > 0 and ((ep + 1) % max(1, cfg.pgd_project_every) == 0):
                # Fallback to L2 ball projection
                _project_to_ball(policy, center_params, eps=cfg.pgd_eps, norm=cfg.pgd_norm)
                _param_box_clip(policy, cfg.param_clip)

        # crude success metric: +1 if episode ended at goal (env gives +1 last step when goal reached)
        succeeded = 1.0 if (rewards and rewards[-1] >= 1.0) else 0.0
        success_window.append(succeeded)
        if len(success_window) > 100:
            success_window.pop(0)

        if (ep + 1) % log_every == 0:
            print(f"[REINFORCE] ep {ep+1:4d}/{cfg.episodes} | "
                  f"len={len(rewards):2d} | return={sum(rewards):+.3f} | "
                  f"success(100ep)={np.mean(success_window)*100:.1f}%")

    # greedy policy for evaluation
    def greedy_policy_fn(s: int) -> int:
        with torch.no_grad():
            s_t = torch.tensor([s], dtype=torch.long, device=device)
            dist = policy(s_t)
            return int(torch.argmax(dist.logits, dim=-1).item())

    return greedy_policy_fn, policy


def evaluate(env: gym.Env, policy_fn, n_episodes=200, seed=123) -> float:
    rng = np.random.default_rng(seed)
    wins = 0
    for _ in range(n_episodes):
        s, _ = env.reset(seed=int(rng.integers(0, 10_000)))
        done = False
        while not done:
            a = policy_fn(s)
            s, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            if terminated and r >= 1.0:
                wins += 1
                break
    return wins / n_episodes

### PGD helpers
# ---- helpers: take a snapshot and project params ----
def _params_snapshot(model: nn.Module):
    return [p.detach().clone() for p in model.parameters() if p.requires_grad]

@torch.no_grad()
def _project_to_ball(model: nn.Module, center_tensors, eps: float, norm: str = "l2"):
    if eps is None or eps <= 0:
        return
    # gather deltas and compute global norm
    if norm == "l2":
        total_sq = 0.0
        deltas = []
        for p, c in zip(model.parameters(), center_tensors):
            if not p.requires_grad:
                deltas.append(None); continue
            d = p.data - c
            deltas.append(d)
            total_sq += float(torch.sum(d * d).item())
        total_norm = np.sqrt(total_sq)
        if total_norm > eps and total_norm > 0:
            scale = eps / total_norm
            for p, c, d in zip(model.parameters(), center_tensors, deltas):
                if d is None: continue
                p.data.copy_(c + d * scale)
    elif norm == "linf":
        for p, c in zip(model.parameters(), center_tensors):
            if not p.requires_grad: continue
            p.data.copy_(torch.clamp(p.data, min=c - eps, max=c + eps))
    else:
        raise ValueError(f"Unsupported pgd_norm={norm!r}; use 'l2' or 'linf'.")

@torch.no_grad()
def _param_box_clip(model: nn.Module, box: float | None):
    if box is None or box <= 0:
        return
    for p in model.parameters():
        if p.requires_grad:
            p.data.clamp_(-box, box)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=4000)
    parser.add_argument("--slip", type=float, default=0.0)
    parser.add_argument("--step_penalty", type=float, default=0.0)
    parser.add_argument("--lava_penalty", type=float, default=0.0)
    parser.add_argument("--shaping_beta", type=float, default=0.0, help="0 disables shaping; 0.2–0.5 often helps")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()


    ### Task 1
    H = W = args.size
    # A simple vertical lava wall with a gap—classic reach-avoid layout
    lava = [(r, W // 2) for r in range(1, H - 1)]  # leave top/bottom as potential gaps
    # open a gap to make it solvable
    if H >= 5:
        gap_r = H // 2
        lava = [cell for cell in lava if cell[0] != gap_r]

    env = ReachAvoidGrid(
        height=H, width=W,
        start=(0, 0),
        goal=(H - 1, W - 1),
        lava=lava,
        slip=args.slip,
        step_penalty=args.step_penalty,
        lava_penalty=args.lava_penalty,
        max_steps=H * W + 10,
        seed=args.seed,
    )

    cfg = TrainCfg(
        episodes=args.episodes,
        shaping_beta=args.shaping_beta,
    )
    policy_fn, policy_network = train_reinforce(env, cfg, device="cpu")

    sr = evaluate(env, policy_fn, n_episodes=300, seed=args.seed + 7)
    print(f"\nEvaluation success rate (Task 1): {sr*100:.1f}% over 300 episodes")

    ### Calculate safe action set for each state
    safe_actions = {}
    for s in range(env.nS):
        r, c = divmod(s, env.W)
        actions = []
        for a in range(env.nA):
            nr, nc = r, c
            if a == 0: nr -= 1        # UP
            elif a == 1: nc += 1      # RIGHT
            elif a == 2: nr += 1      # DOWN
            else: nc -= 1             # LEFT
            if not env._in_bounds(nr, nc):
                nr, nc = r, c  # stays put if hitting wall
            if (nr, nc) not in env.lava:
                actions.append(a)
        safe_actions[s] = actions

    # Optional: print a concise summary
    print("\nSafe actions per state (state_index: [actions])")
    for s in range(env.nS):
        print(f"{s}: {safe_actions[s]}")

    ### Build a dataset in which for each state, a safe action is taken
    # Build (state, action) supervised dataset: pick one safe action per state
    state_action_dataset = []
    for state_idx, action_lst in safe_actions.items():
        if not action_lst:
            continue
        cur_action = random.choice(action_lst)  # label
        state_action_dataset.append((state_idx, cur_action))

    print(f"Number of samples: {len(state_action_dataset)}")
    print("First 5 samples (one-hot index -> action):")
    for s, a in state_action_dataset[:5]:
        print(s, "->", a)

    ### Compute Rashomon set
    # Convert state indices to one-hot vectors since we're using policy_network.net directly
    state_indices = torch.tensor([s for s, _ in state_action_dataset], dtype=torch.long)
    nS = env.observation_space.n
    states = torch.nn.functional.one_hot(state_indices, num_classes=nS).float()
    actions = torch.tensor([a for _, a in state_action_dataset], dtype=torch.long)

    state_action_torch_dataset = TensorDataset(states, actions)
    state_action_loader = DataLoader(state_action_torch_dataset, batch_size=8, shuffle=True)

    print(f"Dataset size: {len(state_action_torch_dataset)}")
    print("States shape:", states.shape, "Actions shape:", actions.shape)

    interval_trainer = IntervalTrainer(
        model=policy_network.net, # policy network's Sequential part
        seed=2025,
    )

    interval_trainer.compute_rashomon_set(
        dataset=state_action_torch_dataset, # states and safe actions; provide one sample per state
        )

    ### Task 2: Train with parameter bounds constraint
    # This will use projected gradient descent to keep parameters within
    # the computed Rashomon set bounds, ensuring certified safety properties
    new_env = ReachAvoidGrid(
        height=H, width=W,
        start=(0, 0),
        goal=(H - 2, W - 2),
        lava=None,
        slip=args.slip,
        step_penalty=args.step_penalty,
        lava_penalty=args.lava_penalty,
        max_steps=H * W + 10,
        seed=args.seed,
    )
    # Extract parameter bounds from the interval trainer
    param_bounds_l = [bound.detach().cpu() for bound in interval_trainer.bounds[0].param_l]
    param_bounds_u = [bound.detach().cpu() for bound in interval_trainer.bounds[0].param_u]
    
    print(f"Using parameter bounds for projected gradient descent:")
    print(f"Number of parameter tensors: {len(param_bounds_l)}")
    for i, (p_l, p_u) in enumerate(zip(param_bounds_l, param_bounds_u)):
        width = (p_u - p_l).abs().mean().item()
        print(f"  Parameter {i}: shape={p_l.shape}, avg_width={width:.6f}")
    
    new_policy_fn, new_policy_network = train_reinforce(
        new_env, cfg, 
        pretrained_policy=policy_network, 
        device="cpu",
        param_bounds_l=param_bounds_l,
        param_bounds_u=param_bounds_u
    )
    
    # Verify final policy parameters are within bounds
    print("\nFinal parameter bounds verification:")
    with torch.no_grad():
        all_within_bounds = True
        for i, param in enumerate(new_policy_network.net.parameters()):
            if i < len(param_bounds_l) and i < len(param_bounds_u):
                p_l = param_bounds_l[i].view(param.shape).to(param.device)
                p_u = param_bounds_u[i].view(param.shape).to(param.device)
                
                within_bounds = ((param.data >= p_l) & (param.data <= p_u)).all()
                violations = ((param.data < p_l) | (param.data > p_u)).sum().item()
                
                print(f"  Parameter {i}: within_bounds={within_bounds}, violations={violations}/{param.numel()}")
                if not within_bounds:
                    all_within_bounds = False
        
        print(f"All parameters within bounds: {all_within_bounds}")

    new_sr = evaluate(new_env, new_policy_fn, n_episodes=300, seed=args.seed + 7)
    old_sr = evaluate(env, new_policy_fn, n_episodes=300, seed=args.seed + 7)
    print(f"\nEvaluation success rate (Task 2): {new_sr*100:.1f}% over 300 episodes")
    print(f"\nEvaluation success rate (Task 1): {old_sr*100:.1f}% over 300 episodes")

if __name__ == "__main__":
    main()
