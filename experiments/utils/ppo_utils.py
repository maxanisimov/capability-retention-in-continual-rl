import torch
from dataclasses import dataclass
import random
import numpy as np
import gymnasium as gym
import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from typing import Any

ActorParamBounds = list[torch.Tensor] | list[list[torch.Tensor]]

#### --- PPO Utilities --- ####
@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 100_000
    eval_episodes: int = 1_000
    rollout_steps: int = 2048
    update_epochs: int = 10
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Early stopping: checked at every periodic evaluation (every 10×rollout_steps steps).
    # Triggers when ALL non-None thresholds are simultaneously satisfied.
    # Deprecated: this flag is ignored; early stopping is enabled automatically
    # when at least one threshold below is set.
    early_stop: bool = False
    # Minimum PPO updates before early-stop checks:
    # - 0: include a pre-update check at step=0
    # - N>=1: start checking only after N PPO updates
    early_stop_min_steps: int = 0
    early_stop_reward_threshold: float | None = None   # stop if mean_reward >= threshold
    early_stop_failure_rate_threshold: float | None = None  # stop if failure_rate <= threshold
    early_stop_success_rate_threshold: float | None = None  # stop if success_rate >= threshold
    # Deprecated (kept for backwards compatibility, ignored in early-stop logic).
    early_stop_deterministic_total_reward_threshold: float | None = None
    early_stop_deterministic_eval_episodes: int = 1
    # Distance norm used to select the nearest convex Rashomon set during PGD projection.
    # Supported values: "l2" (default), "l1", "linf".
    pgd_projection_distance_norm: str = "l2"

def make_actor_critic(
    obs_dim: int, n_actions: int, 
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    continuous_actions: bool = False,
    ):
    """
    Create simple feedforward actor and critic networks.
    
    For discrete actions:
        Actor outputs logits for each action.
    For continuous actions:
        Actor outputs mean values for each action dimension.
        Log std is a separate learnable parameter.
    
    Args:
        obs_dim: Observation space dimension
        n_actions: Number of discrete actions OR dimension of continuous action space
        actor_warm_start: Optional pre-trained actor network
        critic_warm_start: Optional pre-trained critic network
        continuous_actions: Whether the action space is continuous
    """
    if actor_warm_start is not None:
        actor = copy.deepcopy(actor_warm_start)
        assert isinstance(actor, torch.nn.Sequential), "Warm-start actor must be nn.Sequential"
        last = actor[-1]
        assert hasattr(last, "out_features"), "Last layer must expose out_features"
        assert last.out_features == n_actions, "Actor output dim must match env action space"
    else:
        # actor = torch.nn.Sequential(
        #     torch.nn.Linear(obs_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, n_actions),
        # )
        actor = torch.nn.Sequential( # matching SB3 MlpPolicy architecture
            torch.nn.Linear(obs_dim, 256),
            # torch.nn.Tanh(),  # SB3 uses Tanh by default
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            # torch.nn.Tanh(),
            torch.nn.ReLU(),
            torch.nn.Linear(256, n_actions),
        )
    
    if critic_warm_start is not None:
        critic = copy.deepcopy(critic_warm_start)
        assert isinstance(critic, torch.nn.Sequential), "Warm-start critic must be nn.Sequential"
        last = critic[-1]
        assert hasattr(last, "out_features"), "Last layer must expose out_features"
        assert last.out_features == 1, "Critic output dim must be 1"
    else:
        # critic = torch.nn.Sequential(
        #     torch.nn.Linear(obs_dim, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, 1),
        # )
        critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 1),
        )
    
    # For continuous actions, add log_std parameter
    log_std = None
    if continuous_actions:
        log_std = torch.nn.Parameter(torch.zeros(n_actions))
    
    return actor, critic, log_std

def set_seed(env, seed: int):
    """
    Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
        env.action_space.seed(seed)

def evaluate_with_success(
        env: gym.Env, actor: nn.Sequential, episodes=10, seed: int = 2025,
        device: str = 'cpu', render_mode: str | None = None,
        deterministic: bool = True,
        log_std: torch.nn.Parameter | None = None
    ) -> tuple[float, float, float, float]:
    """
    Evaluate policy over episodes and return:
    (mean_total_reward, std_total_reward, failure_rate, success_rate).

    ``failure_rate`` and ``success_rate`` are episode-level rates:
    - failure episode: any step reports unsafe (`safe=False` or `cost>0`)
    - success episode: any step reports `is_success=True`
    """
    if int(episodes) <= 0:
        return 0.0, 0.0, 0.0, 0.0
    assert render_mode in (None, 'rgb_array')
    if deterministic:
        assert log_std is None, "log_std should be None for deterministic evaluation"
    actor.eval()
    scores = []
    failures = 0
    successes = 0
    
    # Determine if actions are continuous
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    
    for episode_num in range(episodes):
        obs, _ = env.reset(seed=seed*episode_num)
        episodic_reward = 0.0
        episode_failed = False
        episode_success = False
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                if continuous_actions:
                    # Continuous actions: use mean from actor
                    if deterministic:
                        action = actor(obs_t).cpu().numpy()[0]
                    else:
                        mean = actor(obs_t)
                        std = torch.exp(log_std) # type: ignore
                        dist = torch.distributions.Normal(mean, std)
                        action = dist.sample().cpu().numpy()[0]
                    action = np.clip(action, env.action_space.low, env.action_space.high) # type: ignore
                else:
                    # Discrete actions
                    logits = actor(obs_t)
                    if deterministic:
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs=probs)
                        action = dist.sample().item()
            
            obs, reward, terminated, truncated, info = env.step(action)
            if render_mode == 'rgb_array':
                plt.imshow(env.render()) # type: ignore
                plt.axis('off')
                plt.show()
            # Episode-level safety/success tracking.
            is_safe = info.get('safe', None)
            if is_safe is None:
                # For Safety Gymnasium: cost > 0 means unsafe
                cost = info.get('cost', 0)
                is_safe = (cost == 0)
            if not is_safe:
                episode_failed = True
            episode_success = episode_success or bool(info.get("is_success", False))
            episodic_reward += reward # type: ignore
            done = terminated or truncated
        scores.append(episodic_reward)
        failures += int(episode_failed)
        successes += int(episode_success)
    actor.train()
    avg_total_reward = float(np.mean(scores))
    std_total_reward = float(np.std(scores))
    failure_rate = failures / episodes
    success_rate = successes / episodes
    return avg_total_reward, std_total_reward, failure_rate, success_rate


def evaluate(
        env: gym.Env, actor: nn.Sequential, episodes=10, seed: int = 2025,
        device: str = 'cpu', render_mode: str | None = None,
        deterministic: bool = True,
        log_std: torch.nn.Parameter | None = None
    ):
    """
    Backward-compatible wrapper returning:
    (mean_total_reward, std_total_reward, failure_rate).
    """
    mean_r, std_r, failure_rate, _ = evaluate_with_success(
        env=env,
        actor=actor,
        episodes=episodes,
        seed=seed,
        device=device,
        render_mode=render_mode,
        deterministic=deterministic,
        log_std=log_std,
    )
    return mean_r, std_r, failure_rate


def _extract_action_mask(info: dict | None, n_actions: int) -> np.ndarray:
    """Return a binary action mask from env info (1 = valid action)."""
    if info is None:
        return np.ones(n_actions, dtype=np.float32)
    raw_mask = info.get("action_mask", None)
    if raw_mask is None:
        return np.ones(n_actions, dtype=np.float32)
    mask = np.asarray(raw_mask, dtype=np.float32).reshape(-1)
    if mask.shape[0] != n_actions:
        raise ValueError(
            f"Expected action_mask with {n_actions} entries, got shape {mask.shape}.",
        )
    mask = (mask > 0).astype(np.float32)
    if mask.sum() <= 0:
        raise ValueError("action_mask has no valid actions (all zeros).")
    return mask


def _apply_action_mask_to_logits(
        logits: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
    """Mask invalid-action logits so Categorical never samples them."""
    if action_mask.ndim == 1:
        action_mask = action_mask.unsqueeze(0)
    valid = action_mask > 0.5
    # Large negative value instead of -inf to avoid inf/nan issues in some ops.
    invalid_logit = torch.full_like(logits, -1e9)
    return torch.where(valid, logits, invalid_logit)


def evaluate_masked_with_success(
        env: gym.Env,
        actor: nn.Sequential,
        episodes: int = 10,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
) -> tuple[float, float, float, float]:
    """Evaluate masked policy and return (mean_reward, std_reward, failure_rate, success_rate)."""
    if int(episodes) <= 0:
        return 0.0, 0.0, 0.0, 0.0
    actor.eval()
    scores = []
    failures = 0
    successes = 0

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError("evaluate_masked only supports Discrete action spaces.")
    n_actions = int(env.action_space.n)

    for episode_num in range(episodes):
        obs, info = env.reset(seed=seed * episode_num)
        action_mask = _extract_action_mask(info, n_actions)
        episodic_reward = 0.0
        episode_failed = False
        episode_success = False
        done = False
        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits = actor(obs_t)
                masked_logits = _apply_action_mask_to_logits(logits, mask_t)
                if deterministic:
                    action = int(torch.argmax(masked_logits, dim=-1).item())
                else:
                    dist = torch.distributions.Categorical(logits=masked_logits)
                    action = int(dist.sample().item())

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episodic_reward += reward # type: ignore

            is_safe = info.get("safe", None)
            if is_safe is None:
                cost = info.get("cost", 0)
                is_safe = (cost == 0)
            if not is_safe:
                episode_failed = True
            episode_success = episode_success or bool(info.get("is_success", False))

            if not done:
                action_mask = _extract_action_mask(info, n_actions)

        scores.append(episodic_reward)
        failures += int(episode_failed)
        successes += int(episode_success)

    actor.train()
    avg_total_reward = float(np.mean(scores))
    std_total_reward = float(np.std(scores))
    failure_rate = failures / episodes
    success_rate = successes / episodes
    return avg_total_reward, std_total_reward, failure_rate, success_rate


def evaluate_masked(
        env: gym.Env,
        actor: nn.Sequential,
        episodes: int = 10,
        seed: int = 2025,
        device: str | torch.device = "cpu",
        deterministic: bool = True,
) -> tuple[float, float, float]:
    """Backward-compatible wrapper returning (mean_reward, std_reward, failure_rate)."""
    mean_r, std_r, failure_rate, _ = evaluate_masked_with_success(
        env=env,
        actor=actor,
        episodes=episodes,
        seed=seed,
        device=device,
        deterministic=deterministic,
    )
    return mean_r, std_r, failure_rate


def _early_stop_thresholds_satisfied(
    cfg: PPOConfig,
    *,
    mean_reward: float,
    failure_rate: float,
    success_rate: float,
) -> tuple[bool, bool, bool]:
    reward_ok = (
        cfg.early_stop_reward_threshold is None
        or mean_reward >= cfg.early_stop_reward_threshold
    )
    failure_ok = (
        cfg.early_stop_failure_rate_threshold is None
        or failure_rate <= cfg.early_stop_failure_rate_threshold
    )
    success_ok = (
        cfg.early_stop_success_rate_threshold is None
        or success_rate >= cfg.early_stop_success_rate_threshold
    )
    return reward_ok, failure_ok, success_ok


def _has_any_early_stop_threshold(cfg: PPOConfig) -> bool:
    return (
        cfg.early_stop_reward_threshold is not None
        or cfg.early_stop_failure_rate_threshold is not None
        or cfg.early_stop_success_rate_threshold is not None
    )


def _is_early_stop_enabled(cfg: PPOConfig) -> bool:
    return _has_any_early_stop_threshold(cfg)


def _warn_if_deprecated_early_stop_settings(cfg: PPOConfig) -> None:
    if (
        cfg.early_stop_deterministic_total_reward_threshold is not None
        or cfg.early_stop_deterministic_eval_episodes != 1
    ):
        warnings.warn(
            "PPOConfig.early_stop_deterministic_total_reward_threshold and "
            "PPOConfig.early_stop_deterministic_eval_episodes are deprecated and "
            "ignored. Use early_stop_reward_threshold / early_stop_failure_rate_threshold / "
            "early_stop_success_rate_threshold instead.",
            FutureWarning,
            stacklevel=2,
        )

    if cfg.early_stop:
        warnings.warn(
            "PPOConfig.early_stop is deprecated and ignored. Early stopping is now "
            "enabled automatically when any threshold is set "
            "(early_stop_reward_threshold / early_stop_failure_rate_threshold / "
            "early_stop_success_rate_threshold).",
            FutureWarning,
            stacklevel=2,
        )


def _validate_and_prepare_param_interval_bounds(
    *,
    actor_params: list[torch.nn.Parameter],
    actor_param_bounds_l: ActorParamBounds,
    actor_param_bounds_u: ActorParamBounds,
    device: torch.device,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    """
    Normalize PGD bounds into set-major interval lists.

    Accepted formats:
      1) Single interval (backward-compatible):
         actor_param_bounds_l/u: list[Tensor] with one lower/upper tensor per parameter.
      2) Multiple intervals:
         - interval-major: list[list[Tensor]] where outer index is interval and
           inner index is parameter.
         - parameter-major: list[list[Tensor]] where outer index is parameter and
           inner index is interval.

    Returns:
      (bounds_l_sets, bounds_u_sets) where each is set-major:
        bounds_*_sets[set_idx][param_idx] -> Tensor
    """
    n_params = len(actor_params)

    def _is_tensor_list(x: object) -> bool:
        return isinstance(x, list) and all(isinstance(v, torch.Tensor) for v in x)

    def _is_nested_tensor_list(x: object) -> bool:
        return (
            isinstance(x, list)
            and all(isinstance(v, list) for v in x)
            and all(all(isinstance(t, torch.Tensor) for t in v) for v in x)
        )

    actor_shapes = [tuple(p.shape) for p in actor_params]

    # Single-interval format: list[Tensor], list[Tensor]
    if _is_tensor_list(actor_param_bounds_l) and _is_tensor_list(actor_param_bounds_u):
        if len(actor_param_bounds_l) != n_params or len(actor_param_bounds_u) != n_params:
            raise ValueError(
                "Single-interval PGD bounds must provide one tensor per actor parameter. "
                f"Expected {n_params}, got lower={len(actor_param_bounds_l)} upper={len(actor_param_bounds_u)}.",
            )

        set_l: list[torch.Tensor] = []
        set_u: list[torch.Tensor] = []
        for p_idx, (lb, ub, expected_shape) in enumerate(
            zip(actor_param_bounds_l, actor_param_bounds_u, actor_shapes),
        ):
            if tuple(lb.shape) != expected_shape or tuple(ub.shape) != expected_shape:
                raise ValueError(
                    f"PGD bound shape mismatch at param index {p_idx}: "
                    f"expected={expected_shape}, lower={tuple(lb.shape)}, upper={tuple(ub.shape)}",
                )
            set_l.append(lb.to(device))
            set_u.append(ub.to(device))
        return [set_l], [set_u]

    # Multi-interval formats: nested lists.
    if not (_is_nested_tensor_list(actor_param_bounds_l) and _is_nested_tensor_list(actor_param_bounds_u)):
        raise TypeError(
            "PGD bounds must be either list[Tensor] (single interval) or list[list[Tensor]] "
            "(multiple intervals).",
        )

    if len(actor_param_bounds_l) != len(actor_param_bounds_u):
        raise ValueError(
            "Lower/upper nested PGD bounds outer lengths do not match: "
            f"lower={len(actor_param_bounds_l)}, upper={len(actor_param_bounds_u)}.",
        )
    if len(actor_param_bounds_l) == 0:
        raise ValueError("Nested PGD bounds must contain at least one interval/parameter group.")

    def _is_valid_interval_major(
        nested_l: list[list[torch.Tensor]],
        nested_u: list[list[torch.Tensor]],
    ) -> bool:
        for int_l, int_u in zip(nested_l, nested_u):
            if len(int_l) != n_params or len(int_u) != n_params:
                return False
            for p_idx, expected_shape in enumerate(actor_shapes):
                if tuple(int_l[p_idx].shape) != expected_shape or tuple(int_u[p_idx].shape) != expected_shape:
                    return False
        return True

    def _is_valid_parameter_major(
        nested_l: list[list[torch.Tensor]],
        nested_u: list[list[torch.Tensor]],
    ) -> bool:
        if len(nested_l) != n_params or len(nested_u) != n_params:
            return False
        if len(nested_l[0]) == 0:
            return False
        n_intervals = len(nested_l[0])
        for p_idx, (param_l, param_u, expected_shape) in enumerate(
            zip(nested_l, nested_u, actor_shapes),
        ):
            if len(param_l) != n_intervals or len(param_u) != n_intervals:
                return False
            for int_idx in range(n_intervals):
                if tuple(param_l[int_idx].shape) != expected_shape or tuple(param_u[int_idx].shape) != expected_shape:
                    return False
        return True

    interval_major_ok = _is_valid_interval_major(actor_param_bounds_l, actor_param_bounds_u)
    parameter_major_ok = _is_valid_parameter_major(actor_param_bounds_l, actor_param_bounds_u)

    # Prefer interval-major when both are syntactically possible.
    if interval_major_ok:
        bounds_l_sets: list[list[torch.Tensor]] = []
        bounds_u_sets: list[list[torch.Tensor]] = []
        for int_l, int_u in zip(actor_param_bounds_l, actor_param_bounds_u):
            bounds_l_sets.append([t.to(device) for t in int_l])
            bounds_u_sets.append([t.to(device) for t in int_u])
        return bounds_l_sets, bounds_u_sets

    if parameter_major_ok:
        n_intervals = len(actor_param_bounds_l[0])
        bounds_l_sets = []
        bounds_u_sets = []
        for int_idx in range(n_intervals):
            set_l = [actor_param_bounds_l[p_idx][int_idx].to(device) for p_idx in range(n_params)]
            set_u = [actor_param_bounds_u[p_idx][int_idx].to(device) for p_idx in range(n_params)]
            bounds_l_sets.append(set_l)
            bounds_u_sets.append(set_u)
        return bounds_l_sets, bounds_u_sets

    raise ValueError(
        "Nested PGD bounds do not match either supported layout:\n"
        "- interval-major: bounds[interval][parameter]\n"
        "- parameter-major: bounds[parameter][interval]",
    )


def _project_actor_to_interval_union(
    actor_params: list[torch.nn.Parameter],
    bounds_l_sets: list[list[torch.Tensor]],
    bounds_u_sets: list[list[torch.Tensor]],
    distance_norm: str = "l2",
) -> int:
    """
    Project actor parameters onto the nearest convex Rashomon set (box) in full parameter space.

    IMPORTANT: This preserves set coupling across parameters.
    """
    norm = str(distance_norm).strip().lower()
    if norm in {"l_inf", "inf", "infty", "infinity"}:
        norm = "linf"
    if norm not in {"l2", "l1", "linf"}:
        raise ValueError(
            "Unsupported projection distance norm. "
            f"Got '{distance_norm}', expected one of: 'l2', 'l1', 'linf'.",
        )

    if len(bounds_l_sets) != len(bounds_u_sets):
        raise ValueError(
            "Set-major lower/upper bounds length mismatch: "
            f"lower_sets={len(bounds_l_sets)}, upper_sets={len(bounds_u_sets)}",
        )
    if len(bounds_l_sets) == 0:
        raise ValueError("At least one convex Rashomon set is required for PGD projection.")

    n_params = len(actor_params)
    for set_idx, (set_l, set_u) in enumerate(zip(bounds_l_sets, bounds_u_sets)):
        if len(set_l) != n_params or len(set_u) != n_params:
            raise ValueError(
                f"Rashomon set {set_idx} must provide bounds for all parameters: "
                f"expected={n_params}, lower={len(set_l)}, upper={len(set_u)}",
            )

    best_set_idx: int | None = None
    best_distance = float("inf")
    best_projected: list[torch.Tensor] | None = None
    best_n_projected = 0

    for set_idx, (set_l, set_u) in enumerate(zip(bounds_l_sets, bounds_u_sets)):
        distance = 0.0
        n_projected = 0
        projected_for_set: list[torch.Tensor] = []

        for p_idx, (param, lb, ub) in enumerate(zip(actor_params, set_l, set_u)):
            p = param.data
            if tuple(lb.shape) != tuple(p.shape) or tuple(ub.shape) != tuple(p.shape):
                raise ValueError(
                    f"Shape mismatch at set {set_idx}, param {p_idx}: "
                    f"param={tuple(p.shape)}, lower={tuple(lb.shape)}, upper={tuple(ub.shape)}",
                )

            projected = torch.maximum(torch.minimum(p, ub), lb)
            outside = (p < lb) | (p > ub)
            n_projected += int(outside.sum().item())
            delta = projected - p
            if norm == "l2":
                # Compare using squared L2 distance (equivalent ordering to L2).
                distance += float(torch.sum(delta * delta).item())
            elif norm == "l1":
                distance += float(torch.sum(torch.abs(delta)).item())
            else:  # norm == "linf"
                distance = max(distance, float(torch.max(torch.abs(delta)).item()))
            projected_for_set.append(projected)

            # Optional pruning: once this candidate is already worse, no need to keep accumulating.
            if distance > best_distance:
                break

        if distance < best_distance:
            best_distance = distance
            best_set_idx = set_idx
            best_projected = projected_for_set
            best_n_projected = n_projected
            if best_distance == 0.0:
                break

    if best_set_idx is None or best_projected is None:
        raise RuntimeError("Failed to select a nearest Rashomon set during PGD projection.")

    for param, projected in zip(actor_params, best_projected):
        param.data.copy_(projected)
    return int(best_n_projected)


def _build_eval_metrics_dict(
    *,
    mean_reward: float,
    std_reward: float,
    failure_rate: float,
    success_rate: float,
) -> dict[str, float]:
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "failure_rate": float(failure_rate),
        "success_rate": float(success_rate),
    }


def _snapshot_actor_policy_params(
    actor: nn.Sequential,
    log_std: torch.nn.Parameter | None = None,
) -> dict[str, torch.Tensor]:
    params_snapshot: dict[str, torch.Tensor] = {
        name: value.detach().cpu().clone()
        for name, value in actor.state_dict().items()
    }
    if log_std is not None:
        params_snapshot["log_std"] = log_std.detach().cpu().clone()
    return params_snapshot


def _write_eval_policy_snapshot(
    *,
    params_snapshot: dict[str, torch.Tensor],
    save_dir: Path,
    checkpoint_idx: int,
    eval_phase: str,
    timestep: int,
    update_idx: int,
) -> Path:
    filename = (
        f"eval_params_{checkpoint_idx:05d}_{eval_phase}_"
        f"step_{int(timestep)}_update_{int(update_idx)}.pt"
    )
    save_path = save_dir / filename
    torch.save(params_snapshot, save_path)
    return save_path


def _append_eval_checkpoint_record(
    eval_checkpoint_records: list[dict[str, Any]],
    *,
    actor: nn.Sequential,
    log_std: torch.nn.Parameter | None,
    timestep: int,
    update_idx: int,
    eval_phase: str,
    mean_reward: float,
    std_reward: float,
    failure_rate: float,
    success_rate: float,
    save_params_to_disk: bool,
    eval_params_save_dir: Path | None,
) -> None:
    metrics = _build_eval_metrics_dict(
        mean_reward=mean_reward,
        std_reward=std_reward,
        failure_rate=failure_rate,
        success_rate=success_rate,
    )
    record: dict[str, Any] = {
        "timestep": int(timestep),
        "update_idx": int(update_idx),
        "eval_phase": eval_phase,
        "metrics": metrics,
    }
    params_snapshot = _snapshot_actor_policy_params(actor=actor, log_std=log_std)
    if save_params_to_disk:
        if eval_params_save_dir is None:
            raise ValueError("eval_params_save_dir must be provided when save_params_to_disk=True.")
        params_path = _write_eval_policy_snapshot(
            params_snapshot=params_snapshot,
            save_dir=eval_params_save_dir,
            checkpoint_idx=len(eval_checkpoint_records),
            eval_phase=eval_phase,
            timestep=timestep,
            update_idx=update_idx,
        )
        record["params_path"] = str(params_path)
    else:
        record["params"] = params_snapshot
    eval_checkpoint_records.append(record)

def ppo_train(
    env: gym.Env, cfg: PPOConfig, 
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    actor_param_bounds_l: ActorParamBounds | None = None,
    actor_param_bounds_u: ActorParamBounds | None = None,
    early_stop_eval_env: gym.Env | None = None,
    return_training_data: bool = False,
    track_eval_params: bool = False,
    return_eval_checkpoint_records: bool = False,
    save_eval_params_to_disk: bool = False,
    eval_params_save_dir: str | Path | None = None,
):
    """
    Train a PPO agent. If actor_param_bounds_l and actor_param_bounds_u are provided,
    uses projected gradient descent to keep actor parameters within bounds.
    
    Args:
        env: Gymnasium environment
        cfg: PPO configuration
        actor_warm_start: Optional pre-trained actor network
        critic_warm_start: Optional pre-trained critic network
        actor_param_bounds_l: Optional lower bounds for actor parameters (for PGD).
            Accepts:
              - list[Tensor]: single interval per parameter (backward-compatible)
              - list[list[Tensor]]: multiple intervals in interval-major or parameter-major layout
        actor_param_bounds_u: Optional upper bounds for actor parameters (for PGD).
            Must match actor_param_bounds_l structure.
        early_stop_eval_env: Optional environment for periodic evaluation and early-stopping checks.
            If None, uses the training env. This is useful when training uses reward shaping but
            early-stopping should use the original sparse reward.
        return_training_data: If True, returns state-action pairs collected during training.
        track_eval_params: If True, track policy-parameter snapshots only at evaluation checkpoints.
        return_eval_checkpoint_records: If True, append evaluation checkpoint records to the returned tuple.
        save_eval_params_to_disk: If True, save eval-checkpoint parameters to disk and store paths in records.
        eval_params_save_dir: Directory used for per-checkpoint parameter files when
            save_eval_params_to_disk=True.
        
    Returns:
        Base returns:
            - Discrete:
                - without return_training_data: (actor, critic)
                - with return_training_data: (actor, critic, training_data)
            - Continuous:
                - without return_training_data: (actor, critic, log_std)
                - with return_training_data: (actor, critic, log_std, training_data)

        If return_eval_checkpoint_records=True, appends eval_checkpoint_records as the
        final return item in the corresponding base return above.

        training_data is a dict containing:
            - 'states': numpy array of shape (N, obs_dim) containing all states visited
            - 'actions': numpy array of shape (N,) containing all actions taken
            - 'terminated': numpy array of shape (N,) containing all termination flags for state-action pairs (1 if terminated, 0 otherwise)
            - 'truncated': numpy array of shape (N,) containing all truncation flags for state-action pairs (1 if truncated, 0 otherwise)
            - 'safe': numpy array of shape (N,) containing all safety flags for state-action pairs (1 if safe, 0 if unsafe)

        eval_checkpoint_records is a list of dictionaries with:
            - 'timestep': global timestep at evaluation
            - 'update_idx': PPO update index at evaluation
            - 'eval_phase': one of {'pre_update', 'periodic', 'final'}
            - 'metrics': dict with mean/std reward and failure/success rates
            - either 'params' (in-memory snapshot) or 'params_path' (disk path)
    """
    # env_kwargs = cfg.env_kwargs if cfg.env_kwargs is not None else {}
    # env = gym.make(cfg.env_id, **env_kwargs)
    set_seed(env, cfg.seed)
    _warn_if_deprecated_early_stop_settings(cfg)
    early_stop_enabled = _is_early_stop_enabled(cfg)
    eval_checkpoint_records: list[dict[str, Any]] = []
    eval_params_dir: Path | None = None
    if save_eval_params_to_disk:
        if eval_params_save_dir is None:
            raise ValueError(
                "eval_params_save_dir must be provided when save_eval_params_to_disk=True.",
            )
        eval_params_dir = Path(eval_params_save_dir)
        eval_params_dir.mkdir(parents=True, exist_ok=True)

    # Determine action space type
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    obs_dim = env.observation_space.shape[0] if isinstance(env.observation_space, gym.spaces.Box) else env.observation_space.n # type: ignore
    n_actions = env.action_space.shape[0] if continuous_actions else env.action_space.n # type: ignore

    actor, critic, log_std = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions, # type: ignore
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        continuous_actions=continuous_actions
    )
    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    device = torch.device(cfg.device)
    actor.to(device)
    critic.to(device)
    if log_std is not None:
        log_std = log_std.to(device)

    # Check if we have parameter bounds for projected gradient descent
    if (actor_param_bounds_l is None) != (actor_param_bounds_u is None):
        raise ValueError(
            "PGD bounds must provide both actor_param_bounds_l and actor_param_bounds_u, or neither.",
        )
    use_pgd = (actor_param_bounds_l is not None and actor_param_bounds_u is not None)
    print('Use PGD:', use_pgd)
    bounds_l_sets = None
    bounds_u_sets = None
    if use_pgd:
        projection_distance_norm = str(cfg.pgd_projection_distance_norm)
        bounds_l_sets, bounds_u_sets = _validate_and_prepare_param_interval_bounds(
            actor_params=actor_params,
            actor_param_bounds_l=actor_param_bounds_l, # type: ignore[arg-type]
            actor_param_bounds_u=actor_param_bounds_u, # type: ignore[arg-type]
            device=device,
        )
        n_sets = len(bounds_l_sets)
        print(
            "Using projected gradient descent with convex-set projection "
            f"(num_sets={n_sets}, distance_norm={projection_distance_norm})",
        )

        # Ensure initial parameters are inside at least one Rashomon set.
        with torch.no_grad():
            init_projections = _project_actor_to_interval_union(
                actor_params,
                bounds_l_sets,
                bounds_u_sets,
                distance_norm=projection_distance_norm,
            )
        if init_projections > 0:
            print(f"Initial PGD projection count: {init_projections}")

    # Create optimizer with actor, critic, and optionally log_std parameters
    optimizer_params = [
        {"params": actor_params, "lr": cfg.lr},
        {"params": critic_params, "lr": cfg.lr},
    ]
    if log_std is not None:
        optimizer_params.append({"params": [log_std], "lr": cfg.lr})
    optimizer = torch.optim.Adam(optimizer_params)

    eval_env = early_stop_eval_env if early_stop_eval_env is not None else env

    obs, _ = env.reset(seed=cfg.seed)
    global_step = 0
    ppo_update_count = 0
    start_time = time.time()
    pgd_projections = 0  # Count total parameter projections for logging
    stop_training_early = False
    
    # Training data tracking
    if return_training_data:
        training_data = {
            'states': [],
            'actions': [],
            'terminated': [],
            'truncated': [],
            'safe': [],
            # 'rewards': []
        }

    # Initial pre-update early-stop check at step=0.
    if early_stop_enabled and cfg.early_stop_min_steps == 0:
        mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
            env=eval_env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        ) # type: ignore
        if track_eval_params:
            _append_eval_checkpoint_record(
                eval_checkpoint_records,
                actor=actor,
                log_std=log_std,
                timestep=global_step,
                update_idx=ppo_update_count,
                eval_phase="pre_update",
                mean_reward=mean_r,
                std_reward=std_r,
                failure_rate=failure_rate,
                success_rate=success_rate,
                save_params_to_disk=save_eval_params_to_disk,
                eval_params_save_dir=eval_params_dir,
            )

        # If eval reuses training env, restore initial rollout state for reproducibility.
        if eval_env is env:
            obs, _ = env.reset(seed=cfg.seed)

        print(
            f"Pre-update check | Steps=0 | meanR={mean_r:.1f} +/- {std_r:.1f} | "
            f"failure_rate={failure_rate:.2f} | success_rate={success_rate:.2f}",
        )
        reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
            cfg,
            mean_reward=mean_r,
            failure_rate=failure_rate,
            success_rate=success_rate,
        )
        if reward_ok and failure_ok and success_ok:
            print(
                "  [Early stop pre-update] step=0 | "
                f"meanR={mean_r:.2f} (threshold={cfg.early_stop_reward_threshold}) | "
                f"failure_rate={failure_rate:.2f} (threshold={cfg.early_stop_failure_rate_threshold}) | "
                f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})"
            )
            stop_training_early = True

    while global_step < cfg.total_timesteps and not stop_training_early:
        # Storage for rollout
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        if continuous_actions:
            act_buf = np.zeros((cfg.rollout_steps, n_actions), dtype=np.float32) # type: ignore
        else:
            act_buf = np.zeros((cfg.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)

        # Collect rollout
        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                value = critic(obs_t).squeeze(-1)
                if continuous_actions:
                    # Continuous actions: use Normal distribution
                    mean = actor(obs_t)
                    std = torch.exp(log_std) # type: ignore
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.sample()
                    logp = dist.log_prob(action).sum(dim=-1)  # Sum log probs across action dims
                    # Clip action to valid range
                    action_np = action.cpu().numpy()[0]
                    action_np = np.clip(action_np, env.action_space.low, env.action_space.high) # type: ignore
                    act = action_np
                else:
                    # Discrete actions: use Categorical distribution
                    logits = actor(obs_t)
                    dist = torch.distributions.Categorical(logits=logits)
                    action = dist.sample()
                    logp = dist.log_prob(action)
                    act = int(action.item())
            
            next_obs, reward, terminated, truncated, info = env.step(act)
            done = terminated or truncated
            
            # Collect state-action pairs if recording training data
            if return_training_data:
                training_data['states'].append(obs.copy()) # type: ignore
                training_data['actions'].append(act) # type: ignore
                training_data['terminated'].append(float(terminated)) # type: ignore
                training_data['truncated'].append(float(truncated)) # type: ignore
                # Safety tracking: check 'safe' flag (custom envs) or 'cost' (Safety Gymnasium)
                is_safe = info.get('safe', None)
                if is_safe is None:
                    # For Safety Gymnasium: cost == 0 means safe, cost > 0 means unsafe
                    cost = info.get('cost', 0)
                    is_safe = 1.0 if cost == 0 else 0.0
                training_data['safe'].append(float(is_safe)) # type: ignore
                # training_data['rewards'].append(float(reward)) # type: ignore
            
            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)
            val_buf[t] = float(value.item())

            obs = next_obs
            global_step += 1
            if done:
                obs, _ = env.reset()

            if global_step >= cfg.total_timesteps:
                # If we hit total_timesteps mid-rollout, cut here
                obs_buf = obs_buf[:t+1]
                act_buf = act_buf[:t+1]
                logp_buf = logp_buf[:t+1]
                rew_buf = rew_buf[:t+1]
                done_buf = done_buf[:t+1]
                val_buf = val_buf[:t+1]
                break

        # Bootstrap with value of last observation
        with torch.no_grad():
            last_val = critic(torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)).item()

        # Compute GAE advantages and returns
        T = len(rew_buf)
        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - done_buf[t]
            next_value = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        # Normalize advantages
        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        if continuous_actions:
            act_t = torch.tensor(act_buf, dtype=torch.float32, device=device)
        else:
            act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)

        # PPO updates
        batch_size = T
        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                if continuous_actions:
                    # Continuous actions
                    mean = actor(mb_obs)
                    std = torch.exp(log_std) # type: ignore
                    dist = torch.distributions.Normal(mean, std)
                    new_logp = dist.log_prob(mb_act).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                else:
                    # Discrete actions
                    logits = actor(mb_obs)
                    dist = torch.distributions.Categorical(logits=logits)
                    new_logp = dist.log_prob(mb_act)
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v = critic(mb_obs).squeeze(-1)
                v_loss = F.mse_loss(v, mb_ret)

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor_params + critic_params, cfg.max_grad_norm)
                optimizer.step()

                # Projected gradient descent: project actor parameters onto interval union.
                if use_pgd:
                    with torch.no_grad():
                        pgd_projections += _project_actor_to_interval_union(
                            actor_params,
                            bounds_l_sets, # type: ignore[arg-type]
                            bounds_u_sets, # type: ignore[arg-type]
                            distance_norm=projection_distance_norm, # type: ignore[arg-type]
                        )
        ppo_update_count += 1

        if global_step % (10 * cfg.rollout_steps) < cfg.rollout_steps:
            mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
                env=eval_env,
                actor=actor,
                device=device,
                episodes=cfg.eval_episodes,
                deterministic=True, # NOTE: eval always uses deterministic policy
            ) # type: ignore
            if track_eval_params:
                _append_eval_checkpoint_record(
                    eval_checkpoint_records,
                    actor=actor,
                    log_std=log_std,
                    timestep=global_step,
                    update_idx=ppo_update_count,
                    eval_phase="periodic",
                    mean_reward=mean_r,
                    std_reward=std_r,
                    failure_rate=failure_rate,
                    success_rate=success_rate,
                    save_params_to_disk=save_eval_params_to_disk,
                    eval_params_save_dir=eval_params_dir,
                )

            # When evaluation reuses the training env, reset to recover rollout state.
            if eval_env is env:
                obs, _ = env.reset()
            elapsed = time.time() - start_time
            log_msg = f"Steps={global_step} | meanR={mean_r:.1f} +/- {std_r:.1f} | elapsed={elapsed:.1f}s"
            log_msg += f" | failure_rate={failure_rate:.2f}"
            log_msg += f" | success_rate={success_rate:.2f}"
            if use_pgd:
                log_msg += f" | PGD projections={pgd_projections}"
            print(log_msg)

            # ── Early stopping ──────────────────────────────────────────
            if early_stop_enabled and ppo_update_count >= cfg.early_stop_min_steps:
                reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
                    cfg,
                    mean_reward=mean_r,
                    failure_rate=failure_rate,
                    success_rate=success_rate,
                )
                if reward_ok and failure_ok and success_ok:
                    print(
                        f"  [Early stop] updates={ppo_update_count} | step={global_step} | "
                        f"meanR={mean_r:.2f} (threshold={cfg.early_stop_reward_threshold}) | "
                        f"failure_rate={failure_rate:.2f} (threshold={cfg.early_stop_failure_rate_threshold}) | "
                        f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})"
                    )
                    break

    if cfg.eval_episodes is not None and cfg.eval_episodes > 0:
        # Final checks and evaluation
        mean_r, std_r, failure_rate, success_rate = evaluate_with_success(
            env=eval_env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        ) # type: ignore
        if track_eval_params:
            _append_eval_checkpoint_record(
                eval_checkpoint_records,
                actor=actor,
                log_std=log_std,
                timestep=global_step,
                update_idx=ppo_update_count,
                eval_phase="final",
                mean_reward=mean_r,
                std_reward=std_r,
                failure_rate=failure_rate,
                success_rate=success_rate,
                save_params_to_disk=save_eval_params_to_disk,
                eval_params_save_dir=eval_params_dir,
            )
        final_msg = f"Final evaluation over {cfg.eval_episodes} episodes: mean_reward={mean_r:.2f} +/- {std_r:.2f}"
        final_msg += f" | failure_rate={failure_rate:.2f}"
        final_msg += f" | success_rate={success_rate:.2f}"
        if use_pgd:
            final_msg += f" | Total PGD projections during training: {pgd_projections}"
        print(final_msg)

    env.close()
    if early_stop_eval_env is not None and early_stop_eval_env is not env:
        early_stop_eval_env.close()

    # Return results
    if return_training_data:
        # Convert lists to numpy arrays
        training_data['states'] = np.array(training_data['states']) # type: ignore
        training_data['actions'] = np.array(training_data['actions']) # type: ignore
        training_data['terminated'] = np.array(training_data['terminated']) # type: ignore
        training_data['truncated'] = np.array(training_data['truncated']) # type: ignore
        training_data['safe'] =  np.array(training_data['safe']) # type: ignore
        # training_data['rewards'] = np.array(training_data['rewards']) # type: ignore
        if continuous_actions:
            result = (actor, critic, log_std, training_data)
        else:
            result = (actor, critic, training_data)
    else:
        if continuous_actions:
            result = (actor, critic, log_std)
        else:
            result = (actor, critic)
    if return_eval_checkpoint_records:
        return (*result, eval_checkpoint_records)
    return result


def ppo_train_maskable(
    env: gym.Env,
    cfg: PPOConfig,
    actor_warm_start: nn.Sequential | None = None,
    critic_warm_start: nn.Sequential | None = None,
    actor_param_bounds_l: ActorParamBounds | None = None,
    actor_param_bounds_u: ActorParamBounds | None = None,
    return_training_data: bool = False,
):
    """
    Train PPO for discrete actions with invalid-action masking.

    This function reads ``info['action_mask']`` from the environment, where:
      - ``1`` means action is valid/executable
      - ``0`` means action is invalid and must not be sampled

    If an action mask is missing from ``info``, a full-valid mask is used.
    """
    set_seed(env, cfg.seed)
    _warn_if_deprecated_early_stop_settings(cfg)
    early_stop_enabled = _is_early_stop_enabled(cfg)

    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(
            "ppo_train_maskable only supports Discrete action spaces.",
        )

    obs_dim = (
        env.observation_space.shape[0]
        if isinstance(env.observation_space, gym.spaces.Box)
        else env.observation_space.n # type: ignore
    )
    n_actions = int(env.action_space.n)

    actor, critic, _ = make_actor_critic(
        obs_dim=obs_dim,
        n_actions=n_actions,
        actor_warm_start=actor_warm_start,
        critic_warm_start=critic_warm_start,
        continuous_actions=False,
    )
    actor_params = list(actor.parameters())
    critic_params = list(critic.parameters())
    device = torch.device(cfg.device)
    actor.to(device)
    critic.to(device)

    if (actor_param_bounds_l is None) != (actor_param_bounds_u is None):
        raise ValueError(
            "PGD bounds must provide both actor_param_bounds_l and actor_param_bounds_u, or neither.",
        )
    use_pgd = (actor_param_bounds_l is not None and actor_param_bounds_u is not None)
    print("Use PGD:", use_pgd)
    bounds_l_sets = None
    bounds_u_sets = None
    if use_pgd:
        projection_distance_norm = str(cfg.pgd_projection_distance_norm)
        bounds_l_sets, bounds_u_sets = _validate_and_prepare_param_interval_bounds(
            actor_params=actor_params,
            actor_param_bounds_l=actor_param_bounds_l, # type: ignore[arg-type]
            actor_param_bounds_u=actor_param_bounds_u, # type: ignore[arg-type]
            device=device,
        )
        n_sets = len(bounds_l_sets)
        print(
            "Using projected gradient descent with convex-set projection "
            f"(num_sets={n_sets}, distance_norm={projection_distance_norm})",
        )
        with torch.no_grad():
            init_projections = _project_actor_to_interval_union(
                actor_params,
                bounds_l_sets,
                bounds_u_sets,
                distance_norm=projection_distance_norm,
            )
        if init_projections > 0:
            print(f"Initial PGD projection count: {init_projections}")

    optimizer = torch.optim.Adam(
        [
            {"params": actor_params, "lr": cfg.lr},
            {"params": critic_params, "lr": cfg.lr},
        ],
    )

    obs, info = env.reset(seed=cfg.seed)
    current_action_mask = _extract_action_mask(info, n_actions)
    global_step = 0
    ppo_update_count = 0
    start_time = time.time()
    pgd_projections = 0
    stop_training_early = False

    if return_training_data:
        training_data = {
            "states": [],
            "actions": [],
            "terminated": [],
            "truncated": [],
            "safe": [],
        }

    # Initial pre-update early-stop check at step=0.
    if early_stop_enabled and cfg.early_stop_min_steps == 0:
        mean_r, std_r, failure_rate, success_rate = evaluate_masked_with_success(
            env=env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        )

        # Restore initial rollout state for reproducibility.
        obs, info = env.reset(seed=cfg.seed)
        current_action_mask = _extract_action_mask(info, n_actions)

        print(
            f"Pre-update check | Steps=0 | meanR={mean_r:.1f} +/- {std_r:.1f} | "
            f"failure_rate={failure_rate:.2f} | success_rate={success_rate:.2f}",
        )
        reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
            cfg,
            mean_reward=mean_r,
            failure_rate=failure_rate,
            success_rate=success_rate,
        )
        if reward_ok and failure_ok and success_ok:
            print(
                "  [Early stop pre-update] step=0 | "
                f"meanR={mean_r:.2f} (threshold={cfg.early_stop_reward_threshold}) | "
                f"failure_rate={failure_rate:.2f} (threshold={cfg.early_stop_failure_rate_threshold}) | "
                f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})"
            )
            stop_training_early = True

    while global_step < cfg.total_timesteps and not stop_training_early:
        obs_buf = np.zeros((cfg.rollout_steps, obs_dim), dtype=np.float32)
        act_buf = np.zeros((cfg.rollout_steps,), dtype=np.int64)
        logp_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        rew_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        done_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        val_buf = np.zeros((cfg.rollout_steps,), dtype=np.float32)
        mask_buf = np.zeros((cfg.rollout_steps, n_actions), dtype=np.float32)

        for t in range(cfg.rollout_steps):
            obs_buf[t] = obs
            mask_buf[t] = current_action_mask

            obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            mask_t = torch.tensor(
                current_action_mask,
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            with torch.no_grad():
                value = critic(obs_t).squeeze(-1)
                logits = actor(obs_t)
                masked_logits = _apply_action_mask_to_logits(logits, mask_t)
                dist = torch.distributions.Categorical(logits=masked_logits)
                action = dist.sample()
                logp = dist.log_prob(action)
                act = int(action.item())

            next_obs, reward, terminated, truncated, step_info = env.step(act)
            done = terminated or truncated

            if return_training_data:
                training_data["states"].append(obs.copy()) # type: ignore
                training_data["actions"].append(act) # type: ignore
                training_data["terminated"].append(float(terminated)) # type: ignore
                training_data["truncated"].append(float(truncated)) # type: ignore
                is_safe = step_info.get("safe", None)
                if is_safe is None:
                    cost = step_info.get("cost", 0)
                    is_safe = 1.0 if cost == 0 else 0.0
                training_data["safe"].append(float(is_safe)) # type: ignore

            act_buf[t] = act
            logp_buf[t] = float(logp.item())
            rew_buf[t] = float(reward)
            done_buf[t] = float(done)
            val_buf[t] = float(value.item())

            global_step += 1
            if done:
                obs, reset_info = env.reset()
                current_action_mask = _extract_action_mask(reset_info, n_actions)
            else:
                obs = next_obs
                current_action_mask = _extract_action_mask(step_info, n_actions)

            if global_step >= cfg.total_timesteps:
                obs_buf = obs_buf[:t + 1]
                act_buf = act_buf[:t + 1]
                logp_buf = logp_buf[:t + 1]
                rew_buf = rew_buf[:t + 1]
                done_buf = done_buf[:t + 1]
                val_buf = val_buf[:t + 1]
                mask_buf = mask_buf[:t + 1]
                break

        with torch.no_grad():
            last_val = critic(
                torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0),
            ).item()

        T = len(rew_buf)
        adv_buf = np.zeros_like(rew_buf)
        last_gae = 0.0
        for t in reversed(range(T)):
            next_nonterminal = 1.0 - done_buf[t]
            next_value = last_val if t == T - 1 else val_buf[t + 1]
            delta = rew_buf[t] + cfg.gamma * next_value * next_nonterminal - val_buf[t]
            last_gae = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_gae
            adv_buf[t] = last_gae
        ret_buf = adv_buf + val_buf

        adv_t = torch.tensor(adv_buf, dtype=torch.float32, device=device)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

        obs_t = torch.tensor(obs_buf, dtype=torch.float32, device=device)
        act_t = torch.tensor(act_buf, dtype=torch.int64, device=device)
        old_logp_t = torch.tensor(logp_buf, dtype=torch.float32, device=device)
        ret_t = torch.tensor(ret_buf, dtype=torch.float32, device=device)
        mask_t = torch.tensor(mask_buf, dtype=torch.float32, device=device)

        batch_size = T
        idxs = np.arange(batch_size)
        for _ in range(cfg.update_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, cfg.minibatch_size):
                end = start + cfg.minibatch_size
                mb_idx = idxs[start:end]
                mb_obs = obs_t[mb_idx]
                mb_act = act_t[mb_idx]
                mb_old_logp = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                mb_mask = mask_t[mb_idx]

                logits = actor(mb_obs)
                masked_logits = _apply_action_mask_to_logits(logits, mb_mask)
                dist = torch.distributions.Categorical(logits=masked_logits)
                new_logp = dist.log_prob(mb_act)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_logp - mb_old_logp)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(
                    ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef,
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v = critic(mb_obs).squeeze(-1)
                v_loss = F.mse_loss(v, mb_ret)

                loss = pg_loss + cfg.vf_coef * v_loss - cfg.ent_coef * entropy

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor_params + critic_params,
                    cfg.max_grad_norm,
                )
                optimizer.step()

                if use_pgd:
                    with torch.no_grad():
                        pgd_projections += _project_actor_to_interval_union(
                            actor_params,
                            bounds_l_sets, # type: ignore[arg-type]
                            bounds_u_sets, # type: ignore[arg-type]
                            distance_norm=projection_distance_norm, # type: ignore[arg-type]
                        )
        ppo_update_count += 1

        if global_step % (10 * cfg.rollout_steps) < cfg.rollout_steps:
            mean_r, std_r, failure_rate, success_rate = evaluate_masked_with_success(
                env=env,
                actor=actor,
                device=device,
                episodes=cfg.eval_episodes,
                deterministic=True,
            )

            obs, info = env.reset()
            current_action_mask = _extract_action_mask(info, n_actions)

            elapsed = time.time() - start_time
            log_msg = f"Steps={global_step} | meanR={mean_r:.1f} +/- {std_r:.1f} | elapsed={elapsed:.1f}s"
            log_msg += f" | failure_rate={failure_rate:.2f}"
            log_msg += f" | success_rate={success_rate:.2f}"
            if use_pgd:
                log_msg += f" | PGD projections={pgd_projections}"
            print(log_msg)

            if early_stop_enabled and ppo_update_count >= cfg.early_stop_min_steps:
                reward_ok, failure_ok, success_ok = _early_stop_thresholds_satisfied(
                    cfg,
                    mean_reward=mean_r,
                    failure_rate=failure_rate,
                    success_rate=success_rate,
                )
                if reward_ok and failure_ok and success_ok:
                    print(
                        f"  [Early stop] updates={ppo_update_count} | step={global_step} | "
                        f"meanR={mean_r:.2f} (threshold={cfg.early_stop_reward_threshold}) | "
                        f"failure_rate={failure_rate:.2f} (threshold={cfg.early_stop_failure_rate_threshold}) | "
                        f"success_rate={success_rate:.2f} (threshold={cfg.early_stop_success_rate_threshold})"
                    )
                    break

    if cfg.eval_episodes is not None and cfg.eval_episodes > 0:
        mean_r, std_r, failure_rate, success_rate = evaluate_masked_with_success(
            env=env,
            actor=actor,
            device=device,
            episodes=cfg.eval_episodes,
            deterministic=True,
        )
        final_msg = (
            f"Final evaluation over {cfg.eval_episodes} episodes: "
            f"mean_reward={mean_r:.2f} +/- {std_r:.2f}"
        )
        final_msg += f" | failure_rate={failure_rate:.2f}"
        final_msg += f" | success_rate={success_rate:.2f}"
        if use_pgd:
            final_msg += f" | Total PGD projections during training: {pgd_projections}"
        print(final_msg)

    env.close()

    if return_training_data:
        training_data["states"] = np.array(training_data["states"]) # type: ignore
        training_data["actions"] = np.array(training_data["actions"]) # type: ignore
        training_data["terminated"] = np.array(training_data["terminated"]) # type: ignore
        training_data["truncated"] = np.array(training_data["truncated"]) # type: ignore
        training_data["safe"] = np.array(training_data["safe"]) # type: ignore
        return actor, critic, training_data # type: ignore
    return actor, critic
