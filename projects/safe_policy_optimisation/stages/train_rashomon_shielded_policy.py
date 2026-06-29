"""Train ProvablySafePPO with a discrete shield and Rashomon bounds."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[3]
for import_path in (REPO_ROOT, REPO_ROOT / "core"):
    path_str = str(import_path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from provably_safe_policy_optimisation import (  # noqa: E402
    ProvablySafePPO,
    Shield,
    projection_target_parameter_names,
)

from projects.safe_policy_optimisation.stages.train_discrete_shielded_policy import (  # noqa: E402
    _episode_rows,
    _parse_env_kwargs,
    _records_to_metrics,
    _training_rows,
    evaluate_shielded_policy,
    evaluate_unshielded_policy,
    load_shield_mask,
    make_unshielded_env,
    validate_shield_for_env,
)
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
)
from projects.safe_policy_optimisation.utils.minipacman_safe_rl import (  # noqa: E402
    aggregate_training_violations,
    aggregate_violations,
    write_json,
)

ALGORITHM_NAME = "rashomon_shielded_ppo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "projects"
    / "safe_policy_optimisation"
    / "artifacts"
    / "rashomon_shielded_policy"
)


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def load_rashomon_bounds(rashomon_dir: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load projection bounds saved by ``compute_shield_rashomon_set.py``."""

    path = rashomon_dir / "rashomon_param_bounds.pt"
    payload = _torch_load(path)
    if "param_bounds_l" not in payload or "param_bounds_u" not in payload:
        raise KeyError(f"Rashomon bounds file must contain param_bounds_l/u; keys={sorted(payload.keys())}.")
    return list(payload["param_bounds_l"]), list(payload["param_bounds_u"])


def load_base_policy_architecture(rashomon_dir: Path) -> dict[str, Any]:
    """Load the saved base-policy architecture metadata."""

    path = rashomon_dir / "base_policy.pt"
    payload = _torch_load(path)
    if "architecture" not in payload:
        raise KeyError(f"Base policy file must contain 'architecture'; keys={sorted(payload.keys())}.")
    architecture = dict(payload["architecture"])
    required = {"input_dim", "n_actions", "hidden_dim", "n_hidden", "activation"}
    missing = sorted(required.difference(architecture))
    if missing:
        raise KeyError(f"Base policy architecture is missing keys: {missing}")
    return architecture


def policy_kwargs_from_base_architecture(architecture: dict[str, Any]) -> dict[str, Any]:
    """Map the saved BC policy architecture to SB3 PPO actor architecture."""

    activation = str(architecture["activation"])
    if activation != "Tanh":
        raise ValueError(f"Unsupported base policy activation {activation!r}; expected 'Tanh'.")
    n_hidden = int(architecture["n_hidden"])
    hidden_dim = int(architecture["hidden_dim"])
    if n_hidden < 0:
        raise ValueError(f"n_hidden must be non-negative; got {n_hidden}.")
    return {
        "net_arch": [hidden_dim] * n_hidden,
        "activation_fn": nn.Tanh,
    }


def _base_to_ppo_actor_name_map(architecture: dict[str, Any]) -> dict[str, str]:
    """Map saved Sequential parameter names to SB3 PPO actor parameter names."""

    n_hidden = int(architecture["n_hidden"])
    mapping: dict[str, str] = {}
    for hidden_idx in range(n_hidden):
        layer_idx = hidden_idx * 2
        prefix = f"mlp_extractor.policy_net.{layer_idx}"
        mapping[f"{layer_idx}.weight"] = f"{prefix}.weight"
        mapping[f"{layer_idx}.bias"] = f"{prefix}.bias"
    final_idx = n_hidden * 2
    mapping[f"{final_idx}.weight"] = "action_net.weight"
    mapping[f"{final_idx}.bias"] = "action_net.bias"
    return mapping


def align_rashomon_bounds_to_ppo_actor(
    architecture: dict[str, Any],
    param_bounds_l: list[torch.Tensor],
    param_bounds_u: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Reorder Sequential Rashomon bounds to PPO's canonical actor target order."""

    name_map = _base_to_ppo_actor_name_map(architecture)
    base_names: list[str] = []
    for hidden_idx in range(int(architecture["n_hidden"])):
        layer_idx = hidden_idx * 2
        base_names.extend([f"{layer_idx}.weight", f"{layer_idx}.bias"])
    final_idx = int(architecture["n_hidden"]) * 2
    base_names.extend([f"{final_idx}.weight", f"{final_idx}.bias"])

    if len(param_bounds_l) != len(base_names) or len(param_bounds_u) != len(base_names):
        raise ValueError(
            "Rashomon bound count does not match saved base-policy architecture: "
            f"lower={len(param_bounds_l)}, upper={len(param_bounds_u)}, expected={len(base_names)}."
        )

    lower_by_target = {
        name_map[base_name]: lower for base_name, lower in zip(base_names, param_bounds_l)
    }
    upper_by_target = {
        name_map[base_name]: upper for base_name, upper in zip(base_names, param_bounds_u)
    }
    target_order = sorted(lower_by_target)
    return [lower_by_target[name] for name in target_order], [upper_by_target[name] for name in target_order]


def validate_rashomon_shapes(
    model: ProvablySafePPO,
    param_bounds_l: list[torch.Tensor],
    param_bounds_u: list[torch.Tensor],
) -> list[dict[str, Any]]:
    """Validate bound tensors against the PPO actor projection target."""

    target_names = projection_target_parameter_names(model)
    named_params = dict(model.policy.named_parameters())
    target_params = [named_params[name] for name in target_names]
    if len(param_bounds_l) != len(target_params) or len(param_bounds_u) != len(target_params):
        raise ValueError(
            "Rashomon bound count does not match PPO actor parameter count: "
            f"lower={len(param_bounds_l)}, upper={len(param_bounds_u)}, target={len(target_params)}."
        )

    rows: list[dict[str, Any]] = []
    for index, (name, param, lower, upper) in enumerate(
        zip(target_names, target_params, param_bounds_l, param_bounds_u)
    ):
        expected = tuple(param.shape)
        lower_shape = tuple(lower.shape)
        upper_shape = tuple(upper.shape)
        if lower_shape != expected or upper_shape != expected:
            raise ValueError(
                "Rashomon bound shape mismatch at actor parameter "
                f"{index} ({name}): expected={expected}, lower={lower_shape}, upper={upper_shape}."
            )
        rows.append(
            {
                "index": index,
                "parameter": name,
                "shape": list(expected),
            }
        )
    return rows


class ExecutedActionSafetyCounterWrapper(gym.Wrapper):
    """Count unsafe primitive actions actually executed in the wrapped env."""

    def __init__(self, env: gym.Env, shield_mask: np.ndarray) -> None:
        super().__init__(env)
        mask = np.asarray(shield_mask) != 0
        if mask.ndim != 2:
            raise ValueError(f"shield_mask must be 2-D, got shape {mask.shape}.")
        self.shield_mask = mask
        self.records: list[dict[str, Any]] = []
        self.total_checks = 0
        self.unsafe_executed_actions = 0
        self._last_obs: Any = None
        self._episode_index = 0
        self._episode_step = 0

    @property
    def episodes(self) -> list[dict[str, Any]]:
        return getattr(self.env, "episodes", [])

    def reset(self, **kwargs: Any):
        obs, info = self.env.reset(**kwargs)
        self._last_obs = obs
        self._episode_step = 0
        return obs, info

    def _state_from_obs(self, obs: Any) -> int:
        return int(np.asarray(obs).item())

    def step(self, action: Any):
        state = self._state_from_obs(self._last_obs)
        action_int = int(np.asarray(action).item())
        is_unsafe = not bool(self.shield_mask[state, action_int])
        self.total_checks += 1
        if is_unsafe:
            self.unsafe_executed_actions += 1

        obs, reward, terminated, truncated, info = self.env.step(action_int)
        record = {
            "episode": self._episode_index,
            "episode_step": self._episode_step,
            "global_step": self.total_checks,
            "state": state,
            "executed_action": action_int,
            "unsafe_executed_action": bool(is_unsafe),
        }
        self.records.append(record)
        self._last_obs = obs
        self._episode_step += 1
        if terminated or truncated:
            self._episode_index += 1
            self._episode_step = 0
        return obs, reward, terminated, truncated, info

    def diagnostics(self) -> dict[str, float]:
        percentage = (
            100.0 * float(self.unsafe_executed_actions) / float(self.total_checks)
            if self.total_checks
            else 0.0
        )
        return {
            "executed_action_checks": int(self.total_checks),
            "executed_unsafe_action_count": int(self.unsafe_executed_actions),
            "executed_unsafe_action_percentage": percentage,
        }


def episode_success(total_reward: float, infos: list[dict[str, Any]], *, reward_threshold: float) -> bool:
    """Determine whether an episode completed the task."""

    for key in ("is_success", "success"):
        for info in reversed(infos):
            if key in info:
                return bool(info[key])
    return float(total_reward) > float(reward_threshold)


def evaluate_success_rate(
    model: ProvablySafePPO,
    env_factory: Callable[[bool], gym.Env],
    *,
    shield_mask: np.ndarray | None,
    eval_policy: str,
    episodes: int,
    seed: int,
    reward_threshold: float,
) -> dict[str, Any]:
    """Evaluate deterministic policy and return task-success metrics."""

    env = env_factory(False)
    if eval_policy not in ("unshielded", "shielded"):
        raise ValueError(f"eval_policy must be 'unshielded' or 'shielded', got {eval_policy!r}.")
    shield = Shield(shield_mask, seed=seed) if eval_policy == "shielded" else None
    rows: list[dict[str, Any]] = []
    try:
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + episode)
            done = False
            total_reward = 0.0
            length = 0
            infos: list[dict[str, Any]] = []
            while not done:
                proposed, _ = model.predict(obs, deterministic=True)
                action = int(np.asarray(proposed).item())
                if shield is not None:
                    state = shield.obs_to_state(np.asarray([obs]))
                    executed = shield.override(state, np.asarray([action]))
                    action = int(executed[0])
                obs, reward, terminated, truncated, info = env.step(action)
                infos.append(dict(info))
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            succeeded = episode_success(total_reward, infos, reward_threshold=reward_threshold)
            rows.append(
                {
                    "episode": episode,
                    "reward": total_reward,
                    "length": length,
                    "success": bool(succeeded),
                }
            )
    finally:
        env.close()

    success_count = sum(1 for row in rows if row["success"])
    return {
        "episodes": int(episodes),
        "success_count": int(success_count),
        "success_rate": float(success_count / int(episodes)) if episodes else 0.0,
        "mean_reward": float(np.mean([row["reward"] for row in rows])) if rows else 0.0,
        "rows": rows,
        "eval_policy": eval_policy,
        "shield_diagnostics": shield.diagnostics() if shield is not None else {},
    }


class EarlyStopOnSuccessCallback(BaseCallback):
    """Stop training when deterministic shielded evaluation reaches a success target."""

    def __init__(
        self,
        *,
        env_factory: Callable[[bool], gym.Env],
        shield_mask: np.ndarray,
        eval_freq: int,
        eval_episodes: int,
        success_rate: float,
        seed: int,
        reward_threshold: float,
        eval_policy: str,
    ) -> None:
        super().__init__()
        if eval_policy not in ("unshielded", "shielded"):
            raise ValueError(f"eval_policy must be 'unshielded' or 'shielded', got {eval_policy!r}.")
        self.env_factory = env_factory
        self.shield_mask = shield_mask
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.target_success_rate = float(success_rate)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.eval_policy = eval_policy
        self.evaluations: list[dict[str, Any]] = []
        self.stop_triggered = False

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        metrics = evaluate_success_rate(
            self.model,
            self.env_factory,
            shield_mask=self.shield_mask,
            eval_policy=self.eval_policy,
            episodes=self.eval_episodes,
            seed=self.seed + self.num_timesteps,
            reward_threshold=self.reward_threshold,
        )
        row = {
            "timesteps": int(self.num_timesteps),
            "episodes": int(metrics["episodes"]),
            "success_count": int(metrics["success_count"]),
            "success_rate": float(metrics["success_rate"]),
            "mean_reward": float(metrics["mean_reward"]),
            "eval_policy": str(metrics["eval_policy"]),
        }
        self.evaluations.append(row)
        if row["success_rate"] >= self.target_success_rate:
            self.stop_triggered = True
            return False
        return True


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _make_env_factory(args: argparse.Namespace, env_kwargs: dict[str, Any], mask: np.ndarray):
    def factory(record_episodes: bool) -> gym.Env:
        env = make_unshielded_env(
            args.env_id,
            env_kwargs=env_kwargs,
            max_episode_steps=args.max_episode_steps,
            cost_limit=args.cost_limit,
            record_episodes=record_episodes,
        )
        if record_episodes:
            env = ExecutedActionSafetyCounterWrapper(env, mask)
        return env

    return factory


def _resolve_curve_eval_freq(args: argparse.Namespace) -> int:
    if args.curve_eval_freq is not None:
        return int(args.curve_eval_freq)
    if int(args.early_stop_eval_freq) > 0:
        return int(args.early_stop_eval_freq)
    return int(args.n_steps)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train ProvablySafePPO with a saved shield and Rashomon set.",
    )
    parser.add_argument("--rashomon-dir", type=Path, required=True)
    parser.add_argument("--shield-path", type=Path, required=True)
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to gym.make.")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--shield-key", default="shield")
    parser.add_argument("--shield-source", choices=("shield", "action_risk"), default="shield")
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument("--shield-action-storage", choices=("proposed", "executed"), default="proposed")
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--total-timesteps", type=int, default=100_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--early-stop-eval-freq", type=int, default=5_000)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=20)
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0)
    parser.add_argument(
        "--tensorboard-log-dir",
        type=Path,
        default=None,
        help="TensorBoard log directory for learning curves. Defaults to <run-dir>/tensorboard.",
    )
    parser.add_argument(
        "--curve-eval-freq",
        type=int,
        default=None,
        help=(
            "Evaluate and log unshielded total reward every N timesteps. "
            "Defaults to --early-stop-eval-freq when positive, otherwise --n-steps. Use 0 to disable."
        ),
    )
    parser.add_argument("--curve-eval-episodes", type=int, default=20)
    parser.add_argument(
        "--evaluation-policy",
        choices=("unshielded", "shielded"),
        default="unshielded",
        help=(
            "Policy used for the final evaluation rollout. 'unshielded' executes the raw greedy "
            "policy and audits whether its proposed actions are shield-safe; 'shielded' applies "
            "the shield before stepping the environment."
        ),
    )
    parser.add_argument(
        "--early-stop-eval-policy",
        choices=("unshielded", "shielded"),
        default="unshielded",
        help=(
            "Policy evaluated by the early-stopping callback. 'unshielded' evaluates the "
            "raw model action without applying the shield; 'shielded' evaluates deployment "
            "with shield overrides."
        ),
    )
    parser.add_argument("--success-reward-threshold", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.env_id is None:
        raise ValueError("--env-id is required for Rashomon-shielded policy training.")
    env_kwargs = _parse_env_kwargs(args.env_kwargs)
    mask = load_shield_mask(
        args.shield_path,
        shield_key=args.shield_key,
        source=args.shield_source,
        risk_threshold=args.risk_threshold,
    )
    raw_param_bounds_l, raw_param_bounds_u = load_rashomon_bounds(args.rashomon_dir)
    architecture = load_base_policy_architecture(args.rashomon_dir)
    param_bounds_l, param_bounds_u = align_rashomon_bounds_to_ppo_actor(
        architecture,
        raw_param_bounds_l,
        raw_param_bounds_u,
    )
    policy_kwargs = policy_kwargs_from_base_architecture(architecture)

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_logger = LearningCurveLogger(
        curve_dir=run_dir / "learning_curves",
        tensorboard_log_dir=args.tensorboard_log_dir or run_dir / "tensorboard",
    )
    curve_eval_freq = _resolve_curve_eval_freq(args)

    env_factory = _make_env_factory(args, env_kwargs, mask)
    train_env = env_factory(True)
    validate_shield_for_env(mask, train_env)
    try:
        expected_input_dim = int(train_env.observation_space.n)
        expected_n_actions = int(train_env.action_space.n)
        if int(architecture["input_dim"]) != expected_input_dim:
            raise ValueError(
                "Base policy input_dim does not match environment observation space: "
                f"architecture={architecture['input_dim']}, env={expected_input_dim}."
            )
        if int(architecture["n_actions"]) != expected_n_actions:
            raise ValueError(
                "Base policy n_actions does not match environment action space: "
                f"architecture={architecture['n_actions']}, env={expected_n_actions}."
            )

        model = ProvablySafePPO(
            "MlpPolicy",
            train_env,
            shield=mask,
            shield_seed=args.seed,
            shield_action_storage=args.shield_action_storage,
            param_bounds_l=param_bounds_l,
            param_bounds_u=param_bounds_u,
            policy_kwargs=policy_kwargs,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
        model.set_exploration_unsafe_action_callback(curve_logger.log_exploration_unsafe)
        projection_targets = validate_rashomon_shapes(model, param_bounds_l, param_bounds_u)
        reward_curve = UnshieldedRewardCurveCallback(
            env_factory=lambda: env_factory(False),
            curve_logger=curve_logger,
            eval_freq=curve_eval_freq,
            eval_episodes=args.curve_eval_episodes,
            seed=args.seed + 30_000,
            reward_threshold=args.success_reward_threshold,
            shield_mask=mask,
        )
        early_stop = EarlyStopOnSuccessCallback(
            env_factory=env_factory,
            shield_mask=mask,
            eval_freq=args.early_stop_eval_freq,
            eval_episodes=args.early_stop_eval_episodes,
            success_rate=args.early_stop_success_rate,
            seed=args.seed + 20_000,
            reward_threshold=args.success_reward_threshold,
            eval_policy=args.early_stop_eval_policy,
        )
        print(f"[{ALGORITHM_NAME}] training for up to {args.total_timesteps} timesteps")
        model.learn(total_timesteps=args.total_timesteps, callback=[reward_curve, early_stop])
        final_curve_evaluation = reward_curve.record_final_evaluation()
        training_records = list(train_env.episodes)
        executed_action_diagnostics = train_env.diagnostics()
        executed_action_records = list(train_env.records)
        training_shield_diagnostics = model.shield_diagnostics()
        projection_diagnostics = model.projection_diagnostics()
        model.save(run_dir / "model.zip")
    finally:
        train_env.close()
        curve_logger.close()

    eval_env = make_unshielded_env(
        args.env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=args.max_episode_steps,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    try:
        if args.evaluation_policy == "shielded":
            eval_shield = Shield(mask, seed=args.seed)
            eval_records = evaluate_shielded_policy(
                model,
                eval_env,
                eval_shield,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_shield_diagnostics = eval_shield.diagnostics()
            eval_action_safety = {
                "proposed_action_checks": int(eval_shield_diagnostics["checked"]),
                "unsafe_proposed_action_count": int(eval_shield_diagnostics["overridden"]),
                "unsafe_proposed_action_percentage": (
                    100.0 * float(eval_shield_diagnostics["overridden"]) / float(eval_shield_diagnostics["checked"])
                    if int(eval_shield_diagnostics["checked"])
                    else 0.0
                ),
            }
        else:
            eval_records, eval_action_safety = evaluate_unshielded_policy(
                model,
                eval_env,
                mask,
                episodes=args.eval_episodes,
                seed=args.seed + 10_000,
            )
            eval_shield_diagnostics = None
    finally:
        eval_env.close()

    config = {
        "algorithm": ALGORITHM_NAME,
        "env_id": args.env_id,
        "env_kwargs": env_kwargs,
        "max_episode_steps": args.max_episode_steps,
        "shield_path": str(args.shield_path),
        "shield_source": args.shield_source,
        "shield_key": args.shield_key,
        "risk_threshold": args.risk_threshold,
        "shield_action_storage": args.shield_action_storage,
        "shield_shape": list(mask.shape),
        "rashomon_dir": str(args.rashomon_dir),
        "base_policy_architecture": architecture,
        "policy_kwargs": {
            "net_arch": policy_kwargs["net_arch"],
            "activation_fn": "Tanh",
        },
        "cost_limit": float(args.cost_limit),
        "total_timesteps": int(args.total_timesteps),
        "training_hyperparameters": {
            "learning_rate": float(args.learning_rate),
            "n_steps": int(args.n_steps),
            "batch_size": int(args.batch_size),
            "n_epochs": int(args.n_epochs),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "clip_range": float(args.clip_range),
            "ent_coef": float(args.ent_coef),
            "vf_coef": float(args.vf_coef),
            "max_grad_norm": float(args.max_grad_norm),
        },
        "eval_episodes": int(args.eval_episodes),
        "evaluation_policy": args.evaluation_policy,
        "early_stop_eval_freq": int(args.early_stop_eval_freq),
        "early_stop_eval_episodes": int(args.early_stop_eval_episodes),
        "early_stop_success_rate": float(args.early_stop_success_rate),
        "early_stop_eval_policy": args.early_stop_eval_policy,
        "success_reward_threshold": float(args.success_reward_threshold),
        "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
        "learning_curve_dir": str(curve_logger.curve_dir),
        "curve_eval_freq": int(curve_eval_freq),
        "curve_eval_episodes": int(args.curve_eval_episodes),
        "seed": int(args.seed),
    }
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "projection_targets.json", projection_targets)

    _write_csv(
        run_dir / "training_episodes.csv",
        _training_rows(training_records),
        [
            "algorithm",
            "episode",
            "end_timestep",
            "reward",
            "cost",
            "length",
            "violated",
            "unsafe_state_visit_count",
            "safe_trajectory",
        ],
    )
    _write_csv(
        run_dir / "episodes.csv",
        _episode_rows(eval_records),
        [
            "algorithm",
            "episode",
            "reward",
            "cost",
            "length",
            "violated",
            "unsafe_state_visit_count",
            "safe_trajectory",
        ],
    )
    _write_csv(
        run_dir / "executed_unsafe_actions.csv",
        executed_action_records,
        ["episode", "episode_step", "global_step", "state", "executed_action", "unsafe_executed_action"],
    )
    _write_csv(
        run_dir / "early_stop_evaluations.csv",
        early_stop.evaluations,
        ["timesteps", "episodes", "success_count", "success_rate", "mean_reward", "eval_policy"],
    )

    summary = {
        "algorithm": ALGORITHM_NAME,
        "model_path": str(run_dir / "model.zip"),
        "final_timesteps": int(model.num_timesteps),
        "total_exploration_steps": int(model.num_timesteps),
        "unsafe_proposed_actions_during_exploration": int(curve_logger.cumulative_unsafe),
        "unshielded_eval_unsafe_action_count": (
            0 if final_curve_evaluation is None else int(final_curve_evaluation.get("unsafe_proposed_action_count", 0))
        ),
        "unshielded_eval_safety_rate": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("safety_rate", 0.0))
        ),
        "unshielded_eval_success_rate": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("success_rate", 0.0))
        ),
        "unshielded_eval_mean_total_reward": (
            0.0 if final_curve_evaluation is None else float(final_curve_evaluation.get("mean_total_reward", 0.0))
        ),
        "early_stop_triggered": bool(early_stop.stop_triggered),
        "last_early_stop_evaluation": early_stop.evaluations[-1] if early_stop.evaluations else None,
        "training": aggregate_training_violations(training_records),
        "evaluation": aggregate_violations(_records_to_metrics(eval_records)),
        "evaluation_policy": args.evaluation_policy,
        "evaluation_proposed_action_safety": eval_action_safety,
        "executed_action_safety": executed_action_diagnostics,
        "training_shield_diagnostics": training_shield_diagnostics,
        "evaluation_shield_diagnostics": eval_shield_diagnostics,
        "projection_diagnostics": projection_diagnostics,
        "learning_curves": {
            "curve_dir": str(curve_logger.curve_dir),
            "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
            "unshielded_reward_evaluations": reward_curve.evaluations,
        },
    }
    write_json(run_dir / "summary.json", summary)
    print(
        "[{algorithm}] executed unsafe actions: {unsafe}/{checked} ({pct:.2f}%)".format(
            algorithm=ALGORITHM_NAME,
            unsafe=executed_action_diagnostics["executed_unsafe_action_count"],
            checked=executed_action_diagnostics["executed_action_checks"],
            pct=executed_action_diagnostics["executed_unsafe_action_percentage"],
        )
    )
    print(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
