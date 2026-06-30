"""Train PPO on an unshielded env using a user-provided discrete shield."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Any


import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

REPO_ROOT = Path(__file__).resolve().parents[3]

from provably_safe_policy_optimisation import ProvablySafePPO, Shield  # noqa: E402

from projects.safe_policy_optimisation.utils import io  # noqa: E402
from projects.safe_policy_optimisation.utils.cli import add_ppo_hyperparameter_args  # noqa: E402
from projects.safe_policy_optimisation.utils.envs import parse_env_kwargs  # noqa: E402
from projects.safe_policy_optimisation.utils.io import write_json  # noqa: E402
from projects.safe_policy_optimisation.utils.metrics import summarise_evaluation  # noqa: E402
from projects.safe_policy_optimisation.utils.learning_curves import (  # noqa: E402
    LearningCurveLogger,
    UnshieldedRewardCurveCallback,
)
from projects.safe_policy_optimisation.utils.safe_rl import (  # noqa: E402
    EpisodeMetrics,
    aggregate_training_violations,
    aggregate_violations,
)
from projects.safe_policy_optimisation.utils.shield import load_shield_mask  # noqa: E402
from projects.safe_policy_optimisation.utils.log import log_info  # noqa: E402

ALGORITHM_NAME = "shielded_ppo"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "projects" / "safe_policy_optimisation" / "artifacts" / "shielded_policy"
)


class EpisodeRecorderWrapper(gym.Wrapper):
    """Record completed episode reward/cost/length for a Gymnasium env."""

    def __init__(self, env: gym.Env, *, cost_limit: float) -> None:
        super().__init__(env)
        self.cost_limit = float(cost_limit)
        self.episodes: list[dict[str, float | int | bool]] = []
        self._episode_index = 0
        self._reset_accumulators()

    def _reset_accumulators(self) -> None:
        self._episode_reward = 0.0
        self._episode_cost = 0.0
        self._episode_unsafe_state_visits = 0
        self._episode_length = 0

    def reset(self, **kwargs: Any):
        self._reset_accumulators()
        obs, info = self.env.reset(**kwargs)
        initial_cost = self._state_cost(obs, dict(info))
        self._episode_cost += initial_cost
        self._episode_unsafe_state_visits += int(initial_cost > 0.0)
        return obs, info

    def _state_cost(self, obs: Any, info: dict[str, Any]) -> float:
        if "cost" in info:
            return float(info["cost"])
        unwrapped = self.unwrapped
        if hasattr(unwrapped, "label_fn") and hasattr(unwrapped, "cost_fn"):
            state = int(np.asarray(obs).item())
            return float(unwrapped.cost_fn(unwrapped.label_fn(state)))
        return 0.0

    def step(self, action: Any):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        cost = self._state_cost(obs, info)
        info["cost"] = cost
        self._episode_reward += float(reward)
        self._episode_cost += cost
        self._episode_length += 1
        if terminated or truncated:
            self.episodes.append(
                {
                    "episode": self._episode_index,
                    "reward": self._episode_reward,
                    "cost": self._episode_cost,
                    "length": self._episode_length,
                    "violated": self._episode_cost > self.cost_limit,
                    "unsafe_state_visit_count": int(self._episode_unsafe_state_visits),
                    "safe_trajectory": bool(self._episode_unsafe_state_visits == 0),
                }
            )
            self._episode_index += 1
            self._reset_accumulators()
        return obs, reward, terminated, truncated, info


def make_unshielded_env(
    env_id: str,
    *,
    env_kwargs: dict[str, Any],
    max_episode_steps: int | None,
    cost_limit: float,
    record_episodes: bool,
    render_mode: str | None = None,
) -> gym.Env:
    """Create the unshielded env; shielding is handled by the PPO model."""

    # Registers the local Custom* MASA-style Gymnasium ids when available.
    try:
        import projects.safe_crl.utils.masa_tabular_envs  # noqa: F401
    except ModuleNotFoundError:
        pass

    kwargs = dict(env_kwargs)
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_id, max_episode_steps=max_episode_steps, **kwargs)
    if record_episodes:
        env = EpisodeRecorderWrapper(env, cost_limit=cost_limit)
    return env


def validate_shield_for_env(mask: np.ndarray, env: gym.Env) -> None:
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        raise ValueError(
            "Generic shielded PPO currently requires a Discrete observation space. "
            f"Got {type(env.observation_space).__name__}."
        )
    if not isinstance(env.action_space, gym.spaces.Discrete):
        raise ValueError(
            "Generic shielded PPO currently requires a Discrete action space. "
            f"Got {type(env.action_space).__name__}."
        )
    if mask.shape != (int(env.observation_space.n), int(env.action_space.n)):
        raise ValueError(
            "Shield shape does not match env spaces: "
            f"shield={mask.shape}, expected={(int(env.observation_space.n), int(env.action_space.n))}."
        )


_write_episode_csv = io.write_record_csv


def _episode_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return io.record_rows(records, algorithm=ALGORITHM_NAME)


def _training_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return io.record_training_rows(records, algorithm=ALGORITHM_NAME)


def _resolve_curve_eval_freq(args: argparse.Namespace) -> int:
    if args.curve_eval_freq is not None:
        return int(args.curve_eval_freq)
    if int(args.early_stop_eval_freq) > 0:
        return int(args.early_stop_eval_freq)
    return int(args.n_steps)


def evaluate_shielded_policy(
    model: ProvablySafePPO,
    env: EpisodeRecorderWrapper,
    shield: Shield,
    *,
    episodes: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Evaluate deterministic policy with the same shield override applied."""

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            proposed, _ = model.predict(obs, deterministic=True)
            state = shield.obs_to_state(np.asarray([obs]))
            executed = shield.override(state, np.asarray([int(np.asarray(proposed).item())]))
            obs, _reward, terminated, truncated, _info = env.step(int(executed[0]))
            done = bool(terminated or truncated)
    return list(env.episodes)


def _action_safety_row(checked: int, unsafe: int) -> dict[str, float | int]:
    return {
        "proposed_action_checks": int(checked),
        "unsafe_proposed_action_count": int(unsafe),
        "unsafe_proposed_action_percentage": float(100.0 * unsafe / checked) if checked else 0.0,
    }


def evaluate_unshielded_policy(
    model: ProvablySafePPO,
    env: EpisodeRecorderWrapper,
    shield_mask: np.ndarray,
    *,
    episodes: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, float | int]]:
    """Evaluate the deterministic raw policy and only audit shield safety."""

    checked = 0
    unsafe = 0
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(np.asarray(action).item())
            state = int(np.asarray(obs).item())
            checked += 1
            unsafe += int(not bool(shield_mask[state, action_int]))
            obs, _reward, terminated, truncated, _info = env.step(action_int)
            done = bool(terminated or truncated)
    return list(env.episodes), _action_safety_row(checked, unsafe)


def episode_success(total_reward: float, infos: list[dict[str, Any]], *, reward_threshold: float) -> bool:
    for key in ("is_success", "success"):
        for info in reversed(infos):
            if key in info:
                return bool(info[key])
    return float(total_reward) > float(reward_threshold)


def evaluate_unshielded_success_rate(
    model: ProvablySafePPO,
    *,
    env_id: str,
    env_kwargs: dict[str, Any],
    max_episode_steps: int | None,
    cost_limit: float,
    episodes: int,
    seed: int,
    reward_threshold: float,
) -> dict[str, Any]:
    env = make_unshielded_env(
        env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=max_episode_steps,
        cost_limit=cost_limit,
        record_episodes=False,
    )
    rewards: list[float] = []
    lengths: list[int] = []
    successes = 0
    try:
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + episode)
            done = False
            total_reward = 0.0
            length = 0
            infos: list[dict[str, Any]] = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(np.asarray(action).item()))
                infos.append(dict(info))
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            rewards.append(total_reward)
            lengths.append(length)
            successes += int(episode_success(total_reward, infos, reward_threshold=reward_threshold))
    finally:
        env.close()
    return {
        "episodes": int(episodes),
        "success_count": int(successes),
        "success_rate": float(successes / int(episodes)) if episodes else 0.0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "eval_policy": "unshielded",
    }


class EarlyStopOnUnshieldedSuccessCallback(BaseCallback):
    def __init__(
        self,
        *,
        env_id: str,
        env_kwargs: dict[str, Any],
        max_episode_steps: int | None,
        cost_limit: float,
        eval_freq: int,
        eval_episodes: int,
        success_rate: float,
        seed: int,
        reward_threshold: float,
    ) -> None:
        super().__init__()
        self.env_id = env_id
        self.env_kwargs = dict(env_kwargs)
        self.max_episode_steps = max_episode_steps
        self.cost_limit = float(cost_limit)
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.target_success_rate = float(success_rate)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.evaluations: list[dict[str, Any]] = []
        self.stop_triggered = False

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        metrics = evaluate_unshielded_success_rate(
            self.model,
            env_id=self.env_id,
            env_kwargs=self.env_kwargs,
            max_episode_steps=self.max_episode_steps,
            cost_limit=self.cost_limit,
            episodes=self.eval_episodes,
            seed=self.seed + self.num_timesteps,
            reward_threshold=self.reward_threshold,
        )
        row = {"timesteps": int(self.num_timesteps), **metrics}
        self.evaluations.append(row)
        if float(metrics["success_rate"]) >= self.target_success_rate:
            self.stop_triggered = True
            return False
        return True


def _records_to_metrics(records: list[dict[str, Any]]) -> list[EpisodeMetrics]:
    return [
        EpisodeMetrics(
            episode=int(record["episode"]),
            reward=float(record["reward"]),
            cost=float(record["cost"]),
            length=int(record["length"]),
            violated=bool(record["violated"]),
            unsafe_state_visit_count=int(record.get("unsafe_state_visit_count", 0)),
            safe_trajectory=bool(record.get("safe_trajectory", float(record.get("cost", 0.0)) <= 0.0)),
        )
        for record in records
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train PPO on an unshielded Gymnasium env with a user-provided shield mask.",
    )
    parser.add_argument("--shield-path", type=Path, required=True)
    parser.add_argument("--env-id", required=True)
    parser.add_argument("--env-kwargs", default=None, help="JSON object passed to gym.make.")
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--shield-key", default="shield")
    parser.add_argument("--shield-source", choices=("shield", "action_risk"), default="shield")
    parser.add_argument("--risk-threshold", type=float, default=None)
    parser.add_argument(
        "--shield-action-storage",
        choices=("proposed", "executed"),
        default="proposed",
        help=(
            "Which action PPO stores/log-probs in the rollout buffer when the shield overrides. "
            "'proposed' matches shielded-env training; 'executed' preserves the old behaviour."
        ),
    )
    parser.add_argument("--cost-limit", type=float, default=0.0)
    parser.add_argument("--total-timesteps", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    add_ppo_hyperparameter_args(parser)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--early-stop-eval-freq", type=int, default=0)
    parser.add_argument("--early-stop-eval-episodes", type=int, default=20)
    parser.add_argument("--early-stop-success-rate", type=float, default=1.0)
    parser.add_argument("--success-reward-threshold", type=float, default=0.0)
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-id", default=None)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    env_kwargs = parse_env_kwargs(args.env_kwargs)
    mask = load_shield_mask(
        args.shield_path,
        shield_key=args.shield_key,
        source=args.shield_source,
        risk_threshold=args.risk_threshold,
    )

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    curve_logger = LearningCurveLogger(
        curve_dir=run_dir / "learning_curves",
        tensorboard_log_dir=args.tensorboard_log_dir or run_dir / "tensorboard",
    )
    curve_eval_freq = _resolve_curve_eval_freq(args)

    train_env = make_unshielded_env(
        args.env_id,
        env_kwargs=env_kwargs,
        max_episode_steps=args.max_episode_steps,
        cost_limit=args.cost_limit,
        record_episodes=True,
    )
    validate_shield_for_env(mask, train_env)
    try:
        model = ProvablySafePPO(
            "MlpPolicy",
            train_env,
            shield=mask,
            shield_seed=args.seed,
            shield_action_storage=args.shield_action_storage,
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
        reward_curve = UnshieldedRewardCurveCallback(
            env_factory=lambda: make_unshielded_env(
                args.env_id,
                env_kwargs=env_kwargs,
                max_episode_steps=args.max_episode_steps,
                cost_limit=args.cost_limit,
                record_episodes=False,
            ),
            curve_logger=curve_logger,
            eval_freq=curve_eval_freq,
            eval_episodes=args.curve_eval_episodes,
            seed=args.seed + 30_000,
            reward_threshold=args.success_reward_threshold,
            shield_mask=mask,
        )
        early_stop = EarlyStopOnUnshieldedSuccessCallback(
            env_id=args.env_id,
            env_kwargs=env_kwargs,
            max_episode_steps=args.max_episode_steps,
            cost_limit=args.cost_limit,
            eval_freq=args.early_stop_eval_freq,
            eval_episodes=args.early_stop_eval_episodes,
            success_rate=args.early_stop_success_rate,
            seed=args.seed + 20_000,
            reward_threshold=args.success_reward_threshold,
        )
        log_info(f"[{ALGORITHM_NAME}] training for {args.total_timesteps} timesteps")
        model.learn(total_timesteps=args.total_timesteps, callback=[reward_curve, early_stop])
        final_curve_evaluation = reward_curve.record_final_evaluation()
        training_records = list(train_env.episodes)
        training_shield_diagnostics = model.shield_diagnostics()
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
            eval_action_safety = _action_safety_row(
                int(eval_shield_diagnostics["checked"]),
                int(eval_shield_diagnostics["overridden"]),
            )
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
        "early_stop_eval_policy": "unshielded",
        "early_stop_eval_freq": int(args.early_stop_eval_freq),
        "early_stop_eval_episodes": int(args.early_stop_eval_episodes),
        "early_stop_success_rate": float(args.early_stop_success_rate),
        "success_reward_threshold": float(args.success_reward_threshold),
        "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
        "learning_curve_dir": str(curve_logger.curve_dir),
        "curve_eval_freq": int(curve_eval_freq),
        "curve_eval_episodes": int(args.curve_eval_episodes),
        "seed": int(args.seed),
    }
    write_json(run_dir / "config.json", config)

    _write_episode_csv(run_dir / "training_episodes.csv", _training_rows(training_records), include_end_timestep=True)
    _write_episode_csv(run_dir / "episodes.csv", _episode_rows(eval_records))
    _write_csv_path = run_dir / "early_stop_evaluations.csv"
    _write_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with _write_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "timesteps",
                "episodes",
                "success_count",
                "success_rate",
                "mean_reward",
                "mean_length",
                "eval_policy",
            ],
        )
        writer.writeheader()
        writer.writerows(early_stop.evaluations)

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
        "training_shield_diagnostics": training_shield_diagnostics,
        "evaluation_shield_diagnostics": eval_shield_diagnostics,
        "learning_curves": {
            "curve_dir": str(curve_logger.curve_dir),
            "tensorboard_log_dir": str(curve_logger.tensorboard_log_dir),
            "unshielded_reward_evaluations": reward_curve.evaluations,
        },
    }
    write_json(run_dir / "summary.json", summary)
    write_json(
        run_dir / "metrics.json",
        summarise_evaluation(
            eval_records,
            success_reward_threshold=float(args.success_reward_threshold),
            cost_limit=float(args.cost_limit),
            algorithm=ALGORITHM_NAME,
        ),
    )
    log_info(
        "[{algorithm}] training overrides: {overridden}/{checked} ({rate:.2%})".format(
            algorithm=ALGORITHM_NAME,
            overridden=training_shield_diagnostics["overridden"],
            checked=training_shield_diagnostics["checked"],
            rate=training_shield_diagnostics["intervention_rate"],
        )
    )
    log_info(f"Artifacts written to {run_dir}")
    return summary


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
