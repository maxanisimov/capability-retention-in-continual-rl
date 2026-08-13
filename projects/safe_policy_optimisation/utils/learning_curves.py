"""TensorBoard and CSV learning-curve logging for shielded policy experiments."""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter

from projects.safe_policy_optimisation.utils.metrics import (
    SUCCESS_MODE_SAFE_TRAJECTORY,
    success_mode_for_env,
)
from projects.safe_policy_optimisation.utils.safe_rl import obs_state_id, state_cost

EXPLORATION_FIELDS = [
    "timestep",
    "unsafe_this_step",
    "checked_this_step",
    "unsafe_rate",
    "cumulative_unsafe",
    "cumulative_checked",
]
EVALUATION_SUMMARY_FIELDS = [
    "eval_index",
    "timestep",
    "episodes",
    "mean_total_reward",
    "min_total_reward",
    "max_total_reward",
    "success_rate",
    "safe_trajectory_count",
    "unsafe_state_visit_count",
    "safety_rate",
    "proposed_action_checks",
    "unsafe_proposed_action_count",
    "cumulative_unsafe_proposed_action_count",
    "unsafe_proposed_action_rate",
    "shield_alignment_rate",
]
EVALUATION_EPISODE_FIELDS = [
    "eval_index",
    "timestep",
    "episode",
    "total_reward",
    "length",
    "success",
    "safe_trajectory",
    "unsafe_state_visit_count",
    "proposed_action_checks",
    "unsafe_proposed_action_count",
    "unsafe_proposed_action_rate",
    "shield_alignment_rate",
]


class _EvaluationChannel:
    """One evaluation curve stream: its CSV pair, TensorBoard prefix and counters.

    A run may log more than one evaluation curve for the same policy -- e.g. the
    ``"unshielded"`` curve (raw policy, measures *intrinsic* safety) alongside a
    ``"shielded"`` curve (the deployed system, with the shield overriding unsafe
    actions). Each keeps its own ``eval_index`` and cumulative counters so the
    two curves never interleave.
    """

    def __init__(self, *, curve_dir: Path, variant: str) -> None:
        self.variant = variant
        self.eval_index = 0
        self.cumulative_eval_unsafe = 0
        self.summary_path = curve_dir / f"evaluation_{variant}_summary.csv"
        self.episodes_path = curve_dir / f"evaluation_{variant}_episodes.csv"
        self._summary_handle = self.summary_path.open("w", newline="", encoding="utf-8")
        self._episodes_handle = self.episodes_path.open("w", newline="", encoding="utf-8")
        self.summary_writer = csv.DictWriter(self._summary_handle, fieldnames=EVALUATION_SUMMARY_FIELDS)
        self.episodes_writer = csv.DictWriter(self._episodes_handle, fieldnames=EVALUATION_EPISODE_FIELDS)
        self.summary_writer.writeheader()
        self.episodes_writer.writeheader()

    def flush(self) -> None:
        self._summary_handle.flush()
        self._episodes_handle.flush()

    def close(self) -> None:
        self._summary_handle.close()
        self._episodes_handle.close()


class LearningCurveLogger:
    """Write requested learning curves to TensorBoard and project-local CSV files."""

    def __init__(self, *, curve_dir: Path, tensorboard_log_dir: Path) -> None:
        self.curve_dir = Path(curve_dir)
        self.tensorboard_log_dir = Path(tensorboard_log_dir)
        self.curve_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.tensorboard_log_dir), flush_secs=10)
        self.cumulative_unsafe = 0
        self.cumulative_checked = 0

        self.exploration_path = self.curve_dir / "exploration_unsafe_actions.csv"
        self._exploration_handle = self.exploration_path.open("w", newline="", encoding="utf-8")
        self._exploration_writer = csv.DictWriter(self._exploration_handle, fieldnames=EXPLORATION_FIELDS)
        self._exploration_writer.writeheader()

        # The unshielded channel is created eagerly so its files exist from the
        # start of the run, as they always have. Extra channels ("shielded") are
        # created on first use, so runs that never log one write no empty files.
        self._channels: dict[str, _EvaluationChannel] = {}
        unshielded = self._channel("unshielded")
        self.evaluation_summary_path = unshielded.summary_path
        self.evaluation_episodes_path = unshielded.episodes_path

    def _channel(self, variant: str) -> _EvaluationChannel:
        channel = self._channels.get(variant)
        if channel is None:
            channel = _EvaluationChannel(curve_dir=self.curve_dir, variant=variant)
            self._channels[variant] = channel
        return channel

    @property
    def eval_index(self) -> int:
        """Unshielded eval counter (kept for backwards compatibility)."""
        return self._channel("unshielded").eval_index

    def log_exploration_unsafe(
        self,
        *,
        timestep: int,
        unsafe_this_step: int,
        checked_this_step: int,
    ) -> dict[str, float | int]:
        """Record unsafe raw actions proposed by the policy during shielded exploration."""

        checked = int(checked_this_step)
        unsafe = int(unsafe_this_step)
        self.cumulative_unsafe += unsafe
        self.cumulative_checked += checked
        row = {
            "timestep": int(timestep),
            "unsafe_this_step": unsafe,
            "checked_this_step": checked,
            "unsafe_rate": float(unsafe / checked) if checked else 0.0,
            "cumulative_unsafe": int(self.cumulative_unsafe),
            "cumulative_checked": int(self.cumulative_checked),
        }
        self._exploration_writer.writerow(row)
        self._exploration_handle.flush()
        self.writer.add_scalar(
            "exploration/cumulative_proposed_unsafe_actions",
            row["cumulative_unsafe"],
            int(timestep),
        )
        self.writer.add_scalar(
            "exploration/cumulative_exploration_steps",
            row["cumulative_checked"],
            int(timestep),
        )
        self.writer.add_scalar("exploration/proposed_unsafe_actions", unsafe, int(timestep))
        self.writer.add_scalar("exploration/proposed_unsafe_action_rate", row["unsafe_rate"], int(timestep))
        return row

    def log_unshielded_evaluation(
        self,
        *,
        timestep: int,
        episode_rows: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Record total rewards from deterministic unshielded policy evaluation."""

        return self._log_evaluation(
            variant="unshielded", timestep=timestep, episode_rows=episode_rows
        )

    def log_shielded_evaluation(
        self,
        *,
        timestep: int,
        episode_rows: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        """Record total rewards from deterministic *shielded* policy evaluation.

        This is the deployed system: unsafe proposed actions are overridden by
        the shield, so ``safety_rate`` should stay at 1.0 while
        ``unsafe_proposed_action_count`` measures how often the shield had to
        intervene.
        """

        return self._log_evaluation(
            variant="shielded", timestep=timestep, episode_rows=episode_rows
        )

    def _log_evaluation(
        self,
        *,
        variant: str,
        timestep: int,
        episode_rows: list[dict[str, Any]],
    ) -> dict[str, float | int]:
        channel = self._channel(variant)
        eval_index = channel.eval_index
        channel.eval_index += 1
        rewards = np.asarray([float(row["total_reward"]) for row in episode_rows], dtype=np.float64)
        success_count = sum(int(bool(row.get("success", False))) for row in episode_rows)
        safe_trajectory_count = sum(int(bool(row.get("safe_trajectory", True))) for row in episode_rows)
        unsafe_state_visits = sum(int(row.get("unsafe_state_visit_count", 0)) for row in episode_rows)
        checked = sum(int(row.get("proposed_action_checks", 0)) for row in episode_rows)
        unsafe = sum(int(row.get("unsafe_proposed_action_count", 0)) for row in episode_rows)
        channel.cumulative_eval_unsafe += unsafe
        unsafe_rate = float(unsafe / checked) if checked else 0.0
        summary = {
            "eval_index": int(eval_index),
            "timestep": int(timestep),
            "episodes": int(len(episode_rows)),
            "mean_total_reward": float(np.mean(rewards)) if rewards.size else 0.0,
            "min_total_reward": float(np.min(rewards)) if rewards.size else 0.0,
            "max_total_reward": float(np.max(rewards)) if rewards.size else 0.0,
            "success_rate": float(success_count / len(episode_rows)) if episode_rows else 0.0,
            "safe_trajectory_count": int(safe_trajectory_count),
            "unsafe_state_visit_count": int(unsafe_state_visits),
            "safety_rate": float(safe_trajectory_count / len(episode_rows)) if episode_rows else 0.0,
            "proposed_action_checks": int(checked),
            "unsafe_proposed_action_count": int(unsafe),
            "cumulative_unsafe_proposed_action_count": int(channel.cumulative_eval_unsafe),
            "unsafe_proposed_action_rate": unsafe_rate,
            "shield_alignment_rate": float(1.0 - unsafe_rate) if checked else 0.0,
        }
        channel.summary_writer.writerow(summary)
        for row in episode_rows:
            episode_row = {
                "eval_index": int(eval_index),
                "timestep": int(timestep),
                "episode": int(row["episode"]),
                "total_reward": float(row["total_reward"]),
                "length": int(row["length"]),
                "success": bool(row["success"]),
                "safe_trajectory": bool(row.get("safe_trajectory", True)),
                "unsafe_state_visit_count": int(row.get("unsafe_state_visit_count", 0)),
                "proposed_action_checks": int(row.get("proposed_action_checks", 0)),
                "unsafe_proposed_action_count": int(row.get("unsafe_proposed_action_count", 0)),
                "unsafe_proposed_action_rate": float(row.get("unsafe_proposed_action_rate", 0.0)),
                "shield_alignment_rate": float(row.get("shield_alignment_rate", 0.0)),
            }
            channel.episodes_writer.writerow(episode_row)
        channel.flush()
        v = channel.variant
        self.writer.add_scalar(f"evaluation/{v}_total_reward_mean", summary["mean_total_reward"], int(timestep))
        self.writer.add_scalar(f"evaluation/{v}_total_reward_min", summary["min_total_reward"], int(timestep))
        self.writer.add_scalar(f"evaluation/{v}_total_reward_max", summary["max_total_reward"], int(timestep))
        self.writer.add_scalar(f"evaluation/{v}_success_rate", summary["success_rate"], int(timestep))
        self.writer.add_scalar(f"evaluation/{v}_safety_rate", summary["safety_rate"], int(timestep))
        self.writer.add_scalar(
            f"evaluation/{v}_unsafe_state_visits",
            summary["unsafe_state_visit_count"],
            int(timestep),
        )
        self.writer.add_scalar(
            f"evaluation/{v}_proposed_action_checks",
            summary["proposed_action_checks"],
            int(timestep),
        )
        self.writer.add_scalar(
            f"evaluation/{v}_unsafe_proposed_actions",
            summary["unsafe_proposed_action_count"],
            int(timestep),
        )
        self.writer.add_scalar(
            f"evaluation/cumulative_{v}_unsafe_actions",
            summary["cumulative_unsafe_proposed_action_count"],
            int(timestep),
        )
        self.writer.add_scalar(
            f"evaluation/{v}_unsafe_proposed_action_rate",
            summary["unsafe_proposed_action_rate"],
            int(timestep),
        )
        self.writer.add_scalar(
            f"evaluation/{v}_shield_alignment_rate",
            summary["shield_alignment_rate"],
            int(timestep),
        )
        self.writer.flush()
        return summary

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()
        self._exploration_handle.close()
        for channel in self._channels.values():
            channel.close()


def episode_success(
    total_reward: float,
    infos: list[dict[str, Any]],
    *,
    reward_threshold: float,
    success_mode: str = "reward_threshold",
    unsafe_state_visits: int = 0,
) -> bool:
    """Determine success from environment info flags, falling back to reward threshold.

    For avoid-only tasks (``success_mode == 'safe_trajectory'``) success instead means
    the episode entered no unsafe zone, i.e. ``unsafe_state_visits == 0``.
    """

    if success_mode == SUCCESS_MODE_SAFE_TRAJECTORY:
        return bool(int(unsafe_state_visits) == 0)
    for key in ("is_success", "success"):
        for info in reversed(infos):
            if key in info:
                return bool(info[key])
    return float(total_reward) > float(reward_threshold)


def evaluate_shielded_total_rewards(
    model: Any,
    env_factory: Callable[[], gym.Env],
    *,
    episodes: int,
    seed: int,
    reward_threshold: float,
    shield_mask: np.ndarray,
) -> list[dict[str, float | int | bool]]:
    """Run deterministic episodes with the shield overriding unsafe actions.

    Mirrors ``Shield.override``'s semantics (keep a safe proposed action, else
    resample uniformly among the safe actions) without calling it, so that
    evaluating a curve point does not perturb the training shield's diagnostics
    counters. The RNG is seeded per evaluation so curve points are reproducible.
    """

    return _evaluate_total_rewards(
        model,
        env_factory,
        episodes=episodes,
        seed=seed,
        reward_threshold=reward_threshold,
        shield_mask=shield_mask,
        apply_shield=True,
    )


def evaluate_unshielded_total_rewards(
    model: Any,
    env_factory: Callable[[], gym.Env],
    *,
    episodes: int,
    seed: int,
    reward_threshold: float,
    shield_mask: np.ndarray | None = None,
) -> list[dict[str, float | int | bool]]:
    """Run deterministic raw-policy episodes and return per-episode total rewards."""

    return _evaluate_total_rewards(
        model,
        env_factory,
        episodes=episodes,
        seed=seed,
        reward_threshold=reward_threshold,
        shield_mask=shield_mask,
        apply_shield=False,
    )


def _evaluate_total_rewards(
    model: Any,
    env_factory: Callable[[], gym.Env],
    *,
    episodes: int,
    seed: int,
    reward_threshold: float,
    shield_mask: np.ndarray | None,
    apply_shield: bool,
) -> list[dict[str, float | int | bool]]:
    """Shared rollout for the shielded and unshielded evaluation curves.

    ``shield_mask`` is used to *count* unsafe proposed actions in both modes;
    with ``apply_shield`` it additionally overrides them before stepping.
    """

    if apply_shield and shield_mask is None:
        raise ValueError("apply_shield=True requires a shield_mask.")
    rng = np.random.default_rng(int(seed))
    env = env_factory()
    success_mode = success_mode_for_env(getattr(getattr(env, "spec", None), "id", None))
    shield = None if shield_mask is None else np.asarray(shield_mask) != 0
    rows: list[dict[str, float | int | bool]] = []
    try:
        for episode in range(int(episodes)):
            obs, _ = env.reset(seed=int(seed) + episode)
            done = False
            total_reward = 0.0
            length = 0
            checked = 0
            unsafe = 0
            initial_cost = state_cost(env, obs)
            unsafe_state_visits = int(initial_cost > 0.0)
            infos: list[dict[str, Any]] = []
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                action_int = int(np.asarray(action).item())
                if shield is not None:
                    state = obs_state_id(env, obs)
                    checked += 1
                    proposed_is_safe = bool(shield[state, action_int])
                    unsafe += int(not proposed_is_safe)
                    if apply_shield and not proposed_is_safe:
                        safe_actions = np.flatnonzero(shield[state])
                        if safe_actions.size:
                            action_int = int(rng.choice(safe_actions))
                obs, reward, terminated, truncated, info = env.step(action_int)
                infos.append(dict(info))
                step_cost = state_cost(env, obs, info)
                unsafe_state_visits += int(step_cost > 0.0)
                total_reward += float(reward)
                length += 1
                done = bool(terminated or truncated)
            rows.append(
                {
                    "episode": int(episode),
                    "total_reward": float(total_reward),
                    "length": int(length),
                    "success": bool(
                        episode_success(
                            total_reward,
                            infos,
                            reward_threshold=reward_threshold,
                            success_mode=success_mode,
                            unsafe_state_visits=unsafe_state_visits,
                        )
                    ),
                    "safe_trajectory": bool(unsafe_state_visits == 0),
                    "unsafe_state_visit_count": int(unsafe_state_visits),
                    "proposed_action_checks": int(checked),
                    "unsafe_proposed_action_count": int(unsafe),
                    "unsafe_proposed_action_rate": float(unsafe / checked) if checked else 0.0,
                    "shield_alignment_rate": float(1.0 - (unsafe / checked)) if checked else 0.0,
                }
            )
    finally:
        env.close()
    return rows


class UnshieldedRewardCurveCallback(BaseCallback):
    """Evaluate and log policy reward curves at a fixed timestep cadence.

    Despite the name (kept for backwards compatibility), this drives either
    evaluation variant. ``apply_shield=True`` evaluates the *deployed* system --
    the shield overrides unsafe proposed actions -- and logs to the ``shielded``
    curve; the default evaluates the raw policy and logs to ``unshielded``.
    Attach one instance per variant to log both curves from a single run.
    """

    def __init__(
        self,
        *,
        env_factory: Callable[[], gym.Env],
        curve_logger: LearningCurveLogger,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
        reward_threshold: float,
        shield_mask: np.ndarray | None = None,
        apply_shield: bool = False,
    ) -> None:
        super().__init__()
        if apply_shield and shield_mask is None:
            raise ValueError("apply_shield=True requires a shield_mask.")
        self.env_factory = env_factory
        self.curve_logger = curve_logger
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.shield_mask = None if shield_mask is None else np.asarray(shield_mask) != 0
        self.apply_shield = bool(apply_shield)
        self.evaluations: list[dict[str, float | int]] = []

    def _logged_evaluation_at(self, timestep: int) -> dict[str, float | int] | None:
        for evaluation in reversed(self.evaluations):
            if int(evaluation["timestep"]) == int(timestep):
                return evaluation
        return None

    def _evaluate_and_log(self, *, timestep: int) -> dict[str, float | int]:
        episode_rows = _evaluate_total_rewards(
            self.model,
            self.env_factory,
            episodes=self.eval_episodes,
            seed=self.seed + int(timestep),
            reward_threshold=self.reward_threshold,
            shield_mask=self.shield_mask,
            apply_shield=self.apply_shield,
        )
        log = (
            self.curve_logger.log_shielded_evaluation
            if self.apply_shield
            else self.curve_logger.log_unshielded_evaluation
        )
        summary = log(timestep=int(timestep), episode_rows=episode_rows)
        self.evaluations.append(summary)
        return summary

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True
        self._evaluate_and_log(timestep=int(self.num_timesteps))
        return True

    def _on_training_end(self) -> None:
        self.record_final_evaluation()

    def record_final_evaluation(self, timestep: int | None = None) -> dict[str, float | int] | None:
        """Record a final evaluation point unless this timestep is already logged."""

        if self.eval_freq <= 0:
            return None
        final_timestep = int(self.num_timesteps if timestep is None else timestep)
        existing = self._logged_evaluation_at(final_timestep)
        if existing is not None:
            return existing
        return self._evaluate_and_log(timestep=final_timestep)


class CallableUnshieldedRewardCurveCallback:
    """Callable reward-curve logger for training loops that accept ``callback(model)``."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], gym.Env],
        curve_logger: LearningCurveLogger,
        eval_freq: int,
        eval_episodes: int,
        seed: int,
        reward_threshold: float,
        shield_mask: np.ndarray | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.curve_logger = curve_logger
        self.eval_freq = int(eval_freq)
        self.eval_episodes = int(eval_episodes)
        self.seed = int(seed)
        self.reward_threshold = float(reward_threshold)
        self.shield_mask = None if shield_mask is None else np.asarray(shield_mask) != 0
        self.evaluations: list[dict[str, float | int]] = []
        self._next_eval_timestep = self.eval_freq if self.eval_freq > 0 else None

    def _logged_evaluation_at(self, timestep: int) -> dict[str, float | int] | None:
        for evaluation in reversed(self.evaluations):
            if int(evaluation["timestep"]) == int(timestep):
                return evaluation
        return None

    def _evaluate_and_log(self, model: Any, *, timestep: int) -> dict[str, float | int]:
        episode_rows = evaluate_unshielded_total_rewards(
            model,
            self.env_factory,
            episodes=self.eval_episodes,
            seed=self.seed + int(timestep),
            reward_threshold=self.reward_threshold,
            shield_mask=self.shield_mask,
        )
        summary = self.curve_logger.log_unshielded_evaluation(
            timestep=int(timestep),
            episode_rows=episode_rows,
        )
        self.evaluations.append(summary)
        return summary

    def _advance_next_eval_timestep(self, current_timestep: int) -> None:
        if self._next_eval_timestep is None:
            return
        while self._next_eval_timestep <= int(current_timestep):
            self._next_eval_timestep += self.eval_freq

    def __call__(self, model: Any) -> bool:
        if self._next_eval_timestep is None:
            return True
        current_timestep = int(model.num_timesteps)
        if current_timestep < self._next_eval_timestep:
            return True
        self._evaluate_and_log(model, timestep=current_timestep)
        self._advance_next_eval_timestep(current_timestep)
        return True

    def record_final_evaluation(
        self,
        model: Any,
        timestep: int | None = None,
    ) -> dict[str, float | int] | None:
        """Record a final evaluation point unless this timestep is already logged."""

        if self.eval_freq <= 0:
            return None
        final_timestep = int(model.num_timesteps if timestep is None else timestep)
        existing = self._logged_evaluation_at(final_timestep)
        if existing is not None:
            return existing
        summary = self._evaluate_and_log(model, timestep=final_timestep)
        self._advance_next_eval_timestep(final_timestep)
        return summary
